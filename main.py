import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse
from uuid import UUID, uuid4

import aiofiles
import aiohttp
import httpx
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field, model_validator

from agents import Agent, Runner, set_tracing_disabled

from AI.group_customer_analyze.create_report_group_c import (
    create_agent_products_state_analysis,
    create_agent_sectioned,
)
from AI.group_customer_analyze.many_customer import (
    get_exported_data_one_file,
    post_get_exported_data_one_file,
)
from AI.group_customer_analyze.preprocess_data_group_c import (
    get_cleaned_catalog,
    get_cleaned_customers,
    prepared_big_data,
    save_df,
)
from AI.group_customer_analyze.report_generate import get_report_generator

from AI.MCP_tools.get_SD_data import get_distributor_data, handle_distributor_data

from AI.single_customer_analyze.endpoint_helper import (
    ensure_sales_report,
    ensure_activities_report,
    ExportError,
)

from AI.utils import (
    TOPIC_CONFIG,
    _process_and_save_file_data,
    analyze_customer_orders_async,
    calculate_cost,
    combine_sections,
    create_response,
    extract_customer_id,
    generate_file_paths,
    get_logger,
    is_data_ready,
    process_fetch_results,
    raw_filename_for,
    read_dataframe_async,
    save_dataframe_async,
    validate_save_results,
)


load_dotenv()
set_tracing_disabled(True)

MCP_PORT = 8001 
MCP_LOCAL_URL = f"http://127.0.0.1:{MCP_PORT}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting MCP Server subprocess...")
    
    mcp_process = subprocess.Popen(
        [sys.executable, "-m", "AI.MCP_tools.List_of_mcp_tools"]
    )
    
    # Block FastAPI startup until MCP is responsive
    is_ready = False
    async with httpx.AsyncClient() as client:
        for attempt in range(15): # 15 seconds
            try:
                response = await client.get(MCP_LOCAL_URL)
                is_ready = True
                print("MCP Server is up and accepting connections.")
                break
            except httpx.RequestError:
                await asyncio.sleep(1)
                
    if not is_ready:
        print("CRITICAL: MCP Server failed to bind to port in time.")

    yield # Yield control back to FastAPI to handle web requests
    
    print("Shutting down MCP Server...")
    mcp_process.terminate()
    try:
        mcp_process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print("MCP Server didn't terminate in time, killing...")
        mcp_process.kill()

# Attach the lifespan to your app
app = FastAPI(lifespan=lifespan)


AllowedEntity = Literal["orders", "activities"]
origins = [
    "https://simply-depo-staging.web.app",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



logger1 = get_logger("logger1", "project_log.log", False)
logger2 = get_logger("logger2", "project_log_many.log", False)

executor = ThreadPoolExecutor()

class AllowedEntity(str, Enum):
    orders = "orders"
    activities = "activities"
 
 
class ReportRequest(BaseModel):
    entity: AllowedEntity
    force: bool = False
 
 
class ChatRequest(BaseModel):
    prompt: str
 
 
@app.post("/generate-reports/{customer_id}")
async def create_reports(customer_id: str, request: ReportRequest):
    """Explicitly (re)generates the report for one entity."""
    entity = request.entity
 
    try:
        if entity == AllowedEntity.orders:
            result = await ensure_sales_report(customer_id, force=request.force)
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Sales report generated successfully",
                    "report": result["full_report"],
                    "sections": result["sections"],
                },
            )
 
        # entity == AllowedEntity.activities
        result = await ensure_activities_report(customer_id, force=request.force)
        return JSONResponse(
            status_code=200,
            content={
                "message": "Report generated successfully",
                "report": result["full_report"],
                "sections": result["sections"],
            },
        )
 
    except ExportError as e:
        logger1.error(f"Export failed generating '{entity}' report for '{customer_id}': {e}")
        raise HTTPException(status_code=502, detail=f"We can't find any information about this customer. Please try again later.")
    except Exception as e:
        logger1.error(f"Error generating '{entity}' report for '{customer_id}': {e}")
        raise HTTPException(status_code=406, detail="Error generating report due to incorrect customer id")
 
 
@app.post("/Ask_ai")
async def ask_ai_endpoint(request: ChatRequest, customer_id: str = Query(...)):
    """
    Answers a free-text question about a customer. Makes sure both reports
    exist first — building only whichever ones are missing, concurrently —
    then streams the agent's answer back over SSE.
    """
    prompt = request.prompt
    pre_prompt = (
        "Use all the tools you need to answer, following the instructions "
        f"carefully. Answer the following questions: {prompt}"
    )
 
    try:
        await asyncio.gather(
            ensure_sales_report(customer_id, force=False),
            ensure_activities_report(customer_id, force=False),
        )
    except ExportError as e:
        logger1.error(f"Could not prepare statistics for customer '{customer_id}': {e}")
        raise HTTPException(status_code=502, detail=f"We can't find any information about this customer. Please try again later.")
    except Exception as e:
        logger1.error(f"Unexpected error preparing statistics for '{customer_id}': {e}")
        raise HTTPException(status_code=500, detail="Error preparing customer statistics")
 
    async def sse_generator():
        try:
            from AI.single_customer_analyze.Ask_ai_single_customer import create_Ask_ai_single_c_agent
            agent, session = await create_Ask_ai_single_c_agent(customer_id)
 
            runner = Runner.run_streamed(agent, input=pre_prompt, session=session)
 
            buffer = ""
            BUFFER_THRESHOLD = 50
 
            async for event in runner.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    buffer += event.data.delta
                    if len(buffer) >= BUFFER_THRESHOLD:
                        yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n"
                        buffer = ""
 
            if buffer:
                yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
 
            try:
                final_cost = calculate_cost(runner, model="gpt-4.1-mini")
            except Exception as cost_err:
                logger2.error(f"Cost calc error: {cost_err}")
                final_cost = "error_calculating"
 
            final_metadata = json.dumps({
                "type": "metadata",
                "cost": final_cost,
                "prompt": prompt,
                "status": "completed",
            })
            yield f"data: {final_metadata}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
 
        except Exception as e:
            logger2.error(f"Error executing LLM: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
 
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

from enum import Enum
class LogFile(str, Enum):
    """Enumeration for the allowed log file names."""
    project = "project_log.log"
    project_many = "project_log_many.log"


@app.get("/logs/last/{num_lines}", response_class=PlainTextResponse)
async def get_last_n_log_lines(
    num_lines: int,
    log_file: LogFile = Query(
        LogFile.project, 
        description="The log file to read from. Defaults to 'project_log.log'."
    )
    # --- END OF CHANGE ---
):
    """
    Return the last `num_lines` from the specified log file.

    If `log_file` is not provided, it defaults to 'project_log.log'.
    """
    
    # Get the actual filename string from the enum value
    LOG_FILE = log_file.value

    # Validate num_lines input
    if num_lines <= 0:
        raise HTTPException(status_code=422, detail="`num_lines` must be a positive integer.")
    
    try:
        with open(LOG_FILE, "r") as file:
            lines = file.readlines()

        if not lines:
            raise HTTPException(status_code=404, detail=f"Log file '{LOG_FILE}' is empty.")

        last_lines = lines[-num_lines:]
        return "".join(last_lines)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Log file '{LOG_FILE}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


async def clean_directories(customer_id: str):
    """
    Remove the data directory for the given customer_id if it exists.
    """
    data_folder = os.path.join("data", customer_id)
    # Only attempt removal if the folder exists
    
    if os.path.isdir(data_folder):
        loop = asyncio.get_event_loop()
        try:
            # Run shutil.rmtree in thread pool to avoid blocking
            await loop.run_in_executor(executor, shutil.rmtree, data_folder)
            logger1.info(f"Deleted user folder: {data_folder}")
        except Exception as e:
            logger1.error(f"Failed to delete {data_folder}: {e}")
    else:
        logger1.info(f"No directory to delete for user {customer_id}")

@app.get("/clean_chat/")
async def clean_chat(customer_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint to schedule directory cleanup for a given customer_id.

    This runs clean_directories in the background and immediately returns a success message.
    """
    background_tasks.add_task(clean_directories, customer_id)
    return {"response": "Chat is cleaned successfully"}


class ReportRequest(BaseModel):
    customer_ids: list[str]  # List of customer IDs
    entity: AllowedEntity    # Single entity, restricted to AllowedEntity values


class AI_Request(BaseModel):
    uuid: str
    prompt: str
    

#____

@app.post("/Ask_ai_many_customers")
async def Ask_ai_many_customers_endpoint(request: AI_Request = Body(...)):
    user_uuid = request.uuid
    prompt = request.prompt

    user_data_folder = os.path.join('data', user_uuid)
    
    if not os.path.exists(user_data_folder):
        logger2.warning(f"Attempt to access non-existent data folder: {user_uuid}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "detail": [
                    {
                        "loc": ["body", "uuid"],
                        "msg": "Invalid UUID provided. Data files not found.",
                        "type": "value_error"
                    }
                ]
            }
        )
    # ----------------------

    pre_prompt = f'Use all the tools you need to answer, following the instructions carefully. Answer the following questions: {prompt} ?'
    try:
        # Use AI function to get response
        #response = await Ask_ai_many_customers(prompt, user_uuid)
        from AI.group_customer_analyze.Ask_ai_many_customers import create_Ask_ai_many_c_agent
        agent, session = await create_Ask_ai_many_c_agent(user_uuid)

        runner = await Runner.run(
                agent, 
                input=pre_prompt,
                session=session
                )

        answer = runner.final_output 
        from pprint import pprint
        #print(answer)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": answer,
                "prompt" : prompt,
                "cost": 'cost'
            }
        )
        
    except Exception as e:
        logger2.error(f"Error executing LLM: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "detail": [
                    {
                        "loc": ["server", "llm_processing"],
                        "msg": "Internal server error while processing request",
                        "type": "internal_server_error"
                    }
                ]
            }
        )


from openai.types.responses import ResponseTextDeltaEvent\

@app.post("/st_Ask_ai_many_customers")
async def st_Ask_ai_many_customers_endpoint(request: AI_Request = Body(...)):
    user_uuid = request.uuid
    prompt = request.prompt

    user_data_folder = os.path.join('data', user_uuid)
    
    # 1. Validation
    if not os.path.exists(user_data_folder):
        logger2.warning(f"Attempt to access non-existent data folder: {user_uuid}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": [{"msg": "Invalid UUID provided.", "type": "value_error"}]}
        )

    pre_prompt = f'Use all the tools you need to answer, following the instructions carefully. Answer the following questions: {prompt} ?'

    # 2. Define the SSE Generator
    async def sse_generator():
        try:
            from AI.group_customer_analyze.Ask_ai_many_customers import create_Ask_ai_many_c_agent

            agent, session = await create_Ask_ai_many_c_agent(user_uuid)

            runner = Runner.run_streamed(
                agent, 
                input=pre_prompt,
                session=session
            )

            # --- BUFFER SETTINGS ---
            buffer = ""
            BUFFER_THRESHOLD = 50  # Send data only when have ~50 chars

            async for event in runner.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Add new token to buffer
                    buffer += event.data.delta

                    # Only yield if buffer is big enough
                    if len(buffer) >= BUFFER_THRESHOLD:
                        chunk_data = json.dumps({
                            "type": "token",
                            "content": buffer
                        })
                        #print(buffer)
                        yield f"data: {chunk_data}\n"
                        buffer = ""  # Reset buffer

            # 2. Flush remaining buffer
            # If the loop ends and there is text left in the buffer, send it now.
            if buffer:
                chunk_data = json.dumps({
                    "type": "token",
                    "content": buffer
                })
                
                yield f"data: {chunk_data}\n\n"

            # 3. Calculate Cost
            try:
                cost_stats = calculate_cost(runner, model="gpt-4.1-mini")
                
                # If calculate_cost returns a dict or object, format it for JSON
                final_cost = cost_stats 
            except Exception as cost_err:
                logger2.error(f"Cost calc error: {cost_err}")
                final_cost = "error_calculating"

            # 4. Send Final Metadata
            final_metadata = json.dumps({
                "type": "metadata",
                "cost": final_cost, 
                "prompt": prompt,
                "status": "completed"
            })
            yield f"data: {final_metadata}\n\n"
            
            # Send Done signal
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            logger2.error(f"Error executing LLM: {str(e)}")
            error_data = json.dumps({
                "type": "error",
                "content": str(e)
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        sse_generator(), 
        media_type="text/event-stream"
    )

# Define an Enum for all allowed report types
class ReportType(str, Enum):
    FULL_REPORT = "full_report"
    KEY_METRICS = "key_metrics"
    DISCOUNT_DISTRIBUTION = "discount_distribution"
    OVERALL_TOTAL_SALES = "overall_total_sales_by_payment_and_delivery_status"
    PAYMENT_STATUS = "payment_status_analysis"
    DELIVERY_FEES = "delivery_fees_analysis"
    FULFILLMENT = "fulfillment_analysis"
    SALES_PERFORMANCE = "sales_performance_overview"
    PRODUCT_PER_STATE_ANALYSIS = "product_per_state_analysis"
    TOP_WORST_PRODUCTS = "top_worst_selling_product"
    KEY_METRICS_ORDERS = "key_metrics_report"
    SALES_PERFORMANCE_ORDERS = "sales_performance_report"
    DISCOUNT_DISTRIBUTION_ORDERS = "discount_report"
    PAYMENT_STATUS_ORDERS = "payment_status_report"
    FULFILLMENT_ORDERS = "fulfillment_report"
    SALES_TRENDS = "sales_trends_report"
    REVENUE_PROFITABILITY = "revenue_profitability"
    INVENTORY_HEALTH_FULFILLMENT_EFFICIENCY = "inventory_fulfillment"
    CROSS_SELL_BUNDLE_ACTIONABILITY = "cross_sell_bundling"
    BUYER_HEALTH = "buyer_health"
    TOP_PERFORMER_DEEP_DIVE = "top_performers"

class AnalysisIdType(str, Enum):
    CUSTOMER = "customer"
    ORDER = "order"
    CATALOG = "catalog"
    
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

class ReportRequest(BaseModel):
    ids: List[str] = Field(
        default_factory=list,
        title="IDs to analyze",
        description=(
            "The list of IDs to analyze. What they refer to is determined "
            "by `id_type` (e.g. customer IDs, order IDs)."
        ),
    )
    id_type: AnalysisIdType = Field(
        default=AnalysisIdType.CUSTOMER,
        title="ID Type",
        description="What kind of IDs are in `ids` - selects which data-fetch strategy is used.",
    )
    report_type: "ReportType" = Field(
        default=ReportType.FULL_REPORT,  # keep your existing default/enum here
        title="Report Type",
        description="Specify which report section to generate. Defaults to the full report.",
    )
    uuid: Optional[str] = None
    entity: "AllowedEntity"  # unchanged - existing field, orthogonal to id_type
 
    # --- Backward compatibility only ---
    customer_ids: Optional[List[str]] = Field(
        default=None,
        deprecated=True,
        description="Deprecated. Use `ids` together with id_type='customer' instead.",
    )
 
    @model_validator(mode="before")
    @classmethod
    def _map_legacy_customer_ids(cls, data: dict) -> dict:
        # Callers still on the old payload shape send `customer_ids` only.
        # Map it onto the new fields so they keep working unchanged.
        if isinstance(data, dict) and data.get("customer_ids") and not data.get("ids"):
            data["ids"] = data["customer_ids"]
            data.setdefault("id_type", AnalysisIdType.CUSTOMER.value)
        return data
 
    @model_validator(mode="after")
    def _ids_required(self) -> "ReportRequest":
        if not self.ids:
            raise ValueError("`ids` must not be empty.")
        return self




def _sync_comparison_logic(df_1: pd.DataFrame, 
                           df_2: pd.DataFrame, 
                           customer_id_s):
    """
    Internal synchronous function to perform blocking Pandas operations.
    Returns a dictionary with the results.
    
    Note:
    - df_1 is assumed to be the 'orders' DataFrame.
    - df_2 is assumed to be the 'customers' DataFrame.
    """
    
    # 1. Check for required columns
    required_cols_df1 = ['customerId']
    required_cols_df2 = ['combinedid', 'name']
    
    if not all(col in df_1.columns for col in required_cols_df1):
        return {"error": f"orders is missing required columns. Needed: {required_cols_df1}"}
    if not all(col in df_2.columns for col in required_cols_df2):
        return {"error": f"customers is missing required columns. Needed: {required_cols_df2}"}

    # 2. Get unique IDs from both DataFrames.
    ids_in_df2 = set(df_2['combinedid'])
    ids_in_df1 = set(df_1['customerId'])
    ids_from_list = set(customer_id_s)

    # --- "Empty customers" (names from df_2) ---
    
    # Find IDs that are in df_2 (customers) but not in df_1 (orders)
    missing_in_df1_ids = ids_in_df2 - ids_in_df1
    
    empty_customer_names = []
    if missing_in_df1_ids:
        # Filter df_2 to get rows with these IDs
        missing_df = df_2[df_2['combinedid'].isin(missing_in_df1_ids)]
        # Get unique names
        empty_customer_names = missing_df['name'].unique().tolist()

    # --- "Invalid IDs" (IDs from the input list) ---
    
    # Find IDs from the list that are not in df_2 (customers)
    missing_in_df2_ids = ids_from_list - ids_in_df2
    invalid_ids_list = list(missing_in_df2_ids)

    # --- Format the result ---
    result = {
        "empty_customers": empty_customer_names,
        "invalid_ids": invalid_ids_list
    }
    
    return result

async def check_customer_ids(df_1: pd.DataFrame, 
                           df_2: pd.DataFrame, 
                           customer_id_s):
    """
    Asynchronous wrapper for checking IDs in DataFrames.
    
    Note: df_1 represents orders, and df_2 represents customers.
    
    Executes blocking Pandas logic in a separate thread and
    returns a dictionary (dict) ready for JSON serialization.
    """
    # Running the heavy synchronous function in a separate thread
    result_dict = await asyncio.to_thread(_sync_comparison_logic, df_1, df_2, customer_id_s)
    
    return result_dict


def _sync_process_merge_logic(orders_df, customer_df, products_df):
    """
    CPU-bound synchronous logic. 
    Runs in a separate thread to avoid blocking the API.
    """
    # 1. Clean Column Names (List comprehension is slightly faster than str.replace for headers)
    orders_df.columns = [c.strip().replace('\ufeff', '') for c in orders_df.columns]
    customer_df.columns = [c.strip().replace('\ufeff', '') for c in customer_df.columns]

    # 2. Merge Orders with Customers
    merged = orders_df.merge(
        customer_df[['combinedid', 'displayedName']],
        left_on='customerId',
        right_on='combinedid',
        how='left'
    )

    # 3. Rename and Drop (Chained for efficiency)
    # Rename 'displayedName' -> 'customer_name', 'customerId' -> 'id'
    orders_final = merged.drop(columns=['combinedid_y']).rename(
        columns={'displayedName': 'customer_name', 'combinedid_x': 'id'}
    )

    # 4. Map Customers to Products
    order_to_customer_map = orders_final.set_index('id')['customer_name'].to_dict()
    products_df['customer_name'] = products_df['orderId'].map(order_to_customer_map)
    
    # Remove products that didn't match a valid order/customer (orphans)
    products_final = products_df.dropna(subset=['customer_name']).drop(columns=['id'], errors='ignore')

    return orders_final, products_final

from AI.group_customer_analyze.fetch_data import get_fetch_strategy

@app.post("/generate-reports-group")
async def create_group_reports_new(request: ReportRequest = Body(...)):
    """
    Generate group reports for multiple customers.
    Optimized to only run analysis for the requested report_type.
    """
    try:
        start_time = time.perf_counter()
        ids = request.ids
        id_type = request.id_type
        report_type = request.report_type
        distributor_id = request.uuid
 
        strategy = get_fetch_strategy(id_type, distributor_id)
        await strategy.validate_ids(ids)
        uuid = request.uuid or str(uuid4())

        # Create directory structure
        user_folder = os.path.join('data', uuid, 'work_data_folder')
        await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)

        # File names
        file_paths = {
            entity: os.path.join(user_folder, raw_filename_for(entity))
            for entity in strategy.entities
        }
        print(f"Step 0 - Starting data fetch for id {uuid}: {time.perf_counter() - start_time:.2f}s")
        try:
            file_paths = await strategy.fetch_and_write(ids, user_folder, distributor_id)
            #print(file_paths)
        except Exception as e:
            error_message = str(e)
            logger2.error(f"Data processing/fetching error ({id_type.value}): {error_message}")
            if "URL component 'query' too long" in error_message:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "Status": "Failed",
                        "Reason": f"Too many {id_type.value} IDs provided. The request URL exceeded the length limit."
                    }
                )
 
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "detail": [
                        {
                            "loc": ["server", "data_fetching"],
                            "msg": f"Critical error during data retrieval: {error_message}",
                            "type": "data_fetch_error"
                        }
                    ]
                }
            )
 
        print(f"Step 1 - Data fetch completed: {time.perf_counter() - start_time:.2f}s")


        if id_type == AnalysisIdType.ORDER or id_type == AnalysisIdType.CUSTOMER:
            if "customer" in file_paths:
                try:
                    check_if_customer_id_correct = pd.read_csv(file_paths["customer"])
                    if check_if_customer_id_correct.empty:
                        logger2.warning(f"Customer data empty for {id_type.value} IDs: {ids}")
                        return JSONResponse(
                            status_code=status.HTTP_404_NOT_FOUND,
                            content={
                                "Status": "Failed",
                                "Reason": f"Incorrect {id_type.value} IDs provided. No customer data found.",
                                "uuid": uuid
                            }
                        )
                except Exception as e:
                    logger2.warning(f"Can not check if {id_type.value} ids are valid: {e}")
            # Preprocess data
            cleaned_paths: Dict[str, str] = {}
            #orders_check_path = file_paths.get("orders") #TODO check orders and customer report_type for this commented
            if strategy.cleanup_entities and all(e in file_paths for e in strategy.cleanup_entities):
                orders_entity, products_entity = strategy.cleanup_entities
                full_cleaned_orders, full_cleaned_products = await prepared_big_data(
                    str(file_paths[orders_entity]),
                    str(file_paths[products_entity]),
                )
                if full_cleaned_orders.empty:
                    logger2.warning(
                        f"prepared_big_data returned an empty orders frame for "
                        f"id_type={id_type.value}, uuid={uuid}"
                    )
                print(f"Step 2 - Data preprocessing completed: {time.perf_counter() - start_time:.2f}s")
                cleaned_orders_path = os.path.join(user_folder, 'cleaned_real_big_orders.csv')
                cleaned_products_path = os.path.join(user_folder, 'cleaned_real_big_products.csv')
                await asyncio.gather(
                    save_df(full_cleaned_orders, str(cleaned_orders_path)),
                    save_df(full_cleaned_products, str(cleaned_products_path))
                )
                cleaned_paths[orders_entity] = cleaned_orders_path
                cleaned_paths[products_entity] = cleaned_products_path
                orders_check_path = cleaned_orders_path
            else:
                # NEXT STEP: no cleanup_entities configured for this strategy -
                # raw fetched files would be used downstream as-is. Hasn't come
                # up yet since both current strategies use the same
                # ("orders", "order_products") pair; flagging here for when it
                # does.
                logger2.info(
                    f"No cleanup step configured/possible for id_type={id_type.value} "
                    f"(entities={strategy.entities}); skipping prepared_big_data."
                )
    except Exception as e:
        logger2.error(f"Data processing error: {e}")

    if id_type == AnalysisIdType.ORDER or id_type == AnalysisIdType.CUSTOMER:
        try:
            # Check if customers ids correct but no data in orders
            try:
                check_if_orders_has_data = pd.read_csv(cleaned_orders_path)
                #print(check_if_orders_has_data.head(3))
                if check_if_orders_has_data.empty:
                    logger2.info("Orders data is empty after processing.")
                    message = """The report cannot be generated based on empty data (No valid orders found). \n
You can create a new order to start analyzing your data - check this guide: [How to Create and Process a New Direct Order](https://scribehow.com/viewer/How_To_Create_And_Process_A_New_Direct_Order__XOZEjF9KTJ2B_C4G32afpQ?referrer=documents)\n
and ask AI agent for help with platform navigation and order creation, or you can clarify with our specialist: [Schedule a Meeting](https://meetings.hubspot.com/john-vasylets/customers)\n
"""
                    return JSONResponse(
                        status_code=status.HTTP_404_NOT_FOUND,
                        content={
                        "Status": "Empty Data",
                        "Reason": message,
                        "incorrect_uuid": request.customer_ids,
                        "uuid": str(uuid)
                    }
                    )
            except Exception as e:
                logger2.warning(f"Can not check if customers orders are empty: {e}")
            # Read dataframes concurrently
            if id_type == AnalysisIdType.CUSTOMER:
                try:
                    read_paths = {**file_paths, **cleaned_paths}
                    entity_names = list(read_paths.keys())
                    raw_dataframes = await asyncio.gather(*(
                        read_dataframe_async(str(read_paths[entity])) for entity in entity_names
                    ))
                    dataframes_by_entity = dict(zip(entity_names, raw_dataframes))
        
                    orders_df = dataframes_by_entity.get("orders", pd.DataFrame())
                    products_df = dataframes_by_entity.get("order_products", pd.DataFrame())
                    customer_df = dataframes_by_entity.get("customer", pd.DataFrame())
    
                    # Clean column names
                    merged_orders, products_df = await asyncio.to_thread(
                        _sync_process_merge_logic, 
                        orders_df, 
                        customer_df, 
                        products_df
                    )
        
                    print(f"Step 3 - Data cleaning completed: {time.perf_counter() - start_time:.2f}s")
                except Exception as e:
                    orders_df, products_df, customer_df = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
                    logger2.error(f"main read_dataframe_async error: {e}")
        
        except Exception as e:
            logger2.error(f"Data cleaning error: {e}")
    
            check_empty = False
        
            try:
                with open(cleaned_orders_path, 'r') as f:
                    content = f.read(5).strip()  # Read a small snippet
                    if not content:
                        logger2.info(f"File contains only whitespace or is effectively empty.")
                        check_empty = True
    
                if check_empty:
                    logger2.error("Empty orders file!")
                    #incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                    return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                            "Status": "The report cannot be generated based on empty data.",
                            "incorrect_ids" : 'incorrect_ids',
                            "sections": {"full_report":"The report cannot be generated based on empty data."},
                            "report": 'The report cannot be generated based on empty data.',
                            "uuid": uuid
                        })
            except Exception as e:
                logger2.error(e)
        
                #incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                # Create and return response
                return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "Status": "The report cannot be generated due to incorrect data",
                            "incorrect_ids" : 'incorrect_ids',
                            "sections": 'sectioned_report',
                            "report": 'full_report',
                            "uuid": uuid
                        })
    else:
        # cleaning catalog case #TODO
        print(f"Step 2 - Data preprocessing completed: {time.perf_counter() - start_time:.2f}s")
        raw_orders_path = os.path.join("data", distributor_id,"work_data_folder" ,"raw_file_orders.csv")
        raw_products_path = os.path.join("data", distributor_id, "work_data_folder", "raw_file_order_products.csv")
        full_cleaned_orders, full_cleaned_products = await prepared_big_data(
                str(raw_orders_path), 
                str(raw_products_path)
        )
        
        catalog_path = os.path.join("data", distributor_id, "work_data_folder", "raw_file_catalog.csv")
        catalog_df, catalog_path = await get_cleaned_catalog(str(catalog_path))

        print(f"Step 2.1 - Catalog preprocessing completed: {time.perf_counter() - start_time:.2f}s")
        # Save cleaned data concurrently
        cleaned_orders_path =  os.path.join('data', distributor_id,  'cleaned_orders.csv') 
        cleaned_products_path =  os.path.join('data', distributor_id,  'cleaned_products.csv')
        cleaned_catalog_path = os.path.join('data', distributor_id,  'cleaned_catalog.csv')

        await asyncio.gather(
            save_df(full_cleaned_orders, str(cleaned_orders_path)),
            save_df(full_cleaned_products, str(cleaned_products_path)),
            save_df(catalog_df, str(cleaned_catalog_path))
        )
        # check if orders empty - custom output how to create a new order if there is no data to analyze
        if full_cleaned_orders.empty:
            logger2.info("Orders data is empty after processing (no data rows found).")
            message = """The report cannot be generated based on empty data (No valid orders found). \n
You can create a new order to start analyzing your data - check this guide: [How to Create and Process a New Direct Order](https://scribehow.com/viewer/How_To_Create_And_Process_A_New_Direct_Order__XOZEjF9KTJ2B_C4G32afpQ?referrer=documents)\n
and ask AI agent for help with platform navigation and order creation, or you can clarify with our specialist: [Schedule a Meeting](https://meetings.hubspot.com/john-vasylets/customers)\n
"""
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "empty_data",
                    "message": message,
                    "distributor_id": distributor_id # Passing back the ID as requested
                }
            )
    

    try:
        report_generator = get_report_generator(id_type)
        if id_type == AnalysisIdType.CUSTOMER:
            sections, full_report = await report_generator.generate(
                report_type, merged_orders, products_df, customer_df, uuid, start_time
            )
        elif id_type == AnalysisIdType.ORDER:
            sections, full_report = await report_generator.generate(
                report_type, cleaned_orders_path, cleaned_products_path, uuid, start_time
            )
            #print(sections.get(report_type, "No section generated for this report type."))
        else:  # id_type == AnalysisIdType.CATALOG
            selected_catalog_path = os.path.join('data', distributor_id, 'work_data_folder', 'raw_file_selected_catalog.csv')
            sections, full_report = await report_generator.generate(
                report_type, selected_catalog_path, cleaned_orders_path, cleaned_products_path, cleaned_catalog_path, uuid, start_time
            )
            #print(sections.get(report_type, "No section generated for this report type."))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "incorrect_ids": 'incorrect_ids',
                "sections": sections,
                "report": full_report,
                "uuid": uuid
            }
        )
    except NotImplementedError as e:
        logger2.warning(f"Report generation not implemented: {e}")
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content={
                "detail": [
                    {"loc": ["server", "report_generation"], "msg": str(e), "type": "not_implemented"}
                ]
            }
        )
    except Exception as e:
        logger2.error(f"Report generation failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {"loc": ["server", "report_generation"], "msg": "Internal server error during report generation", "type": "internal_server_error"}
                ]
            }
        )


# Start of MCP end point

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, get_args
from uuid import UUID

from fastapi import Body, HTTPException, status
from pydantic import BaseModel, Field

from AI.utils import (
    EMPTY_ORDERS_MESSAGE,
    FULL_REPORT,
    ReportContext,
    ReportRunner,
    batch_process_runner,
    build_sales_pipeline,
    error_response,
    files_are_fresh,
    lock_for,
    module_runner,
    ok_response,
    sync_raw_data,
)

import asyncio


RAW_FILES: Dict[str, str] = {
    "customers": "raw_file_customers.csv",
    "orders": "raw_file_orders.csv",
    "order_products": "raw_file_order_products.csv",
    "catalog": "raw_file_catalog.csv",
    "activities": "raw_file_activities.csv",
    "forms": "raw_file_forms.csv", #TODO
    "tasks": "raw_file_tasks.csv",
    "notes": "raw_file_notes.csv",
}

SALES_DATASETS: Tuple[str, ...] = ("customers", "orders", "order_products", "catalog")


@dataclass(frozen=True)
class EntitySpec:
    datasets: Tuple[str, ...]            # raw datasets to fetch + freshness-check
    runner: ReportRunner                 # how the report is actually produced
    needs_sales_pipeline: bool = False   # build the cleaned_* CSVs first
    empty_orders_blocks: bool = False    # return the "no orders" 404 when orders are empty

    def __post_init__(self) -> None:
        unknown = set(self.datasets) - set(RAW_FILES)
        if unknown:
            raise ValueError(f"EntitySpec references unknown datasets: {sorted(unknown)}")
        if self.needs_sales_pipeline:
            missing = set(SALES_DATASETS) - set(self.datasets)
            if missing:
                # Fails at import time, not on a request at 3am.
                raise ValueError(
                    f"needs_sales_pipeline=True also requires datasets {sorted(missing)}"
                )

    @property
    def required_files(self) -> Tuple[str, ...]:
        return tuple(RAW_FILES[d] for d in self.datasets)


def sales_entity(agent_name: Optional[str] = None) -> EntitySpec:
    """Shorthand for the orders / catalog / customers family."""
    return EntitySpec(
        datasets=SALES_DATASETS,
        runner=batch_process_runner(agent_name),
        needs_sales_pipeline=True,
        empty_orders_blocks=True,
    )


ENTITY_SPECS: Dict[str, EntitySpec] = {
    # --- batch / topic-analysis family: all four cleaned CSVs, one agent each ---
    "orders":     sales_entity(),
    "catalog":    sales_entity(),
    "customers":  sales_entity(),

    # --- standalone-module family: own run_report, own raw data ---
    "activities": EntitySpec(
        datasets=("activities", "orders"),
        runner=module_runner("AI.activities"),
    ),
    "forms": EntitySpec(
        datasets=("forms",),
        runner=module_runner(module_path="AI.activities.forms.forms_ai", func_name='analyze_form'),
    ),
    "tasks": EntitySpec(
        datasets=("tasks",),
        runner=module_runner("AI.activities"),
    ),
    "notes": EntitySpec(
        datasets=("notes",),
        runner=module_runner("AI.activities")
    )
}


EntityName = Literal[
    "orders",
    "catalog",
    "customers",
    "activities",
    "forms",
    "tasks",
    "notes",
]

_declared = set(get_args(EntityName))
if _declared != set(ENTITY_SPECS):
    raise RuntimeError(
        "EntityName and ENTITY_SPECS have drifted. "
        f"Only in EntityName: {sorted(_declared - set(ENTITY_SPECS))}; "
        f"only in ENTITY_SPECS: {sorted(set(ENTITY_SPECS) - _declared)}"
    )


class MCPRequest(BaseModel):
    distributor_id: UUID
    entity: EntityName = Field(
        ...,
        title="Entity",
        description="Which analysis pipeline to run.",
    )
    report_type: str = Field(
        default=FULL_REPORT,
        title="Report Type",
        description="Which report section to generate. Defaults to the full report.",
    )
    uuid: Optional[str] = None

    # --- Backward compatibility only ---
    customer_ids: Optional[List[str]] = Field(
        default=None,
        deprecated=True,
        description="Deprecated. Use `ids` together with id_type='customer' instead.",
    )


@app.post("/generate-mcp-reports")
async def create_mcp_reports(request: MCPRequest = Body(...)):
    t0 = time.perf_counter()
    distributor_id = str(request.distributor_id)
    entity = request.entity
    report_type = request.report_type

    try:
        # ---- validate -------------------------------------------------
        spec = ENTITY_SPECS.get(entity)
        if spec is None:  # unreachable through the Literal, but keeps this honest
            return error_response(
                status.HTTP_400_BAD_REQUEST, "unsupported_entity",
                f"No report pipeline registered for entity '{entity}'.",
                distributor_id, allowed_entities=sorted(ENTITY_SPECS),
            )

        allowed_reports = TOPIC_CONFIG.get(entity, [])
        if report_type not in allowed_reports:
            return error_response(
                status.HTTP_400_BAD_REQUEST, "invalid_configuration",
                f"The report type '{report_type}' is not valid for the '{entity}' entity.",
                distributor_id, allowed_reports=allowed_reports,
            )

        # ---- sync (only this entity's datasets; one syncer per distributor) ----
        async with lock_for(f"{distributor_id}:{entity}"):
            fresh = await asyncio.to_thread(files_are_fresh, distributor_id, spec.required_files)
            if not fresh:
                await sync_raw_data(distributor_id, spec.datasets, t0)

        # ---- prepare --------------------------------------------------
        ctx = ReportContext(
            distributor_id=distributor_id,
            entity=entity,
            report_type=report_type,
            request=request,
        )

        if spec.needs_sales_pipeline:
            try:
                ctx.paths, orders_empty = await build_sales_pipeline(
                    distributor_id, RAW_FILES, t0
                )
            except Exception as exc:
                logger2.exception("Data processing error: %s", exc)
                return error_response(
                    status.HTTP_502_BAD_GATEWAY, "data_preprocessing_failed",
                    "Failed to fetch or preprocess distributor data from the upstream "
                    "source. Report generation aborted.",
                    distributor_id,
                )

            if orders_empty and spec.empty_orders_blocks:
                logger2.info("Orders data is empty after processing.")
                return error_response(
                    status.HTTP_404_NOT_FOUND, "empty_data",
                    EMPTY_ORDERS_MESSAGE, distributor_id,
                )

        # ---- run ------------------------------------------------------
        try:
            payload = await spec.runner(ctx)
        except HTTPException:
            raise
        except Exception as exc:
            # Original code logged here and fell through to
            # content={"sections": clean_sections} on an unbound name ->
            # UnboundLocalError -> opaque 500 that hid the real cause.
            logger2.exception("Report generation failed for %s/%s", entity, distributor_id)
            return error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "report_generation_failed",
                str(exc), distributor_id,
            )

        if isinstance(payload.sections, dict) and "error" in payload.sections:
            logger2.error("Report generation aborted: %s", payload.sections["error"])
            return error_response(
                status.HTTP_417_EXPECTATION_FAILED, "report_generation_error",
                payload.sections["error"], distributor_id,
            )

        logger2.info("Step 4 - report done (%s/%s): %.2fs",
                     entity, report_type, time.perf_counter() - t0)
        return ok_response(payload, distributor_id)

    except HTTPException:
        # MUST precede `except Exception`, otherwise every deliberate 404/400
        # raised above is rewritten as "Internal server error during processing".
        raise
    except ValueError as ve:
        logger2.error("Validation/Logic Error in endpoint: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger2.exception("Critical System Error in endpoint")
        raise HTTPException(status_code=500, detail="Internal server error during processing.")




class ChatRequestMCP(BaseModel):
    message: str
    distributor_id: UUID

from fastapi import Request

@app.post("/chat_mcp")
async def chat_endpoint(request: ChatRequestMCP, req: Request):
    from AI.MCP_tools.run_mcp import agent_stream_generator
    logger2.info(f"Received chat request for distributor {request.distributor_id} : {request.message}")
    return StreamingResponse(
        agent_stream_generator(request, req),
        media_type="text/event-stream"
    )

def sse_msg(event_type: str, content: any):
    """Formats data as a Server-Sent Event."""
    return f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"

# --- The Generator Logic ---
async def analyze_state_stream_generator(request):
    try:
        start_time = time.perf_counter()
        customer_ids = request.customer_ids
        uuid = request.uuid or str(uuid4())

        # 1. Setup Directories
        user_folder = os.path.join('data', uuid, 'work_data_folder')
        await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)

        yield sse_msg("status", f"Starting analysis for ID {uuid}...")

        # 2. Define Paths
        filename_orders = 'one_file_orders.csv'
        filename_products = 'one_file_products.csv'
        filename_customers = 'one_file_customers.csv'

        file_path_orders = os.path.join(user_folder, filename_orders)
        file_path_products = os.path.join(user_folder, filename_products)
        file_path_customers = os.path.join(user_folder, filename_customers)

        # 3. Fetch Data
        try:
            yield sse_msg("status", "Fetching customer data...")
            
            entities_orders = ["orders"]
            entities_products = ["order_products"]
            entities_customers = ["customer"]

            # Fetch concurrently
            result_1, result_2, result_3 = await asyncio.gather(
                post_get_exported_data_one_file(customer_ids, entities_orders),
                post_get_exported_data_one_file(customer_ids, entities_products),
                post_get_exported_data_one_file(customer_ids, entities_customers)
            )
        except Exception as e:
            error_message = str(e)
            logger2.error(f"Data fetching error: {error_message}")
            
            # Streaming Error Handling: We cannot return a 413/502 status code here 
            # because the stream has started. We send an error event instead.
            if "URL component 'query' too long" in error_message:
                yield sse_msg("error", {
                    "code": 413,
                    "reason": "Too many customer IDs provided."
                })
            else:
                yield sse_msg("error", {
                    "code": 502,
                    "msg": f"Critical error during data retrieval: {error_message}"
                })
            return  # Stop the generator

        # 4. Save Raw Data
        await asyncio.gather(
            _process_and_save_file_data(result_1, file_path_orders),
            _process_and_save_file_data(result_2, file_path_products),
            _process_and_save_file_data(result_3, file_path_customers)
        )

        # 5. Validation: Check Customers
        try:
            check_customers = pd.read_csv(file_path_customers)
            if check_customers.empty:
                yield sse_msg("error", {
                    "code": 404, 
                    "reason": "Incorrect Customer IDs provided. No customer data found."
                })
                return
        except Exception as e:
            logger2.warning(f"Validation warning: {e}")

        # 6. Preprocess Data
        yield sse_msg("status", "Preprocessing data...")
        
        full_cleaned_orders, full_cleaned_products = await prepared_big_data(
            str(file_path_orders), 
            str(file_path_products)
        )

        cleaned_orders_path = os.path.join(user_folder, 'cleaned_real_big_orders.csv')
        cleaned_products_path = os.path.join(user_folder, 'cleaned_real_big_products.csv')

        await asyncio.gather(
            save_df(full_cleaned_orders, str(cleaned_orders_path)),
            save_df(full_cleaned_products, str(cleaned_products_path))
        )

        # 7. Validation: Check Orders
        try:
            check_orders = pd.read_csv(cleaned_orders_path)
            if check_orders.empty:
                yield sse_msg("error", {
                    "code": 404,
                    "reason": "The report cannot be generated based on empty data."
                })
                return
        except Exception:
            pass

        # 8. Read and Merge Dataframes
        try:
            orders_df, products_df, customer_df = await asyncio.gather(
                read_dataframe_async(str(cleaned_orders_path)),
                read_dataframe_async(str(cleaned_products_path)),
                read_dataframe_async(str(file_path_customers))
            )

            merged_orders, products_df = await asyncio.to_thread(
                _sync_process_merge_logic, 
                orders_df, customer_df, products_df
            )
        except Exception as e:
            logger2.error(f"Merge error: {e}")
            yield sse_msg("error", "Failed to merge dataframes.")
            return

        # 9. Processing for AI
        from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
        from AI.group_customer_analyze.orders_state import async_generate_report, async_process_data

        yield sse_msg("status", "Running analysis algorithms...")
        
        # Save debug CSVs
        products_df['product_variant'] = products_df['name'].astype(str) + ' - ' + products_df['sku'].astype(str)
        await asyncio.gather(
            asyncio.to_thread(merged_orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
            asyncio.to_thread(products_df.to_csv, f'data/{uuid}/pproducts.csv', index=False)
        )

        await async_process_data(uuid)
        await async_generate_report(uuid)

        # 10. AI Agent Streaming
        yield sse_msg("status", "Streaming AI analysis...")
        
        agent = await create_agent_products_state_analysis(uuid)
        answer_buffer = ""      # Stores full answer
        stream_buffer = ""      # Stores pending chunk to send
        BUFFER_THRESHOLD = 50

        try:
            # NOTE: run_streamed is synchronous wrapper or async depending on implementation
            # Assuming openai-agents-python standard usage:
            runner = Runner.run_streamed(
                agent, 
                input="Based on the data return response" 
                # session=session (if needed)
            )

            async for event in runner.stream_events():
             if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                
                # Update buffers
                answer_buffer += token
                stream_buffer += token

                # FLUSH: Only yield if buffer exceeds threshold
                if len(stream_buffer) >= BUFFER_THRESHOLD:
                    yield sse_msg("token", stream_buffer)
                    stream_buffer = ""  # Reset small buffer

        except Exception as e:
            logger2.error(f"AI Agent Error: {e}")
            yield sse_msg("warning", f"AI analysis failed: {str(e)}")
            final_ai_answer = "AI Analysis failed."

        # 11. Finalize and Send Result
        yield sse_msg("status", "Finalizing report...")
        if stream_buffer:
            yield sse_msg("token", stream_buffer)

        # --- Cost Calculation ---
        cost_info = calculate_cost(runner, model="gpt-4o-mini")
        yield sse_msg("metadata", {"cost": cost_info})
        incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
        full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)

        # Save report to file
        async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
            await f.write(full_report.get('full_report', ''))

        final_payload = {
            "incorrect_ids": incorrect_ids,
            "sections": {'product_per_state_analysis': answer_buffer},
            "report": full_report.get('full_report'),
            "uuid": uuid
        }
        
        yield sse_msg("result", final_payload)

    except Exception as e:
        logger2.critical(f"Unhandled Stream Error: {e}")
        yield sse_msg("error", f"Critical system error: {str(e)}")


# --- The Endpoint ---
@app.post("/state-analysis")
async def product_per_state_analysis_func(request: ReportRequest = Body(...)):
    """
    Returns a StreamingResponse.
    The client will receive HTTP 200 immediately, followed by SSE events:
    - 'status': Progress updates
    - 'token': AI text generation tokens
    - 'error': If something goes wrong
    - 'result': The final JSON payload
    """
    return StreamingResponse(
        analyze_state_stream_generator(request),
        media_type="text/event-stream"
    )




if __name__ == '__main__':
    import uvicorn
    from AI.MCP_tools.List_of_mcp_tools import mcp
    mcp.mount()
    uvicorn.run(app, port=8000, host='0.0.0.0')