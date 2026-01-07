from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi import  Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi import Body

import json
import pandas as pd
import os
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from pathlib import Path
from urllib.parse import urlparse

#from db.database_reports import generate_customer_reports 
from agents import Agent, Runner, set_tracing_disabled
from AI.single_customer_analyze.Activities_AI import process_ai_activities_request
from AI.single_customer_analyze.Order_analytics import generate_sales_report
from AI.single_customer_analyze.Activities_analytics import analyze_activities
from AI.single_customer_analyze.Tasks_analytics import tasks_report
from AI.single_customer_analyze.Notes_analytics import notes_report
from AI.group_customer_analyze.create_report_group_c import create_agent_sectioned, create_agent_products_state_analysis

from pydantic import BaseModel, Field
from typing import List
from enum import Enum

from AI.group_customer_analyze.many_customer import get_exported_data_one_file, post_get_exported_data_one_file
from AI.utils import (
    _process_and_save_file_data, read_dataframe_async, save_dataframe_async, combine_sections
)

from AI.group_customer_analyze.preprocess_data_group_c import (
    save_df, prepared_big_data
)
from AI.utils import get_logger, extract_customer_id, process_fetch_results, validate_save_results, generate_file_paths, create_response, \
    analyze_customer_orders_async, calculate_cost

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

import requests
from functools import partial
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
from typing import Literal
import shutil
from typing import List
from concurrent.futures import ProcessPoolExecutor

load_dotenv()
set_tracing_disabled(True)

app = FastAPI()
AllowedEntity = Literal["orders", "activities"]


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


def get_exported_data(customer_id, entity):
    """
    Connects to the API endpoint to export customer profile data and returns the file content.

    Args:
        customer_id (str): The customer's UUID.
        entity (str): The type of data to export ('orders', 'order_products', 'notes', 'tasks', 'customer', or 'activities').

    Returns:
        bytes: The content of the exported data file.

    Raises:
        ValueError: If the entity value is invalid.
        Exception: If the API request or file download fails, including detailed error information.
    """
    # Define allowed entity values
    allowed_entities = ["orders", "order_products", 'customer', "notes", "tasks", "activities"]
    if entity not in allowed_entities:
        raise ValueError(f"Invalid entity: {entity}. Must be one of {allowed_entities}")

    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
    url = SD_API_URL

    # Query parameters
    params = {
        "customer_id": customer_id,
        "entity": entity
    }

    x_api_key = os.getenv('X_API_KEY')
    if not x_api_key:
        raise Exception("X_API_KEY environment variable is not set")
    # Authentication header
    headers = {
        "x-api-key": x_api_key
    }

    def _get_error_detail(resp: requests.Response) -> str:
        """
        Extracts detailed error message from the response, preferring JSON fields.
        """
        try:
            err_json = resp.json()
            for key in ("error", "message", "detail"):  # common keys
                if key in err_json:
                    return f"{key}: {err_json[key]}"
            return str(err_json)
        except ValueError:
            return resp.text or "<no response body>"

    # Make the GET request to the API
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print("Content-Type:", response.headers.get('Content-Type', 'Not specified'))
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    # Handle the response
    if response.status_code == 200:
        # Parse the response as JSON
        try:
            data = response.json()
            print("Parsed JSON data:", data)
            exported_url = data.get("fileUrl")
            if not exported_url:
                raise Exception("No 'fileUrl' found in response")
        except ValueError:
            detail = _get_error_detail(response)
            raise Exception(f"Response is not valid JSON — {detail}")

        # Download the file from the exported URL
        try:
            file_response = requests.get(exported_url, timeout=10)
            if file_response.status_code == 200:
                return file_response.content
            else:
                detail = _get_error_detail(file_response)
                raise Exception(f"Failed to download file: HTTP {file_response.status_code} — {detail}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"File download failed: {e}")
    elif response.status_code == 401:
        detail = _get_error_detail(response)
        raise Exception(f"Authentication error (401): {detail}")
    else:
        detail = _get_error_detail(response)
        raise Exception(f"Error: HTTP {response.status_code} — {detail}")

# Initialize at app startup
@app.on_event("startup")
async def startup_event():
    app.state.process_executor = ProcessPoolExecutor(max_workers=10)  # Adjust based on CPU cores

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    app.state.process_executor.shutdown(wait=True)

from pydantic import BaseModel

class ReportRequest(BaseModel):
    entity: AllowedEntity


@app.post("/generate-reports/{customer_id}")
async def create_reports(customer_id: str, request: ReportRequest):
    entity = request.entity
    """
    Endpoint to generate and save a report for a given customer and entity.
    
    - For 'orders', it generates two files (orders and order_products) and then creates a sales report.
    - For 'activities' it generates activities (+task, notes) report.
    """
    try:
        if entity == "orders":
            # Retrieve file contents for orders and order_products 
            try:
                orders_file_content = get_exported_data(customer_id, "orders")
            except Exception as e:
                logger1.error(f"Error in get_exported_data orders: {e}")
             
            try:
                products_file_content = get_exported_data(customer_id, "order_products")
            except Exception as e:
                logger1.error(f"Error in get_exported_data order_products: {e}")
            # Define directory and file paths for orders and order_products
            orders_dir = os.path.join("data", customer_id, "orders")
            os.makedirs(orders_dir, exist_ok=True)
            orders_path = os.path.join(orders_dir, "orders.csv")
            products_path = os.path.join(orders_dir, "order_products.csv")
            #
            logger1.info(f"Saving orders report for customer '{customer_id}' at {orders_path}")
            async with aiofiles.open(orders_path, "wb") as f:
                await f.write(orders_file_content)
            
            logger1.info(f"Saving order products report for customer '{customer_id}' at {products_path}")
            async with aiofiles.open(products_path, "wb") as f:
                await f.write(products_file_content)
            
            # Generate the sales report using the saved orders and products file paths
            try:
                result  = await generate_sales_report(orders_path, products_path, customer_id)
                report = result["full_report"]
                #print(report)
                report_sections = result["sections"] 
            except Exception as e:
                logger1.error(f"Error in generate_sales_report: {e}")
            
            
            report_dir = os.path.join("data", customer_id)
            path_for_report = os.path.join(report_dir, "report.md")
            async with aiofiles.open(path_for_report, "w") as f:
                await f.write(report)
                
            #print(report)
            return JSONResponse(status_code=200, content={"message": "Sales report generated successfully",
                                                          'file_path_product':products_path,
                                                          'file_path_orders':orders_path,
                                                          "report": report,
                                                          "sections": report_sections})
        
        elif entity == "activities":
            try:
                notes = get_exported_data(customer_id, "notes")
                tasks = get_exported_data(customer_id, "tasks")
                activities = get_exported_data(customer_id, "activities")
            except Exception as e:
                logger1.error(f"Error in get_exported_data activities: {e}")
            
            
            # Build the directory path and file name for non-orders entities
            dir_path = os.path.join("data", customer_id, 'activities')
            os.makedirs(dir_path, exist_ok=True)
            
            file_path_notes = os.path.join(dir_path, "notes.csv")
            file_path_tasks = os.path.join(dir_path, "tasks.csv")
            file_path_activities = os.path.join(dir_path, "activities.csv")
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'notes' at {file_path_notes}")
            async with aiofiles.open(file_path_notes, "wb") as f:
                await f.write(notes)
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'tasks' at {file_path_tasks}")
            async with aiofiles.open(file_path_tasks, "wb") as f:
                await f.write(tasks)
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'activities' at {file_path_activities}")
            async with aiofiles.open(file_path_activities, "wb") as f:
                await f.write(activities)
            
            
            ## Generate the sales report using the saved orders and products file paths
            try:
                report_activities = await analyze_activities(file_path_notes, file_path_tasks, file_path_activities)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_activities_dir, "report_activities.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(report_activities)
            except Exception as e:
                logger1.error(f"Error in analyze_activities: {e}")
                
            try:
                report_task = tasks_report(file_path_tasks)
                #print(report_task)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_activities_dir, "report_task.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(report_task)
            except Exception as e:
                logger1.error(f"Error in tasks_report: {e}")
            try:    
                report_notes = notes_report(file_path_notes)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_notes = os.path.join(report_activities_dir, "report_notes.md")
                async with aiofiles.open(path_for_notes, "w") as f:
                    await f.write(str(report_notes))
            except Exception as e:
                logger1.error(f"Error in notes_report: {e}")
                
            try:
                report_text, section_report = await process_ai_activities_request(customer_id)
                try:
                    full_report = report_text.get('model_answer')
                except Exception as e:
                    full_report = "Could not analyze the activity of your customer"
                    
                #print(report_text.get('model_answer'))
                if report_text['model_answer'] == 'Could not analyze the activity of your customer.':
                    print("Analysis failed - check logs for details")
                else:
                    print("Analysis succeeded:")
            except Exception as e:
                logger1.error(f"Error in process_ai_activities_request: {e}")
            
            
            return JSONResponse(status_code=200, content={"message": f"Report generated successfully", 
                                                          "file_path_activities": file_path_activities,
                                                          "file_path_tasks": file_path_tasks,
                                                          "report": full_report,
                                                          "sections": section_report})

        else:
            raise HTTPException(status_code=406, detail='Incorrect entity: use "activities" or "orders" ')

    except Exception as e:
        logger1.error(f"Error generating report: {e}")
        raise HTTPException(status_code=406, detail=f"Error generating report due to incorrect customer id")
    

class ChatRequest(BaseModel):
    prompt: str


@app.post("/Ask_ai")
async def ask_ai_endpoint(
    request: ChatRequest,
    customer_id: str = Query(...)
):
    prompt = request.prompt
    user_uuid = customer_id
    pre_prompt = f'Use all the tools you need to answer, following the instructions carefully. Answer the following questions: {prompt}'

    try:
        # Use AI function to get response
        #response = await Ask_ai_many_customers(prompt, user_uuid)
        from AI.single_customer_analyze.Ask_ai_single_customer import create_Ask_ai_single_c_agent
        agent, session = await create_Ask_ai_single_c_agent(user_uuid)

        runner = await Runner.run(
                agent, 
                input=pre_prompt,
                session=session
                )

        answer = runner.final_output 

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


@app.post("/st_Ask_ai")
async def st_ask_ai_endpoint(
    request: ChatRequest,
    customer_id: str = Query(...)
):
    prompt = request.prompt
    user_uuid = customer_id
    pre_prompt = f'Use all the tools you need to answer, following the instructions carefully. Answer the following questions: {prompt}'

    async def sse_generator():
        try:
            from AI.single_customer_analyze.Ask_ai_single_customer import create_Ask_ai_single_c_agent
            agent, session = await create_Ask_ai_single_c_agent(user_uuid)

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
@app.post("/analyze_routes/{user_id}")
async def analyze_routes(user_id: str):
    user_data_path = f"data/{user_id}"
    orders_csv_path = f"{user_data_path}/oorders.csv"
    customers_csv_path = f"{user_data_path}/raw_data/concatenated_customers.csv"
    
    # Check if user data directory exists
    if not os.path.exists(user_data_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incorrect user ID: {user_id}"
        )
            
    # Check if files exist
    if not os.path.exists(orders_csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orders file not found."
        )
    
    if not os.path.exists(customers_csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customers file not found."
        )
    
    # Analyze customer orders
    result = await analyze_customer_orders_async(orders_csv_path, customers_csv_path)
    
    # Check if result is empty
    if not result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No data available after analysis"
        )
    
    return result


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
        print(answer)

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

# Update your ReportRequest model
class ReportRequest(BaseModel):
    customer_ids: List[str]
    report_type: ReportType = Field(
        default=ReportType.FULL_REPORT,
        title="Report Type",
        description="Specify which report section to generate. Defaults to the full report."
    )
    uuid: Optional[str] = None
    entity: AllowedEntity    # Single entity, restricted to AllowedEntity values

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





@app.post("/generate-reports-group")
async def create_group_reports_new(request: ReportRequest = Body(...)):
    """
    Generate group reports for multiple customers.
    Optimized to only run analysis for the requested report_type.
    """
    try:
        start_time = time.perf_counter()
        customer_ids = request.customer_ids
        report_type = request.report_type

        uuid = request.uuid or str(uuid4())

        # Create directory structure
        user_folder = os.path.join('data', uuid, 'work_data_folder')
        await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)

        # Fetch data for all customers
        entities = ['orders', 'order_products', 'customer']
        # File names
        filename_orders = 'one_file_orders.csv'
        filename_products = 'one_file_products.csv'
        filename_customers = 'one_file_customers.csv'

        file_path_orders = os.path.join(user_folder,  filename_orders)
        file_path_products = os.path.join(user_folder,  filename_products) 
        file_path_customers = os.path.join(user_folder,  filename_customers) 

        print(f"Step 0 - Starting data fetch for id {uuid}: {time.perf_counter() - start_time:.2f}s")
        try:
            # Fetch data concurrently
            entities_orders = ["orders"]
            entities_products = ["order_products"] 
            entities_customers = ["customer"]

            result_1, result_2, result_3 = await asyncio.gather(
                post_get_exported_data_one_file(customer_ids, entities_orders),
                post_get_exported_data_one_file(customer_ids, entities_products),
                post_get_exported_data_one_file(customer_ids, entities_customers)
            )
        except Exception as e:
            error_message = str(e)
            logger2.error(f"Data processing/fetching error: {error_message}")
            if "URL component 'query' too long" in error_message:
             return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "Status": "Failed",
                    "Reason": "Too many customer IDs provided. The request URL exceeded the length limit."
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

        # Process and save fetched data concurrently
        await asyncio.gather(
            _process_and_save_file_data(result_1, file_path_orders),
            _process_and_save_file_data(result_2, file_path_products),
            _process_and_save_file_data(result_3, file_path_customers)
        )

        try:
            check_if_customer_id_correct = pd.read_csv(file_path_customers)
            #print(check_if_customer_id_correct.head(3))
            if check_if_customer_id_correct.empty:
                logger2.warning(f"Customer data empty for IDs: {customer_ids}")
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "Status": "Failed",
                        "Reason": "Incorrect Customer IDs provided. No customer data found.",
                        "uuid": request.customer_ids
                    }
                )
        except Exception as e:
            logger2.warning(f"Can not check if customers ids are valid: {e}")

        # Preprocess data
        full_cleaned_orders, full_cleaned_products = await prepared_big_data(
            str(file_path_orders), 
            str(file_path_products)
        )

        print(f"Step 2 - Data preprocessing completed: {time.perf_counter() - start_time:.2f}s")


        # Save cleaned data concurrently
        cleaned_orders_path =  os.path.join(user_folder,  'cleaned_real_big_orders.csv') 
        cleaned_products_path =  os.path.join(user_folder,  'cleaned_real_big_products.csv')

        await asyncio.gather(
            save_df(full_cleaned_orders, str(cleaned_orders_path)),
            save_df(full_cleaned_products, str(cleaned_products_path))
        )
    except Exception as e:
        logger2.error(f"Data processing error: {e}")

    try:
        # Check if customers ids correct but no data in orders
        try:
            check_if_orders_has_data = pd.read_csv(cleaned_orders_path)
            #print(check_if_orders_has_data.head(3))
            if check_if_orders_has_data.empty:
                logger2.info("Orders data is empty after processing.")
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                    "Status": "Empty Data",
                    "Reason": "The report cannot be generated based on empty data (No valid orders found).",
                    "uuid": request.customer_ids
                }
                )
        except Exception as e:
            logger2.warning(f"Can not check if customers orders are empty: {e}")
        # Read dataframes concurrently
        try:
            orders_df, products_df, customer_df = await asyncio.gather(
                read_dataframe_async(str(cleaned_orders_path)),
                read_dataframe_async(str(cleaned_products_path)),
                read_dataframe_async(str(file_path_customers))
            )
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
                incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                        "Status": "The report cannot be generated based on empty data.",
                        "incorrect_ids" : incorrect_ids,
                        "sections": {"full_report":"The report cannot be generated based on empty data."},
                        "report": 'The report cannot be generated based on empty data.',
                        "uuid": uuid
                    })
        except Exception as e:
            logger2.error(e)
        
        incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
        # Create and return response
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "Status": "The report cannot be generated due to incorrect data",
                    "incorrect_ids" : incorrect_ids,
                    "sections": 'sectioned_report',
                    "report": 'full_report',
                    "uuid": uuid
                })

    try:
        # Generate analytics reports
        print(f"Step 4 - Before generate report (Type: {report_type.value}): {time.perf_counter() - start_time:.2f}s")
        if report_type.value =="full_report":
            try:
                from AI.group_customer_analyze.create_report_group_c import new_generate_analytics_report_ , main_batch_process
                full_report, sectioned_report = await main_batch_process(merged_orders, products_df, customer_df, uuid)
                #full_report, sectioned_report = await new_generate_analytics_report_(merged_orders, products_df, customer_df, uuid)

                # Save full report
                async with aiofiles.open(f"data/{uuid}/full_report.txt", "w", encoding="utf-8") as f:
                    await f.write(full_report)


                print("Step 5 - after generate report:", time.perf_counter() - start_time)

                incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                # Create and return response
                return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "incorrect_ids" : incorrect_ids,
                    "sections": sectioned_report,
                    "report": full_report,
                    "uuid": uuid
                })
            except Exception as e:
                logger2.warning(f"The problem of displaying of statistics in the 'full_report' block: {e}")
                try:
                    #If agent can not answer then response only statistics:
                    from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
                    raw_stats = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                    incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                        "incorrect_ids" : incorrect_ids,
                        "sections": raw_stats.get('sections'),
                        "report": raw_stats.get('full_report'),
                        "uuid": uuid
                    })
                except Exception as e:
                    logger2.warning(f"The problem of displaying an alternative version of statistics in the “full_report” block: {e}")

        elif report_type.value =="product_per_state_analysis":
            try:
                from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
                from AI.group_customer_analyze.orders_state import async_generate_report, async_process_data

                try:
                    # Run CSV saving in parallel threads
                    products_df['product_variant'] = products_df['name'].astype(str) + ' - ' + products_df['sku'].astype(str)
                    await asyncio.gather(
                        asyncio.to_thread(merged_orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
                        asyncio.to_thread(products_df.to_csv, f'data/{uuid}/pproducts.csv', index=False)
                    )
                except Exception as e:
                    logger2.warning(f"Error saving debug CSVs for {uuid}: {e}")

                await async_process_data(uuid)
                await async_generate_report(uuid)

                agent = await create_agent_products_state_analysis((uuid))

                try:
                    runner = await Runner.run(
                        agent, 
                        input="Based on the data return response"#,  session=session
                    )

                    answer = runner.final_output 
                    from pprint import pprint
                    #print(answer)
                    for i in range(len(runner.raw_responses)):
                        print("Token usage : ", runner.raw_responses[i].usage, '')
                except Exception as e:
                    print(f"Error in product_per_state_analysis runner: {e}")

                try:
                    incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)

                    full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                    # Save full report
                    async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                        await f.write(full_report.get('full_report') )

                    print("Step 5 - after generate report:", time.perf_counter() - start_time)
                except Exception as e:
                    logger2.error(f"full report error in 'state' topic generate: {e}")

                sectioned_report = {'product_per_state_analysis' : answer}
                return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "incorrect_ids" : incorrect_ids,
                    "sections": sectioned_report,
                    "report": full_report.get('full_report'),
                    "uuid": uuid
                })
            except Exception as e:
                logger2.warning(f"The problem of displaying of statistics in the 'product_per_state_analysis' block: {e}")
                try:
                    #If agent can not answer then response only statistics (prepared data):
                    from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
                    incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                    full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                        "incorrect_ids" : incorrect_ids,
                        "sections": {'product_per_state_analysis':'You do not have enough data to analyze the states, or they do not meet the standards. Please try again later.'},
                        "report": full_report.get('full_report'),
                        "uuid": uuid
                    })
                except Exception as e:
                    logger2.warning(f"The problem of displaying an alternative version of statistics in the 'product_per_state_analysis' block: {e}")
        
        else:
            try:
                    topic = report_type.value
                    print(topic)
                    from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
                    #from test_agent_1 import create_agent_sectioned

                    statistics_of_topic = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid, report_type=topic)
                    #print(statistics_of_topic)
                    agent = await create_agent_sectioned(uuid, topic, statistics_of_topic)

                    runner = await Runner.run(
                    agent, 
                    input="Based on the data return response"#,  session=session
                    )

                    answer = runner.final_output 
                    from pprint import pprint
                    #print(statistics_of_topic)
                    #print(answer)
                    sectioned_answer = await combine_sections(topic, statistics_of_topic, answer)

                    calculate_cost(runner, model="gpt-4.1-mini")
                    #for i in range(len(runner.raw_responses)):
                    #    print("Token usage : ", runner.raw_responses[i].usage, '')

                    incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)

                    full_report = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid)
                    # Save full report
                    async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                        await f.write(full_report.get('full_report') )

                    print("Step 5 - after generate report:", time.perf_counter() - start_time)
                
                    #print(sectioned_answer.get(topic))
                    return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "incorrect_ids" : incorrect_ids,
                        "sections": sectioned_answer,
                        "report": full_report.get('full_report'),
                        "uuid": uuid
                    })
            except Exception as e:
                logger2.warning(f"The problem of displaying of statistics in the {report_type.value} block: {e}")
                try:
                    #If agent can not answer then response only statistics:
                    from AI.group_customer_analyze.create_report_group_c import generate_analytics_report_sectioned
                    raw_stats = await generate_analytics_report_sectioned(merged_orders, products_df, customer_df, uuid, report_type.value)

                    incorrect_ids = await check_customer_ids(merged_orders, customer_df, customer_ids)
                    sectioned_report = {f'{report_type.value}' : raw_stats}
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                        "incorrect_ids" : incorrect_ids,
                        "sections": sectioned_report,
                        "report": raw_stats,
                        "uuid": uuid
                    })
                except Exception as e:
                    logger2.warning(f"The problem of displaying an alternative version of statistics in the {report_type.value} block: {e}")
    
        #return await create_response('success_count', len(customer_ids), 'failed_customer_names', 'customer_names_empty', sectioned_report, full_report, uuid)
    
    except Exception as e:
        logger2.error(f"Report generation failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {
                        "loc": ["server", "report_generation"],
                        "msg": "Internal server error during report generation",
                        "type": "internal_server_error"
                    }
                ]
            }
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
    uvicorn.run(app, port=8000, host='0.0.0.0')