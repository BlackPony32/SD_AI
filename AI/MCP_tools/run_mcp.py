import asyncio
import httpx
import aiohttp
import os
import json
import sys
import time
import logging
import uvicorn
from typing import AsyncGenerator, Optional, Tuple, Dict, List
from contextlib import asynccontextmanager, AsyncExitStack

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# --- Third Party Imports ---
from agents import Agent, Runner, OpenAIResponsesModel, AsyncOpenAI
from agents.mcp import MCPServerStreamableHttp, create_static_tool_filter
from agents.extensions.memory import AdvancedSQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

from AI.group_customer_analyze.Agents_rules.prompts import (
    prompt_multi_agent_main, 
    prompt_multi_agent_orders, 
    prompt_multi_agent_customers, 
    prompt_multi_agent_catalog,
    prompt_multi_agent_FAQ
)

# 1. SETUP & CONFIGURATION
load_dotenv()

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger("agents").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING) 

def setup_custom_logger(name: str, log_file: str) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers(): logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger2 = setup_custom_logger("agent_server", "project_log_many.log")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8001/mcp")
USER_ID_DEFAULT = "FULL_DIST_TEST"
llm_model = OpenAIResponsesModel(model='gpt-5.4-mini', openai_client=AsyncOpenAI()) 

# --- TOOL DEFINITIONS ---
ORDER_TOOLS_LIST = [
    "get_top_n_orders",
    "get_order_details",
    "get_financial_metrics_report",
    "get_sales_performance_report",
    "get_discount_distribution_report",
    "get_fulfillment_analysis_report",
    "get_payment_analysis_report",
    "get_sales_trends_orders_report"
]

CUSTOMER_TOOLS_LIST = [
    "get_top_n_customers",
    "get_customers",
    "describe_customer",
    "get_orders_by_customer",
    "get_stopped_ordering_report",
    "get_opportunity_report",
    "get_top_customers_report",
    "get_visits_report"
]

CATALOG_TOOLS_LIST = [
    "get_top_n_products",
    "search_product_catalog",
    "get_product_details",
    "get_product_price",
    "get_catalog_main_info",
    "get_executive_inventory_report",
    "get_product_performance_portfolio_report",
    "get_top_products_customer_insights",
    "get_cross_sell_bundle_report",
    "get_time_based_product_report",
    "get_sales_prospecting_report"
]

FAQ_TOOLS_LIST = [
    "look_up_faq"
]

# 2. CONNECTION MANAGEMENT (Using AsyncExitStack)

def create_client_definition(tool_whitelist: list) -> MCPServerStreamableHttp:
    """
    Returns the client OBJECT, but does not connect yet. 
    The stack will handle the connection.
    """
    return MCPServerStreamableHttp(
        name="sd-ai-mcp",
        client_session_timeout_seconds=30.0,
        params={
            "transport": "streamable_http",
            "url": MCP_URL,
            "headers": {"Accept": "application/json, text/event-stream"},
            "timeout": 30
        },
        tool_filter=create_static_tool_filter(allowed_tool_names=tool_whitelist),
    )

# 3. AGENT FACTORIES
from datetime import datetime

current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")

async def create_orders_agent(mcp_server: MCPServerStreamableHttp, user_id: str) -> Agent:
    instructions = await prompt_multi_agent_orders(user_id, current_date_str)
    return Agent(name="orders_agent", model=llm_model, instructions=instructions, mcp_servers=[mcp_server])

async def create_customer_agent(mcp_server: MCPServerStreamableHttp, user_id: str) -> Agent:
    instructions = await prompt_multi_agent_customers(user_id, current_date_str)
    return Agent(name="customer_agent", model=llm_model, instructions=instructions, mcp_servers=[mcp_server])

async def create_faq_agent(mcp_server: MCPServerStreamableHttp, user_id: str) -> Agent:
    instructions = await prompt_multi_agent_FAQ(user_id)
    return Agent(name="faq_agent", model=llm_model, instructions=instructions, mcp_servers=[mcp_server])

async def create_catalog_agent(mcp_server: MCPServerStreamableHttp, user_id: str) -> Agent:
    instructions = await prompt_multi_agent_catalog(user_id, current_date_str)
    return Agent(name="catalog_agent", model=llm_model, instructions=instructions, mcp_servers=[mcp_server])

async def build_main_agent_session(session_id: str, stack: AsyncExitStack) -> Tuple[Agent, AdvancedSQLiteSession]:
    """
    Builds agents and registers connections into the provided AsyncExitStack.
    """
    logger2.info(f" Establishing FRESH MCP connections for session {session_id}...")

    # 1. Create Client Objects
    client_def_orders = create_client_definition(ORDER_TOOLS_LIST)
    client_def_customers = create_client_definition(CUSTOMER_TOOLS_LIST)
    client_def_catalog = create_client_definition(CATALOG_TOOLS_LIST)
    client_def_faq = create_client_definition(FAQ_TOOLS_LIST)

    # 2. Enter Contexts (Connect) via the Stack
    order_server = await stack.enter_async_context(client_def_orders)
    customer_server = await stack.enter_async_context(client_def_customers)
    catalog_server = await stack.enter_async_context(client_def_catalog)
    faq_server = await stack.enter_async_context(client_def_faq)

    # 3. Create Sub-Agents
    sub_agent_orders = await create_orders_agent(order_server, session_id)
    sub_agent_customers = await create_customer_agent(customer_server, session_id)
    sub_agent_catalog = await create_catalog_agent(catalog_server, session_id)
    sub_agent_faq = await create_faq_agent(faq_server, session_id)

    # 4 Check if data empty - if true then user cn be new or low data quality
    from AI.utils import _is_csv_empty
    df_path = f"data/{session_id}/cleaned_orders.csv"
    if _is_csv_empty(df_path):
        logger2.warning(f"Data quality issue detected for session {session_id}: Empty orders or customers data. This may lead to limited insights.")
        NEW_USER_BOOL = True
    else:
        NEW_USER_BOOL = False

    # 5. Create Main Agent
    main_instructions = await prompt_multi_agent_main(session_id,NEW_USER_BOOL)
    main_agent = Agent(
        name="Lead_Orchestrator",
        instructions=main_instructions,
        model=llm_model,
        tools=[
            sub_agent_orders.as_tool(
                tool_name="orders_agent", 
                max_turns=8, 
                tool_description="Use for macro FINANCIAL and TRANSACTION queries. Routes here for total revenue, sales performance trends, discount distribution, fulfillment/payment statuses, and looking up specific order IDs."
            ),
            sub_agent_customers.as_tool(
                tool_name="customer_agent", 
                max_turns=8, 
                tool_description="Use for WHO is buying. Routes here for customer-specific order history, VIP identification, churn risk (stopped ordering), account growth opportunities, and rep visit reports."
            ),
            sub_agent_catalog.as_tool(
                tool_name="catalog_agent", 
                max_turns=8, 
                tool_description="Use for WHAT is being sold. Routes here for product-level performance, inventory/stock health, cross-sell/bundle recommendations, sales prospecting leads, and identifying new vs stagnant products."
            ),
            sub_agent_faq.as_tool(
                tool_name="faq_agent", 
                max_turns=8, 
                tool_description="Use for STATIC KNOWLEDGE. Routes here for company policies, general business info, FAQ lookups, and SimplyDepo (SD) software documentation."
            ),
        ]
    )
    
    # 6. Initialize Memory
    try:
        session = AdvancedSQLiteSession(
            session_id=session_id,
            create_tables=True,
            db_path=f"data/{session_id}/conversations.db",
            logger=logger2
        )
    except Exception as e:
        logger2.error(f"Error creating DB session: {e}")
        raise e

    return main_agent, session


# 4. HELPER FUNCTIONS

def _extract_tool_info(item) -> str:
    if hasattr(item, "raw_item"):
        raw = item.raw_item
        if hasattr(raw, "name"): return raw.name
        if isinstance(raw, dict) and "name" in raw: return raw["name"]
    if hasattr(item, "function") and hasattr(item.function, "name"): return item.function.name
    if hasattr(item, "tool_call"):
        tc = item.tool_call
        if hasattr(tc, "function") and hasattr(tc.function, "name"): return tc.function.name
        if hasattr(tc, "name"): return tc.name
    return "Unknown Tool"

def _extract_call_id(item) -> Optional[str]:
    if hasattr(item, "raw_item"):
        raw = item.raw_item
        if hasattr(raw, "call_id"): return raw.call_id
        if isinstance(raw, dict) and "call_id" in raw: return raw["call_id"]
    if getattr(item, "id", None): return item.id
    if hasattr(item, "tool_call") and getattr(item.tool_call, "id", None): return item.tool_call.id
    if getattr(item, "call_id", None): return item.call_id
    return None

# FASTAPI & STREAMING
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 1. Define the startup logic to silence the specific Windows error
@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform == "win32":
        # Get the exact loop Uvicorn is using
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()

        def silence_winerror_10054(loop, context):
            exc = context.get('exception')
            # If it's the exact noisy error, ignore it entirely
            if isinstance(exc, ConnectionResetError) and getattr(exc, 'winerror', None) == 10054:
                return 
            
            # Otherwise, handle exceptions normally
            if original_handler:
                original_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        # Attach the silencer
        loop.set_exception_handler(silence_winerror_10054)
        print(" Windows Mode: Suppressed WinError 10054")

    # Yield control to the application
    yield 

# 2. Add the lifespan to your FastAPI app
app = FastAPI(title="Multi-Agent MCP Server", lifespan=lifespan)

class ChatRequestMCP(BaseModel):
    message: str
    distributor_id: str


from fastapi import Request
from AI.group_customer_analyze.preprocess_data_group_c import (
    save_df, prepared_big_data, get_cleaned_catalog, get_cleaned_customers
)
from AI.utils import get_logger, extract_customer_id, process_fetch_results, validate_save_results, generate_file_paths, create_response, \
    analyze_customer_orders_async, calculate_cost, is_data_ready
from AI.MCP_tools.get_SD_data import handle_distributor_data, get_distributor_data


class DataSyncError(Exception):
    """Generic error for upstream sync failures."""
    pass

class UpstreamMissingError(DataSyncError):
    """Specific error for 404s from the upstream provider."""
    def __init__(self, message: str, upstream_detail: dict):
        super().__init__(message)
        self.upstream_detail = upstream_detail

class DataPreprocessingError(Exception):
    """Error for when pandas/CSV processing fails."""
    pass

async def sync_and_process_distributor_data(distributor_id: str) -> bool:
    """
    Checks if data needs to be downloaded, fetches it, and preprocesses it.
    Returns True if a sync occurred, False if data was already ready.
    Raises custom exceptions on failure.
    """
    # Assuming is_data_ready is imported
    should_download_files = is_data_ready(distributor_id, 'ask_ai')
    
    if should_download_files:
        return False # Data is already ready, no sync needed

    # STEP 1: FETCH & DOWNLOAD DATA
    try:
        timeout_config = httpx.Timeout(5.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout_config) as shared_client:
            fetch_tasks = [
                get_distributor_data(distributor_id=distributor_id, entities=["customers"], client=shared_client),
                get_distributor_data(distributor_id=distributor_id, entities=["orders"], client=shared_client),
                get_distributor_data(distributor_id=distributor_id, entities=["order_products"], client=shared_client),
                get_distributor_data(distributor_id=distributor_id, entities=["catalog"], client=shared_client)
            ]
            data, data1, data2, data3 = await asyncio.gather(*fetch_tasks)

        async with aiohttp.ClientSession() as download_session:
            handle_tasks = [
                handle_distributor_data(data, "customers", distributor_id, download_session),
                handle_distributor_data(data1, "orders", distributor_id, download_session),
                handle_distributor_data(data2, "order_products", distributor_id, download_session),
                handle_distributor_data(data3, "catalog", distributor_id, download_session)
            ]
            await asyncio.gather(*handle_tasks)

    except Exception as e:
        error_msg = str(e)
        if "HTTP Error" in error_msg:
            try:
                json_part = error_msg.split("HTTP Error 404: ")[1]
                upstream_detail = json.loads(json_part)
                # Raise our custom 404 error
                raise UpstreamMissingError("Upstream Resource Missing", upstream_detail)
            except (IndexError, json.JSONDecodeError):
                pass # Fall through to generic error

        # Raise generic sync error
        raise DataSyncError(f"Data sync failed: {error_msg}")

    # STEP 2: PREPROCESS DATA
    file_path_orders = os.path.join('data', distributor_id, 'work_data_folder','raw_file_orders.csv')
    file_path_products = os.path.join('data', distributor_id, 'work_data_folder','raw_file_order_products.csv')
    file_path_customers = os.path.join('data', distributor_id,'work_data_folder', 'raw_file_customers.csv')
    file_path_catalog = os.path.join('data', distributor_id,'work_data_folder', 'raw_file_catalog.csv')

    try:
        full_cleaned_orders, full_cleaned_products = await prepared_big_data(
            str(file_path_orders), 
            str(file_path_products)
        )
        catalog_df, catalog_path = await get_cleaned_catalog(file_path_catalog)
        customers_df, customers_path = await get_cleaned_customers(file_path_customers)

        cleaned_orders_path = os.path.join('data', distributor_id, 'cleaned_orders.csv') 
        cleaned_products_path = os.path.join('data', distributor_id, 'cleaned_products.csv')
        cleaned_catalog_path = os.path.join('data', distributor_id, 'cleaned_catalog.csv')
        cleaned_customers_path = os.path.join('data', distributor_id, 'cleaned_customers.csv')

        await asyncio.gather(
            save_df(full_cleaned_orders, str(cleaned_orders_path)),
            save_df(full_cleaned_products, str(cleaned_products_path)),
            save_df(catalog_df, str(cleaned_catalog_path)),
            save_df(customers_df, str(cleaned_customers_path))
        )
    except Exception as e:
        logger2.error(f"Data processing error: {e}")
        raise DataPreprocessingError(str(e))

    return True # Indicates a successful fresh sync

async def agent_stream_generator(request: ChatRequestMCP, req: Request) -> AsyncGenerator[str, None]:
    async with AsyncExitStack() as stack:
        distributor_id = str(request.distributor_id)
        
        try:
            # 1. DATA SYNC & PREPROCESSING
            yield f"data: {json.dumps({'type': 'status', 'content': 'Synchronizing your latest workspace data...'})}\n\n"
            
            did_sync = await sync_and_process_distributor_data(distributor_id)
            if did_sync:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Data up to date. Preparing your analysis...'})}\n\n"

            # 2. AGENT INITIALIZATION
            agent, session = await build_main_agent_session(distributor_id, stack)
            yield f"data: {json.dumps({'type': 'status', 'content': f'Analyzing your request..'})}\n\n"

            runner = Runner.run_streamed(agent, request.message, session=session)

            # State tracking
            tool_timings: Dict[str, float] = {} 
            active_tools: Dict[str, str] = {} 
            buffer = ""
            json_buffer = "" # New dedicated quarantine buffer
            
            # Increased threshold slightly to ensure long JSON keys aren't split across stream chunks
            BUFFER_THRESHOLD = 25 
            
            capturing_json = False
            suggestions_sent = False

            # 3. STREAM PROCESSING LOOP
            async for event in runner.stream_events():
                if await req.is_disconnected():
                    logger2.warning(f"Client disconnected early from session {distributor_id}.")
                    break 

                event_type = getattr(event, "type", "")

                # --- Handle Text Generation ---
                if event_type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta or ""
                    
                    if capturing_json:
                        # Once triggered, NEVER stream to frontend. Quarantine everything.
                        json_buffer += delta
                    else:
                        buffer += delta
                        
                        # Look for JSON markers (Standard Markdown OR Raw Unformatted JSON)
                        trigger_idx = -1
                        potential_triggers = ["```json", "'''json", '{"suggested_prompts"', '{"option_1"']
                        
                        for t in potential_triggers:
                            idx = buffer.find(t)
                            if idx != -1 and (trigger_idx == -1 or idx < trigger_idx):
                                trigger_idx = idx
                                
                        if trigger_idx != -1:
                            # Flush everything right up to the trigger as normal text
                            text_to_flush = buffer[:trigger_idx]
                            if text_to_flush:
                                yield f"data: {json.dumps({'type': 'token', 'content': text_to_flush})}\n\n"
                            
                            # Move the rest into quarantine and activate JSON capture mode
                            json_buffer = buffer[trigger_idx:]
                            buffer = ""
                            capturing_json = True
                        else:
                            # Safely stream standard text, keeping a 25-character trailing margin 
                            # in the buffer so we don't accidentally split a trigger keyword in half.
                            if len(buffer) >= BUFFER_THRESHOLD or "\n" in buffer:
                                split_point = max(0, len(buffer) - 25)
                                if split_point > 0:
                                    yield f"data: {json.dumps({'type': 'token', 'content': buffer[:split_point]})}\n\n"
                                    buffer = buffer[split_point:]

                # --- Handle Tool Interactions ---
                elif event_type == "run_item_stream_event":
                    item = event.item
                    item_type = getattr(item, "type", "")
                    
                    if item_type == "tool_call_item":
                        if buffer: 
                            yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                            buffer = ""
                        
                        tool_name = _extract_tool_info(item)
                        call_id = _extract_call_id(item) 
                        
                        if call_id: 
                            tool_timings[call_id] = time.time()
                            active_tools[call_id] = tool_name
                        
                        logger2.info(f">> START: {tool_name} (ID: {call_id})")
                        TOOL_MESSAGES = {
                            "customer_agent": "Analyzing customer records...",
                            "get_customer_details": "Querying client database...",
                            "orders_agent": "Processing orders activities...",
                            "sales_orchestrator": "Evaluating order metrics...",
                            "catalog_agent": "Reviewing product catalog...",
                            "inventory_lookup": "Analyzing inventory parameters...",
                            "data_analyzer": "Aggregating data points...",
                            "calculator_tool": "Compiling performance metrics..."
                        }
                        display_message = TOOL_MESSAGES.get(tool_name, "Verifying with the knowledge base...")
                        yield f"data: {json.dumps({'type': 'status', 'content': f'{display_message}'})}\n\n"

                    elif item_type == "tool_call_output_item":
                        call_id = _extract_call_id(item)
                        duration_str = "unknown"
                        tool_name = active_tools.get(call_id, "Tool")

                        if call_id and call_id in tool_timings:
                            duration = time.time() - tool_timings.pop(call_id)
                            duration_str = f"{duration:.2f}s"
                            active_tools.pop(call_id, None)
                         
                        logger2.info(f"<< FINISH: {tool_name} (ID {call_id}) | Duration: {duration_str}")
                        yield f"data: {json.dumps({'type': 'status', 'content': f'{tool_name} finished ({duration_str})'})}\n\n"

            # 4. FINAL CLEANUP & METADATA
            # Flush any remaining valid text markdown
            if buffer:
                yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"

            # Try to safely extract and parse the quarantined JSON
            if capturing_json and json_buffer:
                import re
                # Use regex to aggressively find anything resembling a JSON object block,
                # ignoring broken backticks or conversational text trailing at the end.
                json_match = re.search(r'\{.*\}', json_buffer, re.DOTALL)
                
                if json_match:
                    clean_json_str = json_match.group(0)
                    try:
                        parsed_data = json.loads(clean_json_str)
                        
                        # Handle both variations (nested "suggested_prompts" or flat dictionary)
                        target_data = parsed_data.get("suggested_prompts", parsed_data)
                        
                        # Validate it has actual content
                        if target_data and len(target_data) > 0:
                            yield f"data: {json.dumps({'type': 'suggestions', 'content': target_data})}\n\n"
                            suggestions_sent = True
                    except json.JSONDecodeError:
                        logger2.warning("Agent returned malformed JSON. Relying on fallback.")

            # Fallback if no valid suggestions were parsed or generated
            # Notice we DO NOT yield the broken json_buffer text to the frontend. It is simply erased.
            if not suggestions_sent:
                fallback_prompts = {
                    "option_1": "Find growth opportunities",
                    "option_2": "Reduce operational risk"
                }
                yield f"data: {json.dumps({'type': 'suggestions', 'content': fallback_prompts})}\n\n"

            calculated_cost = calculate_cost(runner, model="gpt-5.4-mini")
            final_metadata = json.dumps({"type": "metadata", "cost": calculated_cost, "status": "completed"})
            
            yield f"data: {final_metadata}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        # 5. ERROR HANDLING
        except UpstreamMissingError as e:
            error_payload = {
                "type": "error",
                "content": "Upstream Resource Missing",
                "distributor_id": distributor_id,
                "upstream_message": e.upstream_detail.get("message", "").strip()
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            
        except DataPreprocessingError as e:
            error_payload = {
                "type": "error",
                "content": "Data Preprocessing Failed",
                "message": "Failed to fetch or preprocess distributor data from the upstream source. Report generation aborted.",
                "distributor_id": distributor_id
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            
        except DataSyncError as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        except asyncio.CancelledError:
            logger2.warning(f"Client disconnected session {distributor_id}")
            
        except Exception as e:
            logger2.error(f"Stream Error: {e}", exc_info=True)
            
            error_str = str(e)

            if "MCP server" in error_str or "Could not reach" in error_str:
                friendly_msg = "Our AI service is experiencing a brief connection troubles. Please try asking your question again in a moment!"
            else:
                friendly_msg = "We ran into an unexpected snag while processing your request. Please try again shortly."
                
            # 4. Yield the friendly message to the frontend
            yield f"data: {json.dumps({'type': 'error', 'content': friendly_msg})}\n\n" 

@app.post("/chat_mcp")
async def chat_endpoint(request: ChatRequestMCP, req: Request):
    return StreamingResponse(
        agent_stream_generator(request, req),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    print(" Starting Multi-Agent MCP Server...")
    
    # --- OS-Aware Event Loop Setup ---
    if sys.platform == "win32":
        # LOCAL WINDOWS FIX: Prevents "WinError 10054" when clients disconnect
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        print(" Running on Windows: Using SelectorEventLoop (Safe Mode)")
    else:
        # PRODUCTION LINUX (GCP)
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print(" Running on Linux: Loaded uvloop (High Performance Mode)")
        except ImportError:
            print(" Running on Linux: Standard asyncio loop (uvloop not installed)")

    uvicorn.run(app, host="0.0.0.0", port=8002)