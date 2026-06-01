import asyncio
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
    "get_orders_by_customer",
    "get_stopped_ordering_report",
    "get_opportunity_report",
    "get_top_customers_report",
    "get_visits_report"
]

CATALOG_TOOLS_LIST = [
    "get_top_n_products",
    "get_product_catalog",
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

    # 4. Create Main Agent
    main_instructions = await prompt_multi_agent_main(session_id)
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
    
    # 5. Initialize Memory
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

# 5. FASTAPI & STREAMING
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
from AI.utils import calculate_cost
async def agent_stream_generator(request: ChatRequestMCP, req: Request) -> AsyncGenerator[str, None]:
    async with AsyncExitStack() as stack:
        try:
            session_id = str(request.distributor_id)
            agent, session = await build_main_agent_session(session_id, stack)
            yield f"data: {json.dumps({'type': 'status', 'content': f'Agent {agent.name} is thinking...'})}\n\n"

            runner = Runner.run_streamed(agent, request.message, session=session)

            tool_timings: Dict[str, float] = {} 
            active_tools: Dict[str, str] = {} # Map call_id to tool_name
            buffer = ""
            BUFFER_THRESHOLD = 4 # Reduced for smoother typing effect

            async for event in runner.stream_events():
                if await req.is_disconnected():
                    logger2.warning(f"Client disconnected early from session {request.distributor_id}.")
                    break 

                event_type = getattr(event, "type", "")

                # --- Text Response ---
                if event_type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta or ""
                    buffer += delta
                    if len(buffer) >= BUFFER_THRESHOLD:
                        yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                        buffer = ""

                # --- Tool Interactions ---
                elif event_type == "run_item_stream_event":
                    item = event.item
                    item_type = getattr(item, "type", "")
                    
                    if item_type == "tool_call_item":
                        # Flush buffer before showing tool status
                        if buffer:
                            yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                            buffer = ""
                        
                        tool_name = _extract_tool_info(item)
                        call_id = _extract_call_id(item) 
                        
                        if call_id: 
                            tool_timings[call_id] = time.time()
                            active_tools[call_id] = tool_name
                        
                        logger2.info(f">> START: {tool_name} (ID: {call_id})")
                        yield f"data: {json.dumps({'type': 'status', 'content': f'🛠️ Calling: {tool_name}'})}\n\n"

                    elif item_type == "tool_call_output_item":
                         call_id = _extract_call_id(item)
                         duration_str = "unknown"
                         tool_name = active_tools.get(call_id, "Tool")

                         if call_id and call_id in tool_timings:
                             duration = time.time() - tool_timings.pop(call_id)
                             duration_str = f"{duration:.2f}s"
                             active_tools.pop(call_id, None)
                         
                         logger2.info(f"<< FINISH: {tool_name} (ID {call_id}) | Duration: {duration_str}")
                         # Now the UI knows EXACTLY which tool finished
                         yield f"data: {json.dumps({'type': 'status', 'content': f'✅ {tool_name} finished ({duration_str})'})}\n\n"

            # Flush any remaining text in the buffer
            if buffer:
                yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"

            calculate_cost(runner, model="gpt-5.4-mini")

            final_metadata = json.dumps({"type": "metadata", "cost": 0, "status": "completed"})
            yield f"data: {final_metadata}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except asyncio.CancelledError:
            logger2.warning(f"Client disconnected session {request.distributor_id}")
            
        except Exception as e:
            logger2.error(f"Stream Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred during generation.'})}\n\n"
        

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