import logging

import os
import re
import asyncio
import aiofiles
import pandas as pd
from pathlib import Path
import io
import time

# some util functions
def extract_customer_id(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$", re.I)

    for part in parts:
        if uuid_pattern.match(part):
            return part
    return None  # or raise ValueError("Customer ID not found in path.")

def calculate_cost(runner, model="gpt-4.1-mini"):
    """
    Calculates the estimated cost of an OpenAI Agents SDK session.
    
    Args:
        runner: The agent runner instance containing .raw_responses
        model (str): The model identifier (e.g., "gpt-4.1-mini", "gpt-4o-mini")
        
    Returns:
        float: Total estimated cost in USD.
    """
    # Pricing per 1 Million tokens (USD)
    # Based on Dec 2025 standard pricing
    PRICING = {
        "gpt-4.1-mini": {
            "input": 0.40,
            "cached_input": 0.10,
            "output": 1.60
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "cached_input": 0.075,
            "output": 0.60
        },
        "gpt-4o": {
            "input": 2.50,
            "cached_input": 1.25,
            "output": 10.00
        },
        "gpt-4.1": {
            "input": 2.00,
            "cached_input": 0.5,
            "output": 8.00
        },
        "gpt-5.1": {
            "input": 1.25,
            "cached_input": 0.125,
            "output": 10.00
        },
        "gpt-5.4-mini": {
            "input": 0.75,
            "cached_input": 0.075,
            "output": 4.50
        }
    }

    if model not in PRICING:
        print(f"Warning: Model '{model}' not found in pricing table. Using gpt-4.1-mini rates.")
        rates = PRICING["gpt-4.1-mini"]
    else:
        rates = PRICING[model]

    total_cost = 0.0
    total_input = 0
    total_output = 0
    
    for i, response in enumerate(runner.raw_responses):
        if not hasattr(response, 'usage') or not response.usage:
            continue
            
        usage = response.usage
        
        # Extract token counts
        # Handle cases where attributes might be missing (safety check)
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        
        # Check for cached tokens
        cached_tokens = 0
        if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
            cached_tokens = getattr(usage.input_tokens_details, 'cached_tokens', 0)
        
        # Calculate regular input (Total Input - Cached)
        regular_input_tokens = max(0, input_tokens - cached_tokens)
        
        # Calculate cost for this step
        step_cost = (
            (regular_input_tokens / 1_000_000 * rates["input"]) +
            (cached_tokens / 1_000_000 * rates["cached_input"]) +
            (output_tokens / 1_000_000 * rates["output"])
        )
        
        total_cost += step_cost
        total_input += input_tokens
        total_output += output_tokens
        
        # Optional: Print step detail
        # print(f"Step {i+1}: ${step_cost:.6f} (In: {input_tokens}, Out: {output_tokens})")

    print(f"Total Tokens: {total_input + total_output} (Input: {total_input}, Output: {total_output})")
    print(f"Total Cost:   ${total_cost:.6f}")
    
    return total_cost

def get_logger(name: str, log_file: str, console: bool = True) -> logging.Logger:
    """Create and configure a logger with file and optional console output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Disable propagation to root logger
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger


async def combine_sections(title, var1, var2):
    """
    Combines two markdown sections dynamically:
    - Returns a dict with {title: combined_text}
    
    This works for any title in the list like "Key Metrics", "Discount Distribution", etc.
    """
    lines1 = var1.splitlines(keepends=False)

    processed_var2 = var2.split("\n",2)[2]
    

    combined = '\n'.join(lines1) + '\n---\n' + processed_var2
    response = combined.replace('\n---\n', '\n\n').replace('\n---', '')
    return {title: response}

#some functions for create_group_reports endpoint
from fastapi import Body, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
import aiofiles
from uuid import uuid4
import pandas as pd
import asyncio
import aiofiles
from io import StringIO
from datetime import datetime, timedelta
from collections import defaultdict


# Set up logging
import numpy as np

logger2 = get_logger("logger2", "project_log_many.log", False)


async def process_fetch_results(results, customer_ids, entities):
    """Process fetched data into separate dictionaries for each entity."""
    data_orders = {}
    data_products = {}
    data_customer = {}
    customer_names = {}
    
    for customer_id, payload in results:
        if payload and payload.get("files"):
            data_orders[customer_id] = payload["files"].get("orders")
            data_products[customer_id] = payload["files"].get("order_products")
            data_customer[customer_id] = payload["files"].get("customer")
            customer_names[customer_id] = payload.get("customer_name", f"Unknown ({customer_id})")
        else:
            data_orders[customer_id] = None
            data_products[customer_id] = None
            data_customer[customer_id] = None
            customer_names[customer_id] = f"Unknown ({customer_id})"
    
    return data_orders, data_products, data_customer, customer_names

async def validate_save_results(save_results, customer_ids, customer_names):
    """Validate save results and identify successful/failed customers."""
    save_results_orders, save_results_products, save_results_customer = save_results
    
    success_count = sum(
        1 for customer_id in customer_ids
        if (save_results_orders.get(customer_id, "").endswith(".csv") and
            save_results_products.get(customer_id, "").endswith(".csv") and
            save_results_customer.get(customer_id, "").endswith(".csv"))
    )
    
    failed_customer_names = [
        customer_names.get(customer_id, f"Unknown ({customer_id})")
        for customer_id in customer_ids
        if not (save_results_orders.get(customer_id, "").endswith(".csv") and
                save_results_products.get(customer_id, "").endswith(".csv") and
                save_results_customer.get(customer_id, "").endswith(".csv"))
    ]
    
    return success_count, failed_customer_names

async def generate_file_paths(customer_ids, uuid):
    """Generate file paths for saved data."""
    ord_path = [f"data/{uuid}/raw_data/{customer_id}/orders/orders.csv" for customer_id in customer_ids]
    prod_path = [f"data/{uuid}/raw_data/{customer_id}/order_products/order_products.csv" for customer_id in customer_ids]
    customer_path = [f"data/{uuid}/raw_data/{customer_id}/customer/customer.csv" for customer_id in customer_ids]
    return ord_path, prod_path, customer_path

async def create_response(success_count, total_customers, failed_customer_names, customer_names_empty, sectioned_report, full_report, uuid):
    """Create the JSON response based on processing results."""
    if success_count == 0:
        raise HTTPException(
            status_code=400,
            detail="All customers failed processing. Report cannot be generated."
        )
    
    failed_customers = failed_customer_names + list(set(customer_names_empty))
    
    if full_report == '':
        logger2.error("Some problem with report response")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "message": "The agent was unable to process the data.",
                "failed_customers": failed_customers,
                "sectioned_report": sectioned_report,
                "full_report": full_report,
                "uuid": uuid
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Successfully generated reports for {success_count} of {total_customers} customers",
            "failed_customers": failed_customers,
            "sectioned_report": sectioned_report,
            "full_report": full_report,
            "uuid": uuid
        }
    )


# some functions for analyze_routes endpoints
def convert_to_datetime(df, columns):
    try:
        for col in columns:
            if col in df.columns:
                original = df[col].copy()

                # Enhanced preprocessing
                cleaned = (
                    original
                    .astype(str)
                    .str.split(r'\s*\(.*', n=1).str[0]  # Remove anything after (
                    .str.strip()
                    .str.replace(r'([+-]\d{2}):(\d{2})$', r'\1\2', regex=True)  # Fix tz format
                    .str.replace(r'\b(UTC|GMT)\b', '', regex=True)  # Remove UTC/GMT prefix
                    .str.replace(r'\s+', ' ', regex=True)  # Normalize spaces
                )

                # List of formats to try (order matters!)
                formats = [
                    '%Y-%m-%d %H:%M:%S%z',          # Case: "2025-03-06 13:24:40+0000"
                    '%a %b %d %Y %H:%M:%S %z',      # Case: "Fri Jul 26 2024 18:53:00 +0000"
                    '%Y-%m-%d %H:%M:%S',            # Fallback for tz-naive
                    '%a %b %d %Y %H:%M:%S',         # Fallback for tz-naive
                ]

                parsed = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')

                # Try each format sequentially
                for fmt in formats:
                    mask = parsed.isna()
                    if not mask.any():
                        break
                    
                    # Attempt parsing with current format
                    temp = pd.to_datetime(
                        cleaned[mask],
                        format=fmt,
                        errors='coerce',
                        utc=True
                    )

                    # Only keep successful parses
                    parsed[mask] = temp.dropna()

                df[col] = parsed

                # Report failures
                failed = original[parsed.isna()]


        return df
    except Exception as e:
        logger2.error("Error in convert to datetime: ",e)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

async def analyze_customer_orders_async(orders_csv_path, customers_csv_path):
    """
    Async function to analyze customer orders from CSV files.
    Identifies:
    1. Customers with paid payment status and unfulfilled delivery status
    2. Customers with unpaid payment status and fulfilled delivery status
    3. Customers who haven't checked in for more than 2 weeks
    Groups customers by state for each category and includes order IDs
    """
    try:
        # Read orders file asynchronously
        async with aiofiles.open(orders_csv_path, mode='r', encoding="UTF-8") as file:
            orders_content = await file.read()
        
        # Read customers file asynchronously
        async with aiofiles.open(customers_csv_path, mode='r', encoding="UTF-8") as file:
            customers_content = await file.read()
        
        # Process data with pandas in a thread pool
        loop = asyncio.get_event_loop()
        
        def process_data():
            # Read orders CSV
            orders_df = pd.read_csv(StringIO(orders_content))
            
            # Read customers CSV
            customers_df = pd.read_csv(StringIO(customers_content))
            
            # Filter for paid but unfulfilled orders and include order IDs
            paid_unfulfilled = orders_df[
                (orders_df['paymentStatus'] == 'PAID') & 
                (orders_df['deliveryStatus'] == 'UNFULFILLED')
            ]
            
            # Filter for unpaid but fulfilled orders and include order IDs
            unpaid_fulfilled = orders_df[
                (orders_df['paymentStatus'] == 'UNPAID') & 
                (orders_df['deliveryStatus'] == 'FULFILLED')
            ]
            
            # Get customer names and their order IDs for each condition
            paid_unfulfilled_data = paid_unfulfilled.groupby('customer_name')['customId_customId'].apply(list).to_dict()
            unpaid_fulfilled_data = unpaid_fulfilled.groupby('customer_name')['customId_customId'].apply(list).to_dict()
            
            # Process customers who haven't checked in for more than 2 weeks
            # Use the enhanced datetime conversion function
            customers_df = convert_to_datetime(customers_df, ['lastCheckInAt'])
            
            # Calculate 2 weeks ago
            two_weeks_ago = datetime.now().replace(tzinfo=None) - timedelta(weeks=2)
            
            # Filter customers who haven't checked in for more than 2 weeks
            inactive_customers = customers_df[
                (customers_df['lastCheckInAt'].isna()) | 
                (customers_df['lastCheckInAt'].dt.tz_convert(None) < two_weeks_ago)
            ]
            
            inactive_customer_names = inactive_customers['customer_name'].unique().tolist()
            
            # Create a mapping from customer name to state
            customer_to_state = customers_df.set_index('customer_name')['billingAddress_state'].to_dict()
            
            # Group customers by state for paid/unfulfilled
            paid_unfulfilled_by_state = defaultdict(list)
            for customer, order_ids in paid_unfulfilled_data.items():
                state = customer_to_state.get(customer, 'Unknown')
                paid_unfulfilled_by_state[state].append({
                    'customer': customer,
                    'orders': order_ids
                })
            
            # Group customers by state for unpaid/fulfilled
            unpaid_fulfilled_by_state = defaultdict(list)
            for customer, order_ids in unpaid_fulfilled_data.items():
                state = customer_to_state.get(customer, 'Unknown')
                unpaid_fulfilled_by_state[state].append({
                    'customer': customer,
                    'orders': order_ids
                })
            
            # Group inactive customers by state
            inactive_by_state = defaultdict(list)
            for customer in inactive_customer_names:
                state = customer_to_state.get(customer, 'Unknown')
                inactive_by_state[state].append(customer)
            
            return (
                paid_unfulfilled_data, unpaid_fulfilled_data, inactive_customer_names,
                dict(paid_unfulfilled_by_state), dict(unpaid_fulfilled_by_state), dict(inactive_by_state)
            )
        
        # Execute pandas processing in thread pool
        result_data = await loop.run_in_executor(None, process_data)
        (
            paid_unfulfilled_data, unpaid_fulfilled_data, inactive_customers,
            paid_unfulfilled_by_state, unpaid_fulfilled_by_state, inactive_by_state
        ) = result_data
        
        # Format the summary with order IDs
        def format_summary_with_orders(data, title):
            if not data:
                return f"0 {title}"
            
            result = []
            for customer, order_ids in data.items():
                order_info = f" (Orders: {', '.join(map(str, order_ids))})" if order_ids else ""
                result.append(f"{customer}{order_info}")
            
            return f"{len(data)} {title}: {', '.join(result)}"
        
        # Prepare results
        result = {
            'paid_unfulfilled_data': paid_unfulfilled_data,
            'unpaid_fulfilled_data': unpaid_fulfilled_data,
            'inactive_customers': inactive_customers,
            'paid_unfulfilled_by_state': paid_unfulfilled_by_state,
            'unpaid_fulfilled_by_state': unpaid_fulfilled_by_state,
            'inactive_by_state': inactive_by_state,
            'summary': (
                f"{format_summary_with_orders(paid_unfulfilled_data, 'customer(s) with paid but unfulfilled orders')}. \n"
                f"{format_summary_with_orders(unpaid_fulfilled_data, 'customer(s) with unpaid but fulfilled orders')}. \n"
                f"{len(inactive_customers)} customer(s) haven't checked in for more than 2 weeks: "
                f"{', '.join(inactive_customers)}"
            ),
            'state_summary': (
                f"Paid but unfulfilled by state: {paid_unfulfilled_by_state} \n"
                f"Unpaid but fulfilled by state: {unpaid_fulfilled_by_state} \n"
                f"Inactive customers by state: {inactive_by_state}"
            )
        }
        
        # Convert any NumPy types to Python native types
        result = convert_numpy_types(result)
        
        return result
        
    except FileNotFoundError as e:
        return {'error': f'File not found: {str(e)}'}
    except Exception as e:
        return {'error': f'An error occurred: {str(e)}'}


# functions list for processing one file many customer data

async def save_dataframe_async(df: pd.DataFrame, file_path: str) -> None:
    """Save dataframe asynchronously"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, df.to_csv, file_path)


async def read_dataframe_async(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file asynchronously into a pandas DataFrame using a worker thread.
    """
    print(f"[{filepath}] Reading file asynchronously...")
    
    # Use aiofiles to read the file content asynchronously
    async with aiofiles.open(filepath, mode="r", encoding="utf-8") as afp:
        content = await afp.read()
    
    # Pass the blocking pandas.read_csv operation to a worker thread
    # The io.StringIO object acts like an in-memory file for pandas to read
    df = await asyncio.to_thread(pd.read_csv, io.StringIO(content))
    
    print(f"[{filepath}] DataFrame created in a separate thread.")
    return df


async def write_bytes_to_file_async(file_path: str, data: bytes) -> None:
    """Write bytes data to file asynchronously"""
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(data)


async def _process_and_save_file_data(result: dict, file_path: Path) -> None:
    """Helper function to process file data and save it"""
    combined_bytes = b"".join(
        part if isinstance(part, bytes) else part.encode("utf-8")
        for part in result["files"]["combined"]
    )
    await write_bytes_to_file_async(str(file_path), combined_bytes)

def _is_csv_empty(filepath: str) -> bool:
    """Returns True if the CSV is 0 bytes or only contains a header row."""
    # Check if the file is literally 0 bytes
    if os.path.getsize(filepath) == 0:
        return True
        
    # Open the file and read only the first two lines
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        _ = f.readline() # Read and ignore the header
        first_data_row = f.readline() # Attempt to read the first actual row of data
        
        # If there's no second line, or it's just a blank line/newline, it's empty
        if not first_data_row or not first_data_row.strip():
            return True
            
    return False

def is_data_ready(user_folder: str, entity: str) -> bool:
    """
    Checks if ALL required files exist and are less than 2 hours old.
    Returns True if data is ready (skip download), False otherwise.
    """
    # can be made dynamic later
    entity_file_map = {
        "catalog": ["raw_file_catalog.csv", "raw_file_order_products.csv"],
        "customers": ["raw_file_customers.csv", "raw_file_orders.csv", "raw_file_order_products.csv"],
        "orders": ["raw_file_orders.csv", "raw_file_order_products.csv"],
        "ask_ai": ["raw_file_orders.csv", "raw_file_order_products.csv", "raw_file_customers.csv", "raw_file_catalog.csv"],
        "activities": ["raw_file_activities.csv","raw_file_orders.csv"]
    }
    
    required_files = entity_file_map.get(entity, [])
    
    max_age_seconds = 2 * 60 * 60 # 2 hours in seconds
    current_time = time.time()
    folder_check_path = os.path.join('data', user_folder, 'work_data_folder')

    for filename in required_files:
        file_path = os.path.join(folder_check_path, filename)
        
        # 1. Check if the file exists at all
        if not os.path.exists(file_path):
            print(f"Data Check: Missing required file -> {filename}")
            return False
            
        # 2. Check how old the file is
        # getmtime returns the time of last modification in seconds since the epoch
        file_age_seconds = current_time - os.path.getmtime(file_path)
        
        if file_age_seconds > max_age_seconds:
            print(f"Data Check: File too old -> {filename} is {file_age_seconds / 3600:.2f} hours old.")
            return False

    print("Data Check: All files are present and fresh!")
    return True

from typing import Dict
 
RAW_FILENAME_BY_ENTITY: Dict[str, str] = {
    "orders": "one_file_orders.csv",
    "order_products": "one_file_products.csv",
    "customer": "one_file_customers.csv",
}
 
 
def raw_filename_for(entity: str) -> str:
    return RAW_FILENAME_BY_ENTITY.get(entity, f"one_file_{entity}.csv")

# MCP logic
TOPIC_CONFIG = {
        "customers": [
            "churn_report",
            "refined_opportunity_report",
            "top_customers_report",
            "visits_report",
            "full_report"
        ],
        "catalog": [
            #"key_metrics_report",
            #"sales_performance_report",
            #"fulfillment_report",
            "functional_product_analysis",
            "product_performance",
            "sales_trends_report",
            "bundle_performance_report",
            "top_3_sales_breakdown",
            "time_based_product_report",
            "full_report"
        ],
        "orders": [
            "key_metrics_report",
            "sales_performance_report",
            "discount_report",
            "payment_status_report",
            "fulfillment_report",
            "sales_trends_report",
            "full_report"
        ], 
        "activities": [
            "orders_and_revenue_by_salesperson_report",
            "activities_distribution_report",
            "key_analysis_report",
            "full_report"
        ],
        "tasks": [
            "task_backlog_report",
            "full_report"
        ],
        "notes": [
            "executive_summary_report",
            "action_items_report",
            "full_report"
        ],
        "forms": ["full_report"]
    }


# test for new mcp endpoint redesign

import asyncio
import importlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence, Tuple

import aiohttp
import httpx
from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

logger2 = logging.getLogger(__name__)

DATA_ROOT = "data"
WORK_FOLDER = "work_data_folder"
FRESHNESS_SECONDS = 2 * 60 * 60
FULL_REPORT = "full_report"


# ---------------------------------------------------------------------------
# Paths
def work_dir(distributor_id: str) -> str:
    return os.path.join(DATA_ROOT, distributor_id, WORK_FOLDER)


def raw_file_path(distributor_id: str, filename: str) -> str:
    return os.path.join(work_dir(distributor_id), filename)


def cleaned_path(distributor_id: str, name: str) -> str:
    return os.path.join(DATA_ROOT, distributor_id, f"cleaned_{name}.csv")


# ---------------------------------------------------------------------------
# Freshness check + sync lock
def files_are_fresh(distributor_id: str, filenames: Sequence[str]) -> bool:
    """True if every file exists and is younger than FRESHNESS_SECONDS.

    Blocking (os.stat); call it via asyncio.to_thread from the endpoint.
    This replaces is_data_ready() — the caller passes the file list, derived
    from the entity spec, so the check can never disagree with what is fetched.
    """
    folder = work_dir(distributor_id)
    now = time.time()
    for filename in filenames:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            logger2.info("Data check: missing %s", filename)
            return False
        age = now - os.path.getmtime(path)
        if age > FRESHNESS_SECONDS:
            logger2.info("Data check: %s is %.2fh old", filename, age / 3600)
            return False
    logger2.info("Data check: %d file(s) present and fresh", len(filenames))
    return True


_sync_locks: Dict[str, asyncio.Lock] = {}


def lock_for(key: str) -> asyncio.Lock:
    """Serialise sync per distributor+entity so two requests can't both download
    into the same folder and produce torn CSVs."""
    lock = _sync_locks.get(key)
    if lock is None:
        lock = _sync_locks[key] = asyncio.Lock()
    return lock


# ---------------------------------------------------------------------------
# Upstream errors
_HTTP_ERR = re.compile(r"HTTP Error (\d{3}):\s*(.*)", re.S)


def as_http_exception(exc: Exception, distributor_id: str) -> HTTPException:
    """Turn a stringly-typed upstream failure into a real HTTPException.

    Long-term fix: make get_distributor_data raise a typed UpstreamError with
    .status_code and .payload, then this regex can be deleted.
    """
    msg = str(exc)
    match = _HTTP_ERR.search(msg)
    if match and int(match.group(1)) == status.HTTP_404_NOT_FOUND:
        detail: Any = match.group(2).strip()
        try:
            detail = json.loads(detail).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Upstream Resource Missing",
                "distributor_id": distributor_id,
                "upstream_message": str(detail).strip(),
            },
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Data sync failed: {msg}",
    )


# ---------------------------------------------------------------------------
# Stage 1: fetch + download only the datasets this entity needs
async def sync_raw_data(
    distributor_id: str,
    datasets: Sequence[str],
    t0: float,
) -> None:

    try:
        timeout_config = httpx.Timeout(5.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            fetched = await asyncio.gather(*[
                get_distributor_data(distributor_id=distributor_id, entities=[name], client=client)
                for name in datasets
            ])
        logger2.info("Step 1 - fetch done (%s): %.2fs",
                     ", ".join(datasets), time.perf_counter() - t0)

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[
                handle_distributor_data(payload, name, distributor_id, session)
                for payload, name in zip(fetched, datasets)
            ])
        logger2.info("Step 2 - download done: %.2fs", time.perf_counter() - t0)
    except HTTPException:
        raise
    except Exception as exc:
        raise as_http_exception(exc, distributor_id) from exc


# ---------------------------------------------------------------------------
# Stage 2: the orders/products/catalog/customers cleaning pipeline
# ---------------------------------------------------------------------------
from AI.MCP_tools.get_SD_data import get_distributor_data, handle_distributor_data
from AI.group_customer_analyze.preprocess_data_group_c import (
    get_cleaned_catalog,
    get_cleaned_customers,
    prepared_big_data,
    save_df,
)
async def build_sales_pipeline(
    distributor_id: str,
    raw_files: Mapping[str, str],
    t0: float,
) -> Tuple[Dict[str, str], bool]:
    """Clean + persist the four sales CSVs.

    `raw_files` is the dataset -> filename map (RAW_FILES from the endpoint
    module); this function reads exactly the four keys it needs from it.

    Returns (cleaned_paths, orders_are_empty).
    """

    orders_df, products_df = await prepared_big_data(
        raw_file_path(distributor_id, raw_files["orders"]),
        raw_file_path(distributor_id, raw_files["order_products"]),
    )
    catalog_df, _ = await get_cleaned_catalog(
        raw_file_path(distributor_id, raw_files["catalog"])
    )
    customers_df, _ = await get_cleaned_customers(
        raw_file_path(distributor_id, raw_files["customers"])
    )
    logger2.info("Step 3 - preprocessing done: %.2fs", time.perf_counter() - t0)

    paths = {
        "orders": cleaned_path(distributor_id, "orders"),
        "products": cleaned_path(distributor_id, "products"),
        "catalog": cleaned_path(distributor_id, "catalog"),
        "customers": cleaned_path(distributor_id, "customers"),
    }
    await asyncio.gather(
        save_df(orders_df, paths["orders"]),
        save_df(products_df, paths["products"]),
        save_df(catalog_df, paths["catalog"]),
        save_df(customers_df, paths["customers"]),
    )
    return paths, bool(orders_df.empty)


# ---------------------------------------------------------------------------
# Stage 3: report runners
@dataclass
class ReportContext:
    distributor_id: str
    entity: str
    report_type: str
    request: Any = None
    paths: Dict[str, str] = field(default_factory=dict)  # cleaned_* paths, if built


@dataclass
class ReportPayload:
    report: Any
    sections: Any


ReportRunner = Callable[[ReportContext], Awaitable[ReportPayload]]


def coerce_payload(result: Any) -> ReportPayload:
    """Accept whatever shape a run_report returns, today or after you change it.

    Handles: an object with .report + .to_dict(), an object with .report +
    .sections, a (report, sections) tuple, or a plain dict.
    """
    if isinstance(result, ReportPayload):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        return ReportPayload(report=result[0], sections=result[1])
    if isinstance(result, dict):
        return ReportPayload(report=result.get("report"), sections=result.get("sections", result))

    report = getattr(result, "report", None)
    if hasattr(result, "to_dict"):
        sections = result.to_dict()
    else:
        sections = getattr(result, "sections", None)
    if report is None and sections is None:
        raise TypeError(f"Unrecognised report result type: {type(result)!r}")
    return ReportPayload(report=report, sections=sections)


def module_runner(
    module_path: str,
    func_name: str = "run_report",
    entity_arg: Optional[str] = None,
    pass_entity: bool = True,
) -> ReportRunner:
    """For activities / forms / tasks: `await mod.run_report(distributor_id, entity)`.

    The import stays lazy (as in the original in-function imports) so a heavy AI
    module only loads when that entity is actually requested. If a module later
    changes its signature, adjust func_name / pass_entity here — the endpoint
    does not change.
    """
    async def _run(ctx: ReportContext) -> ReportPayload:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
        args = (ctx.distributor_id, entity_arg or ctx.entity) if pass_entity else (ctx.distributor_id,)
        return coerce_payload(await fn(*args))

    _run.__name__ = f"run_{module_path.rsplit('.', 1)[-1]}"
    return _run


def batch_process_runner(agent_name: Optional[str] = None) -> ReportRunner:
    """For orders / catalog / customers: topic-analysis batch over the cleaned CSVs."""
    async def _run(ctx: ReportContext) -> ReportPayload:
        from AI.MCP_tools.topic_analysis_agents import main_batch_process

        kwargs: Dict[str, Any] = {}
        if ctx.report_type != FULL_REPORT:
            kwargs["specific_topic"] = ctx.report_type

        report, sections = await main_batch_process(
            ctx.paths["orders"],
            ctx.paths["products"],
            ctx.paths["customers"],
            ctx.paths["catalog"],
            ctx.distributor_id,
            agent_name or f"{ctx.entity}_agent",
            **kwargs,
        )
        return ReportPayload(report=report, sections=sections)

    _run.__name__ = "run_batch_process"
    return _run


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

EMPTY_ORDERS_MESSAGE = (
    "The report cannot be generated based on empty data (No valid orders found).\n\n"
    "You can create a new order to start analyzing your data - check this guide: "
    "[How to Create and Process a New Direct Order]"
    "(https://scribehow.com/viewer/How_To_Create_And_Process_A_New_Direct_Order__"
    "XOZEjF9KTJ2B_C4G32afpQ?referrer=documents)\n\n"
    "and ask AI agent for help with platform navigation and order creation, or you can "
    "clarify with our specialist: [Schedule a Meeting]"
    "(https://meetings.hubspot.com/john-vasylets/customers)\n"
)


def error_response(
    code: int,
    kind: str,
    message: Any,
    distributor_id: str,
    **extra: Any,
) -> JSONResponse:
    body = {"error": kind, "message": message, "distributor_id": distributor_id}
    body.update(extra)
    return JSONResponse(status_code=code, content=jsonable_encoder(body))


def ok_response(payload: ReportPayload, distributor_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({
            "sections": payload.sections,
            "report": payload.report,
            "uuid": distributor_id,
        }),
    )