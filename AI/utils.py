import logging

import os
import re

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
    
    return {title: combined}

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
import asyncio
import os
import aiofiles
import pandas as pd
from pathlib import Path
import asyncio
import os
import io
import time


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
