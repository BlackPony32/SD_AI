import asyncio
import logging
import os
import datetime
import json
import functools

import pandas as pd
import traceback
from collections import Counter
import itertools
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from rapidfuzz import process, fuzz


# --- Third Party Imports ---
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from openai import OpenAI
from dotenv import load_dotenv

from AI.MCP_tools.faq_file_search import init_and_load_md, search_md_db, format_search_results
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage

load_dotenv()
mcp = FastMCP("sd-ai-mcp", json_response=True,port=8001)

# 1. ROBUST LOGGER (UTF-8 FORCED)
import logging
import functools
import time
import sys
import traceback

def get_logger(name: str, log_file: str, console: bool = True) -> logging.Logger:
    """Create and configure a clean, robust logger safe for Windows (UTF-8)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        # Clean, aligned formatting
        formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # File handler (UTF-8 strict)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger

# Initialize your global logger
logger2 = get_logger("mcp_tools", "project_log_many.log", console=True)

def log_tool_usage(func):
    """
    Powerful, clean decorator for tracking tool usage.
    Tracks execution time, neatly formats inputs, and prevents silent crashes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.perf_counter()
        
        # 1. Extract User ID smartly (from kwargs or first positional arg)
        user_id = kwargs.get('user_id')
        if not user_id and args:
            user_id = args[0]
        if not user_id:
            user_id = 'SYSTEM'

        # 2. Cleanly format arguments (ignoring None values to reduce noise)
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Safely capture positional args (excluding the user_id if it was args[0])
        pos_args = args[1:] if len(args) > 0 else ()
        
        arg_str = f"Args: {pos_args} | Kwargs: {clean_kwargs}"
        
        # Log Start
        logger2.info(f"▶ START | Tool: '{tool_name}' | User: {user_id} | {arg_str}")

        try:
            # 3. Execute Tool
            result = func(*args, **kwargs)
            
            # 4. Calculate Duration
            duration = time.perf_counter() - start_time
            
            # 5. Clean Result Logging (No multi-line mess)
            res_str = str(result)
            # Replace physical newlines so the log entry stays on a single line in the text file
            res_str_flat = res_str.replace('\n', ' \\n ').replace('\r', '')
            
            # Truncate at 120 characters
            log_preview = res_str_flat[:120] + "..." if len(res_str_flat) > 120 else res_str_flat
            
            # Log Success
            logger2.info(f"✔ END   | Tool: '{tool_name}' | User: {user_id} | Time: {duration:.2f}s | Result: {log_preview}")
            
            return result
            
        except Exception as e:
            # 6. Log Failure cleanly
            duration = time.perf_counter() - start_time
            error_msg = f"Tool '{tool_name}' failed after {duration:.2f}s: {str(e)}"
            
            # exc_info=True automatically attaches the full traceback to the log file neatly
            logger2.error(f" ERROR | User: {user_id} | {error_msg}", exc_info=True)
            
            # Return string to Agent so it knows what happened instead of silently crashing the MCP server
            return f"Tool Execution Error: {str(e)}"
            
    return wrapper

# --- TOOLS ---

@mcp.tool(name="get_top_n_customers")
@log_tool_usage
def get_top_n_customers(
    user_id: Optional[str], 
    n: Optional[int], 
    by_type: Optional[str], 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Gets the top (or bottom) N customers based on aggregated revenue, quantity, or order count,
    calculated ONLY from orders matching the start_date and status criteria.

    Parameters:
    - user_id: User's ID.
    - n: Number of customers to return.
    - by_type: 'revenue', 'totalQuantity', or 'orderCount'.
    - sort_order: 'desc' (Best) or 'asc' (Worst).
    - start_date: Include only orders created ON or AFTER this date (YYYY-MM-DD).
    - end_date: Filter data from start day to this date (YYYY-MM-DD).
    """
    # 1. Path Setup
    csv_path = Path("data") / str(user_id) / "cleaned_orders.csv"
    if not csv_path.exists():
        return f"Error: Data file not found for user {user_id}."

    try:
        dataf = pd.read_csv(csv_path, encoding='utf-8-sig')
        dataf.columns = dataf.columns.str.strip().str.replace('\ufeff', '')
        
        # Ensure critical columns exist
        if 'customer_id' not in dataf.columns:
            return "Error: Orders file missing 'customer_id' column."
            
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')
        if dataf['createdAt'].dt.tz is not None:
             dataf['createdAt'] = dataf['createdAt'].dt.tz_localize(None)
             
        dataf = dataf.dropna(subset=['createdAt'])
        
    except Exception as e:
        return f"Error reading CSV: {e}"

    df_filtered = dataf.copy()

    # 2. Time Filters
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            # Add time to include the entire end date
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            df_filtered = df_filtered[df_filtered['createdAt'] <= end_dt]
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    if df_filtered.empty:
        return "No customer activity found for the specified period/criteria."

    # 3. Aggregation Setup
    # Base aggregation functions
    agg_funcs = {
        'totalAmount': 'sum',
        'totalQuantity': 'sum',
        'id': 'count'
    }
    
    # Dynamically extract all possible identifiers if they exist in the CSV
    ident_cols = ['customer_name', 'customer_displayedName', 'customId_customId']
    present_idents = []
    for col in ident_cols:
        if col in df_filtered.columns:
            agg_funcs[col] = 'first'
            present_idents.append(col)

    # Group by customer_id to ensure accuracy across names/IDs
    customer_agg = df_filtered.groupby('customer_id').agg(agg_funcs).rename(columns={
        'totalAmount': 'totalRevenue',
        'id': 'orderCount'
    }).reset_index()

    # Calculate Average Order Value (AOV)
    customer_agg['averageOrderValue'] = customer_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 4. Sort Logic
    # FIX: Made dictionary keys entirely lowercase so they successfully match by_type.lower()
    sort_map = {
        'revenue': 'totalRevenue',
        'totalquantity': 'totalQuantity',
        'ordercount': 'orderCount'
    }
    
    sort_column = sort_map.get(str(by_type).lower())
    if not sort_column:
        return "Invalid 'by_type'. Choose 'revenue', 'totalQuantity', or 'orderCount'."

    is_ascending = (sort_order.lower() == 'asc')
    top_n_df = customer_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 5. Output Formatting
    filter_info = []
    if start_date: filter_info.append(f"Since: {start_date}")
    if end_date: filter_info.append(f"Until: {end_date}")
    filter_str = f"({', '.join(filter_info)})" if filter_info else "(All Time)"
    direction_label = "Bottom" if is_ascending else "Top"

    lines = []
    lines.append(f"## {direction_label} {n} Customers by {by_type.capitalize()} {filter_str}\n")
    
    # Construct dynamic headers
    headers = ["Customer ID"]
    if 'customId_customId' in present_idents: headers.append("Custom ID")
    headers.append("Name")
    headers.extend(["Revenue", "Qty", "Orders", "Avg Order Val"])
    
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for _, row in top_n_df.iterrows():
        # Clean up missing identifiers
        c_id = str(row['customer_id'])[:15] + "..." if len(str(row['customer_id'])) > 15 else row['customer_id']
        c_name = row.get('customer_displayedName', row.get('customer_name', 'Unknown'))
        if pd.isna(c_name): c_name = 'Unknown'
        
        row_data = [str(c_id)]
        
        if 'customId_customId' in present_idents:
            custom_id = str(row['customId_customId']) if pd.notna(row['customId_customId']) else "-"
            row_data.append(custom_id)
            
        row_data.append(str(c_name)[:30])
        row_data.append(f"${row['totalRevenue']:,.2f}")
        row_data.append(str(int(row.get('totalQuantity', 0))))
        row_data.append(str(int(row['orderCount'])))
        row_data.append(f"${row['averageOrderValue']:,.2f}")
        
        lines.append("| " + " | ".join(row_data) + " |")

    return '\n'.join(lines)

@mcp.tool(name="get_customers")
@log_tool_usage
def get_customers(user_id: Optional[str], search_query: Optional[str] = None) -> str:
    """
    Searches for customers using a hybrid approach:
    1. Exact substring match across ALL columns (finds IDs, exact names, emails).
    2. Fuzzy match on Name/ID columns (handles typos).
    """
    base_path = Path("data") / str(user_id)
    customers_path = base_path / "cleaned_customers.csv"
    orders_path = base_path / "cleaned_orders.csv"
    
    if not customers_path.exists(): 
        return json.dumps({"Error": f"Customer file not found for user {user_id}"})
        
    try:
        # 1. Read master customer list
        df_c = pd.read_csv(customers_path, encoding='utf-8-sig', dtype=str)
        df_c.columns = df_c.columns.str.strip().str.replace('\ufeff', '')
        
        cust_id_col = 'combinedid' if 'combinedid' in df_c.columns else 'id'
        if cust_id_col not in df_c.columns:
            return json.dumps({"Error": "Customer file missing ID column."})
            
        # 2. Get order counts for sorting/display
        order_counts = pd.Series(dtype=int)
        if orders_path.exists():
            df_o = pd.read_csv(orders_path, usecols=['customer_id'], encoding='utf-8-sig')
            order_counts = df_o['customer_id'].value_counts()
            
        # Ensure all NAs are filled with empty strings
        df_c = df_c.fillna('')
            
        # --- 3. APPLY HYBRID SEARCH LOGIC ---
        if search_query:
            search_lower = search_query.lower()
            
            # Strategy A: Global Substring Search (Searches EVERYTHING without fuzzy dilution)
            row_strings = df_c.astype(str).agg(' '.join, axis=1).str.lower()
            exact_mask = row_strings.str.contains(search_lower, regex=False)
            exact_indices = exact_mask[exact_mask].index.tolist()
            
            # Strategy B: Targeted Fuzzy Search (Searches key columns to handle typos)
            targeted_search_series = pd.Series("", index=df_c.index)
            for col in ['name', 'displayedName', 'customId_customId', 'billingAddress_formatted_address']:
                if col in df_c.columns:
                    targeted_search_series += df_c[col] + " "
                    
            matches = process.extract(
                search_lower, 
                targeted_search_series.str.lower().to_dict(), 
                scorer=fuzz.WRatio, 
                limit=None, 
                score_cutoff=70.0
            )
            fuzzy_indices = [match[2] for match in matches]
            
            # Combine the results from both strategies and remove duplicates
            all_matched_indices = list(set(exact_indices + fuzzy_indices))
            
            if not all_matched_indices:
                return json.dumps({"Error": f"No customers found matching '{search_query}'"})
            
            # Filter dataframe to only the matched rows
            df_c = df_c.loc[all_matched_indices]
            
        # 4. Build results list
        results = []
        for _, row in df_c.iterrows():
            c_id = row[cust_id_col]
            
            # Prioritize displayedName, fallback to name
            c_name = row.get('displayedName')
            if not c_name: c_name = row.get('name', 'Unknown')
                
            custom_id = row.get('customId_customId', '')
            count = order_counts.get(c_id, 0)
            
            display_parts = [str(c_name)]
            if custom_id:
                display_parts.append(f"[Custom ID: {custom_id}]")
            display_parts.append(f"({count} orders)")
            
            results.append({
                "display_name": " ".join(display_parts),
                "customer_id": str(c_id),
                "order_count": int(count)
            })
            
        # 5. Sort by most active customers first
        results.sort(key=lambda x: x["order_count"], reverse=True)
        
        # 6. Truncate to top 50 to protect context window
        is_truncated = len(results) > 50
        results = results[:50]
        
        # 7. Format as Dictionary for Agent consumption
        output_dict = {}
        for r in results:
            output_dict[r["display_name"]] = r["customer_id"]
            
        if is_truncated:
            output_dict["_warning"] = f"Found more than 50 matches. Showing top 50 by order count. Refine search_query if needed."
            
        return json.dumps(output_dict, indent=2)

    except Exception as e:
        return json.dumps({"Error": f"Failed: {str(e)}\n{traceback.format_exc()}"})

@mcp.tool(name="describe_customer")
@log_tool_usage
def describe_customer(user_id: Optional[str], search_query: Optional[str]) -> str:
    """
    Finds a specific customer and generates a comprehensive profile including 
    contact details, lifetime value, and an automated 'Health/Engagement' status.
    
    Parameters:
    - user_id (str): The unique identifier for the current user/workspace to locate the correct data files.
    - search_query (str): The specific identifier to search for. This can be an exact system UUID, a custom/internal ID (e.g., '794510'), a full name, or a partial name (e.g., 'bistro').
    
    Returns:
    - A formatted Markdown string containing the customer's profile, health status, and LTV. 
    - If multiple customers match the search_query, it returns a clarification prompt listing the options.
    - If no customer is found, it returns an error message.
    """
    if not search_query:
        return "Error: A search_query must be provided."
        
    base_path = Path("data") / str(user_id)
    customers_path = base_path / "cleaned_customers.csv"
    orders_path = base_path / "cleaned_orders.csv"
    
    sq = str(search_query).strip().lower()
    sq_numeric = sq.replace('.0', '')
    
    if not customers_path.exists():
        return f"Error: Customer file not found for user {user_id}"

    try:
        # --- 1. FAST CUSTOMER RESOLUTION ---
        # Load customers as strings to prevent ID parsing errors
        df_cust = pd.read_csv(customers_path, encoding='utf-8-sig', dtype=str).fillna('')
        df_cust.columns = df_cust.columns.str.strip().str.replace('\ufeff', '')
        cust_id_col = 'combinedid' if 'combinedid' in df_cust.columns else 'id'
        
        # Exact/Substring Match Logic
        mask_id = df_cust[cust_id_col].str.lower() == sq
        mask_custom = df_cust.get('customId_customId', pd.Series('')).str.lower().str.replace(r'\.0$', '', regex=True) == sq_numeric
        mask_name = df_cust.get('name', pd.Series('')).str.lower().str.contains(sq, regex=False)
        mask_disp = df_cust.get('displayedName', pd.Series('')).str.lower().str.contains(sq, regex=False)
        
        matched_custs = df_cust[mask_id | mask_custom | mask_name | mask_disp].copy()
        
        if matched_custs.empty:
            return f"No customers found matching '{search_query}'."
            
        # Handle Multiple Matches
        if len(matched_custs) > 1:
            mask_perfect_name = df_cust.get('name', pd.Series('')).str.lower() == sq
            mask_perfect_disp = df_cust.get('displayedName', pd.Series('')).str.lower() == sq
            perfect_matches = df_cust[mask_id | mask_custom | mask_perfect_name | mask_perfect_disp]
            
            if len(perfect_matches) == 1:
                matched_custs = perfect_matches
            else:
                names = [f"- {r.get('displayedName') or r.get('name')} (ID: {r.get('customId_customId', '')})" for _, r in matched_custs.head(5).iterrows()]
                return f"Multiple customers match '{search_query}'. Please clarify:\n" + "\n".join(names)

        target_customer = matched_custs.iloc[0]
        target_uuid = str(target_customer[cust_id_col]).strip()

        # --- 2. EXTRACT CORE PROFILE DATA ---
        c_name = target_customer.get('displayedName') or target_customer.get('name') or "Unknown"
        c_custom_id = target_customer.get('customId_customId', 'N/A').replace('.0', '')
        c_status = target_customer.get('status', 'UNKNOWN').upper()
        c_terms = target_customer.get('paymentTerms_name', 'Not Set')
        c_discount = target_customer.get('percentDiscount', '0')
        
        # Prioritize shipping address, fallback to billing
        address = target_customer.get('shippingAddress_formatted_address')
        if not address or address == '-':
            address = target_customer.get('billingAddress_formatted_address', 'No address on file')

        # --- 3. LOAD ORDERS (OPTIMIZED FOR SPEED) ---
        # We only load 3 columns to save memory and processing time
        order_count = 0
        total_spent = 0.0
        aov = 0.0
        days_since_last_order = None
        
        if orders_path.exists():
            df_ord = pd.read_csv(orders_path, usecols=lambda c: c in ['customer_id', 'totalAmount', 'createdAt'], encoding='utf-8-sig')
            customer_orders = df_ord[df_ord['customer_id'].astype(str).str.strip() == target_uuid].copy()
            
            if not customer_orders.empty:
                order_count = len(customer_orders)
                customer_orders['totalAmount'] = pd.to_numeric(customer_orders.get('totalAmount', 0), errors='coerce').fillna(0)
                total_spent = customer_orders['totalAmount'].sum()
                aov = total_spent / order_count if order_count > 0 else 0
                
                # Calculate Recency
                if 'createdAt' in customer_orders.columns:
                    customer_orders['createdAt'] = pd.to_datetime(customer_orders['createdAt'], errors='coerce').dt.tz_localize(None)
                    last_order_date = customer_orders['createdAt'].max()
                    if pd.notna(last_order_date):
                        days_since_last_order = (datetime.datetime.now() - last_order_date).days

        # --- 4. CALCULATE HEALTH & ENGAGEMENT FLAG ---
        health_flag = "Prospect (No Orders Yet)"
        if days_since_last_order is not None:
            if days_since_last_order <= 30:
                health_flag = f"Active & Engaged (Last order {days_since_last_order} days ago)"
            elif days_since_last_order <= 90:
                health_flag = f"Slipping / Needs Follow-up (Last order {days_since_last_order} days ago)"
            else:
                health_flag = f"Churn Risk / Inactive (Last order {days_since_last_order} days ago)"

        # Check for missing profile data
        warnings = []
        if address == 'No address on file': warnings.append("Missing Address")
        if c_terms in ['Not Set', '', '-']: warnings.append("Missing Payment Terms")
        warning_str = f"Profile Warnings: {', '.join(warnings)}" if warnings else "✅ Profile Complete"

        # --- 5. FORMAT OUTPUT ---
        lines = [
            f"## Customer Profile: {c_name} (ID: {c_custom_id})",
            f"**Status:** {c_status} | **Engagement:** {health_flag}",
            f"{warning_str}",
            "---",
            f"### Contact & Terms",
            f"- **Address:** {address.replace(chr(10), ' ')}",
            f"- **Payment Terms:** {c_terms}",
            f"- **Global Discount:** {c_discount}%",
            "",
            f"### Lifetime Value (LTV)",
            f"- **Total Orders:** {order_count:,}",
            f"- **Total Spent:** ${total_spent:,.2f}",
            f"- **Average Order Value:** ${aov:,.2f}"
        ]

        return '\n'.join(lines)

    except Exception as e:
        return f"Error analyzing customer: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_orders_by_customer")
@log_tool_usage
def get_orders_by_customer(
    user_id: Optional[str], 
    search_query: Optional[str], 
    limit: int = 10, 
    status_filter: Optional[str] = None,
    sort_by: Optional[str] = 'Date',
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Returns a summary and detailed list of orders for a specific customer.
    Requires the search to resolve to exactly ONE customer to prevent merged reports.
    """
    base_path = Path("data") / str(user_id)
    csv_path = base_path / "cleaned_orders.csv"
    customers_path = base_path / "cleaned_customers.csv"
    
    is_ascending = (sort_order.lower() == 'asc')
    
    if not search_query:
        return "Error: A search_query must be provided."
        
    sq = str(search_query).strip().lower()
    sq_numeric = sq.replace('.0', '')
    
    if not csv_path.exists():
        return f"Error: Orders file not found for user {user_id}"

    try:
        # --- 1. CUSTOMER RESOLUTION STEP ---
        if not customers_path.exists():
            return "Error: Customers file missing. Cannot resolve customer safely."
            
        df_cust = pd.read_csv(customers_path, encoding='utf-8-sig', dtype=str).fillna('')
        df_cust.columns = df_cust.columns.str.strip().str.replace('\ufeff', '')
        cust_id_col = 'combinedid' if 'combinedid' in df_cust.columns else 'id'
        
        # Build loose match masks
        mask_id = df_cust[cust_id_col].str.lower() == sq
        mask_custom = df_cust.get('customId_customId', pd.Series('')).str.lower().str.replace(r'\.0$', '', regex=True) == sq_numeric
        mask_name = df_cust.get('name', pd.Series('')).str.lower().str.contains(sq, regex=False)
        mask_disp = df_cust.get('displayedName', pd.Series('')).str.lower().str.contains(sq, regex=False)
        
        matched_custs = df_cust[mask_id | mask_custom | mask_name | mask_disp].copy()
        
        if matched_custs.empty:
            return f"No customers found matching '{search_query}'. Please try a different name or ID."
            
        # If multiple matches, try to find a single perfect exact match
        if len(matched_custs) > 1:
            mask_perfect_name = df_cust.get('name', pd.Series('')).str.lower() == sq
            mask_perfect_disp = df_cust.get('displayedName', pd.Series('')).str.lower() == sq
            
            perfect_matches = df_cust[mask_id | mask_custom | mask_perfect_name | mask_perfect_disp]
            
            if len(perfect_matches) == 1:
                matched_custs = perfect_matches
            else:
                # Still multiple matches - ABORT and ask user to clarify
                names = []
                for _, r in matched_custs.head(5).iterrows():
                    nm = r.get('displayedName') or r.get('name') or 'Unknown'
                    c_id = r.get('customId_customId', '')
                    names.append(f"- {nm} (Custom ID: {c_id})")
                
                msg = f"Multiple customers found matching '{search_query}'. Please refine your search to exactly one customer. Matches include:\n" + "\n".join(names)
                if len(matched_custs) > 5:
                    msg += f"\n...and {len(matched_custs) - 5} more."
                return msg

        # now EXACTLY ONE verified customer
        target_customer = matched_custs.iloc[0]
        target_uuid = str(target_customer[cust_id_col]).strip()
        display_name = target_customer.get('displayedName') or target_customer.get('name') or "Unknown Customer"


        # --- 2. LOAD & FILTER ORDERS ---
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        if 'customer_id' not in df.columns:
            return "Error: Orders file is missing the 'customer_id' column."
            
        # Match orders strictly to the resolved UUID
        customer_orders = df[df['customer_id'].astype(str).str.strip() == target_uuid].copy()
        
        if customer_orders.empty:
            return f"No orders found for customer: {display_name}"

        # --- 3. APPLY STATUS FILTER ---
        if status_filter:
            if 'orderStatus' in customer_orders.columns:
                customer_orders = customer_orders[
                    customer_orders['orderStatus'].fillna('').str.upper() == status_filter.upper()
                ]
            if customer_orders.empty:
                return f"No orders found for '{display_name}' with status '{status_filter}'"

        # --- 4. CLEANUP AND SORTING ---
        if 'createdAt' in customer_orders.columns:
            customer_orders['createdAt'] = pd.to_datetime(customer_orders['createdAt'], errors='coerce')
            if customer_orders['createdAt'].dt.tz is not None:
                 customer_orders['createdAt'] = customer_orders['createdAt'].dt.tz_localize(None)
             
        # Fallbacks for missing columns
        for col in ['totalAmount', 'totalQuantity']:
            if col not in customer_orders.columns:
                customer_orders[col] = 0.0
            else:
                customer_orders[col] = pd.to_numeric(customer_orders[col], errors='coerce').fillna(0.0)

        sort_mapping = {
            'Date': 'createdAt',
            'Total': 'totalAmount',
            'Qty': 'totalQuantity'
        }
        sort_col = sort_mapping.get(sort_by.capitalize(), 'createdAt')
        
        customer_orders = customer_orders.sort_values(by=sort_col, ascending=is_ascending)

        # --- 5. GENERATE SUMMARY HEADER ---
        total_spent = customer_orders['totalAmount'].sum()
        order_count = len(customer_orders)
        aov = total_spent / order_count if order_count > 0 else 0
        
        lines = []
        lines.append(f"## Customer Report: {display_name}")
        lines.append(f"- **Total Orders:** {order_count:,}")
        lines.append(f"- **Total Lifetime Value:** ${total_spent:,.2f}")
        lines.append(f"- **Average Order Value (AOV):** ${aov:,.2f}")
        if status_filter:
            lines.append(f"- **Status Filter Applied:** {status_filter.upper()}")
            
        lines.append(f"\n*(Showing top {min(limit, order_count)} orders | Sorted by: {sort_by} | Order: {sort_order.upper()})*")
        
        # --- 6. BUILD DISPLAY TABLE ---
        headers = ["Order ID", "Date", "Status", "Payment", "Qty", "Total ($)"]
        lines.append("\n| " + " | ".join(headers) + " |")
        lines.append("|---|---|---|---|---|---|")
        
        order_id_col = 'customId_customId' if 'customId_customId' in customer_orders.columns else 'id'
        display_df = customer_orders.head(limit)
        
        for _, row in display_df.iterrows():
            o_id = str(row.get(order_id_col, '-')).replace('.0', '')
            
            o_date = "-"
            if pd.notna(row.get('createdAt')):
                o_date = row['createdAt'].strftime('%m/%d/%Y')
                
            o_status = str(row.get('orderStatus', '-')).replace('_', ' ').title()
            p_status = str(row.get('paymentStatus', '-')).replace('_', ' ').title()
            o_qty = int(row.get('totalQuantity', 0))
            o_total = float(row.get('totalAmount', 0.0))
            
            lines.append(f"| {o_id} | {o_date} | {o_status} | {p_status} | {o_qty} | ${o_total:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error processing orders: {str(e)}\n{traceback.format_exc()}"
# List of tool from customer block agent

@mcp.tool(name="get_stopped_ordering_report")
@log_tool_usage
def get_stopped_ordering_report(
    user_id: str, 
    churn_threshold_days: Optional[int] = 90, 
    top_n: Optional[int] = 20,
    sort_by: Optional[str] = 'Total Spend',
    sort_order: Optional[str] = 'desc',
    min_orders: Optional[int] = None,
    min_spend: Optional[float] = None
) -> str:
    """
    Generates a churn report identifying customers who haven't ordered recently.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - churn_threshold_days: Number of days without an order to consider a customer 'churned' (default: 90).
    - top_n: Number of inactive customers to return (default: 20).
    - sort_by: Column to sort the report by. Options: 'Total Spend', 'Orders', 'Days Inactive', 'Last Order Date', 'Customer Name' (default: 'Total Spend').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - min_orders: Optional filter to only include customers with at least this many orders.
    - min_spend: Optional filter to only include customers who have spent at least this much.
    """
    base_path = Path("data") / user_id
    orders_path = base_path / "cleaned_orders.csv"
    customers_path = base_path / "cleaned_customers.csv"
    is_ascending = (sort_order.lower() == 'asc')
    
    lines = []
    lines.append("# Customer Inactivity Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d')}\n")

    try:
        if not (orders_path.exists() and customers_path.exists()):
             return f"Error: Data files not found for user {user_id}."

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        
        df_orders.columns = df_orders.columns.str.strip().str.replace('\ufeff', '')
        df_customers.columns = df_customers.columns.str.strip().str.replace('\ufeff', '')
        
        if 'createdAt' not in df_orders.columns:
             return "Error: Orders file missing 'createdAt' column."
             
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])
        if df_orders.empty: return "Error: No valid order data found."

        reference_date = df_orders['createdAt'].max()
        order_stats = df_orders.groupby('customer_id').agg({
            'createdAt': 'max', 'totalAmount': 'sum', 'customer_id': 'count'
        }).rename(columns={'createdAt': 'LastOrderDate', 'totalAmount': 'TotalSpend', 'customer_id': 'OrderCount'})
        
        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        name_col = 'name'
        if name_col not in df_customers.columns:
            name_col = 'displayedName' if 'displayedName' in df_customers.columns else cust_id_col
            
        merged_stats = order_stats.reset_index().merge(
            df_customers[[cust_id_col, name_col]], left_on='customer_id', right_on=cust_id_col, how='left'
        )
        merged_stats['FinalName'] = merged_stats[name_col].fillna(merged_stats['customer_id'])
        merged_stats['DaysSinceLastOrder'] = (reference_date - merged_stats['LastOrderDate']).dt.days
        
        # Base churn condition
        churned = merged_stats[merged_stats['DaysSinceLastOrder'] > churn_threshold_days].copy()
        
        # Apply Additional Filters
        if min_orders is not None:
            churned = churned[churned['OrderCount'] >= min_orders]
        if min_spend is not None:
            churned = churned[churned['TotalSpend'] >= min_spend]
        
        # Map human-readable 'sort_by' inputs to actual dataframe columns
        sort_mapping = {
            'Total Spend': 'TotalSpend',
            'Orders': 'OrderCount',
            'Days Inactive': 'DaysSinceLastOrder',
            'Last Order Date': 'LastOrderDate',
            'Customer Name': 'FinalName'
        }
        
        # Fallback to 'TotalSpend' if an invalid sort column is passed
        sort_col = sort_mapping.get(sort_by, 'TotalSpend')
        
        # Apply sorting logic
        churned = churned.sort_values(by=sort_col, ascending=is_ascending)
        
        total_customers = len(order_stats)
        total_churned = len(churned)
        churn_rate = (total_churned / total_customers * 100) if total_customers > 0 else 0
        lost_revenue = churned['TotalSpend'].sum()
        
        lines.append("## Executive Summary")
        lines.append(f"- **Total Active Customers:** {total_customers}")
        lines.append(f"- **Inactive (Churned) Customers:** {total_churned} ({churn_rate:.1f}%)")
        lines.append(f"- **Total Lifetime Value of Inactive Customers:** ${lost_revenue:,.2f}\n")
        
        # Add a note about applied filters/sorting
        lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        
        lines.append(f"## Inactive Customers List")
        headers = ["Customer Name", "Last Order Date", "Days Inactive", "Total Spend", "Orders"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        if churned.empty:
             lines.append("| No customers found matching criteria | - | - | - | - |")
        else:
            for _, row in churned.head(top_n).iterrows():
                name = str(row['FinalName']).replace("|", "-")
                last = row['LastOrderDate'].strftime('%Y-%m-%d')
                lines.append(f"| {name} | {last} | {int(row['DaysSinceLastOrder'])} | ${row['TotalSpend']:,.2f} | {row['OrderCount']} |")
            
        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating report: {str(e)}"

@mcp.tool(name="get_opportunity_report")
@log_tool_usage
def get_opportunity_report(
    user_id: str, 
    top_products_n: Optional[int] = 15, 
    top_bundles_n: Optional[int] = 5,
    sort_by: Optional[str] = 'Revenue',
    sort_order: Optional[str] = 'desc',
    min_revenue: Optional[float] = None,
    min_orders: Optional[int] = None
) -> str:
    """
    Generates an opportunity report showing top products and cross-selling bundles.
    
    Parameters:
    - user_id: The user's ID.
    - top_products_n: Number of top performing products to analyze (default: 15).
    - top_bundles_n: Number of top cross-sell bundle opportunities to return (default: 5).
    - sort_by: Column to sort products by. Options: 'Revenue', 'Orders', 'Avg Price' (default: 'Revenue').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - min_revenue: Optional filter to only show products with at least this much total revenue.
    - min_orders: Optional filter to only show products that were ordered at least this many times.
    """
    base_path = Path("data") / user_id
    orders_path = base_path / "cleaned_orders.csv"
    customers_path = base_path / "cleaned_customers.csv"
    products_path = base_path / "cleaned_products.csv"
    is_ascending = (sort_order.lower() == 'asc')

    lines = []
    lines.append("# Opportunity Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d')}\n")

    try:
        if not (orders_path.exists() and customers_path.exists() and products_path.exists()):
             return f"Error: Data files missing for user {user_id}."

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')

        for df in [df_orders, df_customers, df_products]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])

        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        name_col = 'name' if 'name' in df_customers.columns else ('displayedName' if 'displayedName' in df_customers.columns else cust_id_col)

        cust_subset = df_customers[[cust_id_col, name_col]].rename(columns={cust_id_col: 'CustomerPK', name_col: 'CustomerNameVal'})
        df_orders_merged = df_orders.merge(cust_subset, left_on='customer_id', right_on='CustomerPK', how='left')
        df_orders_merged['CustomerName'] = df_orders_merged['CustomerNameVal'].fillna(df_orders_merged['customer_id'])

        if 'sku' not in df_products.columns: df_products['sku'] = ''
        df_products['sku'] = df_products['sku'].fillna('')
        df_products['DisplayName'] = df_products.apply(lambda x: f"{x['name']} ({x['sku']})" if x['sku'] else x['name'], axis=1)

        lines.append("## 1. Product Performance")
        prod_stats = df_products.groupby('DisplayName').agg({
            'totalAmount': 'sum', 'orderId': 'nunique', 'price': 'mean' 
        }).rename(columns={'totalAmount': 'Revenue', 'orderId': 'Freq'})
        
        # Grab top 50 strictly by magnitude first to assign roles, then sort the display subset
        top_rev = prod_stats.sort_values(by='Revenue', ascending=False).head(50)
        top_freq = prod_stats.sort_values(by='Freq', ascending=False).head(50)
        combined_index = list(set(top_rev.index) | set(top_freq.index))
        combined_stats = prod_stats.loc[combined_index].copy()
        
        combined_stats['Role'] = combined_stats.apply(
            lambda r: "Star" if (r.name in top_rev.index and r.name in top_freq.index) else 
                      ("Cash Cow" if r.name in top_rev.index else "Traffic Builder"), axis=1
        )
        
        # Apply Filters
        if min_revenue is not None:
            combined_stats = combined_stats[combined_stats['Revenue'] >= min_revenue]
        if min_orders is not None:
            combined_stats = combined_stats[combined_stats['Freq'] >= min_orders]
        
        # Apply Sorting logic
        sort_mapping = {
            'Revenue': 'Revenue',
            'Orders': 'Freq',
            'Avg Price': 'price'
        }
        sort_col = sort_mapping.get(sort_by, 'Revenue')
        combined_stats = combined_stats.sort_values(sort_col, ascending=is_ascending).head(top_products_n)
        
        lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        lines.append("| Product | Role | Revenue | Orders | Avg Price |")
        lines.append("|---|---|---|---|---|")
        for name, row in combined_stats.iterrows():
            lines.append(f"| {name[:40]} | {row['Role']} | ${row['Revenue']:,.2f} | {row['Freq']} | ${row['price']:,.2f} |")

        lines.append("\n## 2. Bundle Opportunities & Cross-Selling")
        basket = df_products.groupby('orderId')['DisplayName'].apply(set)
        pair_counts = Counter()
        for items in basket:
            items_list = sorted(list(items))
            from itertools import combinations
            if len(items_list) > 1: pair_counts.update(combinations(items_list, 2))
                
        candidates = pair_counts.most_common(50)
        valid_opps = []
        
        # FIX 1: Map the actual Order ID to the Customer Name (Previously mapped customer_id to Name, causing failures)
        order_id_col = 'id' if 'id' in df_orders_merged.columns else df_orders_merged.columns[0]
        order_cust_map = df_orders_merged.set_index(order_id_col)['CustomerName'].to_dict()
        
        cust_inv = {}
        for oid, items in basket.items():
            c = order_cust_map.get(oid)
            if c:
                if c not in cust_inv: cust_inv[c] = set()
                cust_inv[c].update(items)

        for (prod_a, prod_b), freq in candidates:
            price_a = prod_stats.loc[prod_a, 'price'] if prod_a in prod_stats.index else 0
            price_b = prod_stats.loc[prod_b, 'price'] if prod_b in prod_stats.index else 0
            bundle_val = price_a + price_b
            
            targets_need_a = []
            targets_need_b = []
            
            # FIX 2: Correctly calculate the missed revenue based ONLY on the item they are missing
            for c, inv in cust_inv.items():
                if (prod_a in inv) and (prod_b not in inv):
                    targets_need_b.append(f"{str(c)} (Needs {prod_b})")
                elif (prod_b in inv) and (prod_a not in inv):
                    targets_need_a.append(f"{str(c)} (Needs {prod_a})")
            
            # They already bought one item, so the potential revenue is just the price of the item they didn't buy
            potential_rev = (len(targets_need_a) * price_a) + (len(targets_need_b) * price_b)
            targets = targets_need_a + targets_need_b
            
            if len(targets) > 0:
                valid_opps.append({
                    'Bundle': f"{prod_a[:20]} + {prod_b[:20]}", 
                    'Freq': freq, 
                    'UnitValue': bundle_val,
                    'PotentialRev': potential_rev, 
                    'TargetList': ", ".join(targets[:3])
                })

        # Apply user sort order to potential revenue
        valid_opps.sort(key=lambda x: x['PotentialRev'], reverse=not is_ascending)
        
        lines.append("| Bundle Pair | Common Orders | Bundle Price | Potential Revenue | Missed Opportunity (Targets) |")
        lines.append("|---|---|---|---|---|")
        
        if valid_opps:
            for op in valid_opps[:top_bundles_n]:
                lines.append(f"| {op['Bundle']} | {op['Freq']} | ${op['UnitValue']:,.2f} | **${op['PotentialRev']:,.2f}** | {op['TargetList']} |")
        else:
            lines.append("| Market saturation reached | - | - | - | - |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating opportunity report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_top_customers_report")
@log_tool_usage
def get_top_customers_report(
    user_id: str, 
    top_n: Optional[int] = 10, 
    vip_n: Optional[int] = 5,
    sort_by: Optional[str] = 'Total Revenue',
    sort_order: Optional[str] = 'desc',
    min_orders: Optional[int] = None,
    min_revenue: Optional[float] = None,
    min_aov: Optional[float] = None,
    max_days_inactive: Optional[int] = None
) -> str:
    """
    Generates a leaderboard and deep dive of top customers based on dynamic criteria.
    
    Parameters:
    - user_id: The user's ID.
    - top_n: Number of customers to show in the leaderboard table (default: 10).
    - vip_n: Number of customers to feature in the deep dive product profile (default: 5).
    - sort_by: Column to sort the leaderboard by. Options: 'Total Revenue', 'AOV', 'Orders', 'Days Inactive' (default: 'Total Revenue').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - min_orders: Optional filter for minimum number of orders.
    - min_revenue: Optional filter for minimum total revenue.
    - min_aov: Optional filter for minimum Average Order Value.
    - max_days_inactive: Optional filter to only include customers who have ordered within the last X days.
    """
    base_path = Path("data") / user_id
    orders_path = base_path / "cleaned_orders.csv"
    customers_path = base_path / "cleaned_customers.csv"
    products_path = base_path / "cleaned_products.csv"
    is_ascending = (sort_order.lower() == 'asc')

    lines = []
    lines.append("# Customer Intelligence Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d')}\n")

    try:
        if not (orders_path.exists() and customers_path.exists() and products_path.exists()):
             return f"Error: Data files missing for user {user_id}."

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')

        for df in [df_orders, df_customers, df_products]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])
        
        if df_orders.empty:
            return "Error: No valid order data found."
            
        reference_date = df_orders['createdAt'].max()

        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        name_col = 'name' if 'name' in df_customers.columns else ('displayedName' if 'displayedName' in df_customers.columns else cust_id_col)

        cust_subset = df_customers[[cust_id_col, name_col]].rename(columns={cust_id_col: 'CustomerPK', name_col: 'CustomerNameVal'})
        df_orders_merged = df_orders.merge(cust_subset, left_on='customer_id', right_on='CustomerPK', how='left')
        df_orders_merged['CustomerName'] = df_orders_merged['CustomerNameVal'].fillna(df_orders_merged['customer_id'])

        if 'sku' not in df_products.columns: df_products['sku'] = ''
        df_products['sku'] = df_products['sku'].fillna('')
        df_products['DisplayName'] = df_products.apply(lambda x: f"{x['name']} ({x['sku']})" if x['sku'] else x['name'], axis=1)

        # Link products to customer names for the VIP section
        order_id_col = 'id' if 'id' in df_orders_merged.columns else df_orders_merged.columns[0]
        df_products_linked = df_products.merge(
            df_orders_merged[[order_id_col, 'CustomerName']], left_on='orderId', right_on=order_id_col, how='left'
        )

        # Base Aggregation
        customer_stats = df_orders_merged.groupby('CustomerName').agg({
            'totalAmount': 'sum',
            'customer_id': 'count',
            'createdAt': 'max'
        }).rename(columns={
            'totalAmount': 'TotalRevenue', 
            'customer_id': 'OrderCount', 
            'createdAt': 'LastOrderDate'
        })
        
        customer_stats['AOV'] = customer_stats['TotalRevenue'] / customer_stats['OrderCount']
        customer_stats['DaysSinceLastOrder'] = (reference_date - customer_stats['LastOrderDate']).dt.days
        
        # Apply Filters
        if min_orders is not None:
            customer_stats = customer_stats[customer_stats['OrderCount'] >= min_orders]
        if min_revenue is not None:
            customer_stats = customer_stats[customer_stats['TotalRevenue'] >= min_revenue]
        if min_aov is not None:
            customer_stats = customer_stats[customer_stats['AOV'] >= min_aov]
        if max_days_inactive is not None:
            customer_stats = customer_stats[customer_stats['DaysSinceLastOrder'] <= max_days_inactive]

        # Apply Sorting
        sort_mapping = {
            'Total Revenue': 'TotalRevenue',
            'AOV': 'AOV',
            'Orders': 'OrderCount',
            'Days Inactive': 'DaysSinceLastOrder'
        }
        sort_col = sort_mapping.get(sort_by, 'TotalRevenue')
        customer_stats = customer_stats.sort_values(by=sort_col, ascending=is_ascending)

        top_customers = customer_stats.head(top_n)

        # Extract VIP profiles from the top of the currently filtered/sorted list
        vip_names = top_customers.head(vip_n).index.tolist()
        vip_profiles = []
        for cust in vip_names:
            stats = customer_stats.loc[cust]
            
            # FIX: Added .copy() here to avoid SettingWithCopyWarning
            cust_prods = df_products_linked[df_products_linked['CustomerName'] == cust].copy()
            
            fav_str = "No product data"
            if not cust_prods.empty and 'quantity' in cust_prods.columns:
                cust_prods['quantity'] = pd.to_numeric(cust_prods['quantity'], errors='coerce').fillna(1)
                fav_prods = cust_prods.groupby('DisplayName')['quantity'].sum().sort_values(ascending=False).head(3)
                fav_str = ", ".join([f"{p_name} ({int(qty)})" for p_name, qty in fav_prods.items()])
            
            vip_profiles.append({
                'Customer': cust, 'Revenue': stats['TotalRevenue'], 'Orders': stats['OrderCount'],
                'LastSeen': f"{int(stats['DaysSinceLastOrder'])} days ago", 'Favorites': fav_str
            })
            
        df_vip = pd.DataFrame(vip_profiles)

        # Render output
        lines.append(f"*(Filtered by applied criteria | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        
        lines.append(f"## Top Customers Leaderboard")
        lines.append("| Rank | Customer | Total Revenue | Orders | AOV | Last Order Date | Days Inactive |")
        lines.append("|---|---|---|---|---|---|---|")
        
        if top_customers.empty:
            lines.append("| No customers found matching criteria | - | - | - | - | - | - |")
        else:
            for i, (name, row) in enumerate(top_customers.iterrows(), 1):
                last_date = row['LastOrderDate'].strftime('%Y-%m-%d')
                lines.append(f"| {i} | {name[:40]} | ${row['TotalRevenue']:,.2f} | {row['OrderCount']} | ${row['AOV']:,.2f} | {last_date} | {int(row['DaysSinceLastOrder'])} |")

        lines.append(f"\n## VIP Deep Dive (Top {len(df_vip)})")
        lines.append("| Customer | Revenue | Orders | Last Seen | Top 3 Favorite Products |")
        lines.append("|---|---|---|---|---|")
        if not df_vip.empty:
            for _, r in df_vip.iterrows():
                lines.append(f"| {r['Customer'][:30]} | ${r['Revenue']:,.2f} | {r['Orders']} | {r['LastSeen']} | {r['Favorites']} |")
        else:
            lines.append("| No VIPs to display based on criteria | - | - | - | - |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_visits_report")
@log_tool_usage
def get_visits_report(
    user_id: str, 
    churn_days: Optional[int] = 90, 
    revisit_days: Optional[int] = 60, 
    top_n: Optional[int] = 10,
    sort_by: Optional[str] = 'Total Revenue',
    sort_order: Optional[str] = 'desc',
    min_orders: Optional[int] = None,
    min_revenue: Optional[float] = None
) -> str:
    """
    Generates a report comparing visited vs. unvisited customers and highlights at-risk accounts.
    
    Parameters:
    - user_id: The user's ID.
    - churn_days: Days since last order to consider a customer at-risk/churned (default: 90).
    - revisit_days: Days since last visit to prompt a re-visit warning (default: 60).
    - top_n: Number of customers to display in each category (default: 10).
    - sort_by: Column to sort by. Options: 'Total Revenue', 'Orders', 'Days Since Visit', 'Days Since Order' (default: 'Total Revenue').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - min_orders: Optional filter to only include customers with at least this many orders.
    - min_revenue: Optional filter to only include customers who have spent at least this much.
    """
    base_path = Path("data") / user_id
    orders_path = base_path / "cleaned_orders.csv"
    customers_path = base_path / "cleaned_customers.csv"
    is_ascending = (sort_order.lower() == 'asc')

    lines = []
    lines.append("# Visited vs Not Visited Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        if not (orders_path.exists() and customers_path.exists()):
             return f"Error: Data files missing for user {user_id}."

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        
        df_orders.columns = df_orders.columns.str.strip().str.replace('\ufeff', '')
        df_customers.columns = df_customers.columns.str.strip().str.replace('\ufeff', '')

        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        if df_orders['createdAt'].dt.tz is not None:
            df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])
        reference_date = pd.Timestamp.now().replace(tzinfo=None)
        
        if 'lastCheckInAt' in df_customers.columns:
            df_customers['lastCheckInAt'] = pd.to_datetime(
                df_customers['lastCheckInAt'], errors='coerce',
                format='%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)'
            )
            if df_customers['lastCheckInAt'].isna().all():
                 df_customers['lastCheckInAt'] = pd.to_datetime(df_customers['lastCheckInAt'], errors='coerce')
            if df_customers['lastCheckInAt'].dt.tz is not None:
                df_customers['lastCheckInAt'] = df_customers['lastCheckInAt'].dt.tz_localize(None)
        else:
            df_customers['lastCheckInAt'] = pd.NaT

        if 'salesDuplicate_name' in df_orders.columns:
            last_reps = df_orders.sort_values('createdAt', ascending=False).groupby('customer_id')['salesDuplicate_name'].first().reset_index()
            last_reps.columns = ['customer_id', 'LastRep']
        else:
            last_reps = pd.DataFrame(columns=['customer_id', 'LastRep'])

        order_stats = df_orders.groupby('customer_id').agg({
            'totalAmount': 'sum', 'id': 'count', 'createdAt': 'max'
        }).rename(columns={'totalAmount': 'TotalRevenue', 'id': 'OrderCount', 'createdAt': 'LastOrderDate'}).reset_index()

        order_stats = order_stats.merge(last_reps, on='customer_id', how='left')
        order_stats['LastRep'] = order_stats['LastRep'].fillna("Unknown")

        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        df_merged = df_customers.merge(order_stats, left_on=cust_id_col, right_on='customer_id', how='left')
        
        name_col = 'name' if 'name' in df_merged.columns else ('displayedName' if 'displayedName' in df_merged.columns else cust_id_col)
        df_merged['CustomerName'] = df_merged[name_col].fillna(df_merged[cust_id_col]).astype(str)

        df_merged['TotalRevenue'] = df_merged['TotalRevenue'].fillna(0)
        df_merged['OrderCount'] = df_merged['OrderCount'].fillna(0)
        df_merged['HasOrders'] = df_merged['OrderCount'] > 0
        df_merged['LastRep'] = df_merged['LastRep'].fillna("No Orders")
        
        df_merged['IsVisited'] = df_merged['lastCheckInAt'].notna()
        df_merged['DaysSinceVisit'] = df_merged['lastCheckInAt'].apply(lambda x: (reference_date - x).days if pd.notna(x) else float('inf'))
        df_merged['DaysSinceOrder'] = df_merged['LastOrderDate'].apply(lambda x: (reference_date - x).days if pd.notna(x) else float('inf'))
        df_merged['IsChurned'] = (df_merged['DaysSinceOrder'] > churn_days) | (df_merged['OrderCount'] == 0)

        # Apply Additional Filters
        if min_orders is not None:
            df_merged = df_merged[df_merged['OrderCount'] >= min_orders]
        if min_revenue is not None:
            df_merged = df_merged[df_merged['TotalRevenue'] >= min_revenue]

        # Map human-readable 'sort_by' inputs to actual dataframe columns
        sort_mapping = {
            'Total Revenue': 'TotalRevenue',
            'Orders': 'OrderCount',
            'Days Since Visit': 'DaysSinceVisit',
            'Days Since Order': 'DaysSinceOrder'
        }
        sort_col = sort_mapping.get(sort_by, 'TotalRevenue')

        # Apply sorting logic based on the mapped column
        unvisited_gold = df_merged[~df_merged['IsVisited'] & (df_merged['TotalRevenue'] > 0)].sort_values(by=sort_col, ascending=is_ascending).head(top_n)
        revisit_candidates = df_merged[(df_merged['IsVisited']) & (df_merged['DaysSinceVisit'] > revisit_days) & (df_merged['TotalRevenue'] > 0)].sort_values(by=sort_col, ascending=is_ascending).head(top_n)
        failed_visits = df_merged[(df_merged['IsVisited']) & (df_merged['DaysSinceVisit'] < churn_days) & (df_merged['IsChurned']) & (df_merged['TotalRevenue'] > 0)].sort_values(by=sort_col, ascending=is_ascending).head(top_n)

        lines.append(f"*(Filtered by applied criteria | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        lines.append(f"## 1. Top Opportunities (Unvisited)")
        lines.append("| # | Customer | Total Revenue | Orders | Last Order |")
        lines.append("|---|---|---|---|---|")
        if not unvisited_gold.empty:
            for i, (_, r) in enumerate(unvisited_gold.iterrows(), 1):
                last_order = r['LastOrderDate'].strftime('%Y-%m-%d') if pd.notna(r['LastOrderDate']) else "Never"
                lines.append(f"| {i} | {r['CustomerName'][:40]} | ${r['TotalRevenue']:,.2f} | {int(r['OrderCount'])} | {last_order} |")
        else:
            lines.append("| - | No unvisited opportunities found matching criteria | - | - | - |")

        lines.append(f"\n## 2. Re-Visit Candidates (> {revisit_days} days)")
        lines.append("| # | Customer | Days Since Visit | Total Revenue | Last Order |")
        lines.append("|---|---|---|---|---|")
        if not revisit_candidates.empty:
            for i, (_, r) in enumerate(revisit_candidates.iterrows(), 1):
                lines.append(f"| {i} | {r['CustomerName'][:40]} | {int(r['DaysSinceVisit'])} days | ${r['TotalRevenue']:,.2f} | {r['LastOrderDate'].strftime('%Y-%m-%d')} |")
        else:
            lines.append("| - | No re-visit candidates found matching criteria | - | - | - |")

        lines.append(f"\n## 3. At-Risk Visits")
        lines.append("| # | Customer | Visit Date | Last Order | Days Since Order | Last Rep | Revenue Risk |")
        lines.append("|---|---|---|---|---|---|---|")
        if not failed_visits.empty:
            for i, (_, r) in enumerate(failed_visits.iterrows(), 1):
                visit_date = r['lastCheckInAt'].strftime('%Y-%m-%d') if pd.notna(r['lastCheckInAt']) else "Unknown"
                last_order = r['LastOrderDate'].strftime('%Y-%m-%d') if pd.notna(r['LastOrderDate']) else "Never"
                lines.append(f"| {i} | {r['CustomerName'][:40]} | {visit_date} | {last_order} | {int(r['DaysSinceOrder'])} | {r['LastRep']} | ${r['TotalRevenue']:,.2f} |")
        else:
            lines.append("| - | No at-risk visits found matching criteria | - | - | - | - | - |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating visits report: {str(e)}\n{traceback.format_exc()}"


# List of tools from orders block agent

@mcp.tool(name="get_financial_metrics_report")
@log_tool_usage
def get_financial_metrics_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_status_breakdown: Optional[bool] = True,
    group_by_period: Optional[str] = None
) -> str:
    """
    Generates an executive summary of key financial metrics, with delivery fee analysis, trend analysis, and status breakdowns.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (YYYY-MM-DD).
    - end_date: Optional end date filter (YYYY-MM-DD).
    - include_status_breakdown: Whether to include fulfillment and payment status tables (default: True).
    - group_by_period: Optional time grouping for trend analysis. Options: 'day', 'week', 'month', 'quarter', 'year'.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'createdAt' not in df.columns or 'totalAmount' not in df.columns:
            return "Error: Required columns ('createdAt', 'totalAmount') missing from orders data."

        # Clean Dates & Timezones
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        if df['createdAt'].dt.tz is not None:
             df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df = df.dropna(subset=['createdAt'])

        # Apply Date Filters
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df = df[df['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df = df[df['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

        if df.empty:
            return "No order data found for the specified parameters."

        # Ensure numeric columns safely
        df['totalAmount'] = pd.to_numeric(df['totalAmount'], errors='coerce').fillna(0)
        
        has_discount = 'totalDiscountValue' in df.columns
        if has_discount:
            df['totalDiscountValue'] = pd.to_numeric(df['totalDiscountValue'], errors='coerce').fillna(0)
            
        has_delivery = 'deliveryFee' in df.columns
        if has_delivery:
            df['deliveryFee'] = pd.to_numeric(df['deliveryFee'], errors='coerce').fillna(0)

        # 1. Base Metrics
        total_revenue = df['totalAmount'].sum()
        total_orders = len(df)
        avg_order_value = df['totalAmount'].mean() if total_orders > 0 else 0
        std_order_value = df['totalAmount'].std() if total_orders > 1 else 0

        # 2. Fees & Discounts
        total_discounts = df['totalDiscountValue'].sum() if has_discount else 0
        total_delivery = df['deliveryFee'].sum() if has_delivery else 0
        
        orders_with_delivery = (df['deliveryFee'] > 0).sum() if has_delivery else 0
        pct_with_delivery = (orders_with_delivery / total_orders * 100) if total_orders > 0 else 0
        
        # Calculate avg delivery ONLY for orders that actually had a delivery fee > 0
        avg_delivery = df.loc[df['deliveryFee'] > 0, 'deliveryFee'].mean() if orders_with_delivery > 0 else 0
        std_delivery = df['deliveryFee'].std() if total_orders > 1 and has_delivery else 0

        lines = [
            f"# Executive Financial Metrics",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d')}",
            f"**Date Range:** {start_date or 'All Time'} to {end_date or 'Present'}\n",
            "## 1. Executive Key Metrics",
            "*Overview of financial performance and order variability.*\n",
            "### Sales & Orders",
            f"- **Total Sales:** ${total_revenue:,.2f}",
            f"- **Total Orders:** {total_orders}",
            f"- **Average Order Value:** ${avg_order_value:,.2f}",
            f"- **Order Value Standard Deviation:** ${std_order_value:,.2f}",
            "> *Standard Deviation measures consistency. A high number means order sizes vary wildly; a low number means most orders are around the average.*\n",
            "### Fees & Discounts",
            f"- **Total Discounts Given:** ${total_discounts:,.2f}",
            f"- **Total Delivery Fees Collected:** ${total_delivery:,.2f}",
            f"- **Orders with Delivery Fee:** {orders_with_delivery} ({pct_with_delivery:.1f}%)",
            f"- **Average Delivery Fee:** ${avg_delivery:,.2f}",
            f"- **Delivery Fee Standard Deviation:** ${std_delivery:,.2f}\n"
        ]

        # Trend Analysis (Time Period Breakdown)
        if group_by_period:
            period_mapping = {
                'day': 'D', 'week': 'W', 'month': 'ME', 
                'quarter': 'QE', 'year': 'YE'
            }
            alias = period_mapping.get(group_by_period.lower(), 'ME')
            try:
                period_col = df['createdAt'].dt.to_period(alias.replace('E', '')) 
            except ValueError:
                period_col = df['createdAt'].dt.to_period(alias[0])
                
            df_trend = df.groupby(period_col).agg(
                Orders=('id', 'count'),
                Revenue=('totalAmount', 'sum'),
                Discounts=('totalDiscountValue', 'sum') if has_discount else ('id', 'count'), # dummy if missing
                DeliveryFees=('deliveryFee', 'sum') if has_delivery else ('id', 'count')
            ).reset_index()
            
            df_trend['AOV'] = df_trend['Revenue'] / df_trend['Orders']
            
            lines.append(f"## 2. Trend Analysis (Grouped by {group_by_period.capitalize()})")
            
            # Dynamic headers based on available columns
            headers = ["Period", "Orders", "Revenue", "AOV"]
            if has_discount: headers.append("Discounts")
            if has_delivery: headers.append("Delivery Fees")
                
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            
            for _, row in df_trend.iterrows():
                period_str = str(row[period_col.name])
                row_data = [
                    period_str,
                    str(int(row['Orders'])),
                    f"${row['Revenue']:,.2f}",
                    f"${row['AOV']:,.2f}"
                ]
                if has_discount: row_data.append(f"${row['Discounts']:,.2f}")
                if has_delivery: row_data.append(f"${row['DeliveryFees']:,.2f}")
                
                lines.append("| " + " | ".join(row_data) + " |")
                
            lines.append("")

        # Status Breakdowns
        if include_status_breakdown:
            section_num = "3" if group_by_period else "2"
            lines.append(f"## {section_num}. Operational Breakdowns\n")
            
            for col, title in [('deliveryStatus', 'Fulfillment'), ('paymentStatus', 'Payment')]:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
                    stats = df.groupby(col).agg(Count=('id', 'count'), Revenue=('totalAmount', 'sum')).sort_values('Count', ascending=False)
                    
                    lines.extend([f"### {title} Status", f"| {title} Status | Orders | % of Total | Revenue |", "|---|---|---|---|"])
                    for status, row in stats.iterrows():
                        pct = (row['Count'] / total_orders * 100) if total_orders > 0 else 0
                        clean_status = str(status).replace('_', ' ').title()
                        lines.append(f"| {clean_status} | {int(row['Count'])} | {pct:.1f}% | ${row['Revenue']:,.2f} |")
                    lines.append("")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating metrics report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_sales_performance_report")
@log_tool_usage
def get_sales_performance_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = 'Month',
    sort_order: Optional[str] = 'desc',
    min_orders: Optional[int] = None,
    min_revenue: Optional[float] = None
) -> str:
    """
    Calculates monthly sales performance with Month-over-Month (MoM) % change.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (YYYY-MM-DD).
    - end_date: Optional end date filter (YYYY-MM-DD).
    - sort_by: Column to sort the table by. Options: 'Month', 'Total Sales', 'Orders', 'AOV', '% Change' (default: 'Month').
    - sort_order: 'desc' (default) or 'asc'.
    - min_orders: Optional filter to only show months with at least this many orders.
    - min_revenue: Optional filter to only show months with at least this much total sales.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'createdAt' not in df.columns or 'totalAmount' not in df.columns:
            return "Error: Required columns ('createdAt', 'totalAmount') missing from orders data."

        # 1. Clean Dates & Apply Date Filters
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        if df['createdAt'].dt.tz is not None:
             df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df = df.dropna(subset=['createdAt'])

        if start_date:
            try:
                df = df[df['createdAt'] >= pd.to_datetime(start_date)]
            except Exception:
                return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df = df[df['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

        if df.empty:
            return "No order data found for the specified period."

        df['totalAmount'] = pd.to_numeric(df['totalAmount'], errors='coerce').fillna(0)
        
        # 2. Extract Canonical Month (YYYY-MM format)
        df['Month_Str'] = df['createdAt'].dt.strftime('%Y-%m')
        
        # Use order ID or customer ID for counting
        count_col = 'id' if 'id' in df.columns else 'customer_id'

        # 3. Group by Month
        monthly = df.groupby('Month_Str').agg(
            total_sales=('totalAmount', 'sum'),
            order_count=(count_col, 'count')
        ).reset_index()

        # 4. CRITICAL: Calculate % Change in strict chronological order BEFORE applying other sorts/filters
        monthly = monthly.sort_values('Month_Str')
        monthly['pct_change'] = monthly['total_sales'].pct_change() * 100
        monthly['aov'] = monthly['total_sales'] / monthly['order_count']

        # 5. Apply Value Filters
        if min_orders is not None:
            monthly = monthly[monthly['order_count'] >= min_orders]
        if min_revenue is not None:
            monthly = monthly[monthly['total_sales'] >= min_revenue]

        # 6. Apply Dynamic Sorting
        sort_mapping = {
            'Month': 'Month_Str',
            'Total Sales': 'total_sales',
            'Orders': 'order_count',
            'AOV': 'aov',
            '% Change': 'pct_change'
        }
        
        sort_col = sort_mapping.get(sort_by, 'Month_Str')
        monthly = monthly.sort_values(by=sort_col, ascending=is_ascending)

        # 7. Formatting output
        lines = [
            f"## Sales Performance & Trends",
            f"*Monthly breakdown of revenue and order volume.*"
        ]
        
        filter_str = []
        if start_date or end_date: 
            filter_str.append(f"Dates: {start_date or 'All'} to {end_date or 'Present'}")
        if min_revenue: filter_str.append(f"Min Rev: ${min_revenue}")
        if min_orders: filter_str.append(f"Min Orders: {min_orders}")
        
        if filter_str:
            lines.append(f"*(Filters applied: {', '.join(filter_str)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        lines.append("| Month | Total Sales | Orders | Avg Sales/Order | % Change (MoM) |")
        lines.append("|---|---|---|---|---|")

        if monthly.empty:
            lines.append("| No data found matching criteria | - | - | - | - |")
        else:
            for _, row in monthly.iterrows():
                # Format % Change safely
                if pd.notna(row['pct_change']):
                    change_str = f"{row['pct_change']:+.1f}%"
                else:
                    change_str = "-"
                    
                lines.append(f"| {row['Month_Str']} | ${row['total_sales']:,.2f} | {int(row['order_count'])} | ${row['aov']:,.2f} | {change_str} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating sales performance report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_discount_distribution_report")
@log_tool_usage
def get_discount_distribution_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = 'Orders',
    sort_order: Optional[str] = 'desc',
    min_orders: Optional[int] = None,
    min_revenue: Optional[float] = None
) -> str:
    """
    Calculates discount distribution, efficiency, and performance vs baseline (No Discount).
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - sort_by: Column to sort the table by. Options: 'Discount Type', 'Orders', 'Total Discount', 'AOV', 'Lift' (default: 'Orders').
    - sort_order: 'desc' (default) or 'asc'.
    - min_orders: Optional filter to only show discount types used at least this many times.
    - min_revenue: Optional filter to only show discount types that generated at least this much total sales.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 1. Date Cleaning & Filtering
        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
            if df['createdAt'].dt.tz is not None:
                df['createdAt'] = df['createdAt'].dt.tz_localize(None)
            df = df.dropna(subset=['createdAt'])

            if start_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df['createdAt'] >= start_dt]
                except Exception:
                    return "Error: Invalid start_date format. Use MM/DD/YYYY."

            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                    df = df[df['createdAt'] <= end_dt]
                except Exception:
                    return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df.empty:
            return "No order data found for the specified period."

        # 2. Ensure Numeric Columns
        df['totalAmount'] = pd.to_numeric(df.get('totalAmount', 0), errors='coerce').fillna(0)
        df['totalDiscountValue'] = pd.to_numeric(df.get('totalDiscountValue', 0), errors='coerce').fillna(0)
        
        # 3. Categorize Discounts
        if 'appliedDiscountsType' not in df.columns:
            df['DiscountType'] = np.where(df['totalDiscountValue'] > 0, 'Generic', 'No Discount')
        else:
            df['DiscountType'] = df['appliedDiscountsType'].fillna('No Discount').astype(str)
            df.loc[df['DiscountType'].str.upper() == 'NONE', 'DiscountType'] = 'No Discount'
            # Catch edge cases where it says "No Discount" but has a discount value
            df.loc[(df['DiscountType'] == 'No Discount') & (df['totalDiscountValue'] > 0), 'DiscountType'] = 'Custom/Other'

        total_orders = len(df)
        num_with_disc = (df['totalDiscountValue'] > 0).sum()
        pct_with_disc = (num_with_disc / total_orders * 100) if total_orders > 0 else 0

        # 4. Aggregation
        # Count using whichever ID column is available
        id_col = 'id' if 'id' in df.columns else ('customer_id' if 'customer_id' in df.columns else df.columns[0])
        
        stats = df.groupby('DiscountType').agg(
            Count=(id_col, 'count'),
            TotalDiscount=('totalDiscountValue', 'sum'),
            TotalSales=('totalAmount', 'sum') 
        ).reset_index()
        
        stats['AvgOrderValue'] = stats['TotalSales'] / stats['Count']

        # 5. Calculate Baseline & Lift analytically before sorting
        baseline_row = stats[stats['DiscountType'] == 'No Discount']
        baseline_aov = baseline_row['AvgOrderValue'].iloc[0] if not baseline_row.empty else 0
        
        stats['Lift'] = np.where(
            (stats['DiscountType'] != 'No Discount') & (baseline_aov > 0),
            (stats['AvgOrderValue'] - baseline_aov) / baseline_aov * 100,
            np.nan
        )

        # 6. Apply Filters
        if min_orders is not None:
            stats = stats[stats['Count'] >= min_orders]
        if min_revenue is not None:
            stats = stats[stats['TotalSales'] >= min_revenue]

        # 7. Dynamic Sorting
        sort_mapping = {
            'Discount Type': 'DiscountType',
            'Orders': 'Count',
            'Total Discount': 'TotalDiscount',
            'AOV': 'AvgOrderValue',
            'Lift': 'Lift'
        }
        
        sort_col = sort_mapping.get(sort_by, 'Count')
        # Fill NaN lift with -infinity for sorting purposes so they drop to the bottom if descending
        stats['Sort_Helper'] = stats[sort_col].fillna(-float('inf') if not is_ascending else float('inf'))
        stats = stats.sort_values(by='Sort_Helper', ascending=is_ascending).drop(columns=['Sort_Helper'])

        # 8. Formatting
        lines = [
            f"## Discount Distribution & Efficiency",
            f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}"
        ]
        
        filter_str = []
        if start_date or end_date: 
            filter_str.append(f"Dates: {start_date or 'All'} to {end_date or 'Present'}")
        if min_revenue: filter_str.append(f"Min Rev: ${min_revenue}")
        if min_orders: filter_str.append(f"Min Orders: {min_orders}")
        
        if filter_str:
            lines.append(f"*(Filters applied: {', '.join(filter_str)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        lines.append(f"- **Total Orders with Discounts:** {num_with_disc} ({pct_with_disc:.1f}%)\n")
        
        lines.append("| Discount Type | Orders | Total Discount Given | Avg Order Value | Performance vs Baseline |")
        lines.append("|---|---|---|---|---|")

        if stats.empty:
            lines.append("| No data found matching criteria | - | - | - | - |")
        else:
            for _, row in stats.iterrows():
                # Clean up the status text
                raw_name = str(row['DiscountType'])
                name = raw_name if raw_name in ['No Discount', 'Custom/Other'] else raw_name.replace('_', ' ').title()
                
                count = int(row['Count'])
                disc_val = f"${row['TotalDiscount']:,.2f}"
                aov = f"${row['AvgOrderValue']:,.2f}"
                
                if raw_name == 'No Discount':
                    perf = "(Baseline)"
                elif pd.notna(row['Lift']):
                    perf = f"{row['Lift']:+.1f}% Lift"
                else:
                    perf = "-"
                    
                lines.append(f"| {name} | {count} | {disc_val} | {aov} | {perf} |")

        lines.extend([
            "",
            "> **What is Performance vs Baseline?**",
            "> This compares the Average Order Value (AOV) of this specific discount type against the 'No Discount' baseline.",
            "> - **Positive Lift (+):** Customers using this discount actually spend *more* than full-price customers.",
            "> - **Negative Lift (-):** Customers using this discount spend *less* than average."
        ])
            
        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating discount report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_fulfillment_analysis_report")
@log_tool_usage
def get_fulfillment_analysis_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = 'Orders',
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Calculates a breakdown of orders and revenue by delivery/fulfillment status.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - sort_by: Column to sort by. Options: 'Status', 'Orders', 'Percentage', 'Revenue' (default: 'Orders').
    - sort_order: 'desc' (default) or 'asc'.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 1. Date Filtering
        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
            if df['createdAt'].dt.tz is not None:
                df['createdAt'] = df['createdAt'].dt.tz_localize(None)
            df = df.dropna(subset=['createdAt'])

            if start_date:
                try:
                    df = df[df['createdAt'] >= pd.to_datetime(start_date)]
                except Exception:
                    return "Error: Invalid start_date format. Use MM/DD/YYYY."
            if end_date:
                try:
                    df = df[df['createdAt'] <= pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)]
                except Exception:
                    return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df.empty:
            return "No order data found for the specified period."

        # 2. Aggregation
        df['totalAmount'] = pd.to_numeric(df.get('totalAmount', 0), errors='coerce').fillna(0)
        df['deliveryStatus'] = df.get('deliveryStatus', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
        
        total_orders = len(df)
        id_col = 'id' if 'id' in df.columns else ('customer_id' if 'customer_id' in df.columns else df.columns[0])

        stats = df.groupby('deliveryStatus').agg(
            Count=(id_col, 'count'),
            Revenue=('totalAmount', 'sum')
        ).reset_index()
        
        stats['Percentage'] = (stats['Count'] / total_orders) * 100

        # 3. Dynamic Sorting
        sort_mapping = {
            'Status': 'deliveryStatus',
            'Orders': 'Count',
            'Percentage': 'Percentage',
            'Revenue': 'Revenue'
        }
        sort_col = sort_mapping.get(sort_by, 'Count')
        stats = stats.sort_values(by=sort_col, ascending=is_ascending)

        # 4. Formatting
        lines = [
            "## Fulfillment Analysis",
            "*Breakdown of orders and revenue by delivery status.*"
        ]
        
        filter_str = []
        if start_date or end_date: 
            filter_str.append(f"Dates: {start_date or 'All Time'} to {end_date or 'Present'}")
            
        if filter_str:
            lines.append(f"*(Filters applied: {', '.join(filter_str)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        lines.append("| Delivery Status | Orders | Percentage | Revenue |")
        lines.append("|---|---|---|---|")

        if stats.empty:
            lines.append("| No data found matching criteria | - | - | - |")
        else:
            for _, row in stats.iterrows():
                raw_status = str(row['deliveryStatus'])
                clean_status = raw_status.replace('_', ' ').title()
                lines.append(f"| {clean_status} | {int(row['Count'])} | {row['Percentage']:.1f}% | ${row['Revenue']:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating fulfillment report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_payment_analysis_report")
@log_tool_usage
def get_payment_analysis_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = 'Orders',
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Calculates a breakdown of orders and revenue by payment status.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - sort_by: Column to sort by. Options: 'Status', 'Orders', 'Percentage', 'Revenue' (default: 'Orders').
    - sort_order: 'desc' (default) or 'asc'.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 1. Date Filtering
        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
            if df['createdAt'].dt.tz is not None:
                df['createdAt'] = df['createdAt'].dt.tz_localize(None)
            df = df.dropna(subset=['createdAt'])

            if start_date:
                try:
                    df = df[df['createdAt'] >= pd.to_datetime(start_date)]
                except Exception:
                    return "Error: Invalid start_date format. Use MM/DD/YYYY."
            if end_date:
                try:
                    df = df[df['createdAt'] <= pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)]
                except Exception:
                    return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df.empty:
            return "No order data found for the specified period."

        # 2. Aggregation
        df['totalAmount'] = pd.to_numeric(df.get('totalAmount', 0), errors='coerce').fillna(0)
        df['paymentStatus'] = df.get('paymentStatus', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
        
        total_orders = len(df)
        id_col = 'id' if 'id' in df.columns else ('customer_id' if 'customer_id' in df.columns else df.columns[0])

        stats = df.groupby('paymentStatus').agg(
            Count=(id_col, 'count'),
            Revenue=('totalAmount', 'sum')
        ).reset_index()
        
        stats['Percentage'] = (stats['Count'] / total_orders) * 100

        # 3. Dynamic Sorting
        sort_mapping = {
            'Status': 'paymentStatus',
            'Orders': 'Count',
            'Percentage': 'Percentage',
            'Revenue': 'Revenue'
        }
        sort_col = sort_mapping.get(sort_by, 'Count')
        stats = stats.sort_values(by=sort_col, ascending=is_ascending)

        # 4. Formatting
        lines = [
            "## Payment Status Analysis",
            "*Breakdown of orders and revenue by payment status.*"
        ]
        
        filter_str = []
        if start_date or end_date: 
            filter_str.append(f"Dates: {start_date or 'All Time'} to {end_date or 'Present'}")
            
        if filter_str:
            lines.append(f"*(Filters applied: {', '.join(filter_str)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        lines.append("| Payment Status | Orders | Percentage | Revenue |")
        lines.append("|---|---|---|---|")

        if stats.empty:
            lines.append("| No data found matching criteria | - | - | - |")
        else:
            for _, row in stats.iterrows():
                raw_status = str(row['paymentStatus'])
                clean_status = raw_status.replace('_', ' ').title()
                lines.append(f"| {clean_status} | {int(row['Count'])} | {row['Percentage']:.1f}% | ${row['Revenue']:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating payment report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_sales_trends_orders_report")
@log_tool_usage
def get_sales_trends_orders_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    monthly_sort_by: Optional[str] = 'Month',
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Generates a comprehensive sales & customer quality report including Quarterly, Monthly, Day of Week, and Cohort analysis.
    
    Parameters:
    - user_id: The user's ID to locate data files.
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - monthly_sort_by: Column to sort the Monthly table by. Options: 'Month', 'Revenue', 'Growth', 'Orders', 'AOV' (default: 'Month').
    - sort_order: 'desc' (default) or 'asc'. Applies to the timeline of Quarterly/Cohort tables, and the selected metric in the Monthly table.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    try:
        if not orders_path.exists():
            return f"Error: Orders file not found for user {user_id}."

        df = pd.read_csv(orders_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'createdAt' not in df.columns or 'totalAmount' not in df.columns:
            return "Error: Required columns ('createdAt', 'totalAmount') missing from orders data."

        # 1. Clean Dates & Apply Date Filters
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        if df['createdAt'].dt.tz is not None:
             df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df = df.dropna(subset=['createdAt'])
        
        # Calculate dataset bounds for the header
        data_min_date = df['createdAt'].min()
        data_max_date = df['createdAt'].max()

        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df = df[df['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use MM/DD/YYYY."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df = df[df['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df.empty:
            return "No order data found for the specified period."

        df['totalAmount'] = pd.to_numeric(df['totalAmount'], errors='coerce').fillna(0)
        
        # Determine ID column for counting
        id_col = 'customer_id' if 'customer_id' in df.columns else ('id' if 'id' in df.columns else df.columns[0])

        # --- Features Extraction ---
        df['Month_Str'] = df['createdAt'].dt.strftime('%Y-%m')
        df['Quarter'] = df['createdAt'].dt.to_period('Q').astype(str)
        df['DayOfWeek'] = df['createdAt'].dt.day_name()

        # --- A. Monthly Trends ---
        monthly_stats = df.groupby('Month_Str').agg(
            Revenue=('totalAmount', 'sum'),
            Orders=(id_col, 'count')
        ).reset_index()
        
        # Strictly chronological for Growth calculation
        monthly_stats = monthly_stats.sort_values('Month_Str')
        monthly_stats['AOV'] = monthly_stats['Revenue'] / monthly_stats['Orders']
        monthly_stats['Rev_Growth'] = monthly_stats['Revenue'].pct_change() * 100
        
        chrono_monthly = monthly_stats.copy() # Save chronological state for Exec Summary

        # Apply user sort to Monthly Table
        sort_mapping = {
            'Month': 'Month_Str',
            'Revenue': 'Revenue',
            'Growth': 'Rev_Growth',
            'Orders': 'Orders',
            'AOV': 'AOV'
        }
        sort_col = sort_mapping.get(monthly_sort_by, 'Month_Str')
        
        # Fix NaN growth sorting by filling temporarily
        monthly_stats['Sort_Helper'] = monthly_stats[sort_col].fillna(-float('inf') if not is_ascending else float('inf'))
        monthly_stats = monthly_stats.sort_values(by='Sort_Helper', ascending=is_ascending).drop(columns=['Sort_Helper'])

        # --- B. Quarterly Summary ---
        quarterly_stats = df.groupby('Quarter').agg(
            Revenue=('totalAmount', 'sum'),
            Orders=(id_col, 'count')
        ).reset_index()
        quarterly_stats = quarterly_stats.sort_values('Quarter', ascending=is_ascending)

        # --- C. Day of Week Analysis ---
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)
        dow_stats = df.groupby('DayOfWeek', observed=False).agg(
            TotalOrders=(id_col, 'count'),
            AvgOrderValue=('totalAmount', 'mean')
        ).reset_index()
        dow_stats['AvgOrderValue'] = dow_stats['AvgOrderValue'].fillna(0)

        # --- D. Customer Quality (Cohort Analysis) ---
        cust_stats = df.groupby(id_col).agg(
            FirstOrder=('createdAt', 'min'),
            LifetimeRevenue=('totalAmount', 'sum')
        ).reset_index()
        cust_stats['JoinYear'] = cust_stats['FirstOrder'].dt.year
        
        cohort_stats = cust_stats.groupby('JoinYear').agg(
            NewCustomers=(id_col, 'count'),
            AvgLifetimeValue=('LifetimeRevenue', 'mean')
        ).reset_index()
        cohort_stats = cohort_stats.sort_values('JoinYear', ascending=is_ascending)

        # --- formatting Markdown Report ---
        lines = []
        lines.append("# Comprehensive Sales & Customer Quality Report")
        lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
        
        start_str = start_date if start_date else data_min_date.strftime('%m/%d/%Y')
        end_str = end_date if end_date else data_max_date.strftime('%m/%d/%Y')
        lines.append(f"**Data Range:** {start_str} to {end_str}\n")
        
        # 1. Executive Summary
        lines.append("## 1. Executive Trend Summary")
        if len(chrono_monthly) >= 2:
            curr_m = chrono_monthly.iloc[-1]
            prev_m = chrono_monthly.iloc[-2]
            trend_icon = "📈" if curr_m['Revenue'] > prev_m['Revenue'] else "📉"
            lines.append(f"- **Latest Month ({curr_m['Month_Str']}):** ${curr_m['Revenue']:,.2f} ({curr_m['Rev_Growth']:+.1f}% vs prev) {trend_icon}")
        else:
            lines.append("- *Not enough monthly data for trend analysis.*")
            
        overall_aov = df['totalAmount'].sum() / len(df) if len(df) > 0 else 0
        lines.append(f"- **Overall Average Order Value (AOV):** ${overall_aov:,.2f}\n")

        # 2. Quarterly
        lines.append("## 2. Quarterly Performance")
        lines.append("| Quarter | Revenue | Orders | Avg Revenue/Order |")
        lines.append("|---|---|---|---|")
        for _, row in quarterly_stats.iterrows():
            avg = row['Revenue'] / row['Orders'] if row['Orders'] > 0 else 0
            lines.append(f"| {row['Quarter']} | ${row['Revenue']:,.2f} | {int(row['Orders'])} | ${avg:,.2f} |")
        lines.append("")
        
        # 3. Monthly
        lines.append(f"## 3. Monthly Sales History")
        lines.append(f"*(Sorted by: {monthly_sort_by} | Order: {sort_order.upper()})*")
        lines.append("| Month | Revenue | Growth | Orders | AOV |")
        lines.append("|---|---|---|---|---|")
        for _, row in monthly_stats.head(24).iterrows(): # Show up to 24 rows to keep it readable
            growth = f"{row['Rev_Growth']:+.1f}%" if pd.notna(row['Rev_Growth']) else "-"
            lines.append(f"| {row['Month_Str']} | ${row['Revenue']:,.2f} | {growth} | {int(row['Orders'])} | ${row['AOV']:,.2f} |")
        lines.append("")

        # 4. Operational
        lines.append("## 4. Operational Insights: Day of Week")
        lines.append("| Day | Total Orders | Avg Order Size |")
        lines.append("|---|---|---|")
        for _, row in dow_stats.iterrows():
            lines.append(f"| {row['DayOfWeek']} | {int(row['TotalOrders'])} | ${row['AvgOrderValue']:,.2f} |")
        lines.append("")
        
        # 5. Cohorts
        lines.append("## 5. Customer Quality (Cohort Analysis)")
        lines.append("*Analyzing the value of new customers acquired each year.*")
        if start_date:
            lines.append("*(Note: 'Join Year' and Lifetime Value reflect data within the filtered date range only)*")
        lines.append("| Join Year | New Customers Acquired | Avg Lifetime Value (CLV) |")
        lines.append("|---|---|---|")
        for _, row in cohort_stats.iterrows():
            lines.append(f"| {int(row['JoinYear'])} | {int(row['NewCustomers'])} | ${row['AvgLifetimeValue']:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error generating sales trends report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_top_n_orders")
@log_tool_usage
def get_top_n_orders(
    user_id: str, 
    n: int = 10, 
    sort_by: Optional[str] = 'Total', 
    status_filter: Optional[str] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Gets the top (or bottom) N orders based on specified criteria, optionally filtered by date and status.

    Parameters:
    - user_id: The user's ID.
    - n: Number of records to return (default: 10).
    - sort_by: Column to sort by. Options: 'Total', 'Quantity', 'Discount', 'Delivery' (default: 'Total').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - start_date: Filter orders created ON or AFTER this date. Format: 'MM/DD/YYYY'.
    - end_date: Filter data from start day to this date. Format: 'MM/DD/YYYY'.
    - status_filter: Filter by specific order status (e.g., 'COMPLETED', 'PENDING').
    """
    csv_path = Path("data") / str(user_id) / "cleaned_orders.csv"
    is_ascending = (sort_order.lower() == 'asc')

    if not csv_path.exists():
        return f"Error: Orders file not found for user {user_id}."

    try:
        # Load data safely
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        if 'createdAt' not in df.columns:
            return "Error: Required column 'createdAt' missing from orders data."

        # 1. Clean Dates & Timezones
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        if df['createdAt'].dt.tz is not None:
             df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df = df.dropna(subset=['createdAt'])

        # 2. Apply Date Filters (MM/DD/YYYY)
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df = df[df['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use 'MM/DD/YYYY'."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df = df[df['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use 'MM/DD/YYYY'."

        # 3. Apply Status Filter
        if status_filter and 'orderStatus' in df.columns:
            s_filter = status_filter.upper()
            df = df[df['orderStatus'].fillna('').str.upper() == s_filter]
            if df.empty:
                return f"No orders found matching status '{status_filter}' in the specified date range."

        if df.empty:
            return "No orders found matching the applied criteria."

        # 4. Ensure Numeric Columns for Sorting
        df['totalAmount'] = pd.to_numeric(df.get('totalAmount', 0), errors='coerce').fillna(0)
        df['totalQuantity'] = pd.to_numeric(df.get('totalQuantity', 0), errors='coerce').fillna(0)
        df['totalDiscountValue'] = pd.to_numeric(df.get('totalDiscountValue', 0), errors='coerce').fillna(0)
        df['deliveryFee'] = pd.to_numeric(df.get('deliveryFee', 0), errors='coerce').fillna(0)

        # 5. Dynamic Sorting
        sort_mapping = {
            'total': 'totalAmount',
            'revenue': 'totalAmount',
            'quantity': 'totalQuantity',
            'discount': 'totalDiscountValue',
            'delivery': 'deliveryFee'
        }
        
        sort_col = sort_mapping.get(str(sort_by).lower(), 'totalAmount')
        top_n_df = df.sort_values(by=sort_col, ascending=is_ascending).head(n)

        # 6. Output Formatting
        lines = [f"## Top {len(top_n_df)} Orders"]
        
        filter_info = []
        if start_date or end_date:
            filter_info.append(f"Dates: {start_date or 'All'} to {end_date or 'Present'}")
        if status_filter:
            filter_info.append(f"Status: {status_filter.upper()}")
            
        if filter_info:
            lines.append(f"*(Filters: {', '.join(filter_info)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        headers = ["Order ID", "Date", "Customer", "Status", "Qty", "Total ($)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|---|---|---|---|---|---|")

        # Determine best columns for IDs and Names
        order_id_col = 'customId_customId' if 'customId_customId' in df.columns else 'id'
        
        for _, row in top_n_df.iterrows():
            o_id = str(row.get(order_id_col, '-'))
            o_date = row['createdAt'].strftime('%m/%d/%Y')
            
            c_name = row.get('customer_displayedName')
            if pd.isna(c_name) or not c_name:
                c_name = row.get('customer_name', 'Unknown')
                
            o_status = str(row.get('orderStatus', '-')).replace('_', ' ').title()
            o_qty = int(row['totalQuantity'])
            o_total = float(row['totalAmount'])

            lines.append(f"| {o_id} | {o_date} | {str(c_name)[:30]} | {o_status} | {o_qty} | ${o_total:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error processing top orders report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_order_details")
@log_tool_usage
def get_order_details(user_id: str, order_identifier: str) -> str:
    """
    Gets complete information about an order.
    
    Searches using the following priority:
    1. The internal 'customId' (e.g., 771657) - PREFERRED.
    2. The system UUID (e.g., 'ab16c21f-e705...')
    3. The Shopify Order ID.
    
    Args:
        user_id (str): The user's ID.
        order_identifier (str or int): The ID to search for (can be numeric string or UUID).
        
    Returns:
        str: Formatted Markdown order details or "Not Found" message.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    
    if not orders_path.exists():
        return f"Error: Orders file not found for user {user_id}."

    try:
        # Load Orders securely
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_orders.columns = df_orders.columns.str.strip().str.replace('\ufeff', '')
        
        # Clean the search value (Remove trailing .0 if pandas casted a user input)
        search_val = str(order_identifier).strip().replace('.0', '')
        order_row = None
        
        # STRATEGY A: Check 'customId_customId'
        if 'customId_customId' in df_orders.columns:
            # Safely cast to string and drop .0 for exact matching
            df_orders['search_custom'] = df_orders['customId_customId'].astype(str).str.replace(r'\.0$', '', regex=True)
            matches = df_orders[df_orders['search_custom'] == search_val]
            if not matches.empty:
                order_row = matches.iloc[0]

        # STRATEGY B: Check UUID ('id')
        if order_row is None and 'id' in df_orders.columns:
            matches = df_orders[df_orders['id'].astype(str).str.strip() == search_val]
            if not matches.empty:
                order_row = matches.iloc[0]
                
        # STRATEGY C: Check Shopify Order ID
        if order_row is None and 'shopifyOrderId' in df_orders.columns:
            df_orders['search_shopify'] = df_orders['shopifyOrderId'].astype(str).str.replace(r'\.0$', '', regex=True)
            matches = df_orders[df_orders['search_shopify'] == search_val]
            if not matches.empty:
                order_row = matches.iloc[0]

        # Result Check
        if order_row is None:
            return f"Error: Order '{order_identifier}' not found in the database."

        # 3. Retrieve Line Items
        internal_order_id = str(order_row.get('id', ''))
        order_products = pd.DataFrame()
        
        if products_path.exists():
            try:
                df_products = pd.read_csv(products_path, encoding='utf-8-sig')
                df_products.columns = df_products.columns.str.strip().str.replace('\ufeff', '')
                if 'orderId' in df_products.columns:
                    order_products = df_products[df_products['orderId'].astype(str) == internal_order_id]
            except Exception:
                pass # Proceed gracefully without line items if file fails

        # 4. Format Output
        lines = []
        
        # --- Header ---
        cust_id = str(order_row.get('customId_customId', internal_order_id)).replace('.0', '')
        
        # Format Dates (MM/DD/YYYY)
        created_dt = pd.to_datetime(order_row.get('createdAt'), errors='coerce')
        created_at = created_dt.strftime('%m/%d/%Y %H:%M') if pd.notna(created_dt) else 'Unknown'
        
        due_dt = pd.to_datetime(order_row.get('paymentDue'), errors='coerce')
        due_date = due_dt.strftime('%m/%d/%Y') if pd.notna(due_dt) else 'Unknown'

        # Get Display Name safely
        c_name = order_row.get('customer_displayedName')
        if pd.isna(c_name) or not c_name:
            c_name = order_row.get('customer_name', 'Unknown')
            
        lines.append(f"## Order Report: #{cust_id}")
        lines.append(f"- **Customer:** {c_name}")
        lines.append(f"- **Date Created:** {created_at}")
        lines.append(f"- **Order Type:** {str(order_row.get('type', 'Direct')).title()}")
        
        # --- Statuses ---
        status = str(order_row.get('orderStatus', 'N/A')).replace('_', ' ').title()
        pay_status = str(order_row.get('paymentStatus', 'N/A')).replace('_', ' ').title()
        del_status = str(order_row.get('deliveryStatus', 'N/A')).replace('_', ' ').title()
        
        lines.append("\n### Order Status")
        lines.append(f"- **Fulfillment:** {status} / {del_status}")
        lines.append(f"- **Payment:** {pay_status}")
        
        terms = str(order_row.get('paymentTermsDuplicate_name', 'N/A'))
        if terms and terms != 'nan':
            lines.append(f"- **Payment Terms:** {terms} (Due: {due_date})")

        # --- Financials ---
        lines.append("\n### Financial Summary")
        subtotal = float(order_row.get('totalAmountWithoutDelivery', 0))
        discount = float(order_row.get('totalDiscountValue', 0))
        del_fee = float(order_row.get('deliveryFee', 0))
        total = float(order_row.get('totalAmount', 0))
        
        lines.append(f"- **Subtotal:** ${subtotal:,.2f}")
        if discount > 0:
            disc_type = str(order_row.get('totalOrderDiscountType', '')).title()
            lines.append(f"- **Discount:** -${discount:,.2f} ({disc_type})")
        if del_fee > 0:
            lines.append(f"- **Delivery Fee:** ${del_fee:,.2f}")
        lines.append(f"- **GRAND TOTAL:** **${total:,.2f}**")

        # --- Line Items ---
        lines.append(f"\n### Line Items ({len(order_products)})")
        
        if order_products.empty:
            lines.append("*No products linked to this order.*")
        else:
            lines.append("| Product Name | SKU | Qty | Total |")
            lines.append("|---|---|---|---|")
            
            for _, prod in order_products.iterrows():
                # Prefer product_variant, fallback to name
                name = prod.get('product_variant')
                if pd.isna(name) or not name:
                    name = prod.get('name', 'Unknown')
                
                name = str(name)[:45] + ".." if len(str(name)) > 45 else str(name)
                sku = str(prod.get('sku', '-'))
                if sku == 'nan': sku = '-'
                
                qty = int(pd.to_numeric(prod.get('quantity', 0), errors='coerce'))
                line_total = float(pd.to_numeric(prod.get('totalAmount', 0), errors='coerce'))
                
                lines.append(f"| {name} | {sku} | {qty} | ${line_total:,.2f} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error retrieving order details: {str(e)}\n{traceback.format_exc()}"


# List of tools from catalog block agent

@mcp.tool(name="get_top_n_products")
@log_tool_usage
def get_top_n_products(
    user_id: str, 
    n: int = 10, 
    by_type: Optional[str] = 'revenue', 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    sort_order: Optional[str] = 'desc', 
    group_by: Optional[str] = 'variant'
) -> str:
    """
    Gets top N products, categories, or manufacturers based on revenue, quantity, or orders.
    
    Parameters:
    - user_id: User ID.
    - n: Number of items to return (default: 10).
    - by_type: Metric to sort by. Options: 'revenue', 'quantity', 'orders' (default: 'revenue').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - start_date: Filter data from this date (MM/DD/YYYY).
    - end_date: Filter data up to this date (MM/DD/YYYY).
    - group_by: Aggregation level. Options: 'variant' (Specific Product), 'category' (Product Category), 'manufacturer' (Brand/Manufacturer).
    """
    csv_path = Path("data") / str(user_id) / "cleaned_products.csv"
    is_ascending = (sort_order.lower() == 'asc')
    
    if not csv_path.exists():
        return f"Error: Products file not found for user {user_id}."

    try:
        # Load Safely
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'createdAt' not in df.columns:
            return "Error: Required column 'createdAt' missing from products data."

        # 1. Clean Dates & Timezones
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        if df['createdAt'].dt.tz is not None:
            df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df = df.dropna(subset=['createdAt'])

        # 2. Date Filtering (MM/DD/YYYY)
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df = df[df['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use MM/DD/YYYY."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df = df[df['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df.empty:
            return "No product data found for the specified period."

        # 3. Determine Grouping Column dynamically
        group_by_lower = str(group_by).lower()
        if group_by_lower in ['variant', 'product']:
            group_col = 'product_variant' if 'product_variant' in df.columns else 'name'
            label = "Product Variant"
        elif group_by_lower == 'category':
            group_col = 'productCategoryName' if 'productCategoryName' in df.columns else 'category'
            label = "Category"
        elif group_by_lower in ['manufacturer', 'brand']:
            group_col = 'manufacturerName' if 'manufacturerName' in df.columns else 'manufacturer'
            label = "Manufacturer"
        else:
            return "Invalid 'group_by'. Use 'variant', 'category', or 'manufacturer'."

        if group_col not in df.columns:
            return f"Error: Grouping column '{group_col}' not found in data."

        # Fill N/A to avoid losing valid transaction rows
        df[group_col] = df[group_col].fillna('Unknown').astype(str)

        # Ensure numeric columns
        df['totalAmount'] = pd.to_numeric(df.get('totalAmount', 0), errors='coerce').fillna(0)
        df['quantity'] = pd.to_numeric(df.get('quantity', 0), errors='coerce').fillna(0)
        
        # 4. Aggregation
        has_orders = 'orderId' in df.columns
        agg_dict = {
            'totalAmount': 'sum',
            'quantity': 'sum'
        }
        if has_orders:
            agg_dict['orderId'] = 'nunique'
            
        product_agg = df.groupby(group_col).agg(agg_dict).rename(columns={
            'totalAmount': 'totalRevenue',
            'quantity': 'totalQuantity',
            'orderId': 'orderCount'
        }).reset_index()

        if not has_orders:
            # Fallback if orderId column is missing
            counts = df.groupby(group_col).size().reset_index(name='orderCount')
            product_agg = product_agg.merge(counts, on=group_col)

        # Calculate Avg Revenue Per Order
        product_agg['avgRevenuePerOrder'] = product_agg.apply(
            lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
            axis=1
        )

        # 5. Sort Logic
        sort_map = {
            'revenue': 'totalRevenue',
            'quantity': 'totalQuantity',
            'totalquantity': 'totalQuantity',
            'orders': 'orderCount',
            'ordercount': 'orderCount'
        }
        sort_column = sort_map.get(str(by_type).lower(), 'totalRevenue')
        top_n_df = product_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

        # 6. Formatting Output
        direction_label = "Bottom" if is_ascending else "Top"
        lines = [f"## {direction_label} {n} {label}s"]
        
        filter_info = []
        if start_date or end_date:
            filter_info.append(f"Dates: {start_date or 'All'} to {end_date or 'Present'}")
            
        if filter_info:
            lines.append(f"*(Filters: {', '.join(filter_info)} | Sorted by: {by_type.capitalize()} | Order: {sort_order.upper()})*\n")
        else:
            lines.append(f"*(Sorted by: {by_type.capitalize()} | Order: {sort_order.upper()})*\n")

        headers = [label, "Revenue", "Qty", "Orders", "Avg Rev/Order"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|---|---|---|---|---|")

        if top_n_df.empty:
            lines.append("| No data found | - | - | - | - |")
        else:
            for _, row in top_n_df.iterrows():
                # Clean up variant name if it's too long
                item_name = str(row[group_col])
                display_name = item_name[:50] + ".." if len(item_name) > 50 else item_name
                
                rev = f"${row['totalRevenue']:,.2f}"
                qty = f"{int(row['totalQuantity']):,}"
                orders = f"{int(row['orderCount']):,}"
                avg_rev = f"${row['avgRevenuePerOrder']:,.2f}"
                
                lines.append(f"| {display_name} | {rev} | {qty} | {orders} | {avg_rev} |")

        return '\n'.join(lines)

    except Exception as e:
        return f"Error processing products report: {str(e)}\n{traceback.format_exc()}"

def _narrow(combos, column, value, label, filters, notes, sku_mode=False):
    """
    Apply one filter step, but skip it (with an explanatory note) if a prior
    filter already narrowed the working set to zero rows - avoids firing a
    misleading "no match found for X" against results that were already empty
    for an unrelated reason.
    """
    if combos.empty:
        notes.append(
            f"Skipped {label} filter ('{value}') because earlier filters already "
            f"narrowed results to zero matches."
        )
        return combos
    return apply_filter(combos, column, value, label, filters, notes, sku_mode=sku_mode)

def fuzzy_blob_search(
    df: pd.DataFrame,
    query: str,
    columns=("manufacturerName", "productCategoryName", "name", "sku"),
    score_cutoff: int = 75,
    limit: int = 15,
):
    """
    Free-text search across a combined text blob built from `columns`, joined
    per row. Splits the query into individual words and scores each row by the
    average of each query word's best per-token match anywhere in that row's
    combined text - so word order doesn't matter and the query doesn't need to
    map cleanly onto a single field.
 
    Returns (matched_df, notes):
      - matched_df: rows with a "match_score" column, sorted descending (may
        be more than one row - this is a *search*, not a single resolved
        value, so ties and near-ties are all surfaced rather than forced to
        pick one).
      - notes: a one-line summary of what matched, or why nothing did.
    """
    if df.empty or not query:
        return df.iloc[0:0], []
 
    query_tokens = _tokenize(query)
    if not query_tokens:
        return df.iloc[0:0], []
 
    working = df.copy()
    present_cols = [c for c in columns if c in working.columns]
    working["_blob"] = working[present_cols].fillna("").astype(str).agg(" ".join, axis=1)
    working["match_score"] = working["_blob"].apply(
        lambda blob: _token_overlap_score(query_tokens, _tokenize(blob))
    )
    working = working.drop(columns=["_blob"])
 
    matched = working[working["match_score"] >= score_cutoff].sort_values(
        "match_score", ascending=False
    ).head(limit)
 
    if matched.empty:
        return df.iloc[0:0], [
            f"No catalog entries matched free-text search '{query}' "
            f"(searched name/sku/category/manufacturer combined, per-word fuzzy matching)."
        ]
 
    top_score = matched["match_score"].iloc[0]
    note = (
        f"Free-text search '{query}' matched {len(matched)} catalog entrie(s) "
        f"(best match {top_score:.0f}%)."
    )
    return matched, [note]
 
def resolve_value(
    user_input: str,
    choices: list,
    score_cutoff: int = 80,
    ambiguity_gap: int = 5,
):
    """
    Try to resolve `user_input` to the closest value in `choices`.
 
    Returns a tuple: (resolved_value, note, ambiguous_candidates)
      - resolved_value: the best matching choice, or None if nothing cleared the cutoff
                         or the match was ambiguous.
      - note: human-readable string describing the substitution, or None if the
              match was exact (case-insensitive) and needs no explanation.
      - ambiguous_candidates: list of near-tied candidate values (empty if not ambiguous).
    """
    if not choices:
        return None, None, []
 
    matches = process.extract(
        user_input, choices, scorer=fuzz.WRatio, limit=3, score_cutoff=score_cutoff
    )
    if not matches:
        return None, None, []
 
    top_value, top_score, _ = matches[0]
    close = [m for m in matches if top_score - m[1] <= ambiguity_gap]
 
    if len(close) > 1:
        # Too close to call - don't guess, ask instead.
        return None, None, [m[0] for m in close]
 
    note = None
    if top_value.strip().lower() != user_input.strip().lower():
        note = f"No exact match for '{user_input}' — using closest match '{top_value}' ({top_score:.0f}% match)."
 
    return top_value, note, []
 
def _tokenize(text: str) -> list:
    return [t.strip(",;:") for t in str(text).lower().split() if t.strip(",;:")]
 
 
def _token_overlap_score(query_tokens: list, row_tokens: list) -> float:
    """ 
    Returns the mean of per-token best scores (0 if row has no tokens).
    """
    if not row_tokens:
        return 0.0
    scores = []
    for qt in query_tokens:
        match = process.extractOne(qt, row_tokens, scorer=fuzz.ratio)
        scores.append(match[1] if match else 0.0)
    return sum(scores) / len(scores)
 
def apply_filter(
    df: pd.DataFrame,
    column: str,
    user_value: str,
    label: str,
    filters: list,
    notes: list,
    sku_mode: bool = False,
) -> pd.DataFrame:
    """
    Filter `df` on `column` matching `user_value`.
    Tries exact/substring match first; falls back to fuzzy matching against the
    column's unique values only if the substring match returns nothing.
 
    Note: whether a filter argument was PROVIDED is tracked separately by the
    caller. This function only determines whether the provided value resolved
    to anything - it must not be used to infer "no filter was passed".
    """
    if column not in df.columns:
        notes.append(f"Column '{column}' not found in data - skipping {label} filter.")
        return df
 
    # 1. Exact / substring match first (fast, 100% precise when it hits)
    exact = df[df[column].fillna("").astype(str).str.contains(user_value, case=False, na=False)]
    if not exact.empty:
        filters.append(f"{label}='{user_value}'")
        return exact
 
    # 2. Fuzzy fallback - only against unique values, not every row
    cutoff = 92 if sku_mode else 80
    unique_values = df[column].dropna().astype(str).unique().tolist()
    resolved, note, ambiguous = resolve_value(user_value, unique_values, score_cutoff=cutoff)
 
    if ambiguous:
        candidates = ", ".join(f"'{c}'" for c in ambiguous)
        notes.append(
            f"'{user_value}' matched multiple {label} values ({candidates}) — "
            f"please specify which one you meant."
        )
        return df.iloc[0:0]  # empty on purpose: force clarification instead of guessing
 
    if resolved is None:
        # Filter WAS provided, it just didn't match anything - say so explicitly
        # rather than leaving both `filters` and `notes` empty, which would look
        # identical to "no filter was ever passed".
        notes.append(f"No match found for {label}='{user_value}' (checked exact and fuzzy match).")
        return df.iloc[0:0]
 
    if note:
        notes.append(note)
    filters.append(f"{label}='{resolved}'")
    return df[df[column].fillna("").astype(str).str.contains(resolved, case=False, na=False)]

@mcp.tool(name="search_product_catalog")
@log_tool_usage
def search_product_catalog(
    user_id: str,
    manufacturer: Optional[str] = None,
    category: Optional[str] = None,
    product_name: Optional[str] = None,
    sku: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """
    Returns lists of unique product attributes with detailed variants from the
    active catalog. Reads strictly from the catalog file and correctly maps
    parent attributes (manufacturer, category, name) down to child variants.
 
    Use this to browse "what's in the catalog", to confirm the exact spelling
    of a name/sku/category/manufacturer, or to narrow down candidates before
    calling get_product_details.
 
    Args:
        user_id (str): The user's ID.
        manufacturer (str): Narrow to this manufacturer (partial match, fuzzy fallback).
        category (str): Narrow to this category (partial match, fuzzy fallback).
        product_name (str): Narrow to this product name (partial match, fuzzy fallback).
        sku (str): Narrow to this SKU (near-exact only; strict fuzzy cutoff, since
                   near-miss SKUs are usually different products, not typos).
        query (str): Free-text search fallback for when you are NOT sure which
                     field a term belongs to, or the term seems to combine more
                     than one attribute (e.g. a product name plus a variant/SKU
                     fragment, like "cola hanukkah"). Searches name, sku,
                     category, and manufacturer together and does not require
                     the words to be in field order or even in the right field -
                     use this instead of guessing which structured parameter to
                     force the phrase into. Can be combined with the structured
                     filters above (applied as an additional AND narrowing step),
                     but is most useful on its own.
    """
    catalog_path = Path("data") / str(user_id) / "cleaned_catalog.csv"
 
    if not catalog_path.exists():
        return f"Error: Catalog file not found for user {user_id}."
 
    try:
        # 1. Load data safely
        df_catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
        df_catalog.columns = df_catalog.columns.str.strip().str.replace("\ufeff", "")
 
        # Standardize column names
        rename_map = {
            "manufacturer_name": "manufacturerName",
            "productCategory_name": "productCategoryName",
        }
        df = df_catalog.rename(columns=rename_map)
 
        # Ensure required columns exist
        for col in ["id", "parentProductId", "manufacturerName", "productCategoryName", "name", "sku"]:
            if col not in df.columns:
                df[col] = None
 
        # 2. Build Parent Inheritance Mapping
        parent_map = df.set_index("id")[["manufacturerName", "productCategoryName", "name"]].to_dict("index")
 
        def get_inherited_value(row, col_name):
            val = row[col_name]
            if pd.isna(val) or str(val).strip() == "":
                pid = row["parentProductId"]
                if pd.notna(pid) and pid in parent_map:
                    parent_val = parent_map[pid].get(col_name)
                    if pd.notna(parent_val) and str(parent_val).strip() != "":
                        return parent_val
            return val
 
        df["manufacturerName"] = df.apply(lambda r: get_inherited_value(r, "manufacturerName"), axis=1)
        df["productCategoryName"] = df.apply(lambda r: get_inherited_value(r, "productCategoryName"), axis=1)
        df["name"] = df.apply(lambda r: get_inherited_value(r, "name"), axis=1)
 
        # 3. Clean up NaNs
        df[["manufacturerName", "productCategoryName", "name", "sku"]] = df[
            ["manufacturerName", "productCategoryName", "name", "sku"]
        ].fillna("Unknown")
 
    except Exception as e:
        return f"Error reading catalog data: {str(e)}\n{traceback.format_exc()}"
 
    # 4. Build the browsable unique-combination table (post-inheritance)
    combos = df[["manufacturerName", "productCategoryName", "name", "sku"]].drop_duplicates()
    combos = combos[~((combos["name"] == "Unknown") & (combos["sku"] == "Unknown"))]
 
    # 5. Free-text search first (if given) - broad recall across all fields combined
    filters: list = []
    notes: list = []
 
    if query:
        combos, query_notes = fuzzy_blob_search(combos, query)
        notes.extend(query_notes)
        if query_notes:
            filters.append(f"Query='{query}'")
 
    # 6. Apply structured narrowing filters on top (exact -> fuzzy fallback,
    if manufacturer:
        combos = _narrow(combos, "manufacturerName", manufacturer, "Manufacturer", filters, notes)
    if category:
        combos = _narrow(combos, "productCategoryName", category, "Category", filters, notes)
    if product_name:
        combos = _narrow(combos, "name", product_name, "Name", filters, notes)
    if sku:
        combos = _narrow(combos, "sku", sku, "SKU", filters, notes, sku_mode=True)
 
    # 7. Build clean attribute lists from the (possibly narrowed) combos
    has_scores = "match_score" in combos.columns
 
    def _variant_line(row):
        base = (
            f"manufacturerName: {row.manufacturerName}, "
            f"productCategoryName: {row.productCategoryName}, "
            f"name: {row.name}, "
            f"sku: {row.sku};"
        )
        if has_scores:
            base = f"[match {row.match_score:.0f}%] " + base
        return base
 
    if has_scores:
        # Preserve score-descending order (combos was already sorted by scorein fuzzy_blob_search) 
        detailed_variants = [_variant_line(row) for row in combos.itertuples()]
    else:
        detailed_variants = sorted(_variant_line(row) for row in combos.itertuples())
 
    mfg_list = sorted(m for m in combos["manufacturerName"].astype(str).unique() if m != "Unknown")
    cat_list = sorted(c for c in combos["productCategoryName"].astype(str).unique() if c != "Unknown")
    name_list = sorted(n for n in combos["name"].astype(str).unique() if n != "Unknown")
    sku_list = sorted(s for s in combos["sku"].astype(str).unique() if s != "Unknown")
 
    # 8. Build the response. filters/notes are always present (even when empty)
    catalog = {
        "filters_applied": filters,
        "notes": notes,
        "total_variants_matched": len(combos),
        "all_product_variants": detailed_variants,
        "all_product_names": name_list,
        "all_skus": sku_list,
        "all_categories": cat_list,
        "all_manufacturers": mfg_list,
    }
 
    return json.dumps(catalog, indent=2)


def _generate_product_report(df_to_report: pd.DataFrame, filters: list, period_msg: str, total_stock: int = 0, prices: list = None) -> str:
    """
    Helper function to generate a detailed Markdown report with breakdown.
    Now includes active catalog stock and pricing data.
    """
    lines = []
    
    filter_str = ", ".join(filters) if filters else "All Products"
    lines.append(f"## Product Analysis Report: {filter_str}")
    lines.append(f"**Period:** {period_msg}\n")

    # 1. Safe Numeric Aggregation
    df_to_report['totalAmount'] = pd.to_numeric(df_to_report.get('totalAmount', 0), errors='coerce').fillna(0)
    df_to_report['quantity'] = pd.to_numeric(df_to_report.get('quantity', 0), errors='coerce').fillna(0)
    
    total_revenue = df_to_report['totalAmount'].sum()
    total_quantity = df_to_report['quantity'].sum()
    total_orders = df_to_report['orderId'].nunique() if 'orderId' in df_to_report.columns else 0
    
    lines.append("### Sales Summary")
    lines.append(f"- **Total Revenue:** ${total_revenue:,.2f}")
    lines.append(f"- **Total Units Sold:** {int(total_quantity):,}")
    lines.append(f"- **Total Orders:** {total_orders:,}")

    # 2. Current Catalog Status (Replacing Averages)
    lines.append(f"- **Current Stock (Catalog):** {int(total_stock):,} units")
    
    if prices:
        if len(prices) == 1:
            lines.append(f"- **Catalog Price:** ${prices[0]:,.2f}")
        else:
            min_p = min(prices)
            max_p = max(prices)
            lines.append(f"- **Catalog Price Range:** ${min_p:,.2f} - ${max_p:,.2f}")
    else:
        lines.append("- **Catalog Price:** N/A")

    lines.append("")

    # 3. Attributes Summary (Show ranges if multiple)
    lines.append("### Attributes Included")
    if 'manufacturerName' in df_to_report.columns:
        manufacturers = df_to_report['manufacturerName'].dropna().unique()
        m_str = ', '.join(str(m) for m in manufacturers[:3]) + ("..." if len(manufacturers) > 3 else "")
        lines.append(f"- **Manufacturer(s):** {m_str if m_str else 'N/A'}")
        
    if 'productCategoryName' in df_to_report.columns:
        categories = df_to_report['productCategoryName'].dropna().unique()
        c_str = ', '.join(str(c) for c in categories[:3]) + ("..." if len(categories) > 3 else "")
        lines.append(f"- **Category(s):** {c_str if c_str else 'N/A'}")

    # 4. Top Variants
    var_col = 'product_variant' if 'product_variant' in df_to_report.columns else 'name'
    if var_col in df_to_report.columns:
        unique_variants = df_to_report[var_col].nunique()
        
        if unique_variants > 1:
            lines.append(f"\n### Top 5 Products in this Group (Out of {unique_variants})")
            
            top_vars = df_to_report.groupby(var_col).agg(
                rev=('totalAmount', 'sum'),
                qty=('quantity', 'sum')
            ).sort_values(by='rev', ascending=False).head(5)
            
            lines.append("| Product Variant | Revenue | Qty |")
            lines.append("|---|---|---|")
            
            for name, row in top_vars.iterrows():
                display_name = str(name)[:45] + ".." if len(str(name)) > 45 else str(name)
                lines.append(f"| {display_name} | ${row['rev']:,.2f} | {int(row['qty']):,} |")

    # 5. Customers Who Bought This
    has_customer = 'customer_displayedName' in df_to_report.columns or 'customer_name' in df_to_report.columns
    if has_customer:
        c_col = 'customer_displayedName' if 'customer_displayedName' in df_to_report.columns else 'customer_name'
        df_to_report[c_col] = df_to_report[c_col].fillna('Unknown')
        
        cust_stats = df_to_report.groupby(c_col).agg(
            rev=('totalAmount', 'sum'),
            qty=('quantity', 'sum'),
            orders=('orderId', 'nunique')
        ).sort_values('rev', ascending=False)
        
        unique_cust_count = len(cust_stats)
        lines.append(f"\n### Top Customers ({unique_cust_count} Total Unique Buyers)")
        
        if unique_cust_count > 0:
            lines.append("| Customer | Units Bought | Total Spent | Orders |")
            lines.append("|---|---|---|---|")
            for c_name, row in cust_stats.head(10).iterrows():
                display_name = str(c_name)[:40] + ".." if len(str(c_name)) > 40 else str(c_name)
                if display_name == 'Unknown': continue
                lines.append(f"| {display_name} | {int(row['qty']):,} | ${row['rev']:,.2f} | {int(row['orders']):,} |")
                
            if unique_cust_count > 10:
                lines.append(f"*(...and {unique_cust_count - 10} more buyers)*")
        else:
            lines.append("*No customer data available for these transactions.*")

    return '\n'.join(lines)

def resolve_value(
    user_input: str,
    choices: list,
    score_cutoff: int = 80,
    ambiguity_gap: int = 5,
):
    """
    Try to resolve `user_input` to the closest value in `choices`.

    Returns a tuple: (resolved_value, note, ambiguous_candidates)
      - resolved_value: the best matching choice, or None if nothing cleared the cutoff
                         or the match was ambiguous.
      - note: human-readable string describing the substitution, or None if the
              match was exact (case-insensitive) and needs no explanation.
      - ambiguous_candidates: list of near-tied candidate values (empty if not ambiguous).
    """
    if not choices:
        return None, None, []

    matches = process.extract(
        user_input, choices, scorer=fuzz.WRatio, limit=3, score_cutoff=score_cutoff
    )
    if not matches:
        return None, None, []

    top_value, top_score, _ = matches[0]
    close = [m for m in matches if top_score - m[1] <= ambiguity_gap]

    if len(close) > 1:
        # Too close to call - don't guess, ask instead.
        return None, None, [m[0] for m in close]

    note = None
    if top_value.strip().lower() != user_input.strip().lower():
        note = f"No exact match for '{user_input}' — using closest match '{top_value}' ({top_score:.0f}% match)."

    return top_value, note, []


def apply_filter(
    df: pd.DataFrame,
    column: str,
    user_value: str,
    label: str,
    filters: list,
    notes: list,
    sku_mode: bool = False,
) -> pd.DataFrame:
    """
    Filter `df` on `column` matching `user_value`.
    Tries exact/substring match first; falls back to fuzzy matching against the
    column's unique values only if the substring match returns nothing.

    Note: whether a filter argument was PROVIDED is tracked separately by the
    caller (`provided_filters` in get_product_details). This function only
    determines whether the provided value resolved to anything - it must not
    be used to infer "no filter was passed".
    """
    if column not in df.columns:
        notes.append(f"Column '{column}' not found in data - skipping {label} filter.")
        return df

    # 1. Exact / substring match first (fast, 100% precise when it hits)
    exact = df[df[column].fillna("").str.contains(user_value, case=False, na=False)]
    if not exact.empty:
        filters.append(f"{label}='{user_value}'")
        return exact

    # 2. Fuzzy fallback - only against unique values, not every row
    cutoff = 92 if sku_mode else 80
    unique_values = df[column].dropna().unique().tolist()
    resolved, note, ambiguous = resolve_value(user_value, unique_values, score_cutoff=cutoff)

    if ambiguous:
        candidates = ", ".join(f"'{c}'" for c in ambiguous)
        notes.append(
            f"'{user_value}' matched multiple {label} values ({candidates}) — "
            f"please specify which one you meant."
        )
        return df.iloc[0:0]  # empty on purpose: force clarification instead of guessing

    if resolved is None:
        notes.append(f"No match found for {label}='{user_value}' (checked exact and fuzzy match).")
        return df.iloc[0:0]

    if note:
        notes.append(note)
    filters.append(f"{label}='{resolved}'")
    return df[df[column].fillna("").str.contains(resolved, case=False, na=False)]


@mcp.tool(name="get_product_details")
@log_tool_usage
def get_product_details(
    user_id: str,
    product_name: Optional[str] = None,
    sku: Optional[str] = None,
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Provides a detailed report for specific products, categories, or manufacturers,
    including sales metrics, stock counts, and a list of top customers who purchased them.

    Uses exact/substring matching first. If that finds nothing, falls back to fuzzy
    matching (typos, plural/singular, minor wording differences) against the known
    catalog values, and reports when a substitution or ambiguity was involved instead
    of silently guessing.

    Args:
        user_id (str): The user's ID.
        product_name (str): Filter by product name (partial match, with fuzzy fallback).
        sku (str): Filter by SKU (near-exact only; fuzzy fallback uses a strict cutoff
                   since near-miss SKUs are usually different products, not typos).
        category (str): Filter by category (partial match, with fuzzy fallback).
        manufacturer (str): Filter by manufacturer (partial match, with fuzzy fallback).
        start_date (str): 'MM/DD/YYYY'.
        end_date (str): 'MM/DD/YYYY'.
    """
    base_path = Path("data") / str(user_id)
    products_path = base_path / "cleaned_products.csv"
    orders_path = base_path / "cleaned_orders.csv"
    catalog_path = base_path / "cleaned_catalog.csv"

    if not products_path.exists():
        return f"Error: Products file not found for user {user_id}."

    try:
        df_products = pd.read_csv(products_path, encoding="utf-8-sig")
        df_products.columns = df_products.columns.str.strip().str.replace("\ufeff", "")

        if "createdAt" not in df_products.columns:
            return "Error: Required column 'createdAt' missing from products data."

        # 1. Clean Dates & Timezones safely
        df_products["createdAt"] = pd.to_datetime(df_products["createdAt"], errors="coerce")
        if df_products["createdAt"].dt.tz is not None:
            df_products["createdAt"] = df_products["createdAt"].dt.tz_localize(None)
        df_products = df_products.dropna(subset=["createdAt"])

    except Exception as e:
        return f"Error reading file: {str(e)}\n{traceback.format_exc()}"

    # 2. Time Filtering (MM/DD/YYYY)
    period_msg = "All Time"

    if start_date:
        try:
            s_dt = pd.to_datetime(start_date)
            df_products = df_products[df_products["createdAt"] >= s_dt]
            period_msg = f"From {start_date}"
        except Exception:
            return "Error: Invalid start_date format. Use MM/DD/YYYY."

    if end_date:
        try:
            e_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
            df_products = df_products[df_products["createdAt"] <= e_dt]
            if start_date:
                period_msg += f" To {end_date}"
            else:
                period_msg = f"Up to {end_date}"
        except Exception:
            return "Error: Invalid end_date format. Use MM/DD/YYYY."

    if df_products.empty:
        return f"No sales data found for the period: {period_msg}"

    # 3. Attribute Filtering (exact match first, fuzzy fallback second)
    provided_filters = {
        "product_name": product_name,
        "sku": sku,
        "category": category,
        "manufacturer": manufacturer,
    }
    if not any(provided_filters.values()):
        return "Error: Please provide at least one filter (name, sku, category, or manufacturer)."

    filters: list = []
    notes: list = []
    filtered_df = df_products.copy()

    if product_name:
        filtered_df = apply_filter(
            filtered_df, "name", product_name, "Name", filters, notes
        )

    if sku:
        filtered_df = apply_filter(
            filtered_df, "sku", sku, "SKU", filters, notes, sku_mode=True
        )

    if category:
        filtered_df = apply_filter(
            filtered_df, "productCategoryName", category, "Category", filters, notes
        )

    if manufacturer:
        filtered_df = apply_filter(
            filtered_df, "manufacturerName", manufacturer, "Brand", filters, notes
        )

    if filtered_df.empty:
        if notes:
            return "\n".join(notes)
        return f"No products found matching the given filters ({period_msg})"


    total_stock = 0
    prices = []
    df_cat = pd.DataFrame()

    if catalog_path.exists():
        try:
            df_cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
            df_cat.columns = df_cat.columns.str.strip().str.replace("\ufeff", "")
        except Exception:
            df_cat = pd.DataFrame()  # fails gracefully; no active-catalog filtering applied below

    if not df_cat.empty and "productId" in filtered_df.columns:
        active_ids = set(df_cat["id"].dropna())

        has_pid = filtered_df["productId"].notna()
        is_active = filtered_df["productId"].isin(active_ids)

        # Only drop rows where we KNOW the product is inactive (has a productId that isn't in the current catalog).
        inactive_mask = has_pid & ~is_active
        dropped = filtered_df[inactive_mask]

        if not dropped.empty:
            dropped_names = dropped["name"].dropna().unique().tolist()[:5]
            dropped_revenue = pd.to_numeric(dropped.get("totalAmount", 0), errors="coerce").fillna(0).sum()
            dropped_qty = pd.to_numeric(dropped.get("quantity", 0), errors="coerce").fillna(0).sum()
            names_preview = ", ".join(f"'{n}'" for n in dropped_names)
            notes.append(
                f"Excluded {len(dropped)} order line(s) (${dropped_revenue:,.2f}, {int(dropped_qty)} units) "
                f"for product(s) no longer in the active catalog: {names_preview}."
            )

        filtered_df = filtered_df[~inactive_mask]

    if filtered_df.empty:
        if notes:
            return "\n".join(notes)
        return f"No products found matching the given filters ({period_msg})"

    # 5. Extract Real Catalog Data (Stock & Pricing)
    if not df_cat.empty and "productId" in filtered_df.columns:
        try:
            matched_pids = filtered_df["productId"].dropna().unique()
            matched_cat = df_cat[df_cat["id"].isin(matched_pids)]

            total_stock = pd.to_numeric(
                matched_cat.get("inventory_onHand", 0), errors="coerce"
            ).fillna(0).sum()

            prices = (
                pd.to_numeric(matched_cat.get("wholesalePrice", 0), errors="coerce")
                .dropna()
                .unique()
                .tolist()
            )
        except Exception:
            pass  # Fails gracefully if catalog is malformed, leaving stock at 0

    # 6. Merge Orders data to get Customer Names
    if orders_path.exists():
        try:
            df_orders = pd.read_csv(
                orders_path,
                encoding="utf-8-sig",
                usecols=lambda c: c in ["id", "customer_name", "customer_displayedName"],
            )
            df_orders.columns = df_orders.columns.str.strip().str.replace("\ufeff", "")
            filtered_df = pd.merge(
                filtered_df, df_orders, left_on="orderId", right_on="id", how="left"
            )
        except Exception:
            pass  # Fail gracefully if orders can't be loaded

    # 7. Generate Report
    try:
        report = _generate_product_report(filtered_df, filters, period_msg, total_stock, prices)
    except Exception as e:
        return f"Error generating report: {str(e)}\n{traceback.format_exc()}"

    # 8. Prepend any fuzzy-match / ambiguity / inactive-product notes so nothing is silent
    if notes:
        note_block = "\n".join(f"⚠ {n}" for n in notes)
        return f"{note_block}\n\n{report}"

    return report  

@mcp.tool(name="get_catalog_main_info")
@log_tool_usage
def get_catalog_main_info(user_id: str) -> str:
    """
    Analyzes the product catalog to return a high-level overview including 
    data quality, financial metrics (stock value), and demand insights.
    
    Parameters:
    - user_id: User ID to locate the correct catalog file.
    """
    
    # 1. Path Setup
    catalog_path = Path("data") / user_id / "cleaned_catalog.csv"
    if not catalog_path.exists():
        return f"Error: Catalog file not found for user {user_id}."

    # 2. Data Loading & Cleaning
    try:
        df = pd.read_csv(catalog_path)
    except pd.errors.EmptyDataError:
        return "Error: The catalog file is empty."
    except Exception as e:
        return f"Error reading catalog CSV: {e}\n{traceback.format_exc()}"

    if df.empty:
        return "The catalog is empty."

    try:
        # Map parent names for variants safely
        if 'parentProductId' in df.columns and 'id' in df.columns and 'name' in df.columns:
            parent_map = df.set_index('id')['name'].to_dict()
            df['parent_name'] = df['parentProductId'].map(parent_map)
        else:
            df['parent_name'] = None

        # Base Name
        df['base_name'] = df.get('name', pd.Series([None]*len(df)))
        if 'sku' in df.columns and 'parent_name' in df.columns:
            df['base_name'] = df['base_name'].fillna(df['parent_name'] + ' (' + df['sku'].fillna('Variant') + ')')
        if 'sku' in df.columns:
            df['base_name'] = df['base_name'].fillna(df['sku'])
        df['base_name'] = df['base_name'].fillna('Unknown Product')
        
        # Display Name Builder
        def build_full_name(row):
            parts = [str(row['base_name'])]
            if 'size' in row and pd.notna(row['size']) and str(row['size']) not in parts[0]:
                parts.append(str(row['size']))
            if 'color' in row and pd.notna(row['color']) and str(row['color']) not in parts[0]:
                parts.append(str(row['color']))
            return " - ".join(parts)
            
        df['display_name'] = df.apply(build_full_name, axis=1)
        
        # Convert numeric columns
        inventory = pd.to_numeric(df.get('inventory_onHand', 0), errors='coerce').fillna(0)
        wholesale = pd.to_numeric(df.get('wholesalePrice', 0), errors='coerce').fillna(0)
        allocated = pd.to_numeric(df.get('inventory_allocated', 0), errors='coerce').fillna(0)
        
        # Stock Value calculation
        df['stock_value'] = np.where(inventory > 0, inventory * wholesale, 0)
        df['inventory_allocated_clean'] = allocated

    except Exception as e:
        return f"Error cleaning catalog data: {e}\n{traceback.format_exc()}"

    # 3. Calculate Metrics
    total_records = len(df)
    missing_skus = df['sku'].isna().sum() if 'sku' in df.columns else total_records
    missing_barcodes = df['barcode'].isna().sum() if 'barcode' in df.columns else total_records
    missing_prices = df['wholesalePrice'].isna().sum() if 'wholesalePrice' in df.columns else total_records
    
    total_inventory_value = df['stock_value'].sum()
    total_on_hand = inventory.sum()
    total_allocated = allocated.sum()
    
    # Aggregations
    value_by_cat = pd.Series(dtype=float)
    if 'productCategory_name' in df.columns:
        value_by_cat = df.groupby('productCategory_name')['stock_value'].sum().sort_values(ascending=False).head(5)
        
    top_allocated = df.sort_values(by='inventory_allocated_clean', ascending=False)[['display_name', 'inventory_allocated_clean']].head(5)
    overall_alloc_rate = (total_allocated / total_on_hand * 100) if total_on_hand > 0 else 0

    # 4. Output Formatting
    md = []
    md.append("## Catalog Health & Data Quality")
    md.append(f"* **Total Products**: {total_records}")
    md.append(f"* **Data Gaps**: Missing SKUs: {missing_skus} | Missing Barcodes: {missing_barcodes} | Missing Prices: {missing_prices}\n")
    
    md.append("## Financial & Capital Allocation")
    md.append(f"* **Total Estimated Stock Value**: ${total_inventory_value:,.2f}")
    md.append("* **Capital Concentration (Top Categories)**:")
    if not value_by_cat.empty and value_by_cat.sum() > 0:
        for cat, val in value_by_cat.items():
            if val > 0:
                percentage = (val / total_inventory_value) * 100 if total_inventory_value > 0 else 0
                cat_name = cat if pd.notna(cat) else "Uncategorized"
                md.append(f"  * **{cat_name}**: ${val:,.2f} ({percentage:.1f}%)")
    else:
        md.append("  * No category valuation data available.")
    md.append("")
        
    md.append("## Demand Insights (Current Allocations)")
    md.append(f"* **Overall Allocation Rate**: {overall_alloc_rate:.1f}% of current stock is allocated to orders.")
    md.append("* **Top Allocated Products (High Demand)**:")
    
    has_allocated = False
    for index, row in top_allocated.iterrows():
        if row['inventory_allocated_clean'] > 0:
            md.append(f"  * {row['display_name']} ({row['inventory_allocated_clean']:.0f} units allocated)")
            has_allocated = True
            
    if not has_allocated:
         md.append("  * No active allocations found.")

    return '\n'.join(md)

@mcp.tool(name="get_product_price")
@log_tool_usage
def get_product_price(
    user_id: str, 
    name: Optional[str] = None, 
    sku: Optional[str] = None,
    manufacturer: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> str:
    """
    Fetches product pricing and attributes based on various filters.
    Returns results in the format: name - sku - manufacture - size - color - Price

    Args:
        user_id (str): The user's ID to locate the data files.
        name (str): Filter by product name (partial match).
        sku (str): Filter by SKU (partial match).
        manufacturer (str): Filter by manufacturer name (partial match).
        size (str): Filter by product size (partial match).
        color (str): Filter by product color (partial match).
        min_price (float): Minimum price.
        max_price (float): Maximum price.
    """
    base_path = Path("data") / str(user_id)
    catalog_path = base_path / "cleaned_catalog.csv"
    
    if not catalog_path.exists():
        return f"Error: Required catalog file not found for user {user_id}."
        
    try:
        # Load catalog data (bypassing the old products file dependency)
        df = pd.read_csv(catalog_path, encoding='utf-8-sig', low_memory=False)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        # Fix: Inherit missing `name` and `manufacturer_name` from parent products for variants
        if 'parentProductId' in df.columns:
            parents = df[df['parentProductId'].isna()]
            if 'name' in df.columns:
                name_map = parents.set_index('id')['name']
                df['name'] = df['name'].fillna(df['parentProductId'].map(name_map))
            if 'manufacturer_name' in df.columns:
                mfg_map = parents.set_index('id')['manufacturer_name']
                df['manufacturer_name'] = df['manufacturer_name'].fillna(df['parentProductId'].map(mfg_map))
                
        # Extract only necessary columns from the consolidated catalog
        analysis_df = pd.DataFrame()
        analysis_df['Name'] = df.get('name', pd.Series(dtype='str'))
        analysis_df['SKU'] = df.get('sku', pd.Series(dtype='str'))
        analysis_df['Manufacturer'] = df.get('manufacturer_name', pd.Series(dtype='str'))
        analysis_df['Size'] = df.get('size', pd.Series(dtype='str'))
        analysis_df['Color'] = df.get('color', pd.Series(dtype='str'))
        # Using wholesalePrice as the primary price from the new catalog
        analysis_df['Price'] = df.get('wholesalePrice', pd.Series(dtype='float'))
        
        # Drop exact duplicate rows to keep output concise
        analysis_df.drop_duplicates(inplace=True)

    except Exception as e:
        return f"Error processing data: {str(e)}\n{traceback.format_exc()}"

    # Clean data types for filtering and display
    analysis_df['Name'] = analysis_df['Name'].fillna('Unknown').astype(str)
    analysis_df['SKU'] = analysis_df['SKU'].fillna('N/A').astype(str)
    analysis_df['Manufacturer'] = analysis_df['Manufacturer'].fillna('Unknown').astype(str)
    analysis_df['Size'] = analysis_df['Size'].fillna('N/A').astype(str)
    analysis_df['Color'] = analysis_df['Color'].fillna('N/A').astype(str)
    analysis_df['Price'] = pd.to_numeric(analysis_df['Price'], errors='coerce').fillna(0.0)

    # Apply Filters
    if name:
        analysis_df = analysis_df[analysis_df['Name'].str.contains(name, case=False, na=False)]
    if sku:
        analysis_df = analysis_df[analysis_df['SKU'].str.contains(sku, case=False, na=False)]
    if manufacturer:
        analysis_df = analysis_df[analysis_df['Manufacturer'].str.contains(manufacturer, case=False, na=False)]
    if size:
        analysis_df = analysis_df[analysis_df['Size'].str.contains(size, case=False, na=False)]
    if color:
        analysis_df = analysis_df[analysis_df['Color'].str.contains(color, case=False, na=False)]
    if min_price is not None:
        analysis_df = analysis_df[analysis_df['Price'] >= min_price]
    if max_price is not None:
        analysis_df = analysis_df[analysis_df['Price'] <= max_price]

    if analysis_df.empty:
        return "No pricing data found matching your criteria."

    # Sort results for readability
    analysis_df.sort_values(by=['Manufacturer', 'Name', 'Price'], inplace=True)

    # Generate Output in exact requested format:
    # Add a clear header so that it is obvious what each column represents
    header = "Name - SKU - Manufacturer - Size - Color - Price"
    separator = "-" * len(header)
    
    report_lines = [header, separator]
    
    for _, row in analysis_df.iterrows():
        # Formatting to handle empty sizes/colors nicely, but adhering to the string sequence
        r_name = row['Name']
        r_sku = row['SKU'] if row['SKU'] != 'N/A' else 'No SKU'
        r_mfg = row['Manufacturer']
        r_size = row['Size'] if row['Size'] != 'N/A' else 'No Size'
        r_color = row['Color'] if row['Color'] != 'N/A' else 'No Color'
        r_price = f"${row['Price']:.2f}"
        
        report_lines.append(f"{r_name} - {r_sku} - {r_mfg} - {r_size} - {r_color} - {r_price}")

    return "\n".join(report_lines)

@mcp.tool(name="get_executive_inventory_report")
@log_tool_usage
def get_executive_inventory_report(
    user_id: str,
    top_n: int = 5,
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
    sort_by: Optional[str] = 'Revenue at Risk',
    sort_order: Optional[str] = 'desc'
) -> str:
    """
    Generates a dense, business-focused markdown report of inventory health, capital inefficiency, and fulfillment risks.
    
    Parameters:
    - user_id: User ID to locate the correct catalog file.
    - top_n: Number of items to show in the critical liability and inefficiency tables (default: 5).
    - category: Optional filter to only analyze a specific category.
    - manufacturer: Optional filter to only analyze a specific manufacturer.
    - sort_by: Column to sort the tables by. Options: 'On Hand', 'Allocated', 'Available', 'Revenue at Risk', 'Tied Capital' (default: 'Revenue at Risk').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    """
    catalog_path = Path("data") / str(user_id) / "cleaned_catalog.csv"
    is_ascending = (sort_order.lower() == 'asc')
    
    if not catalog_path.exists():
        return f"Error: Catalog file not found for user {user_id}."

    try:
        # 1. Load Data Safely
        df = pd.read_csv(catalog_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        if df.empty:
            return "Error: The catalog file is empty."
            
        # 2. Flexible Column Matching & Filtering
        cat_col = 'productCategory_name' if 'productCategory_name' in df.columns else 'productCategoryName'
        mfg_col = 'manufacturer_name' if 'manufacturer_name' in df.columns else 'manufacturerName'
        
        filters_applied = []
        if category and cat_col in df.columns:
            df = df[df[cat_col].fillna('').str.contains(category, case=False, na=False)]
            filters_applied.append(f"Category: '{category}'")
            
        if manufacturer and mfg_col in df.columns:
            df = df[df[mfg_col].fillna('').str.contains(manufacturer, case=False, na=False)]
            filters_applied.append(f"Manufacturer: '{manufacturer}'")
            
        if df.empty:
            return f"No catalog items found matching the filters: {', '.join(filters_applied)}"

        # 3. Naming Logic (Parent Map & Display Name)
        if 'parentProductId' in df.columns and 'id' in df.columns and 'name' in df.columns:
            parent_map = df.set_index('id')['name'].to_dict()
            df['parent_name'] = df['parentProductId'].map(parent_map)
        else:
            df['parent_name'] = None
            
        df['base_name'] = (df.get('name', pd.Series([None]*len(df)))
                           .fillna(df['parent_name'] + ' (' + df.get('sku', pd.Series([None]*len(df))).fillna('Variant') + ')')
                           .fillna(df.get('sku', pd.Series([None]*len(df))))
                           .fillna('Unknown Product'))
        
        def build_full_name(row):
            parts = [str(row['base_name'])]
            if 'size' in row and pd.notna(row['size']) and str(row['size']) not in parts[0]:
                parts.append(str(row['size']))
            if 'color' in row and pd.notna(row['color']) and str(row['color']) not in parts[0]:
                parts.append(str(row['color']))
            return " - ".join(parts)
            
        df['display_name'] = df.apply(build_full_name, axis=1)

        # 4. Financial & Inventory Metrics
        df['inventory_onHand'] = pd.to_numeric(df.get('inventory_onHand', 0), errors='coerce').fillna(0)
        df['inventory_allocated'] = pd.to_numeric(df.get('inventory_allocated', 0), errors='coerce').fillna(0)
        df['wholesalePrice'] = pd.to_numeric(df.get('wholesalePrice', 0), errors='coerce').fillna(0)
        
        # Calculate active capital (positive stock) and backorder liabilities (negative stock)
        df['active_stock_value'] = np.where(df['inventory_onHand'] > 0, df['inventory_onHand'] * df['wholesalePrice'], 0)
        
        # True Deficit: If onHand is negative, or if pending allocations exceed positive onHand
        df['fulfillment_deficit'] = np.maximum(
            0, 
            df['inventory_allocated'] - df['inventory_onHand']
        )
        
        df['revenue_at_risk'] = df['fulfillment_deficit'] * df['wholesalePrice']
        df['available'] = df['inventory_onHand'] - df['inventory_allocated']

        # 5. Macro Health Metrics
        total_capital = df['active_stock_value'].sum()
        total_liability = df['revenue_at_risk'].sum()
        
        positive_on_hand_sum = df[df['inventory_onHand'] > 0]['inventory_onHand'].sum()
        overall_alloc_rate = (df['inventory_allocated'].sum() / positive_on_hand_sum * 100) if positive_on_hand_sum > 0 else 0

        # 6. Apply Dynamic Sorting
        sort_map_liabilities = {
            'on hand': 'inventory_onHand',
            'allocated': 'inventory_allocated',
            'available': 'available',
            'revenue at risk': 'revenue_at_risk',
            'tied capital': 'revenue_at_risk' # Fallback mapping
        }
        
        sort_map_inefficient = {
            'on hand': 'inventory_onHand',
            'allocated': 'inventory_allocated',
            'available': 'available',
            'revenue at risk': 'active_stock_value', # Map risk to tied capital for this table
            'tied capital': 'active_stock_value'
        }
        
        sort_col_liab = sort_map_liabilities.get(str(sort_by).lower(), 'revenue_at_risk')
        sort_col_ineff = sort_map_inefficient.get(str(sort_by).lower(), 'active_stock_value')

        liabilities_df = df[df['fulfillment_deficit'] > 0].sort_values(by=sort_col_liab, ascending=is_ascending).head(top_n)
        inefficient_df = df[(df['inventory_allocated'] == 0) & (df['inventory_onHand'] > 0)].sort_values(by=sort_col_ineff, ascending=is_ascending).head(top_n)

        # 7. Build Markdown Report
        md = ["## Executive Inventory & Fulfillment Report"]
        
        if filters_applied:
            md.append(f"*(Filters applied: {', '.join(filters_applied)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*")
        else:
            md.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*")
            
        md.extend([
            "\n### Portfolio Health & Capital Allocation",
            f"- **Active Capital Tied in Inventory:** ${total_capital:,.2f}",
            f"- **Total Revenue at Risk (Backorders & Deficits):** ${total_liability:,.2f}",
            f"- **Pipeline Utilization:** {overall_alloc_rate:.1f}% of active inventory is currently pending fulfillment.",
            
            f"\n### Critical Fulfillment Liabilities (Top {top_n})",
            "*Products with the highest unfulfilled demand or negative inventory balances, ranked by total wholesale revenue at risk.*",
            "| Product | ON HAND | ALLOCATED | AVAILABLE | Revenue at Risk |",
            "|---|---|---|---|---|"
        ])

        if not liabilities_df.empty:
            for _, row in liabilities_df.iterrows():
                # Truncate extremely long names
                name = str(row['display_name'])[:40] + '..' if len(str(row['display_name'])) > 40 else str(row['display_name'])
                md.append(f"| **{name}** | {row['inventory_onHand']:,.0f} | {row['inventory_allocated']:,.0f} | {row['available']:,.0f} | **${row['revenue_at_risk']:,.2f}** |")
        else:
            md.append("| _No immediate fulfillment liabilities detected._ | - | - | - | - |")

        md.extend([
            f"\n### Capital Inefficiency Alerts (Top {top_n})",
            "*Products tying up the most capital with zero pending orders. Consider targeted promotions or liquidation.*",
            "| Product | Current Stock | Wholesale Price | Tied Capital |",
            "|---|---|---|---|"
        ])

        if not inefficient_df.empty:
            for _, row in inefficient_df.iterrows():
                name = str(row['display_name'])[:40] + '..' if len(str(row['display_name'])) > 40 else str(row['display_name'])
                md.append(f"| **{name}** | {row['inventory_onHand']:,.0f} | ${row['wholesalePrice']:,.2f} | **${row['active_stock_value']:,.2f}** |")
        else:
            md.append("| _No highly inefficient capital allocation detected._ | - | - | - |")

        return '\n'.join(md)

    except Exception as e:
        return f"Error generating executive inventory report: {str(e)}\n{traceback.format_exc()}"

@mcp.tool(name="get_product_performance_portfolio_report")
@log_tool_usage
def get_product_performance_portfolio_report(
    user_id: str,
    top_n: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_order: Optional[str] = 'desc',
    min_revenue: Optional[float] = None,
    min_units: Optional[int] = None,
    min_orders: Optional[int] = None,
    min_buyers: Optional[int] = None,
    min_price: Optional[float] = None,
    min_stock: Optional[int] = None,
    min_engagement: Optional[float] = None
) -> str:
    """
    Analyzes product performance by merging catalog, orders, and products data.
    Categorizes products into Top Revenue Drivers, High Penetration Opportunities, and Underperforming Assets.
    
    Parameters:
    - user_id: User ID to locate the data files.
    - top_n: Number of products to show in each category breakdown (default: 5).
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - sort_order: 'desc' (default) or 'asc'. Flips the sorting logic of the resulting tables.
    - min_revenue: Filter products with at least this much total revenue.
    - min_units: Filter products with at least this many units sold.
    - min_orders: Filter products with at least this many orders.
    - min_buyers: Filter products with at least this many unique buyers.
    - min_price: Filter products with at least this average price.
    - min_stock: Filter products with at least this much stock on hand.
    - min_engagement: Filter products with at least this engagement score.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    catalog_path = base_path / "cleaned_catalog.csv"
    is_ascending = (sort_order.lower() == 'asc')

    # Check for missing files
    missing_files = [p.name for p in [orders_path, products_path, catalog_path] if not p.exists()]
    if missing_files:
        return f"Error: Missing required files for user {user_id}: {', '.join(missing_files)}"

    try:
        # 1. Load Data Safely
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig', usecols=lambda c: c in ['id', 'customer_name', 'createdAt'])
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')
        df_catalog = pd.read_csv(catalog_path, encoding='utf-8-sig')

        for df in [df_orders, df_products, df_catalog]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'id' not in df_orders.columns or 'productId' not in df_products.columns or 'id' not in df_catalog.columns:
            return "Error: Required ID columns missing from datasets."

        # 2. Date Filtering (Apply to Orders)
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        if df_orders['createdAt'].dt.tz is not None:
             df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])

        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df_orders = df_orders[df_orders['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use MM/DD/YYYY."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df_orders = df_orders[df_orders['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df_orders.empty:
            return "No orders found in the specified date range."

        # 3. Setup Catalog Dictionary
        valid_catalog_ids = set(df_catalog['id'].dropna().unique())
        
        for col in ['size', 'color', 'inventory_onHand', 'name']:
            if col not in df_catalog.columns:
                df_catalog[col] = None
                
        cat_dict = df_catalog.set_index('id')[['size', 'color', 'inventory_onHand', 'name']].to_dict('index')

        # 4. Filter Products to Active Catalog and Valid Date Orders
        df_products_active = df_products[
            (df_products['productId'].isin(valid_catalog_ids)) & 
            (df_products['orderId'].isin(df_orders['id']))
        ].copy()

        if df_products_active.empty:
            return "No matching products found between the filtered orders and the active catalog."

        if 'name' not in df_products_active.columns:
            df_products_active['name'] = df_products_active.get('product_variant', pd.Series([None]*len(df_products_active)))

        # Detailed Naming Logic
        def get_detailed_name(row):
            name = row.get('name')
            if pd.isna(name): return "Unknown Product"
            pid = row.get('productId')
            if pid in cat_dict:
                cat_name = cat_dict[pid].get('name')
                if pd.notna(cat_name): name = cat_name
                size = cat_dict[pid].get('size')
                color = cat_dict[pid].get('color')
                if pd.notna(size): return f"{name} {size}"
                elif pd.notna(color): return f"{name} {color}"
            return str(name)

        df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        # 5. Merge with Orders to get customer names for "Unique Buyers"
        merged = pd.merge(df_products_active, df_orders[['id', 'customer_name']], left_on='orderId', right_on='id', how='inner')

        # 6. Calculate Metrics globally
        merged['totalAmount'] = pd.to_numeric(merged.get('totalAmount', 0), errors='coerce').fillna(0)
        merged['quantity'] = pd.to_numeric(merged.get('quantity', 0), errors='coerce').fillna(0)
        merged['price'] = pd.to_numeric(merged.get('price', 0), errors='coerce').fillna(0)

        metrics = merged.groupby(['productId', 'detailed_name']).agg(
            total_revenue=('totalAmount', 'sum'),
            total_units=('quantity', 'sum'),
            order_count=('orderId', 'nunique'),
            unique_customers=('customer_name', 'nunique'),
            avg_price=('price', 'mean')
        ).reset_index()

        metrics['stock'] = metrics['productId'].apply(
            lambda pid: int(pd.to_numeric(cat_dict[pid].get('inventory_onHand', 0), errors='coerce')) if pid in cat_dict and pd.notna(cat_dict[pid].get('inventory_onHand')) else 0
        )
        
        # Pre-calculate engagement score for all products to allow filtering
        denominator = np.log1p(metrics['total_revenue'])
        denominator = np.where(denominator == 0, 1, denominator) 
        metrics['engagement_score'] = (metrics['unique_customers'] * metrics['order_count']) / denominator

        # 7. Apply Dynamic Value Filters
        filters_applied = []
        if start_date or end_date: 
            filters_applied.append(f"Dates: {start_date or 'All Time'} to {end_date or 'Present'}")
            
        if min_revenue is not None:
            metrics = metrics[metrics['total_revenue'] >= min_revenue]
            filters_applied.append(f"Min Revenue: ${min_revenue}")
        if min_units is not None:
            metrics = metrics[metrics['total_units'] >= min_units]
            filters_applied.append(f"Min Units: {min_units}")
        if min_orders is not None:
            metrics = metrics[metrics['order_count'] >= min_orders]
            filters_applied.append(f"Min Orders: {min_orders}")
        if min_buyers is not None:
            metrics = metrics[metrics['unique_customers'] >= min_buyers]
            filters_applied.append(f"Min Buyers: {min_buyers}")
        if min_price is not None:
            metrics = metrics[metrics['avg_price'] >= min_price]
            filters_applied.append(f"Min Price: ${min_price}")
        if min_stock is not None:
            metrics = metrics[metrics['stock'] >= min_stock]
            filters_applied.append(f"Min Stock: {min_stock}")
        if min_engagement is not None:
            metrics = metrics[metrics['engagement_score'] >= min_engagement]
            filters_applied.append(f"Min Engagement: {min_engagement}")

        if metrics.empty:
            return f"No products found matching the applied criteria: {', '.join(filters_applied)}"

        # 8. Define Performance Categories
        
        # A. Top Revenue Drivers
        heavyweights = metrics.sort_values('total_revenue', ascending=is_ascending).head(top_n)
        top_revenue_pids = set(heavyweights['productId'])

        # B. High Penetration Opportunities
        potential_gems = metrics[~metrics['productId'].isin(top_revenue_pids)].copy()
        hidden_gems = potential_gems.sort_values('engagement_score', ascending=is_ascending).head(top_n)

        # C. Underperforming Assets (Reverses logic based on sort_order)
        dead_weight = metrics.sort_values(['order_count', 'total_revenue'], ascending=[not is_ascending, not is_ascending]).head(top_n)

        # 9. Build the Markdown Report
        md = [
            "## Product Performance & Portfolio Insights",
            f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}"
        ]
        
        if filters_applied:
            md.append(f"*(Filters applied: {', '.join(filters_applied)} | Order: {sort_order.upper()})*\n")

        # Top Revenue Drivers
        md.extend([
            f"### Top Revenue Drivers (Top {len(heavyweights)})",
            "*These products generate the highest gross revenue. It is critical to maintain adequate inventory levels to protect these revenue streams.*",
            "| Product | Total Revenue | Units Sold | Orders | Unique Buyers | Stock | Engagement Score |",
            "|---|---|---|---|---|---|---|"
        ])
        if heavyweights.empty:
            md.append("| No data | - | - | - | - | - | - |")
        else:
            for _, row in heavyweights.iterrows():
                name = str(row['detailed_name'])[:35] + ".." if len(str(row['detailed_name'])) > 35 else str(row['detailed_name'])
                md.append(f"| **{name}** | ${row['total_revenue']:,.2f} | {int(row['total_units']):,} | {int(row['order_count']):,} | {int(row['unique_customers']):,} | {row['stock']:,} | {row['engagement_score']:.2f} |")

        # High Penetration Opportunities
        md.extend([
            f"\n### High Penetration Opportunities (Top {len(hidden_gems)})",
            "*(Engagement Score = (Unique Buyers * Total Orders) / ln(Total Revenue + 1))*",
            "*These products demonstrate high purchase frequency and broad customer appeal but yield lower overall revenue.*",
            "| Product | Engagement Score | Unique Buyers | Orders | Total Revenue | Avg Price | Stock |",
            "|---|---|---|---|---|---|---|"
        ])
        if hidden_gems.empty:
            md.append("| No data | - | - | - | - | - | - |")
        else:
            for _, row in hidden_gems.iterrows():
                name = str(row['detailed_name'])[:35] + ".." if len(str(row['detailed_name'])) > 35 else str(row['detailed_name'])
                md.append(f"| **{name}** | {row['engagement_score']:.2f} | {int(row['unique_customers']):,} | {int(row['order_count']):,} | ${row['total_revenue']:,.2f} | ${row['avg_price']:.2f} | {row['stock']:,} |")

        # Underperforming Assets
        md.extend([
            f"\n### Underperforming Assets (Bottom {len(dead_weight)})",
            "*These products exhibit minimal order volume and low revenue, tying up capital and inventory space.*",
            "| Product | Orders | Total Revenue | Units Sold | Unique Buyers | Stock | Engagement Score |",
            "|---|---|---|---|---|---|---|"
        ])
        if dead_weight.empty:
            md.append("| No data | - | - | - | - | - | - |")
        else:
            for _, row in dead_weight.iterrows():
                name = str(row['detailed_name'])[:35] + ".." if len(str(row['detailed_name'])) > 35 else str(row['detailed_name'])
                md.append(f"| **{name}** | {int(row['order_count']):,} | ${row['total_revenue']:,.2f} | {int(row['total_units']):,} | {int(row['unique_customers']):,} | {row['stock']:,} | {row['engagement_score']:.2f} |")

        return '\n'.join(md)

    except Exception as e:
        return f"Error generating product performance report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_product_customer_insights_report")
@log_tool_usage
def get_product_customer_insights_report(
    user_id: str,
    top_n: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_product: Optional[str] = None,
    sort_by: Optional[str] = 'Revenue',
    sort_order: Optional[str] = 'desc',
    min_revenue: Optional[float] = None,
    min_units: Optional[int] = None,
    min_orders: Optional[int] = None,
    min_buyers: Optional[int] = None,
    min_avg_units: Optional[float] = None,
    min_basket_halo: Optional[float] = None
) -> str:
    """
    Generates advanced customer insights and purchasing behavior for the top products (or a specific product).
    
    Parameters:
    - user_id: User ID to locate data files.
    - top_n: Number of top products to analyze (default: 3). Ignored if specific_product is provided.
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - specific_product: Optional product name filter to analyze a single specific product.
    - sort_by: 'Revenue', 'Units', 'Orders', 'Buyers', 'Avg Units', 'Basket Halo' (default: 'Revenue').
    - sort_order: 'desc' (default) or 'asc'.
    - min_revenue: Filter to products generating at least this much revenue.
    - min_units: Filter to products selling at least this many units.
    - min_orders: Filter to products included in at least this many orders.
    - min_buyers: Filter to products bought by at least this many unique customers.
    - min_avg_units: Filter to products with at least this many average units per buyer.
    - min_basket_halo: Filter to products with an average total basket size of at least this much.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    catalog_path = base_path / "cleaned_catalog.csv"
    is_ascending = (sort_order.lower() == 'asc')

    # Check for missing files
    missing_files = [p.name for p in [orders_path, products_path, catalog_path] if not p.exists()]
    if missing_files:
        return f"Error: Missing required files for user {user_id}: {', '.join(missing_files)}"

    try:
        # 1. Load Data
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig', usecols=lambda c: c in ['id', 'customer_name', 'createdAt', 'totalAmount'])
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')
        df_catalog = pd.read_csv(catalog_path, encoding='utf-8-sig')

        for df in [df_orders, df_products, df_catalog]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'id' not in df_orders.columns or 'productId' not in df_products.columns or 'id' not in df_catalog.columns:
            return "Error: Required ID columns missing from datasets."

        # 2. Date Filtering (Apply to Orders)
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        if df_orders['createdAt'].dt.tz is not None:
             df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])

        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df_orders = df_orders[df_orders['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use MM/DD/YYYY."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df_orders = df_orders[df_orders['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df_orders.empty:
            return "No orders found in the specified date range."

        # Rename order amount to avoid collision
        df_orders = df_orders.rename(columns={'totalAmount': 'order_totalAmount'})

        # 3. Setup Catalog Dictionary for Naming
        for col in ['size', 'color', 'name']:
            if col not in df_catalog.columns:
                df_catalog[col] = None
                
        cat_dict = df_catalog.set_index('id')[['size', 'color', 'name']].to_dict('index')

        # 4. Filter Products to Active Orders
        df_products_active = df_products[df_products['orderId'].isin(df_orders['id'])].copy()
        if df_products_active.empty:
            return "No products found in the specified orders."

        def get_detailed_name(row):
            name = row.get('name')
            if pd.isna(name): return "Unknown Product"
            pid = row.get('productId')
            if pid in cat_dict:
                cat_name = cat_dict[pid].get('name')
                if pd.notna(cat_name): name = cat_name
                size = cat_dict[pid].get('size')
                color = cat_dict[pid].get('color')
                if pd.notna(size): return f"{name} {size}"
                elif pd.notna(color): return f"{name} {color}"
            return str(name)

        if 'name' not in df_products_active.columns:
            df_products_active['name'] = df_products_active.get('product_variant', pd.Series([None]*len(df_products_active)))
            
        df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        # 5. Merge with Orders
        merged = pd.merge(df_products_active, df_orders, left_on='orderId', right_on='id', how='inner')
        
        merged['totalAmount'] = pd.to_numeric(merged.get('totalAmount', 0), errors='coerce').fillna(0)
        merged['quantity'] = pd.to_numeric(merged.get('quantity', 0), errors='coerce').fillna(0)
        merged['order_totalAmount'] = pd.to_numeric(merged.get('order_totalAmount', 0), errors='coerce').fillna(0)

        # 6. Global Metric Calculation (Pre-calculate for all products to allow filtering)
        prod_stats = merged.groupby(['productId', 'detailed_name']).agg(
            total_rev=('totalAmount', 'sum'),
            total_qty=('quantity', 'sum'),
            unique_buyers=('customer_name', 'nunique'),
            order_count=('orderId', 'nunique')
        ).reset_index()

        prod_stats['avg_units_per_buyer'] = np.where(prod_stats['unique_buyers'] > 0, prod_stats['total_qty'] / prod_stats['unique_buyers'], 0)

        # Basket Halo Vectorized Calculation
        # Drop duplicate orders per product to ensure we only count each basket once per item
        basket_halo = merged.drop_duplicates(subset=['productId', 'orderId']).groupby('productId')['order_totalAmount'].mean().reset_index()
        basket_halo = basket_halo.rename(columns={'order_totalAmount': 'avg_basket_size'})
        
        prod_stats = prod_stats.merge(basket_halo, on='productId', how='left')
        prod_stats['avg_basket_size'] = prod_stats['avg_basket_size'].fillna(0)

        # 7. Apply Dynamic Value Filters
        filters_applied = []
        if min_revenue is not None:
            prod_stats = prod_stats[prod_stats['total_rev'] >= min_revenue]
            filters_applied.append(f"Min Rev: ${min_revenue}")
        if min_units is not None:
            prod_stats = prod_stats[prod_stats['total_qty'] >= min_units]
            filters_applied.append(f"Min Units: {min_units}")
        if min_orders is not None:
            prod_stats = prod_stats[prod_stats['order_count'] >= min_orders]
            filters_applied.append(f"Min Orders: {min_orders}")
        if min_buyers is not None:
            prod_stats = prod_stats[prod_stats['unique_buyers'] >= min_buyers]
            filters_applied.append(f"Min Buyers: {min_buyers}")
        if min_avg_units is not None:
            prod_stats = prod_stats[prod_stats['avg_units_per_buyer'] >= min_avg_units]
            filters_applied.append(f"Min Avg Units: {min_avg_units}")
        if min_basket_halo is not None:
            prod_stats = prod_stats[prod_stats['avg_basket_size'] >= min_basket_halo]
            filters_applied.append(f"Min Basket Halo: ${min_basket_halo}")

        if prod_stats.empty:
            return f"No products found matching the applied criteria: {', '.join(filters_applied)}"

        # 8. Sort and Determine Target Products
        sort_mapping = {
            'revenue': 'total_rev',
            'units': 'total_qty',
            'orders': 'order_count',
            'buyers': 'unique_buyers',
            'avg units': 'avg_units_per_buyer',
            'basket halo': 'avg_basket_size'
        }
        sort_col = sort_mapping.get(sort_by.lower(), 'total_rev')
        prod_stats = prod_stats.sort_values(by=sort_col, ascending=is_ascending)

        target_products = pd.DataFrame()
        if specific_product:
            matches = prod_stats[prod_stats['detailed_name'].str.contains(specific_product, case=False, na=False)]
            if matches.empty:
                return f"No product sales found matching '{specific_product}' after applying filters."
            target_products = matches.head(1)
            report_title = f"# Customer Insights: '{specific_product}'"
        else:
            target_products = prod_stats.head(top_n)
            report_title = f"# Top {len(target_products)} Products: Customer Insights & Demographics"

        if target_products.empty:
            return "No valid products found to analyze."

        # 9. Generate Insights Report
        md = [report_title]
        
        if start_date or end_date: 
            filters_applied.insert(0, f"Dates: {start_date or 'All Time'} to {end_date or 'Present'}")
            
        if filters_applied:
            md.append(f"*(Filters applied: {', '.join(filters_applied)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            md.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        for i, row in enumerate(target_products.itertuples(), 1):
            pid = row.productId
            p_name = row.detailed_name

            # Formatting Section
            md.append(f"## {i if not specific_product else ''}. {p_name}")
            md.append(f"- **Product Revenue:** ${row.total_rev:,.2f} | **Units Sold:** {int(row.total_qty):,}")
            md.append(f"- **Unique Buyers:** {row.unique_buyers:,} | **Orders:** {int(row.order_count):,}")
            md.append(f"- **Avg. Units per Buyer:** {row.avg_units_per_buyer:.1f} units")
            md.append(f"- **Basket Halo Effect (Avg Order Value containing this item):** ${row.avg_basket_size:,.2f}\n")
            
            # Top Customers for this specific product
            p_sales = merged[merged['productId'] == pid]
            top_custs = p_sales.groupby('customer_name').agg(
                Units=('quantity', 'sum'),
                Spend=('totalAmount', 'sum')
            ).sort_values('Spend', ascending=False).head(3)
            
            md.append("#### Top 3 Buyers of this Product:")
            md.append("| Customer | Units Bought | Total Spent on Item |")
            md.append("|---|---|---|")
            for c_name, c_row in top_custs.iterrows():
                md.append(f"| {str(c_name)[:40]} | {int(c_row['Units']):,} | ${c_row['Spend']:,.2f} |")
                
            md.append("\n---\n")

        return '\n'.join(md)

    except Exception as e:
        return f"Error generating customer insights report: {str(e)}\n{traceback.format_exc()}"


@mcp.tool(name="get_cross_sell_bundle_report")
@log_tool_usage
def get_cross_sell_bundle_report(
    user_id: str,
    top_n: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = 'Potential Value',
    sort_order: Optional[str] = 'desc',
    min_common_orders: Optional[int] = 1
) -> str:
    """
    Analyzes cross-category product pairings, customer purchasing overlap, and calculates untapped bundle revenue.
    
    Parameters:
    - user_id: User ID to locate data files.
    - top_n: Number of bundle pairs to return (default: 3).
    - start_date: Optional start date filter (MM/DD/YYYY).
    - end_date: Optional end date filter (MM/DD/YYYY).
    - sort_by: Column to sort bundles by. Options: 'Common Orders', 'Potential Value' (default: 'Potential Value').
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - min_common_orders: Minimum number of times the items must have been bought together to be considered a bundle.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    catalog_path = base_path / "cleaned_catalog.csv"
    is_ascending = (sort_order.lower() == 'asc')

    missing_files = [p.name for p in [orders_path, products_path, catalog_path] if not p.exists()]
    if missing_files:
        return f"Error: Missing required files for user {user_id}: {', '.join(missing_files)}"

    try:
        # 1. Load Data
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig', usecols=lambda c: c in ['id', 'customer_name', 'createdAt', 'totalAmount'])
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')
        df_catalog = pd.read_csv(catalog_path, encoding='utf-8-sig')

        for df in [df_orders, df_products, df_catalog]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 2. Date Filtering (Apply to Orders)
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        if df_orders['createdAt'].dt.tz is not None:
             df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])

        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df_orders = df_orders[df_orders['createdAt'] >= start_dt]
            except Exception:
                return "Error: Invalid start_date format. Use MM/DD/YYYY."

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
                df_orders = df_orders[df_orders['createdAt'] <= end_dt]
            except Exception:
                return "Error: Invalid end_date format. Use MM/DD/YYYY."

        if df_orders.empty:
            return "No orders found in the specified date range."

        # 3. Setup Catalog Dictionary
        valid_catalog_ids = set(df_catalog['id'].dropna().unique())
        for col in ['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name']:
            if col not in df_catalog.columns:
                df_catalog[col] = None
        
        cat_dict = df_catalog.set_index('id')[['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name']].to_dict('index')

        def get_detailed_name(row):
            name = row.get('name')
            if pd.isna(name): return "Unknown Product"
            pid = row.get('productId')
            if pid in cat_dict:
                cat_name = cat_dict[pid].get('name')
                if pd.notna(cat_name): name = cat_name
                size = cat_dict[pid].get('size')
                color = cat_dict[pid].get('color')
                if pd.notna(size): return f"{name} {size}"
                elif pd.notna(color): return f"{name} {color}"
            return str(name)

        # 4. Filter Products to Active Catalog and valid Orders
        df_products_active = df_products[
            (df_products['productId'].isin(valid_catalog_ids)) & 
            (df_products['orderId'].isin(df_orders['id']))
        ].copy()

        if df_products_active.empty:
            return "No matching products found between orders and the active catalog."

        if 'name' not in df_products_active.columns:
            df_products_active['name'] = df_products_active.get('product_variant', pd.Series([None]*len(df_products_active)))
            
        df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        # 5. Gather Product Stats (Pricing, Stock, Category)
        df_products_active['price'] = pd.to_numeric(df_products_active.get('price', 0), errors='coerce').fillna(0)
        df_products_active['quantity'] = pd.to_numeric(df_products_active.get('quantity', 0), errors='coerce').fillna(0)
        
        prod_stats = df_products_active.groupby('detailed_name').agg({
            'price': 'mean',
            'quantity': 'mean',
            'productCategoryName': lambda x: x.mode()[0] if not x.mode().empty else 'Uncategorized',
            'productId': 'first'
        }).reset_index()

        prod_info = {}
        for _, row in prod_stats.iterrows():
            d_name = row['detailed_name']
            pid = row['productId']
            price = row['price']
            stock = 0
            if pid in cat_dict:
                cat_stock = cat_dict[pid].get('inventory_onHand')
                cat_price = cat_dict[pid].get('wholesalePrice')
                stock = cat_stock if pd.notna(cat_stock) else 0
                if pd.notna(cat_price) and cat_price > 0:
                    price = cat_price
                    
            prod_info[d_name] = {
                'price': float(price),
                'stock': float(stock),
                'category': str(row['productCategoryName'])
            }

        # 6. Find Top Product Pairs (Cross-Category Only)
        order_prods_df = df_products_active.dropna(subset=['detailed_name']).groupby('orderId')['detailed_name'].unique()
        pairs = []
        for prods in order_prods_df:
            if len(prods) > 1:
                valid_pairs = []
                for p1, p2 in itertools.combinations(sorted(prods), 2):
                    if prod_info[p1]['category'] != prod_info[p2]['category']:
                        valid_pairs.append((p1, p2))
                pairs.extend(valid_pairs)

        if not pairs:
            return "No valid cross-category product pairings found in the data."

        pair_counts = pd.Series(pairs).value_counts().reset_index()
        pair_counts.columns = ['pair', 'common_orders']
        
        if min_common_orders is not None:
            pair_counts = pair_counts[pair_counts['common_orders'] >= min_common_orders]

        if pair_counts.empty:
            return f"No bundles found with at least {min_common_orders} common orders."

        # 7. Merge for Customer-Level Insights
        merged = pd.merge(df_products_active, df_orders[['id', 'customer_name']], left_on='orderId', right_on='id', how='left')
        cust_prods = merged.groupby('customer_name')['detailed_name'].unique().to_dict()

        # 8. Calculate Target Value & Build Bundle Objects
        bundles_data = []
        order_groups = merged.groupby('orderId').agg({'detailed_name': lambda x: set(x), 'customer_name': 'first'})

        for _, row in pair_counts.iterrows():
            prod_a, prod_b = row['pair']
            common_orders = row['common_orders']
            
            info_a = prod_info[prod_a]
            info_b = prod_info[prod_b]
            
            success_orders = order_groups[order_groups['detailed_name'].apply(lambda x: prod_a in x and prod_b in x)]
            success_custs = success_orders['customer_name'].dropna().unique().tolist()
            
            targets = []
            
            # Target buyers of A who haven't bought B
            buyers_a = merged[merged['detailed_name'] == prod_a]
            if not buyers_a.empty:
                agg_a = buyers_a.groupby('customer_name').agg({'quantity': 'mean'})
                for cust, cust_row in agg_a.iterrows():
                    if cust in cust_prods and prod_b not in cust_prods[cust]:
                        targets.append({
                            'customer': cust, 'source': prod_a, 'pitch': prod_b,
                            'qty': cust_row['quantity'], 'val': cust_row['quantity'] * info_b['price']
                        })
                        
            # Target buyers of B who haven't bought A
            buyers_b = merged[merged['detailed_name'] == prod_b]
            if not buyers_b.empty:
                agg_b = buyers_b.groupby('customer_name').agg({'quantity': 'mean'})
                for cust, cust_row in agg_b.iterrows():
                    if cust in cust_prods and prod_a not in cust_prods[cust]:
                        targets.append({
                            'customer': cust, 'source': prod_b, 'pitch': prod_a,
                            'qty': cust_row['quantity'], 'val': cust_row['quantity'] * info_a['price']
                        })

            targets_df = pd.DataFrame(targets)
            if not targets_df.empty:
                targets_df = targets_df.sort_values('val', ascending=False).drop_duplicates(subset=['customer'])
                
            total_targets = len(targets_df) if not targets_df.empty else 0
            total_missed_val = targets_df['val'].sum() if total_targets > 0 else 0
            
            bundles_data.append({
                'prod_a': prod_a, 'prod_b': prod_b,
                'info_a': info_a, 'info_b': info_b,
                'common_orders': common_orders,
                'success_custs': success_custs,
                'targets_df': targets_df,
                'total_targets': total_targets,
                'total_missed_val': total_missed_val
            })

        # 9. Dynamic Sorting
        sort_mapping = {
            'common orders': 'common_orders',
            'potential value': 'total_missed_val'
        }
        sort_col = sort_mapping.get(sort_by.lower(), 'total_missed_val')
        
        # Sort list of dicts
        bundles_data.sort(key=lambda x: x[sort_col], reverse=not is_ascending)
        
        # Limit to Top N
        bundles_data = bundles_data[:top_n]

        # 10. Build Markdown Output
        md_lines = [
            "# Cross-Selling Bundle Opportunities",
            f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}"
        ]
        
        filter_str = []
        if start_date or end_date: 
            filter_str.append(f"Dates: {start_date or 'All Time'} to {end_date or 'Present'}")
        if min_common_orders and min_common_orders > 1:
            filter_str.append(f"Min Common Orders: {min_common_orders}")
            
        if filter_str:
            md_lines.append(f"*(Filters applied: {', '.join(filter_str)} | Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")
        else:
            md_lines.append(f"*(Sorted by: {sort_by} | Order: {sort_order.upper()})*\n")

        for row in bundles_data:
            prod_a, prod_b = row['prod_a'], row['prod_b']
            info_a, info_b = row['info_a'], row['info_b']
            
            stock_a, stock_b = int(info_a['stock']), int(info_b['stock'])
            stock_a_str = f"**In Stock ({stock_a} units)**" if stock_a > 0 else f"**Out of Stock ({stock_a} units)**"
            stock_b_str = f"**In Stock ({stock_b} units)**" if stock_b > 0 else f"**Out of Stock ({stock_b} units)**"
            
            md_lines.append(f"## {prod_a} & {prod_b}")
            md_lines.append("\n**Stock Availability & Standalone Pricing:**")
            md_lines.append(f"- **{prod_a} ({info_a['category']}):** {stock_a_str} (Price: ${info_a['price']:,.2f})")
            md_lines.append(f"- **{prod_b} ({info_b['category']}):** {stock_b_str} (Price: ${info_b['price']:,.2f})\n")
            
            md_lines.append(f"**The Synergy:** Found together organically in **{row['common_orders']}** historical order(s).")
            
            if row['success_custs']:
                success_display = ", ".join(row['success_custs'][:3])
                if len(row['success_custs']) > 3:
                    success_display += f", and {len(row['success_custs'])-3} others"
                md_lines.append(f"- **Proven Traction:** Customers like **{success_display}** have already purchased these together, validating the cross-category demand.")
                
            md_lines.append("\n### How to Scale This Bundle")
            if row['total_targets'] > 0:
                discount_pct = 10
                conversion_rate = 0.30
                projected_rev = row['total_missed_val'] * conversion_rate * (1 - discount_pct/100)
                
                md_lines.append(f"You have **{row['total_targets']} customers** who buy one of these items, but not the other. The total untapped potential is **${row['total_missed_val']:,.2f}**.")
                md_lines.append(f"1. **Action:** Create a **{discount_pct}% Off Bundle Discount** in SimplyDepo specifically pairing these two items.")
                md_lines.append(f"2. **Outreach:** Pitch this new incentive to the target list below.")
                md_lines.append(f"3. **Impact:** If just 30% of these targets convert using the discount, you generate **~${projected_rev:,.2f}** in immediate incremental revenue.")
            else:
                md_lines.append("1. **Action:** Create a **Bundle Discount** in SimplyDepo for these items.")
                md_lines.append("2. **Impact:** Even with a small customer base, incentivizing cross-category purchases increases Average Order Value (AOV) and establishes new, more profitable purchasing habits across your territory.")
                
            md_lines.append("\n**Top Target Customers (Missed Opportunities):**")
            if row['total_targets'] > 0:
                for _, t in row['targets_df'].head(5).iterrows():
                    md_lines.append(f"- **{t['customer']}:** Buys ~{int(t['qty'])}x {t['source']} per order → **Pitch: {int(t['qty'])}x {t['pitch']}** (Upsell Value: **${t['val']:,.2f}**)")
            else:
                md_lines.append("- *All current buyers of these products already purchase them together! This is a highly mature bundle. Expand your reach to completely new accounts.*")
                
            md_lines.append("\n---\n")
            
        return '\n'.join(md_lines)

    except Exception as e:
        return f"Error generating bundle report: {str(e)}\n{traceback.format_exc()}"



@mcp.tool(name="get_time_based_product_report")
@log_tool_usage
def get_time_based_product_report(
    user_id: str,
    top_n: int = 10,
    recent_days: int = 180,
    new_days: int = 180,
    sort_by_new: Optional[str] = 'Total Revenue',
    sort_order_new: Optional[str] = 'desc',
    sort_by_stagnant: Optional[str] = 'Last Sold Date',
    sort_order_stagnant: Optional[str] = 'desc',
    min_revenue: Optional[float] = None
) -> str:
    """
    Generates a time-based performance report identifying newly added products gaining traction and historical products that have stopped selling.
    
    Parameters:
    - user_id: User ID to locate data files.
    - top_n: Number of products to show in each table (default: 10).
    - recent_days: Days without an order to consider a product 'stagnant' (default: 180).
    - new_days: Days since catalog creation to consider a product 'new' (default: 180).
    - sort_by_new: Sort New Products by 'Date Added', 'Total Revenue', 'Units Sold', 'Orders', 'Unique Buyers', 'Available'.
    - sort_order_new: 'desc' or 'asc'.
    - sort_by_stagnant: Sort Stagnant Products by 'Last Sold Date', 'Lifetime Orders', 'Total Revenue', 'Available'.
    - sort_order_stagnant: 'desc' or 'asc'.
    - min_revenue: Optional filter to only include products with at least this much lifetime revenue.
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    catalog_path = base_path / "cleaned_catalog.csv"

    # Check for missing files
    missing_files = [p.name for p in [orders_path, products_path, catalog_path] if not p.exists()]
    if missing_files:
        return f"Error: Missing required files for user {user_id}: {', '.join(missing_files)}"

    try:
        # 1. Load Data Safely
        catalog_df = pd.read_csv(catalog_path, encoding='utf-8-sig')
        orders_df = pd.read_csv(orders_path, encoding='utf-8-sig', usecols=lambda c: c in ['id', 'customer_name', 'createdAt'])
        products_df = pd.read_csv(products_path, encoding='utf-8-sig')

        for df in [catalog_df, orders_df, products_df]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 2. Clean and Parse Dates
        catalog_df['createdAt'] = pd.to_datetime(
            catalog_df['createdAt'].astype(str).str.replace(r' GMT\+0000 \(Coordinated Universal Time\)', '', regex=True), 
            errors='coerce'
        )
        if catalog_df['createdAt'].dt.tz is not None:
             catalog_df['createdAt'] = catalog_df['createdAt'].dt.tz_localize(None)

        orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], errors='coerce')
        if orders_df['createdAt'].dt.tz is not None:
             orders_df['createdAt'] = orders_df['createdAt'].dt.tz_localize(None)
        orders_df = orders_df.dropna(subset=['createdAt'])

        if orders_df.empty:
            return "No valid order dates found to perform time-based analysis."

        # Establish dynamic boundaries based on data reality
        max_order_date = orders_df['createdAt'].max()
        max_cat_date = catalog_df['createdAt'].max()
        
        if pd.isna(max_cat_date): max_cat_date = pd.Timestamp.now()
        
        recent_cutoff = max_order_date - pd.Timedelta(days=recent_days)
        new_product_cutoff = max_cat_date - pd.Timedelta(days=new_days)

        # 3. Build Catalog Dictionary
        valid_catalog_ids = set(catalog_df['id'].dropna().unique())
        
        for col in ['size', 'color', 'inventory_onHand', 'inventory_allocated', 'name', 'createdAt']:
            if col not in catalog_df.columns:
                catalog_df[col] = None
                
        cat_dict = catalog_df.set_index('id')[['size', 'color', 'inventory_onHand', 'inventory_allocated', 'name', 'createdAt']].to_dict('index')

        df_products_active = products_df[products_df['productId'].isin(valid_catalog_ids)].copy()
        if df_products_active.empty:
            return "No matching products found between orders and active catalog."

        # Naming Logic
        if 'name' not in df_products_active.columns:
            df_products_active['name'] = df_products_active.get('product_variant', pd.Series([None]*len(df_products_active)))

        def get_detailed_name(row):
            name = row.get('name')
            if pd.isna(name): return "Unknown Product"
            pid = row.get('productId')
            if pid in cat_dict:
                cat_name = cat_dict[pid].get('name')
                if pd.notna(cat_name): name = cat_name
                size = cat_dict[pid].get('size')
                color = cat_dict[pid].get('color')
                if pd.notna(size): return f"{name} {size}"
                elif pd.notna(color): return f"{name} {color}"
            return str(name)

        df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        # 4. Merge Products with Orders
        merged = pd.merge(df_products_active, orders_df, left_on='orderId', right_on='id', how='inner', suffixes=('', '_order'))
        
        merged['totalAmount'] = pd.to_numeric(merged.get('totalAmount', 0), errors='coerce').fillna(0)
        merged['quantity'] = pd.to_numeric(merged.get('quantity', 0), errors='coerce').fillna(0)

        # 5. Group and Calculate Time-Based Metrics
        metrics = merged.groupby(['productId', 'detailed_name']).agg(
            total_revenue=('totalAmount', 'sum'),
            total_units=('quantity', 'sum'),
            order_count=('orderId', 'nunique'),
            unique_customers=('customer_name', 'nunique'),
            last_order_date=('createdAt_order', 'max')
        ).reset_index()

        # Calculate 'Available' (onHand - allocated) and map creation dates
        metrics['available'] = metrics['productId'].apply(
            lambda pid: int(
                (pd.to_numeric(cat_dict[pid].get('inventory_onHand', 0), errors='coerce') or 0) - 
                (pd.to_numeric(cat_dict[pid].get('inventory_allocated', 0), errors='coerce') or 0)
            ) if pid in cat_dict else 0
        )
        metrics['catalog_created_at'] = metrics['productId'].apply(
            lambda pid: cat_dict[pid].get('createdAt') if pid in cat_dict else pd.NaT
        )

        # 6. Apply Filters
        if min_revenue is not None:
            metrics = metrics[metrics['total_revenue'] >= min_revenue]

        # 7. Identify Targets and Apply Dynamic Sorting
        
        # New Products
        sort_map_new = {
            'date added': 'catalog_created_at',
            'total revenue': 'total_revenue',
            'units sold': 'total_units',
            'orders': 'order_count',
            'unique buyers': 'unique_customers',
            'available': 'available'
        }
        col_new = sort_map_new.get(sort_by_new.lower(), 'total_revenue')
        asc_new = sort_order_new.lower() == 'asc'
        
        # FIX: Added .copy() here
        new_products = metrics[metrics['catalog_created_at'] >= new_product_cutoff].copy()
        if not new_products.empty:
            new_products['Sort_Helper'] = new_products[col_new].fillna(pd.Timestamp.min if asc_new else pd.Timestamp.max)
            new_products = new_products.sort_values(by='Sort_Helper', ascending=asc_new).head(top_n)

        # Stagnant Products
        sort_map_stag = {
            'last sold date': 'last_order_date',
            'lifetime orders': 'order_count',
            'total revenue': 'total_revenue',
            'available': 'available'
        }
        col_stag = sort_map_stag.get(sort_by_stagnant.lower(), 'last_order_date')
        asc_stag = sort_order_stagnant.lower() == 'asc'
        
        # FIX: Added .copy() here
        stopped_selling_all = metrics[(metrics['last_order_date'] < recent_cutoff) & (metrics['order_count'] > 0)].copy()
        stopped_selling = pd.DataFrame()
        
        if not stopped_selling_all.empty:
            stopped_selling_all['Sort_Helper'] = stopped_selling_all[col_stag].fillna(pd.Timestamp.min if asc_stag else pd.Timestamp.max)
            stopped_selling = stopped_selling_all.sort_values(by='Sort_Helper', ascending=asc_stag).head(top_n)

        # 8. Build the Markdown Report
        md = [
            "# Time-Based Product Performance Report",
            f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}"
        ]
        if min_revenue:
            md.append(f"*(Global Filter: Lifetime Revenue >= ${min_revenue})*\n")

        # Table 1: New Products Analysis
        md.append("## 1. New Products Analysis (Recently Added)")
        md.append(f"*Products added to the catalog since **{new_product_cutoff.strftime('%m/%d/%Y')}** ({new_days} days). Analyzed to determine if they are gaining traction and beneficial to the business.*")
        md.append(f"*(Sorted by: {sort_by_new} | Order: {sort_order_new.upper()})*")
        md.append("| Product | Date Added | Total Revenue | Units Sold | Orders | Unique Buyers | Available |")
        md.append("|---|---|---|---|---|---|---|")
        
        if new_products.empty:
            md.append("| No new products found in this timeframe | - | - | - | - | - | - |")
        else:
            for _, row in new_products.iterrows():
                date_added = row['catalog_created_at'].strftime('%m/%d/%Y') if pd.notna(row['catalog_created_at']) else 'N/A'
                name = str(row['detailed_name'])[:35] + '..' if len(str(row['detailed_name'])) > 35 else str(row['detailed_name'])
                md.append(f"| **{name}** | {date_added} | ${row['total_revenue']:,.2f} | {int(row['total_units']):,} | {int(row['order_count']):,} | {int(row['unique_customers']):,} | {int(row['available']):,} |")
        md.append("\n---\n")

        # Table 2: Stagnant Products
        md.append("## 2. Stagnant Products (Stopped Selling)")
        md.append(f"*Products with historical sales but no orders since **{recent_cutoff.strftime('%m/%d/%Y')}** ({recent_days} days).*")
        md.append(f"*(Sorted by: {sort_by_stagnant} | Order: {sort_order_stagnant.upper()})*")
        md.append("| Product | Last Sold Date | Lifetime Orders | Total Revenue | Available |")
        md.append("|---|---|---|---|---|")
        
        if stopped_selling.empty:
            md.append("| No stagnant products found in this timeframe | - | - | - | - |")
        else:
            for _, row in stopped_selling.iterrows():
                last_sold = row['last_order_date'].strftime('%m/%d/%Y') if pd.notna(row['last_order_date']) else 'N/A'
                name = str(row['detailed_name'])[:40] + '..' if len(str(row['detailed_name'])) > 40 else str(row['detailed_name'])
                md.append(f"| **{name}** | {last_sold} | {int(row['order_count']):,} | ${row['total_revenue']:,.2f} | {int(row['available']):,} |")

            # Dynamic notation if there are more than N stagnant products
            if len(stopped_selling_all) > top_n:
                extra_count = len(stopped_selling_all) - top_n
                md.append(f"\n*Note: There are {extra_count:,} more stagnant products not shown here based on current filters.*")

        return '\n'.join(md)

    except Exception as e:
        return f"Error generating time-based report: {str(e)}\n{traceback.format_exc()}"    


@mcp.tool(name="get_sales_prospecting_report")
@log_tool_usage
def get_sales_prospecting_report(
    user_id: str,
    product_name: str,
    top_n: int = 5
) -> str:
    """
    Generates a proactive sales prospecting list for a specific product, identifying restock 
    candidates and high-probability cross-sell targets based on basket analysis.
    
    Parameters:
    - user_id: User's ID.
    - product_name: The name (or partial name) of the product you want to sell.
    - top_n: Number of prospects to return in each category (default: 5).
    """
    base_path = Path("data") / str(user_id)
    orders_path = base_path / "cleaned_orders.csv"
    products_path = base_path / "cleaned_products.csv"
    catalog_path = base_path / "cleaned_catalog.csv"

    # Check for missing files
    missing_files = [p.name for p in [orders_path, products_path, catalog_path] if not p.exists()]
    if missing_files:
        return f"Error: Missing required files for user {user_id}: {', '.join(missing_files)}"

    try:
        # 1. Load Data
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig', usecols=lambda c: c in ['id', 'customer_name', 'createdAt'])
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')
        df_catalog = pd.read_csv(catalog_path, encoding='utf-8-sig')

        for df in [df_orders, df_products, df_catalog]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 2. Date Cleaning (Apply to Orders)
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        if df_orders['createdAt'].dt.tz is not None:
             df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])

        # 3. Catalog Dictionary & Naming
        valid_catalog_ids = set(df_catalog['id'].dropna().unique())
        for col in ['size', 'color', 'inventory_onHand', 'inventory_allocated', 'wholesalePrice', 'name']:
            if col not in df_catalog.columns:
                df_catalog[col] = None
                
        cat_dict = df_catalog.set_index('id')[['size', 'color', 'inventory_onHand', 'inventory_allocated', 'wholesalePrice', 'name']].to_dict('index')

        def get_detailed_name(row):
            name = row.get('name')
            if pd.isna(name): return "Unknown Product"
            pid = row.get('productId')
            if pid in cat_dict:
                cat_name = cat_dict[pid].get('name')
                if pd.notna(cat_name): name = cat_name
                size = cat_dict[pid].get('size')
                color = cat_dict[pid].get('color')
                if pd.notna(size): return f"{name} {size}"
                elif pd.notna(color): return f"{name} {color}"
            return str(name)

        df_products_active = df_products[df_products['productId'].isin(valid_catalog_ids)].copy()
        if 'name' not in df_products_active.columns:
            df_products_active['name'] = df_products_active.get('product_variant', pd.Series([None]*len(df_products_active)))
            
        df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        # 4. Merge with Orders safely
        # Fix for KeyError: Drop 'createdAt' from products so it doesn't collide with orders' 'createdAt'
        df_products_active = df_products_active.drop(columns=['createdAt'], errors='ignore')
        
        merged = pd.merge(df_products_active, df_orders, left_on='orderId', right_on='id', how='inner')
        merged['quantity'] = pd.to_numeric(merged.get('quantity', 0), errors='coerce').fillna(0)
        merged['totalAmount'] = pd.to_numeric(merged.get('totalAmount', 0), errors='coerce').fillna(0)

        # 5. Find Target Product
        matches = merged[merged['detailed_name'].str.contains(product_name, case=False, na=False)]
        if matches.empty:
            return f"No sales history found for a product matching '{product_name}'."
            
        # If multiple matches, take the highest revenue one
        target_pid = matches.groupby('productId')['totalAmount'].sum().idxmax()
        target_name = matches[matches['productId'] == target_pid]['detailed_name'].iloc[0]
        
        # Get target product stock & price
        t_stock = 0
        t_price = 0.0
        if target_pid in cat_dict:
            on_hand = pd.to_numeric(cat_dict[target_pid].get('inventory_onHand', 0), errors='coerce') or 0
            alloc = pd.to_numeric(cat_dict[target_pid].get('inventory_allocated', 0), errors='coerce') or 0
            t_stock = int(on_hand - alloc)
            t_price = float(pd.to_numeric(cat_dict[target_pid].get('wholesalePrice', 0), errors='coerce') or 0)

        # 6. Basket Analysis (What sells with this?)
        target_orders = merged[merged['productId'] == target_pid]['orderId'].unique()
        basket_items = merged[merged['orderId'].isin(target_orders) & (merged['productId'] != target_pid)]
        
        top_cross_sells = pd.DataFrame()
        cross_sell_pids = []
        if not basket_items.empty:
            top_cross_sells = basket_items.groupby(['productId', 'detailed_name']).agg(
                CommonOrders=('orderId', 'nunique')
            ).reset_index().sort_values('CommonOrders', ascending=False).head(3)
            cross_sell_pids = top_cross_sells['productId'].tolist()

        # 7. Prospect Segment A: Restock Candidates (Warm Leads)
        target_sales = merged[merged['productId'] == target_pid]
        restock_candidates = target_sales.groupby('customer_name').agg(
            TotalOrders=('orderId', 'nunique'),
            AvgQty=('quantity', 'mean'),
            LastOrderDate=('createdAt', 'max')
        ).reset_index().sort_values('LastOrderDate', ascending=True) # Oldest first (most likely due for restock)
        
        # 8. Prospect Segment B: Cross-Sell Targets (Untapped)
        cross_sell_targets = pd.DataFrame()
        if cross_sell_pids:
            # Customers who bought the cross-sell items
            buyers_of_cross_sells = merged[merged['productId'].isin(cross_sell_pids)]
            
            # Customers who bought the target item
            buyers_of_target = target_sales['customer_name'].unique()
            
            # Filter to customers who bought cross-sells BUT NOT the target item
            untapped_sales = buyers_of_cross_sells[~buyers_of_cross_sells['customer_name'].isin(buyers_of_target)]
            
            if not untapped_sales.empty:
                # Group to find the best targets (those who buy the complementary items most often)
                cross_sell_targets = untapped_sales.groupby('customer_name').agg(
                    ComplementaryOrders=('orderId', 'nunique'),
                    FavoriteComplement=('detailed_name', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
                ).reset_index().sort_values('ComplementaryOrders', ascending=False)

        # 9. Format the Report
        md = [
            f"# Sales Prospecting Report: {target_name}",
            f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}",
            f"**Available Stock:** {t_stock:,} units | **Price:** ${t_price:,.2f}"
        ]
        
        if t_stock <= 0:
            md.append("\n **WARNING: This product is currently out of stock or fully allocated. Prospecting may result in backorders.**")

        # Cross-Sell Insights
        md.append("\n## 🛒 Basket Analysis: Perfect Pairings")
        if top_cross_sells.empty:
            md.append("*No distinct complementary products found in historical orders.*")
        else:
            md.append("*When customers buy this item, they frequently buy these products in the same order:*")
            for _, row in top_cross_sells.iterrows():
                md.append(f"- **{row['detailed_name']}** (Found in {row['CommonOrders']} shared orders)")

        # Restock Candidates
        md.append(f"\n##  Warm Leads: Restock Candidates (Top {top_n})")
        md.append("*Historical buyers sorted by the oldest last-order date. These customers already love the product and may be running low.*")
        md.append("| Customer | Last Bought | Total Orders | Suggested Pitch Qty | Pitch Value |")
        md.append("|---|---|---|---|---|")
        
        if restock_candidates.empty:
            md.append("| No historical buyers found | - | - | - | - |")
        else:
            for _, row in restock_candidates.head(top_n).iterrows():
                last_dt = row['LastOrderDate'].strftime('%m/%d/%Y')
                rec_qty = int(row['AvgQty'])
                val = rec_qty * t_price
                c_name = str(row['customer_name'])[:35] + '..' if len(str(row['customer_name'])) > 35 else str(row['customer_name'])
                md.append(f"| **{c_name}** | {last_dt} | {row['TotalOrders']} | {rec_qty:,} units | **${val:,.2f}** |")

        # Untapped Prospects
        md.append(f"\n##  Net-New Prospects: Cross-Sell Targets (Top {top_n})")
        md.append("*Customers who frequently buy the 'Perfect Pairings' listed above, but have **never** bought the target product. Highly likely to convert on a bundle or trial pitch.*")
        md.append("| Customer | Pitch Angle (They already buy...) | Orders of Complement | Suggested Pitch Qty |")
        md.append("|---|---|---|---|")
        
        if cross_sell_targets.empty:
            md.append("| No untapped cross-sell targets found | - | - | - |")
        else:
            # Suggest a conservative trial quantity (e.g., 1 case/unit or average restock size if known)
            avg_trial_qty = max(1, int(restock_candidates['AvgQty'].median() if not restock_candidates.empty else 1))
            
            for _, row in cross_sell_targets.head(top_n).iterrows():
                c_name = str(row['customer_name'])[:30] + '..' if len(str(row['customer_name'])) > 30 else str(row['customer_name'])
                comp_name = str(row['FavoriteComplement'])[:40] + '..' if len(str(row['FavoriteComplement'])) > 40 else str(row['FavoriteComplement'])
                md.append(f"| **{c_name}** | Loves *{comp_name}* | {row['ComplementaryOrders']} | {avg_trial_qty:,} units (Trial) |")

        
        return '\n'.join(md)

    except Exception as e:
        return f"Error generating prospecting report: {str(e)}\n{traceback.format_exc()}"



def _mode_or_none(series: pd.Series):
    """Most frequent non-null value in a Series, or None if there isn't one."""
    counts = series.dropna().value_counts()
    return counts.index[0] if not counts.empty else None


def _resolve_date_window(start_date, end_date, lookback_days, reference_date, notes):
    """
    reference_date = latest activity actually present in the data - used as
    "today" for lookback_days. Returns (start_dt, end_dt, period_msg); both
    None means All Time.
    """
    if (start_date or end_date) and lookback_days:
        notes.append(
            "Both an explicit date range and lookback_days were provided — "
            "using the explicit start_date/end_date and ignoring lookback_days."
        )
        lookback_days = None

    if lookback_days:
        end_dt = reference_date
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        period_msg = (
            f"Last {lookback_days} days ({start_dt.date()} to {end_dt.date()}, anchored to "
            f"the most recent activity in your data: {reference_date.date()})"
        )
        return start_dt, end_dt, period_msg

    start_dt = end_dt = None
    period_msg = "All Time"

    if start_date:
        start_dt = pd.to_datetime(start_date)
        period_msg = f"From {start_date}"
    if end_date:
        end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
        period_msg = f"{period_msg} To {end_date}" if start_date else f"Up to {end_date}"

    return start_dt, end_dt, period_msg


@mcp.tool(name="get_cross_sell_prospects")
@log_tool_usage
def get_cross_sell_prospects(
    user_id: str,
    product_name: Optional[str] = None,
    sku: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: Optional[int] = None,
    top_n: int = 25,
    min_category_orders: int = 1,
    lapsed_threshold_days: int = 60,
) -> str:
    """
    Finds who to offer a given product to, segmented by purchase history:

    - All Time: "New Prospects" (never bought it) and "Previous Buyers"
      (bought it before) - both shown, nobody excluded.
    - A specific period (start_date/end_date or lookback_days): anyone who
      bought the product DURING that period is excluded (they just got it).
      The rest are split into "New Prospects" (never bought it, ever) and
      "Reorder Candidates" (bought it before, just not in this period - a
      win-back/repeat-purchase target).

    Every customer shown also gets a lifecycle tag - "New Customer", "Active",
    or "Lapsed (Nd)" - based on their overall order history across ALL
    products, so you can tell a brand-new lead apart from a long-time
    customer who's gone quiet on everything.

    Args:
        user_id (str): The user's ID.
        product_name (str): Product to analyze (partial match, fuzzy fallback).
        sku (str): Alternative/additional way to pin down the exact product
                   (near-exact match only). At least one of product_name or
                   sku must be provided.
        start_date (str): 'MM/DD/YYYY'. Ignored if lookback_days is given.
        end_date (str): 'MM/DD/YYYY'. Ignored if lookback_days is given.
        lookback_days (int): Shortcut for "last N days" (e.g. 30, 60). Anchored
                              to the most recent activity in the data, not the
                              real calendar date (see module docstring).
        top_n (int): Number of customers to return PER section (default 25).
        min_category_orders (int): Minimum category orders for the New
                                    Prospects section, to filter out one-off
                                    noise (default 1 = no filtering).
        lapsed_threshold_days (int): Days with no purchases of anything before
                                      a customer is tagged "Lapsed" instead of
                                      "Active" (default 60).
    """
    if not product_name and not sku:
        return "Error: Please provide product_name and/or sku to identify the product."
    if top_n <= 0:
        top_n = 25

    base_path = Path("data") / str(user_id)
    products_path = base_path / "cleaned_products.csv"
    orders_path = base_path / "cleaned_orders.csv"
    catalog_path = base_path / "cleaned_catalog.csv"

    if not products_path.exists():
        return f"Error: Products file not found for user {user_id}."
    if not orders_path.exists():
        return f"Error: Orders file not found for user {user_id} (needed to identify customers)."

    notes: list = []
    filters: list = []

    # 1. Load & clean products
    try:
        df_products = pd.read_csv(products_path, encoding="utf-8-sig")
        df_products.columns = df_products.columns.str.strip().str.replace("\ufeff", "")
        df_products["createdAt"] = pd.to_datetime(df_products["createdAt"], errors="coerce")
        if df_products["createdAt"].dt.tz is not None:
            df_products["createdAt"] = df_products["createdAt"].dt.tz_localize(None)
        df_products = df_products.dropna(subset=["createdAt"])
    except Exception as e:
        return f"Error reading products file: {str(e)}\n{traceback.format_exc()}"

    # 2. Load just enough of orders to attribute line items to customers
    try:
        df_orders = pd.read_csv(
            orders_path, encoding="utf-8-sig",
            usecols=lambda c: c in ["id", "customer_id", "customer_name", "customer_displayedName"],
        )
        df_orders.columns = df_orders.columns.str.strip().str.replace("\ufeff", "")
    except Exception as e:
        return f"Error reading orders file: {str(e)}\n{traceback.format_exc()}"

    df_all = pd.merge(
        df_products, df_orders, left_on="orderId", right_on="id", how="left", suffixes=("", "_order")
    )
    if "customer_id" not in df_all.columns:
        return "Error: Could not attribute order lines to customers (missing customer_id after merge)."

    reference_date = df_all["createdAt"].max()

    # 3. Resolve the target product across ALL history
    matched_all = apply_filter(df_all, "name", product_name, "Name", filters, notes) if product_name else df_all
    if sku:
        matched_all = apply_filter(matched_all, "sku", sku, "SKU", filters, notes, sku_mode=True)

    if matched_all.empty:
        if notes:
            return "\n".join(notes)
        return "No product found matching the given name/SKU."

    resolved_names = matched_all["name"].dropna().unique().tolist()
    category_name = _mode_or_none(matched_all["productCategoryName"])
    manufacturer_name = _mode_or_none(matched_all["manufacturerName"])
    product_ids = set(matched_all["productId"].dropna().unique())

    unattributed_target = int(matched_all["customer_id"].isna().sum())
    if unattributed_target:
        notes.append(
            f"{unattributed_target} historical order line(s) for this product couldn't be "
            f"attributed to a customer (missing/unresolved order record) and were ignored "
            f"when checking purchase history."
        )
    matched_all = matched_all[matched_all["customer_id"].notna()]
    buyer_ids_all = set(matched_all["customer_id"].unique())

    # 4. Sanity-check the product: is it still active / in stock?
    total_stock = None
    if catalog_path.exists():
        try:
            df_cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
            df_cat.columns = df_cat.columns.str.strip().str.replace("\ufeff", "")
            cat_match = df_cat[df_cat["id"].isin(product_ids)]
            if product_ids and cat_match.empty:
                notes.append(
                    f"'{', '.join(resolved_names[:3])}' does not appear in the active catalog "
                    f"(may be discontinued) — consider whether it should still be recommended."
                )
            elif not cat_match.empty:
                total_stock = pd.to_numeric(
                    cat_match.get("inventory_onHand", 0), errors="coerce"
                ).fillna(0).sum()
                if total_stock <= 0:
                    notes.append(
                        f"Current on-hand stock for this product is {int(total_stock)} — "
                        f"check availability before running outreach."
                    )
        except Exception:
            pass

    # 5. Resolve the analysis window (anchored to data's own latest activity)
    start_dt, end_dt, period_msg = _resolve_date_window(start_date, end_date, lookback_days, reference_date, notes)
    is_windowed = start_dt is not None or end_dt is not None

    # 6. Who bought the target product DURING the window vs before it
    matched_window = matched_all
    if start_dt is not None:
        matched_window = matched_window[matched_window["createdAt"] >= start_dt]
    if end_dt is not None:
        matched_window = matched_window[matched_window["createdAt"] <= end_dt]
    in_period_buyer_ids = set(matched_window["customer_id"].unique())
    prior_buyer_ids = buyer_ids_all - in_period_buyer_ids  # only meaningful when windowed

    if is_windowed and in_period_buyer_ids:
        notes.append(
            f"{len(in_period_buyer_ids)} customer(s) already bought this product during "
            f"{period_msg} and were excluded from the lists below (they just bought it)."
        )

    # 7. Lifecycle tag - based on ALL products, all time
    lifetime_orders = df_all.groupby("customer_id")["orderId"].nunique()
    last_order_overall = df_all.groupby("customer_id")["createdAt"].max()

    def lifecycle_tag(cid):
        total = lifetime_orders.get(cid, 0)
        last = last_order_overall.get(cid)
        if total <= 1:
            return "New Customer"
        if pd.notna(last):
            days_inactive = (reference_date - last).days
            if days_inactive > lapsed_threshold_days:
                return f"Lapsed ({days_inactive}d)"
        return "Active"

    name_cols = [c for c in ["customer_displayedName", "customer_name"] if c in df_all.columns]
    names_df = (
        df_all.drop_duplicates("customer_id").set_index("customer_id")[name_cols] if name_cols else None
    )

    def display_name(cid):
        if names_df is None or cid not in names_df.index:
            return cid
        row = names_df.loc[cid]
        for col in name_cols:
            val = row[col] if len(name_cols) > 1 else row
            if pd.notna(val) and str(val).strip():
                return val
        return cid

    # 8. Bucket 1 - New Prospects: never bought this product, ever
    df_window_pool = df_all
    if start_dt is not None:
        df_window_pool = df_window_pool[df_window_pool["createdAt"] >= start_dt]
    if end_dt is not None:
        df_window_pool = df_window_pool[df_window_pool["createdAt"] <= end_dt]

    pool = df_window_pool[
        (df_window_pool["productCategoryName"] == category_name)
        | (df_window_pool["manufacturerName"] == manufacturer_name)
    ]
    pool = pool[~pool["customer_id"].isin(buyer_ids_all) & pool["customer_id"].notna()]

    new_prospects = pd.DataFrame()
    fallback_used = False
    if not pool.empty:
        grouped = pool.groupby("customer_id").agg(
            spend=("totalAmount", "sum"), orders=("orderId", "nunique")
        ).reset_index()
        grouped = grouped[grouped["orders"] >= min_category_orders]
        new_prospects = grouped

    if new_prospects.empty:
        fallback_used = True
        fallback_pool = df_window_pool[
            ~df_window_pool["customer_id"].isin(buyer_ids_all) & df_window_pool["customer_id"].notna()
        ]
        if not fallback_pool.empty:
            new_prospects = fallback_pool.groupby("customer_id").agg(
                spend=("totalAmount", "sum"), orders=("orderId", "nunique")
            ).reset_index()
            notes.append(
                f"No customers found with '{category_name}'/'{manufacturer_name}' affinity during "
                f"{period_msg} — New Prospects below are ranked by overall spend instead."
            )

    if not new_prospects.empty:
        new_prospects["display_name"] = new_prospects["customer_id"].apply(display_name)
        new_prospects["lifecycle"] = new_prospects["customer_id"].apply(lifecycle_tag)
        new_prospects = new_prospects.sort_values(["spend", "orders"], ascending=[False, False])

    # 9. Bucket 2 - Previous Buyers (All Time) / Reorder Candidates (windowed)
    reorder_candidates = pd.DataFrame()
    if is_windowed and prior_buyer_ids:
        prior_df = matched_all[matched_all["customer_id"].isin(prior_buyer_ids)]
    elif not is_windowed and buyer_ids_all:
        prior_df = matched_all
    else:
        prior_df = None

    if prior_df is not None and not prior_df.empty:
        reorder_candidates = prior_df.groupby("customer_id").agg(
            product_spend=("totalAmount", "sum"),
            product_orders=("orderId", "nunique"),
            last_target_purchase=("createdAt", "max"),
        ).reset_index()
        reorder_candidates["display_name"] = reorder_candidates["customer_id"].apply(display_name)
        reorder_candidates["lifecycle"] = reorder_candidates["customer_id"].apply(lifecycle_tag)
        reorder_candidates = reorder_candidates.sort_values("product_spend", ascending=False)

    if new_prospects.empty and reorder_candidates.empty:
        if notes:
            return "\n".join(notes)
        return f"No prospective customers found for the given product and period ({period_msg})."

    # 10. Build the report
    lines = []
    title = resolved_names[0] if resolved_names else (product_name or sku)
    lines.append(f"=== Cross-Sell / Reorder Prospects: {title} ===")
    lines.append(f"Category: {category_name or 'Unknown'} | Manufacturer: {manufacturer_name or 'Unknown'}")
    if total_stock is not None:
        lines.append(f"Current stock on hand: {int(total_stock)} units")
    lines.append(f"Period analyzed: {period_msg}")
    lines.append("")

    ranking_basis = "overall spend (fallback)" if fallback_used else f"{category_name} category/brand affinity + spend"
    lines.append(f"--- New Prospects ({ranking_basis}) ---")
    if new_prospects.empty:
        lines.append("(none found)")
    else:
        for rank, row in enumerate(new_prospects.head(top_n).itertuples(), start=1):
            lines.append(
                f"{rank}. {row.display_name} — ${row.spend:,.2f} across {row.orders} order(s) | "
                f"Customer status: {row.lifecycle}"
            )
    lines.append("")

    section_label = (
        "Reorder Candidates (bought before, not during this period)" if is_windowed else "Previous Buyers (all time)"
    )
    lines.append(f"--- {section_label} ---")
    if reorder_candidates.empty:
        lines.append("(none found)")
    else:
        for rank, row in enumerate(reorder_candidates.head(top_n).itertuples(), start=1):
            last_dt = (
                row.last_target_purchase.strftime("%Y-%m-%d") if pd.notna(row.last_target_purchase) else "Unknown"
            )
            lines.append(
                f"{rank}. {row.display_name} — ${row.product_spend:,.2f} lifetime on this product across "
                f"{row.product_orders} order(s) | Last bought: {last_dt} | Customer status: {row.lifecycle}"
            )

    report = "\n".join(lines)

    if notes:
        note_block = "\n".join(f"⚠ {n}" for n in notes)
        return f"{note_block}\n\n{report}"

    return report

# FAQ Tool

@mcp.tool(name="look_up_faq")
@log_tool_usage
def look_up_faq(query: Optional[str]) -> str:
    """
    Searches the FAQ (Frequently Asked Questions) database 
    to find answers to user questions about policies or features.

    Args:
        query: The specific question or topic the user is asking about.
    """
    # 1. Provide a fallback if query is None
    if not query:
        return "No query provided. Please ask a specific question."

    # 2. Fix the function call to match your search_md_db definition
    file_to_parse = "AI/group_customer_analyze/Agents_rules/SD_FAQ.md"
    init_and_load_md(file_to_parse)
    try:
        response = search_md_db(query_text=query, n_results=5)
        formatted_string = format_search_results(response)

        return formatted_string
    except Exception as e:
        return f"Database Search Error: {str(e)}"



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
    mcp.run(transport="streamable-http")