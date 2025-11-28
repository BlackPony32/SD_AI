import asyncio
import aiofiles
import os
import pandas as pd
import time

from AI.utils import get_logger

#from langchain.agents import tool
#from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain_openai import ChatOpenAI
#from langchain_community.callbacks import get_openai_callback

logger2 = get_logger("logger2", "project_log_many.log", False)


    
#Test block of new Ai conversation
import logging
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession
from agents.extensions.memory import AdvancedSQLiteSession
import asyncio
import aiofiles
import os
from dotenv import load_dotenv
load_dotenv()
from agents.extensions.memory import AdvancedSQLiteSession
from pathlib import Path
from typing import List, AsyncGenerator, Tuple, Any
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_create_full_report, prompt_agent_create_sectioned, prompt_for_state_agent
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm_model = OpenAIResponsesModel(model='gpt-4.1', openai_client=AsyncOpenAI()) 


@function_tool
def General_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    from create_report_group_c import generate_analytics_report_sectioned
    logger2.info(f"Tool 'General_statistics_tool' called ")
    data_path = f"data/{user_id}/full_report.md"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            statistics =  f.read()
        logger2.info(f"Successfully read statistics from {data_path}")
        return statistics
    except FileNotFoundError:
        logger2.error(f"Statistics file not found at: {data_path}")
        return "Error: Statistics file not found."
    except Exception as e:
        logger2.error(f"Error reading {data_path}: {e}")
        return f"Error: {e}"

@function_tool
def Detailed_statistics_tool(user_id:str, query: str) -> str:
    '''Use this tool to fetch detailed customer statistics and insights from the overall report. Input should be a specific topic or question.'''
    logger2.info(f"Tool 'Detailed_statistics_tool' called called for: {query}")
    data_path = f"data/{user_id}/overall_report.txt"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            statistics =  f.read()
        logger2.info(f"Successfully read statistics from {data_path}")
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(statistics)

        # Create vector store
        emb = OpenAIEmbeddings()
        index = FAISS.from_texts(chunks, emb)

        docs = index.similarity_search(query, k=10)
        return "\n".join(d.page_content for d in docs)
    
    except FileNotFoundError:
        logger2.error(f"Statistics file not found at: {data_path}")
        return "Error: Statistics file not found."
    except Exception as e:
        logger2.error(f"Error reading {data_path}: {e}")
        return f"Error: {e}"

     
@function_tool
def get_top_n_orders(
    user_id: str, 
    n: int, 
    by_type: str, 
    sort_order: str = 'desc', 
    start_date: str = None, 
    end_date: str = None,
    status_filter: str = None
) -> str:
    """
    Gets the top (or bottom) N orders based on revenue or quantity, 
    optionally filtered by a start date and order status.

    Parameters:
    - user_id: The user's ID.
    - n: Number of records to return.
    - by_type: 'revenue' or 'totalQuantity'.
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - start_date: Filter orders created ON or AFTER this date. Format: 'YYYY-MM-DD'.
    - end_date: Filter data from start day to this date (YYYY-MM-DD).
    - status_filter: Filter by specific order status. 
      VALID VALUES: orderStatus - ['COMPLETED', 'PENDING' or None].
    """
    logger2.info(f"Tool 'get_top_n_orders' called for: {user_id} order: {sort_order},     by_type: {by_type}, start_date: {start_date},  status_filter: {status_filter}, end_date: {end_date}")
    # 1. Path Setup
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return f"Error: File not found."

    try:
        # Load specific columns including Date and Statuses
        # Adjusted to match your CSV structure
        relevant_cols = [
            'customId_customId', 'customer_name', 'totalAmount', 
            'totalQuantity', 'createdAt', 'orderStatus' 
        ]
        dataf = pd.read_csv(csv_path, usecols=relevant_cols)
        
        # Convert Date Column (Handling timezone like in your file: 2025-04-14 15:21:27+00:00)
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')

    except Exception as e:
        return f"Error processing CSV: {e}"

    # 2. Filtering Logic
    df_filtered = dataf.copy()

    # A) Time Filter (Start Date -> Now)
    if start_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            start_dt = pd.to_datetime(start_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_filtered = df_filtered[df_filtered['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_filtered = df_filtered[df_filtered['createdAt'] <= end_dt]
    
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    # B) Status Filter (Strict Matching)
    if status_filter:
        # Normalize to upper case to match CSV content
        s_filter = status_filter.upper()
        # Check if such status exists in the filtered data to avoid returning empty list silently
        if not df_filtered['orderStatus'].str.contains(s_filter, case=False, na=False).any():
             return f"Warning: No orders found with status '{status_filter}'."
        
        df_filtered = df_filtered[df_filtered['orderStatus'].str.upper() == s_filter]

    if df_filtered.empty:
        return "No orders found matching these criteria."

    # 3. Sort Logic (Standard)
    sort_col = 'totalAmount' if by_type == 'revenue' else 'totalQuantity'
    is_ascending = (sort_order == 'asc')
    
    top_n_df = df_filtered.sort_values(by=sort_col, ascending=is_ascending).head(n)

    # 4. Output Formatting
    output = [f"Found {len(top_n_df)} orders (Sorted by {by_type}, {sort_order}):"]
    for _, row in top_n_df.iterrows():
        # Clean date for display (removing time part for readability)
        d_str = row['createdAt'].strftime('%Y-%m-%d') if pd.notnull(row['createdAt']) else "N/A"
        output.append(
            f"ID: {row['customId_customId']} | Date: {d_str} | "
            f"Customer: {row['customer_name']} | ${row['totalAmount']:.2f}"
        )

    return '\n'.join(output)

@function_tool
def get_top_n_customers(
    user_id: str, 
    n: int, 
    by_type: str, 
    sort_order: str = 'desc', 
    start_date: str = None,
    end_date: str = None,
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
    logger2.info(f"Tool 'get_top_n_customers' called for: {user_id} order: {sort_order},   by_type: {by_type}, start_date: {start_date}, end_date: {end_date}")
    # 1. Path Setup
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return f"Error: File not found."

    try:
        # Load data with Date and Status columns needed for filtering
        relevant_cols = [
            'customer_name', 'totalAmount', 'totalQuantity', 
            'id', 'createdAt', 'orderStatus'
        ]
        dataf = pd.read_csv(csv_path, usecols=relevant_cols)
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')
    except Exception as e:
        return f"Error reading CSV: {e}"

    # 2. Filtering (CRITICAL: Filter BEFORE Aggregation)
    df_filtered = dataf.copy()

    # A) Time Filter
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            df_filtered = df_filtered[df_filtered['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_filtered = df_filtered[df_filtered['createdAt'] <= end_dt]
    
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    if df_filtered.empty:
        return "No customer activity found for the specified period/status."

    # 3. Aggregation (Grouping filtered data)
    # We group by customer name and sum up the remaining rows
    customer_agg = df_filtered.groupby('customer_name').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('totalQuantity', 'sum'),
        orderCount=('id', 'count')
    ).reset_index()

    # Calculate Average Order Value (AOV)
    customer_agg['averageOrderValue'] = customer_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 4. Sort Logic
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return "Invalid 'by_type'. Choose 'revenue', 'totalQuantity', or 'orderCount'."

    is_ascending = (sort_order == 'asc')
    top_n_df = customer_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 5. Formatting
    filter_info = f" (Since {start_date})" if start_date else " (All Time)"

    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    
    output_strings = [
        f"--- {direction_label} {n} Customers by {by_type.capitalize()}{filter_info} ---",
        "Customer Name | Revenue | Qty | Orders | Avg Order Val",
        "-" * 80
    ]

    for _, row in top_n_df.iterrows():
        formatted_row = (
            f"{row['customer_name']} | "
            f"${row['totalRevenue']:,.2f} | "
            f"{int(row['totalQuantity'])} | "
            f"{row['orderCount']} | "
            f"${row['averageOrderValue']:,.2f}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_top_n_products(
    user_id: str, 
    n: int, 
    by_type: str, 
    sort_order: str = 'desc', 
    start_date: str = None,
    end_date: str = None,
    group_by: str = 'variant'
) -> str:
    """
    Gets top N products, categories, or manufacturers based on revenue/quantity.
    
    Parameters:
    - user_id: User ID.
    - n: Number of items to return.
    - by_type: 'revenue', 'totalQuantity', 'orderCount'.
    - sort_order: 'desc' (Best) or 'asc' (Worst).
    - start_date: Filter data from this date (YYYY-MM-DD).
    - end_date: Filter data from start day to this date (YYYY-MM-DD).
    - group_by: Aggregation level. Options: 
        'variant' (Specific Product), 
        'category' (Product Category), 
        'manufacturer' (Brand/Manufacturer).
    """
    logger2.info(f"Tool 'get_top_n_products' called for: {user_id} order: {sort_order},     by_type: {by_type}, start_date: {start_date},  group_by: {group_by}, end_date: {end_date}")
    
    # 1. Path Setup
    csv_path = Path("data") / user_id / "pproducts.csv"
    if not csv_path.exists():
        return f"Error: File not found."

    try:
        dataf = pd.read_csv(csv_path)
        # Ensure date column is datetime
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')
        
        relevant_cols = [
            'product_variant', 'productCategoryName', 'manufacturerName',
            'totalAmount', 'quantity', 'orderId', 'customer_name', 'createdAt'
        ]
        # Check if columns exist (dynamic check because CSVs vary)
        existing_cols = [c for c in relevant_cols if c in dataf.columns]
        df_copy = dataf[existing_cols].copy()
        
    except Exception as e:
        return f"Error reading CSV: {e}"

    # 2. Time Filter
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            df_copy = df_copy[df_copy['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_copy = df_copy[df_copy['createdAt'] <= end_dt]
    
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    if df_copy.empty:
        return "No product data found for this period."


    # 3. Determine Grouping Column
    if group_by == 'variant':
        group_col = 'product_variant'
        label = "Product Variant"
    elif group_by == 'category':
        group_col = 'productCategoryName'
        label = "Category"
    elif group_by == 'manufacturer':
        group_col = 'manufacturerName'
        label = "Manufacturer"
    else:
        return "Invalid 'group_by'. Use 'variant', 'category', or 'manufacturer'."

    # Safety check if column exists
    if group_col not in df_copy.columns:
        return f"Error: Column for {group_by} not found in data."

    # 4. Aggregation
    # Fill N/A in group column to avoid losing data
    df_copy[group_col] = df_copy[group_col].fillna('Unknown')

    product_agg = df_copy.groupby(group_col).agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('quantity', 'sum'),
        orderCount=('orderId', 'nunique'),
        customerCount=('customer_name', 'nunique')
    ).reset_index()

    # Calculate Metrics
    product_agg['avgRevenuePerOrder'] = product_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 5. Sort Logic
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return "Invalid 'by_type'."

    is_ascending = (sort_order == 'asc')
    top_n_df = product_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 6. Formatting
    period_info = f" (Since {start_date})" if start_date else " (All Time)"
    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    
    output_strings = [
        f"--- {direction_label} {n} {label}s by {by_type.capitalize()}{period_info} ---",
        f"{label} | Revenue | Qty | Orders | Customers | Avg Rev/Order",
        "-" * 100
    ]

    for _, row in top_n_df.iterrows():
        formatted_row = (
            f"{row[group_col]} | "
            f"${row['totalRevenue']:,.2f} | "
            f"{int(row['totalQuantity'])} | "
            f"{row['orderCount']} | "
            f"{row['customerCount']} | "
            f"${row['avgRevenuePerOrder']:.2f}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_order_details(user_id: str, order_identifier: str) -> str:
    """
    Gets complete information about an order.
    
    CRITICAL: This tool performs a 'Smart Search'. You can pass:
    1. The internal 'customId' (e.g., 771657) - PREFERRED.
    2. The system UUID (e.g., 'ab16c21f-e705...')
    3. The Shopify Order ID.
    
    Args:
        user_id (str): The user's ID.
        order_identifier (str or int): The ID to search for (can be numeric string or UUID).
        
    Returns:
        str: Formatted order details or "Not Found" message.
    """
    logger2.info(f"Tool 'get_order_details' called for: {user_id} order_identifier: {order_identifier}")
    
    # 1. Path Setup & Data Loading
    base_path = Path("data") / user_id
    orders_path = base_path / "oorders.csv"
    products_path = base_path / "pproducts.csv"
    
    if not orders_path.exists():
        return "Error: Orders file not found."

    try:
        # Load Orders
        df_orders = pd.read_csv(orders_path)
        
        # Prepare columns for searching (ensure types match)
        # Convert customId to string/numeric to handle matching safely
        if 'customId_customId' in df_orders.columns:
             df_orders['customId_customId'] = pd.to_numeric(df_orders['customId_customId'], errors='coerce')
        
    except Exception as e:
        return f"Error reading order file: {e}"

    # 2. SMART SEARCH LOGIC
    # We try to find the row by checking multiple columns
    
    order_row = None
    search_val = str(order_identifier).strip()
    
    # Strategy A: Check if it's a numeric Custom ID
    if search_val.isdigit():
        matches = df_orders[df_orders['customId_customId'] == int(search_val)]
        if not matches.empty:
            order_row = matches.iloc[0]

    # Strategy B: If not found, check UUID ('id')
    if order_row is None:
        matches = df_orders[df_orders['id'] == search_val]
        if not matches.empty:
            order_row = matches.iloc[0]
            
    # Strategy C: Check Shopify Order ID (if column exists)
    if order_row is None and 'shopifyOrderId' in df_orders.columns:
        matches = df_orders[df_orders['shopifyOrderId'].astype(str) == search_val]
        if not matches.empty:
            order_row = matches.iloc[0]

    # Result Check
    if order_row is None:
        return f"--- Order '{order_identifier}' not found in database ---"

    # 3. Retrieve Products (Only if order found)
    internal_order_id = order_row['id']
    order_products = pd.DataFrame()
    
    if products_path.exists():
        try:
            df_products = pd.read_csv(products_path)
            order_products = df_products[df_products['orderId'] == internal_order_id]
        except Exception:
            pass # Continue without products if file read fails

    # 4. Format Output
    output = []
    
    # --- Header & Status ---
    cust_id = order_row.get('customId_customId', 'N/A')
    created_at = pd.to_datetime(order_row.get('createdAt')).strftime('%Y-%m-%d %H:%M') if pd.notnull(order_row.get('createdAt')) else 'N/A'
    
    output.append(f"=== ORDER REPORT: #{cust_id} ===")
    output.append(f"Customer:    {order_row.get('customer_name', 'Unknown')}")
    output.append(f"Created:     {created_at}")
    output.append(f"Type:        {order_row.get('type', 'Direct')}")
    
    # Status Block
    status = order_row.get('orderStatus', 'N/A')
    pay_status = order_row.get('paymentStatus', 'N/A')
    del_status = order_row.get('deliveryStatus', 'N/A')
    output.append(f"Statuses:    [Order: {status}] [Payment: {pay_status}] [Delivery: {del_status}]")
    
    # Payment Terms (Useful Context)
    terms = order_row.get('paymentTermsDuplicate_name', 'N/A')
    due_date = order_row.get('paymentDue', 'N/A')
    output.append(f"Terms:       {terms} (Due: {due_date})")

    # --- Financials ---
    output.append("\n--- FINANCIAL SUMMARY ---")
    subtotal = order_row.get('totalAmountWithoutDelivery', 0)
    discount = order_row.get('totalDiscountValue', 0)
    del_fee = order_row.get('deliveryFee', 0)
    total = order_row.get('totalAmount', 0)
    
    output.append(f"Subtotal:       ${subtotal:,.2f}")
    if discount > 0:
        output.append(f"Discount:      -${discount:,.2f} ({order_row.get('totalOrderDiscountType', '')})")
    output.append(f"Delivery Fee:   ${del_fee:,.2f}")
    output.append(f"GRAND TOTAL:    ${total:,.2f}")

    # --- Line Items ---
    output.append(f"\n--- LINE ITEMS ({len(order_products)}) ---")
    
    if order_products.empty:
        output.append("No products linked to this order.")
    else:
        output.append(f"{'Product Name':<35} | {'Sku':<10} | {'Qty':<5} | {'Total':<10}")
        output.append("-" * 75)
        
        for _, prod in order_products.iterrows():
            name = str(prod.get('product_variant', 'N/A'))
            sku = str(prod.get('sku', ''))
            qty = int(prod.get('quantity', 0))
            line_total = prod.get('totalAmount', 0)
            
            # Truncate long names
            display_name = (name[:32] + '..') if len(name) > 32 else name
            
            output.append(f"{display_name:<35} | {sku:<10} | {qty:<5} | ${line_total:<10.2f}")

    return '\n'.join(output)


@function_tool
def get_product_catalog(user_id: str):
    """
    Returns lists of unique product attributes (variants, names, skus, categories, manufacturers).
    """
    logger2.info(f"Tool 'get_product_catalog' called for: {user_id} ")
    
    try:
        path = Path("data") / user_id / "pproducts.csv"
        df_products = pd.read_csv(path)
        
        # Safe column extraction
        def get_unique(col):
            if col in df_products.columns:
                return sorted(df_products[col].dropna().unique().tolist())
            return []

        catalog = {
            "all_product_variants": get_unique('product_variant'),
            "all_product_names": get_unique('name'),
            "all_skus": get_unique('sku'),
            "all_categories": get_unique('productCategoryName'),
            # --- NEW ADDITION ---
            "all_manufacturers": get_unique('manufacturerName') 
        }
        return catalog

    except Exception as e:
        return f"Error loading catalog: {e}"


def _generate_product_report(df_to_report, report_title):
    """
    Helper function to generate a detailed report with breakdown.
    """
    output_strings = [report_title]

    # 1. Aggregated Sales Info
    total_revenue = df_to_report['totalAmount'].sum()
    total_quantity = df_to_report['quantity'].sum()
    total_orders = df_to_report['orderId'].nunique()
    
    output_strings.append("\n--- SALES SUMMARY (Selected Period) ---")
    output_strings.append(f"Total Revenue:     ${total_revenue:,.2f}")
    output_strings.append(f"Total Units Sold:  {total_quantity:,}")
    output_strings.append(f"Total Orders:      {total_orders:,}")

    # 2. Averages
    if total_orders > 0:
        avg_rev = total_revenue / total_orders
        output_strings.append(f"Avg. Order Value:  ${avg_rev:,.2f}")
    
    if total_quantity > 0:
        avg_price = total_revenue / total_quantity
        output_strings.append(f"Avg. Price/Unit:   ${avg_price:.2f}")

    # 3. Attributes Summary (Show ranges if multiple)
    manufacturers = df_to_report['manufacturerName'].unique()
    categories = df_to_report['productCategoryName'].dropna().unique()
    
    output_strings.append("\n--- ATTRIBUTES ---")
    output_strings.append(f"Manufacturer(s):   {', '.join(manufacturers[:3])}" + ("..." if len(manufacturers)>3 else ""))
    output_strings.append(f"Category(s):       {', '.join(categories[:3])}" + ("..." if len(categories)>3 else ""))

    # --- NEW FEATURE: Top Contributors Breakdown ---
    # If the search result contains multiple products (e.g. searching for a Category),
    # show which specific products are selling the most.
    
    unique_variants = df_to_report['product_variant'].nunique()
    
    if unique_variants > 1:
        output_strings.append(f"\n--- TOP 5 PRODUCTS IN THIS GROUP ({unique_variants} found) ---")
        
        # Group by variant to find leaders
        top_vars = df_to_report.groupby('product_variant').agg(
            rev=('totalAmount', 'sum'),
            qty=('quantity', 'sum')
        ).sort_values(by='rev', ascending=False).head(5)
        
        output_strings.append(f"{'Product Variant':<35} | {'Rev':<10} | {'Qty':<5}")
        output_strings.append("-" * 60)
        
        for name, row in top_vars.iterrows():
            display_name = (name[:32] + '..') if len(str(name)) > 32 else str(name)
            output_strings.append(f"{display_name:<35} | ${row['rev']:<9.0f} | {int(row['qty']):<5}")
            
    return '\n'.join(output_strings)

@function_tool
def get_product_details(
    user_id: str, 
    name: str = None, 
    sku: str = None, 
    category: str = None, 
    manufacturer: str = None,
    start_date: str = None,
    end_date: str = None
) -> str:
    """
    Provides a detailed report for products, categories, or manufacturers.
    Supports time filtering to analyze specific periods.

    Args:
        name (str): Filter by product name (partial match).
        sku (str): Filter by SKU.
        category (str): Filter by category.
        manufacturer (str): Filter by manufacturer (e.g., 'Coca-Cola').
        start_date (str): 'YYYY-MM-DD'.
        end_date (str): 'YYYY-MM-DD'.
    """
    logger2.info(f"Tool 'get_product_details' called for: {user_id} name: {name},     sku: {sku}, category: {category},  manufacturer: {manufacturer} , start_date: {start_date}, end_date: {end_date}")
    
    path = Path("data") / user_id / "pproducts.csv"
    if not path.exists():
        return "Error: Products file not found."
        
    try:
        df_products = pd.read_csv(path)
        df_products['createdAt'] = pd.to_datetime(df_products['createdAt'], errors='coerce')
    except Exception as e:
        return f"Error reading file: {e}"

    # 1. Time Filtering
    period_msg = "All Time"
    if start_date:
        s_dt = pd.to_datetime(start_date).tz_localize('UTC')
        df_products = df_products[df_products['createdAt'] >= s_dt]
        period_msg = f"From {start_date}"
    if end_date:
        e_dt = pd.to_datetime(end_date).tz_localize('UTC')
        df_products = df_products[df_products['createdAt'] <= e_dt]
        period_msg += f" To {end_date}"

    if df_products.empty:
        return f"No sales data found for the period: {period_msg}"

    # 2. Attribute Filtering
    filters = []
    filtered_df = df_products.copy()

    if name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(name, case=False, na=False)]
        filters.append(f"Name='{name}'")
    if sku:
        filtered_df = filtered_df[filtered_df['sku'].str.contains(sku, case=False, na=False)]
        filters.append(f"SKU='{sku}'")
    if category:
        filtered_df = filtered_df[filtered_df['productCategoryName'].str.contains(category, case=False, na=False)]
        filters.append(f"Category='{category}'")
    if manufacturer:
        filtered_df = filtered_df[filtered_df['manufacturerName'].str.contains(manufacturer, case=False, na=False)]
        filters.append(f"Brand='{manufacturer}'")

    if not filters:
        return "Error: Please provide at least one filter (name, sku, category, or manufacturer)."

    if filtered_df.empty:
        return f"No products found matching: {', '.join(filters)} ({period_msg})"

    # 3. Generate Report
    report_title = f"--- REPORT: {', '.join(filters)} [{period_msg}] ---"
    return _generate_product_report(filtered_df, report_title)

@function_tool
def get_customers(user_id: str, search_name: str = None) -> dict:
    """
    Returns a dictionary mapping 'Customer Name' to 'Customer ID'.
    
    IMPROVEMENT: Supports partial search to reduce noise. 
    If duplicates exist (same name, diff ID), appends ID to name.
    """
    logger2.info(f"Tool 'get_customers' called for: {user_id} search_name: {search_name}")
    
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return {"Error": "File not found"}
        
    try:
        df = pd.read_csv(csv_path, usecols=['customerId', 'customer_name', 'id'])
        
        # Filter if search term provided (Save tokens!)
        if search_name:
            # Case-insensitive search
            df = df[df['customer_name'].str.contains(search_name, case=False, na=False)]
            if df.empty:
                return {"Error": f"No customers found matching '{search_name}'"}

        # Calculate Order Count per customer for context
        # This helps the Agent choose the "real" customer if there are duplicates
        counts = df.groupby(['customerId', 'customer_name']).size().reset_index(name='order_count')
        
        customer_dict = {}
        # Track names to handle duplicates
        name_tracker = {} 
        
        for _, row in counts.iterrows():
            name = str(row['customer_name'])
            c_id = str(row['customerId'])
            count = row['order_count']
            
            # Formatting: "John Doe (5 orders)"
            # This gives the Agent crucial context immediately
            display_name = f"{name} ({count} orders)"
            
            # Handle duplicate names (rare but possible)
            if name in name_tracker:
                # If name exists, append partial ID to differentiate
                display_name = f"{name} [{c_id[-4:]}] ({count} orders)"
            
            name_tracker[name] = True
            customer_dict[display_name] = c_id
            
        return customer_dict

    except Exception as e:
        return {"Error": f"Failed to load customers: {e}"}

@function_tool
def get_orders_by_customer(
    user_id: str, 
    customer_id: str, 
    limit: int = 10,
    status_filter: str = None
) -> str:
    """
    Returns a summary and detailed list of orders for a specific customer.
    Sorted by Date (Newest first).
    """
    logger2.info(f"Tool 'get_orders_by_customer' called for: {user_id} customer_id: {customer_id},     limit: {limit}, status_filter: {status_filter}")
    
    csv_path = Path("data") / user_id / "oorders.csv"
    
    try:
        df = pd.read_csv(csv_path)
        
        # 1. Filter by Customer ID
        # Ensure strict string comparison
        customer_orders = df[df['customerId'].astype(str) == str(customer_id)].copy()
        
        if customer_orders.empty:
            return f"No orders found for customer_id: {customer_id}"

        # 2. Apply Filters
        if status_filter:
            customer_orders = customer_orders[
                customer_orders['orderStatus'].str.upper() == status_filter.upper()
            ]

        # 3. Sort (Newest First) & Date Parsing
        customer_orders['createdAt'] = pd.to_datetime(customer_orders['createdAt'], errors='coerce')
        customer_orders = customer_orders.sort_values(by='createdAt', ascending=False)

        # 4. Generate Summary Header
        total_spent = customer_orders['totalAmount'].sum()
        order_count = len(customer_orders)
        
        # Get customer name safely
        c_name = customer_orders['customer_name'].iloc[0] if not customer_orders.empty else "Unknown"
        
        output = [
            f"=== CUSTOMER REPORT: {c_name} ===",
            f"Total Orders: {order_count} | Total Lifetime Value: ${total_spent:,.2f}",
            f"Showing last {min(limit, order_count)} orders:",
            "-" * 80
        ]

        # 5. Select & Format Columns for Table
        # We rename columns to be Agent-friendly (human readable)
        cols_map = {
            'customId_customId': 'Order ID',
            'createdAt': 'Date',
            'orderStatus': 'Status',
            'paymentStatus': 'Payment',
            'totalAmount': 'Total ($)',
            'totalQuantity': 'Qty'
        }
        
        # Slice to limit
        display_df = customer_orders.head(limit).copy()
        
        # Format Date nicely
        display_df['createdAt'] = display_df['createdAt'].dt.strftime('%Y-%m-%d')
        
        # Select and Rename
        table_data = display_df[cols_map.keys()].rename(columns=cols_map)
        
        # Convert to Markdown
        output.append(table_data.to_markdown(index=False, floatfmt=".2f"))
        
        return '\n'.join(output)

    except Exception as e:
        return f"Error processing orders: {e}"

import os
import numpy as np
import faiss
from openai import OpenAI
from agents import function_tool  # The specific decorator from the SDK

# Initialize standard OpenAI client for embeddings
client = OpenAI()

class SimpleVectorStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.is_initialized = False

    def _cosine_similarity(self, vec_a, matrix_b):
        """Calculates cosine similarity between vector A and all vectors in Matrix B."""
        # Normalize vector A
        norm_a = np.linalg.norm(vec_a)
        if norm_a == 0: return np.zeros(len(matrix_b))
        vec_a_norm = vec_a / norm_a

        # Normalize Matrix B (all chunks)
        norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
        matrix_b_norm = np.divide(matrix_b, norm_b, where=norm_b!=0)

        # Dot product
        return np.dot(matrix_b_norm, vec_a_norm)

    def load_and_index(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print("Loading FAQ file...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split by double newline (paragraphs)
        self.chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        if not self.chunks:
            print("Warning: No chunks found in file.")
            return

        print(f"Embedding {len(self.chunks)} entries...")
        
        # Get embeddings in one batch for speed
        response = client.embeddings.create(
            input=self.chunks,
            model="text-embedding-3-small"
        )
        
        # Store as numpy array
        self.embeddings = np.array([d.embedding for d in response.data]).astype('float32')
        self.is_initialized = True
        print("Indexing complete.")

    def search(self, query: str, top_k: int = 2):
        if not self.is_initialized:
            self.load_and_index()
            
        # Embed query
        query_embedding = client.embeddings.create(
            input=[query], 
            model="text-embedding-3-small"
        ).data[0].embedding
        
        query_vec = np.array(query_embedding).astype('float32')
        
        # Calculate similarities
        scores = self._cosine_similarity(query_vec, self.embeddings)
        
        # Get top_k indices (sorted high to low)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Optional: Filter low relevance (e.g., score < 0.3)
            if scores[idx] > 0.3:
                results.append(self.chunks[idx])
                
        return "\n---\n".join(results) if results else "No relevant info found."

FAQ_FILE_PATH = Path("AI/group_customer_analyze/Agents_rules/QA_SD2.txt")
faq_engine = SimpleVectorStore(FAQ_FILE_PATH)

@function_tool
def look_up_faq(question: str) -> str:
    """
    Searches the FAQ (Frequently Asked Questions) text file 
    to find answers to user questions about policies or features.

    Args:
        question: The specific question or topic the user is asking about.
    """
    logger2.info(f"Tool 'look_up_faq' called for: {question}")
    try:
        return faq_engine.search(question)
            
    except Exception as e:
        return f"Error retrieving FAQ: {str(e)}"

async def create_Ask_ai_many_c_agent(USER_ID:str) -> Tuple[Agent, AdvancedSQLiteSession]:
    """Initializes a new Inventory agent and session."""

    try:
        from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_Ask_ai_many

        session_db = AdvancedSQLiteSession(
            session_id=USER_ID,
            create_tables=True,
            db_path=f"data/{USER_ID}/conversations.db",
            logger=logger2
        )
    except Exception as e:
        logger2.error(f"error creating session: {e}")

    try:
        instructions = await prompt_agent_Ask_ai_many(USER_ID)
        agent = Agent(
            name="Warehouse_Inventory_Assistant",
            instructions=instructions,
            model=llm_model,
            tools=[
            General_statistics_tool,
            get_top_n_customers,
            get_top_n_orders,
            get_top_n_products,
            get_order_details,
            get_product_catalog,
            get_product_details,
            get_customers,
            get_orders_by_customer,
            look_up_faq]

        )
        session = session_db

    except Exception as e:
        logger2.error(f"error creating agent: {e}")
        agent, session = None
    return agent, session