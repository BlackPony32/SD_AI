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
def get_top_n_orders(user_id: str, n: int, by_type: str, sort_order: str = 'desc') -> str:
    """
    Gets the top (or bottom) N orders based on revenue or quantity.
    """
    logger2.info(f"Tool 'get_top_n_orders' called for: {user_id} order: {sort_order}")
    
    # 1. Path Setup
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return f"Error: File not found at {csv_path}"

    try:
        dataf = pd.read_csv(csv_path)
        relevant_cols = ['customId_customId', 'customer_name', 'totalAmount', 'totalQuantity']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing column: {e}. Available: {dataf.columns.tolist()}"
    except Exception as e:
        return f"Error reading CSV: {e}"

    # 2. Determine sort column
    if by_type == 'revenue':
        sort_column = 'totalAmount'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    else:
        return "Invalid 'by_type'. Choose 'revenue' or 'totalQuantity'."

    # 3. Sort Logic
    is_ascending = True if sort_order == 'asc' else False
    top_n_df = df_copy.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 4. Formatting
    top_n_df_filled = top_n_df.fillna({'customId_customId': 'N/A', 'customer_name': 'N/A'})
    
    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    output_strings = [
        f"--- {direction_label} {n} Orders by {by_type.capitalize()} ---",
        "\nCustom ID - Customer Name - Revenue - Total Quantity",
        "-" * 60
    ]

    for _, row in top_n_df_filled.iterrows():
        formatted_row = (
            f"{row['customId_customId']} - "
            f"{row['customer_name']} - "
            f"${row['totalAmount']:.2f} - "
            f"{int(row['totalQuantity'])}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_top_n_customers(user_id: str, n: int, by_type: str, sort_order: str = 'desc') -> str:
    """
    Gets the top (or bottom) N customers based on aggregated revenue or quantity.
    """
    logger2.info(f"Tool 'get_top_n_customers' called for: {user_id} order: {sort_order}")
    
    # 1. Path Setup
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return f"Error: File not found at {csv_path}"

    try:
        dataf = pd.read_csv(csv_path)
        relevant_cols = ['customer_name', 'totalAmount', 'totalQuantity', 'id']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing column: {e}. Available: {dataf.columns.tolist()}"

    # 2. Aggregation
    customer_agg = df_copy.groupby('customer_name').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('totalQuantity', 'sum'),
        orderCount=('id', 'count')
    ).reset_index()

    customer_agg['averageOrderValue'] = customer_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 3. Determine sort column
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return "Invalid 'by_type'. Choose 'revenue', 'totalQuantity', or 'orderCount'."

    # 4. Sort Logic
    is_ascending = True if sort_order == 'asc' else False
    top_n_df = customer_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)
    
    # 5. Formatting
    top_n_df_filled = top_n_df.fillna({'customer_name': 'N/A'})
    
    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    output_strings = [
        f"--- {direction_label} {n} Customers by {by_type.capitalize()} ---",
        "\nCustomer Name - Total Revenue - Total Quantity - Order Count - Avg. Order Value",
        "-" * 80
    ]

    for _, row in top_n_df_filled.iterrows():
        formatted_row = (
            f"{row['customer_name']} - "
            f"${row['totalRevenue']:,.2f} - "
            f"{int(row['totalQuantity'])} - "
            f"{row['orderCount']} - "
            f"${row['averageOrderValue']:,.2f}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_top_n_products(user_id: str, n: int, by_type: str, sort_order: str = 'desc') -> str:
    """
    Gets the top (or bottom) N products based on aggregated revenue or quantity.
    """
    logger2.info(f"Tool 'get_top_n_products' called for: {user_id} order: {sort_order}")
    
    # 1. Path Setup
    csv_path = Path("data") / user_id / "pproducts.csv"
    if not csv_path.exists():
        return f"Error: File not found at {csv_path}"

    try:
        dataf = pd.read_csv(csv_path)
        relevant_cols = ['product_variant', 'totalAmount', 'quantity', 'orderId', 'customer_name']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing column: {e}. Available: {dataf.columns.tolist()}"

    # 2. Aggregation
    product_agg = df_copy.groupby('product_variant').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('quantity', 'sum'),
        orderCount=('orderId', 'nunique'),
        customerCount=('customer_name', 'nunique')
    ).reset_index()

    product_agg['avgRevenuePerOrder'] = product_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )
    product_agg['avgQuantityPerOrder'] = product_agg.apply(
        lambda row: row['totalQuantity'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 3. Determine sort column
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return "Invalid 'by_type'. Choose 'revenue', 'totalQuantity', or 'orderCount'."

    # 4. Sort Logic
    is_ascending = True if sort_order == 'asc' else False
    top_n_df = product_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 5. Formatting
    top_n_df_filled = top_n_df.fillna({'product_variant': 'N/A'})

    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    output_strings = [
        f"--- {direction_label} {n} Products by {by_type.capitalize()} ---",
        "\nProduct - Total Revenue - Total Quantity - Order Count - Unique Customers - Avg. Qty/Order",
        "-" * 100
    ]

    for _, row in top_n_df_filled.iterrows():
        formatted_row = (
            f"{row['product_variant']} - "
            f"${row['totalRevenue']:,.2f} - "
            f"{int(row['totalQuantity'])} - "
            f"{row['orderCount']} - "
            f"{row['customerCount']} - "
            f"{row['avgQuantityPerOrder']:.1f}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_order_details(order_custom_id:int, user_id:str):
    """
    Gets complete information about an order by its customId_customId,
    using data from both DataFrames.

    Args:
        order_custom_id (int or str): The unique 'customId_customId' of the order.
        user_id (str): Use given user id.

    Returns:
        str: A formatted string with the complete order information.
    """
    
    output_strings = []
    # Select and copy the relevant columns
    try:
        logger2.info(f"Tool 'get_order_details' called called for: {user_id}")
        df_orders = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
        df_products = pd.read_csv(os.path.join("data", user_id, "pproducts.csv"))
        df_orders['customId_customId'] = pd.to_numeric(df_orders['customId_customId'], errors='coerce')

    except Exception as e:
        print(f"Error loading files: {e}")
    if 'customId_customId' not in df_orders.columns:
        print("Error: The 'customId_customId' column was not found in oorders.csv.")
        df_orders = None # Set to None to prevent errors
    # 1. Find the order in df_orders
    try:
        # Convert ID to a numeric type for comparison
        numeric_custom_id = int(order_custom_id)
    except ValueError:
        return f"Error: 'order_custom_id' must be numeric. Received: {order_custom_id}"

    # Filter df_orders
    order_details = df_orders[df_orders['customId_customId'] == numeric_custom_id]
    
    if order_details.empty:
        return f"--- Order with Custom ID: {order_custom_id} not found ---"
        
    # Get the first (and likely only) row
    order_row = order_details.iloc[0]
    
    # Get the internal ID to link with products
    internal_order_id = order_row['id']

    # --- 2. Format Main Information ---
    output_strings.append(f"--- Order Details (Custom ID: {numeric_custom_id}) ---")
    output_strings.append(f"Customer:       {order_row.get('customer_name', 'N/A')}")
    
    try:
        # Try to get just the date
        order_date = pd.to_datetime(order_row.get('createdAt')).strftime('%Y-%m-%d')
    except:
        order_date = order_row.get('createdAt', 'N/A')
        
    output_strings.append(f"Date:           {order_date}")
    output_strings.append(f"Status:         {order_row.get('orderStatus', 'N/A')}")
    output_strings.append(f"Payment:        {order_row.get('paymentStatus', 'N/A')}")
    output_strings.append(f"Delivery:       {order_row.get('deliveryStatus', 'N/A')}")
    
    # --- 3. Financial Information ---
    output_strings.append("\n--- Financials ---")
    output_strings.append(f"Subtotal (excl. delivery): ${order_row.get('totalAmountWithoutDelivery', 0):.2f}")
    output_strings.append(f"Discount:              ${order_row.get('totalDiscountValue', 0):.2f}")
    output_strings.append(f"Delivery Fee:          ${order_row.get('deliveryFee', 0):.2f}")
    output_strings.append(f"GRAND TOTAL:           ${order_row.get('totalAmount', 0):.2f}")
    output_strings.append(f"Total Units:           {int(order_row.get('totalQuantity', 0))}")

    # --- 4. Find products in df_products ---
    order_products = df_products[df_products['orderId'] == internal_order_id]
    
    output_strings.append(f"\n--- Products in Order ({len(order_products)}) ---")
    
    if order_products.empty:
        output_strings.append("Products for this order were not found.")
    else:
        # Products table header
        output_strings.append(f"{'Product':<40} | {'Qty':<5} | {'Price':<10} | {'Total':<10}")
        output_strings.append("-" * 70)
        
        for index, prod_row in order_products.iterrows():
            product_name = prod_row.get('product_variant', 'N/A')
            # Truncate the name if it's too long
            if len(product_name) > 38:
                product_name = product_name[:35] + "..."
            
            quantity = int(prod_row.get('quantity', 0))
            price = prod_row.get('price', 0)
            total_line_amount = prod_row.get('totalAmount', 0)
            
            output_strings.append(f"{product_name:<40} | {quantity:<5} | ${price:<9.2f} | ${total_line_amount:<9.2f}")
            
    return '\n'.join(output_strings)


# --- Function 1: Get Product List ---
@function_tool
def get_product_catalog(user_id:str)-> str:
    """
    Parses the products DataFrame and returns a dictionary
    of unique product attributes (variants, names, SKUs).
    
    Args:
        user_id (str): Use given user id.

    Returns:
        dict: A dictionary containing lists of unique
              product_variants, names, and skus.
              Returns None if the DataFrame is invalid.
    """
    try:
        logger2.info(f"Tool 'get_product_catalog' called called for: {user_id}")
        df_products = pd.read_csv(os.path.join("data", user_id, "pproducts.csv"))
    
        # Clean the 'combined﻿id' column name if it exists
        if 'combined\ufeffid' in df_products.columns:
            df_products.rename(columns={'combined\ufeffid': 'combined_id'}, inplace=True)

    except Exception as e:
        print(f"Error loading pproducts.csv: {e}")

    if df_products is None:
        return None
        
    try:
        catalog = {
            "all_product_variants": df_products['product_variant'].unique().tolist(),
            "all_product_names": df_products['name'].unique().tolist(),
            "all_skus": df_products['sku'].unique().tolist(),
            # --- NEWLY ADDED ---
            "all_categories": df_products['productCategoryName'].dropna().unique().tolist()
        }
        
        # Sort lists for easier reading
        for key in catalog:
            catalog[key].sort()
            
        return catalog
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        return None
    except Exception as e:
        print(f"An error occurred in get_product_catalog: {e}")
        return None

# --- Function 2: Get Product Details ---
def _generate_product_report(df_to_report, report_title):
    """
    Helper function to generate a detailed report from a DataFrame.
    """
    output_strings = [report_title]

    # --- 1. Static Info ---
    output_strings.append("\n--- Matched Attributes ---")
    output_strings.append(f"Base Names:        {', '.join(df_to_report['name'].unique())}")
    output_strings.append(f"SKUs:              {', '.join(df_to_report['sku'].unique())}")
    output_strings.append(f"Manufacturers:     {', '.join(df_to_report['manufacturerName'].unique())}")
    
    valid_categories = df_to_report['productCategoryName'].dropna().unique()
    if len(valid_categories) > 0:
        output_strings.append(f"Categories:        {', '.join(valid_categories)}")
    else:
        output_strings.append("Categories:        N/A")
        
    items_per_case_vals = df_to_report['itemsPerCase'].unique()
    output_strings.append(f"Items per Case:    {', '.join(map(str, items_per_case_vals))}")

    # --- 2. Aggregated Sales Info ---
    output_strings.append("\n--- Lifetime Sales Summary (for this group) ---")
    total_revenue = df_to_report['totalAmount'].sum()
    total_quantity = df_to_report['quantity'].sum()
    total_orders = df_to_report['orderId'].nunique()
    total_customers = df_to_report['customer_name'].nunique()
    
    output_strings.append(f"Total Revenue:     ${total_revenue:,.2f}")
    output_strings.append(f"Total Units Sold:  {total_quantity:,}")
    output_strings.append(f"Total Orders:      {total_orders:,}")
    output_strings.append(f"Unique Customers:  {total_customers:,}")

    if total_quantity > 0:
        avg_price_per_unit = total_revenue / total_quantity
        output_strings.append(f"Avg. Price / Unit: ${avg_price_per_unit:.2f}")
    
    if total_orders > 0:
        avg_revenue_per_order = total_revenue / total_orders
        output_strings.append(f"Avg. Revenue / Order: ${avg_revenue_per_order:,.2f}")

    # --- 3. Top Customers for this group ---
    output_strings.append(f"\n--- Top 5 Customers (for this group) ---")
    customer_sales = df_to_report.groupby('customer_name')['totalAmount'].sum().sort_values(ascending=False).head(5)
    
    if customer_sales.empty:
        output_strings.append("No customer sales data available.")
    else:
        for customer, revenue in customer_sales.items():
            output_strings.append(f"{customer:<30} | ${revenue:,.2f}")

    # --- 4. Order Dates ---
    try:
        first_order = pd.to_datetime(df_to_report['createdAt']).min().strftime('%Y-%m-%d')
        last_order = pd.to_datetime(df_to_report['createdAt']).max().strftime('%Y-%m-%d')
        output_strings.append("\n--- Order History (for this group) ---")
        output_strings.append(f"First Order Date:  {first_order}")
        output_strings.append(f"Last Order Date:   {last_order}")
    except Exception:
        output_strings.append("\n--- Order History (for this group) ---")
        output_strings.append("Could not parse order dates.")

    return '\n'.join(output_strings)

@function_tool
def get_product_details(user_id:str, name:str=None, sku:str=None, category:str=None)-> str:
    """
    Call this tool after get_product_catalog, using the validated identifiers.
    Purpose: To provide the user with a detailed report based on their specific query.

    Example Scenarios for Step 2:
    Case 1 (Name Only): User asks for "all Mars products."
    Call: get_product_details(user_id, name='Mars')
    Case 2 (SKU Only): User asks about "SKU 12345."
    Call: get_product_details(user_id, sku='12345')
    Case 3 (Category Only): User asks for "everything in the Sodas category."
    Call: get_product_details(user_id, category='Sodas')
    Case 4 (Name + SKU): User asks for "Coke Original."
    Call: get_product_details(user_id, name='Coca Cola', sku='Original')
    Case 5 (Name + Category): User asks for "Coke products in the Sodas category."
    Call: get_product_details(user_id, name='Coca Cola', category='Sodas')

    Args:
        user_id (str): Use given user id.
        name (str, optional): A string to match in the 'name' column.
        sku (str, optional): A string to match in the 'sku' column.
        category (str, optional): A string to match in the 
                                  'productCategoryName' column.
    Returns:
        str: A formatted string with full product info.
    """
    logger2.info(f"Tool 'get_product_details' called called for: {user_id} and name = {name} sku = {sku} category = {category}")
    df_products = pd.read_csv(os.path.join("data", user_id, "pproducts.csv"))
    if df_products is None:
        return "Error: The products DataFrame is None. Please check file loading."
        
    # --- UPDATED ---
    if name is None and sku is None and category is None:
        return "Error: Please provide at least one filter (name, sku, or category)."

    filtered_df = df_products.copy()
    filters_applied = []

    # Apply filters
    if name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(name, case=False, na=False)]
        filters_applied.append(f"Name containing '{name}'")
        
    if sku:
        filtered_df = filtered_df[filtered_df['sku'].str.contains(sku, case=False, na=False)]
        filters_applied.append(f"SKU containing '{sku}'")

    # --- UPDATED ---
    if category:
        # Use str.contains for partial, case-insensitive matching
        # na=False ensures that rows with NaN categories are skipped
        filtered_df = filtered_df[filtered_df['productCategoryName'].str.contains(category, case=False, na=False)]
        filters_applied.append(f"Category containing '{category}'")
    # --- END UPDATE ---

    # Check for results
    if filtered_df.empty:
        return f"--- No products found matching: {', '.join(filters_applied)} ---"

    # --- Generate Report ---
    report_title = f"--- Product Report For: {', '.join(filters_applied)} ---"
    report_string = _generate_product_report(filtered_df, report_title)

    # --- "See Also" Logic ---
    variants_found = filtered_df['product_variant'].unique()
    
    if len(variants_found) > 1:
        report_string += "\n\n--- Specific Variants Found ---"
        report_string += "\nYour search returned multiple variants. You may be interested in a detailed report on one of these:"
        
        limit = 10
        for variant in variants_found[:limit]:
            report_string += f"\n- {variant}"
            
        if len(variants_found) > limit:
            report_string += f"\n...and {len(variants_found) - limit} more."

    return report_string


@function_tool
def get_customers(user_id:str)-> dict[str, Any]:
    """
    Returns a dictionary of unique customers (customerId: customer_name).
    """
    logger2.info(f"Tool 'get_customers' called called for: {user_id}")
    dataframe = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
    # Select relevant columns and drop duplicates
    customer_df = dataframe[['customerId', 'customer_name']].drop_duplicates()
    
    # Convert to dictionary
    customer_dict = pd.Series(customer_df.customer_name.values, index=customer_df.customerId).to_dict()
    
    return customer_dict

@function_tool
def get_orders_by_customer_id(user_id:str, customer_id:str)-> str:
    """
    Returns a DataFrame in md format with specific order details for a given customer_id - use get_customers tool before.
    """
    logger2.info(f"Tool 'get_orders_by_customer' called called for: {user_id} and {customer_id}")
    # Filter DataFrame by customer_id
    dataframe = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
    customer_orders_df = dataframe[dataframe['customerId'] == customer_id].copy()
    
    # Define columns requested
    requested_columns = [
        'id',
        'customId_customId',
        'totalOrderDiscountAmount',
        'totalOrderDiscountType',
        'createdAt',
        'orderStatus',
        'deliveryStatus',
        'paymentStatus',
        'totalAmount',
        'totalQuantity'
    ]
    
    # Select and rename columns
    customer_orders_summary = customer_orders_df[requested_columns]
    
    # Rename columns as per user request
    column_rename_map = {
        'id': 'order_id',
        'customId_customId': 'custom_order_id'
    }
    customer_orders_summary = customer_orders_summary.rename(columns=column_rename_map)
    customer_orders_summary_md = customer_orders_summary.to_markdown(index=False)
    return customer_orders_summary_md


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
            get_orders_by_customer_id,
            look_up_faq]

        )
        session = session_db

    except Exception as e:
        logger2.error(f"error creating agent: {e}")
        agent, session = None
    return agent, session