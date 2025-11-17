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

def _create_advice_tool(user_uuid: str):
    """ More detailed information about each customer"""
    try:
       path_for_customers_Overall_report = os.path.join("data", user_uuid, "overall_report.txt")
       
       with open(path_for_customers_Overall_report, "r", encoding="utf-8") as f:
           advice_text = f.read()
    except FileNotFoundError:
        logger2.error("overall_report file not found")
        return None
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(advice_text)
    
    # Create vector store
    emb = OpenAIEmbeddings()
    index = FAISS.from_texts(chunks, emb)
    
    # Define advice retrieval tool
    @tool("AdviceTool")
    def get_data(query: str) -> str:
        '''Use this tool to fetch detailed customer statistics and insights from the overall report. Input should be a specific topic or question.'''
        docs = index.similarity_search(query, k=12)
        return "\n".join(d.page_content for d in docs)
        
    return get_data


async def Ask_ai_many_customers(prompt: str, user_uuid: str):
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        file_path_product = os.path.join('data', user_uuid, 'oorders.csv')
        file_path_orders = os.path.join('data', user_uuid, 'pproducts.csv')
        for encoding in encodings:
            try:
                df1 = pd.read_csv(file_path_product, encoding=encoding, low_memory=False)
                df2 = pd.read_csv(file_path_orders, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                logger2.warning(f"Failed decoding attempt with encoding: {encoding}")

        llm = ChatOpenAI(model='gpt-4.1-mini') #model='o3-mini'  gpt-4.1-mini

        # Create advice tool
        advice_tool = _create_advice_tool(user_uuid)

        # Create agent with extra tool
        agent = create_pandas_dataframe_agent(
            llm,
            [df1, df2],
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True,
            number_of_head_rows=5,
            max_iterations=5,
            extra_tools=[advice_tool] if advice_tool else []  # Add tool if available
        )
         

        try:
            full_report_path = os.path.join('data', user_uuid, 'full_report.md')
            with open(full_report_path, "r") as file:
                full_report = file.read()
        except Exception as e:
            full_report = 'No data given'
            logger2.warning(f"Can not read additional_info.md due to {e} ")
        
        try:
            promo_rules_path = os.path.join('AI', 'group_customer_analyze', 'promo_rules.txt')
            with open(promo_rules_path, "r") as file:
                recommendations = file.read()
        except Exception as e:
            recommendations = 'No data given'
            logger2.warning(f"Can not read additional_info.md due to {e} ")


        formatted_prompt = f"""
        You are an AI assistant which answers users' questions about the data you have 
        providing business insights based on two related datasets and files:  
        - **df1** (Orders) contains critical order-related data or can be user activities data.  
        - **df2** (Products) contains details about products within each order or can be user tasks data.  

        Calculated data for all customers in a joint report: {full_report}

        some additional data that you can use to make recommendations to the customer: {recommendations}
        
        **Important Rules to Follow:**  
        - **Unique Values:** When answering questions about orders or products, always consider unique values.  
        - **Neutral Wording:** Do not mention "df1" or "df2" in your response. Instead, phrase answers as "According to the user's data."  
        - **No Column/File References:** Do not refer to specific file names or column names—focus on insights and conclusions.  
        - **Well-Structured Markdown Formatting:** Ensure responses are clear and organized using appropriate Markdown formatting.  
        - **No Code or Visualizations:** Do not include Python code or suggest data visualizations in your answers.  
        - If you are sure that the question has nothing to do with the data, answer - "Your question is not related to the analysis of your data, please ask another question."

        
        **Example Response:**  
        **Sales Trends**  
        - **Peak sales month:** **2023-04** (**$1,474.24**)  
        According to the user's data, overall sales reflect a steady momentum underpinned by a balanced mix of confirmed transactions and
        those in earlier stages. Completed orders with confirmed payments form a solid base, suggesting that key customer
        segments are both engaged and reliable. Pending transactions indicate opportunities for growth, while recurring
        product lines highlight sustained customer interest. The pricing strategy, with consistent margins between wholesale
        and retail values, supports profitability and long-term stability.  

        -  **Use AdviceTool**: It may already contain the answer to the question. or if the question concerns more detailed data on customers or products. 
        This tool searches the txt file for data that matches the question.

        **Question:**  

        {prompt}"""

        logger2.info("\n===== Metadata =====")
        MODEL_RATES = {
            "gpt-4.1":      {"prompt": 2.00,   "completion": 8.00},
            "gpt-4.1-mini": {"prompt": 0.40,   "completion": 1.60},
            "gpt-4.1-nano": {"prompt": 0.10,   "completion": 0.40},
            "gpt-4o":       {"prompt": 2.50,   "completion": 10.00},
            "gpt-4o-mini":  {"prompt": 0.15,   "completion": 0.60},
            "gpt-4.5":      {"prompt": 75.00,  "completion": 150.00},
            "o3":           {"prompt": 2.00,   "completion": 8.00},
            "o3-mini":      {"prompt": 1.10,   "completion": 4.40},
            "o4-mini":      {"prompt": 1.10,   "completion": 4.40},
            "gpt-5":        {"prompt": 1.25,   "completion": 10.00},
        }
        
        def calculate_cost(model_name, prompt_tokens, completion_tokens):
            rates = MODEL_RATES.get(model_name)
            if not rates:
                raise ValueError(f"Unknown model: {model_name}")
            # assume rates are $ per 1 000 000 tokens:
            cost_per_prompt_token     = rates["prompt"]     / 1_000_000
            cost_per_completion_token = rates["completion"] / 1_000_000
        
            input_cost  = prompt_tokens    * cost_per_prompt_token
            output_cost = completion_tokens * cost_per_completion_token
            total_cost  = input_cost + output_cost
            return total_cost, input_cost, output_cost
        
        with get_openai_callback() as cb:
            agent.agent.stream_runnable = False
            start_time = time.time()
            result = agent.invoke({"input": formatted_prompt})
            execution_time = time.time() - start_time
        
            in_toks, out_toks = cb.prompt_tokens, cb.completion_tokens
            cost, in_cost, out_cost = calculate_cost(llm.model_name, in_toks, out_toks)
        
            logger2.info("Agent for func:  Ask_ai_many_customers")
            logger2.info(f"Input Cost:  ${in_cost:.6f}")
            logger2.info(f"Output Cost: ${out_cost:.6f}")
            logger2.info(f"Total Cost:  ${cost:.6f}")
        
            result['metadata'] = {
                'total_tokens': in_toks+out_toks,
                'prompt_tokens': in_toks,
                'completion_tokens': out_toks,
                'execution_time': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }

        
        for k, v in result['metadata'].items():
            logger2.info(f"{k.replace('_', ' ').title()}: {v}")

        return {"output": result.get('output'), "cost": cost}

    except Exception as e:
        logger2.error(f"Error in AI processing: {str(e)}")
        return {"error": 'invalid_uuid', "cost": 0}
    
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

from typing import List, AsyncGenerator, Tuple, Any
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_create_full_report, prompt_agent_create_sectioned, prompt_for_state_agent
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm_model = OpenAIResponsesModel(model='gpt-4.1-mini', openai_client=AsyncOpenAI()) 


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
def get_top_n_orders(user_id:str, n : int, by_type:str) -> str:
    """
    Gets the top N orders from the DataFrame based on revenue or quantity.

    Args:
        user_id str: Use given user id.
        n (int): The number of top orders to return.
        by_type (str): The criteria to sort by. 
                       Must be 'revenue' or 'totalQuantity'.

    Returns:
        str: A formatted string of the top N orders.
    """
    
    # Select and copy the relevant columns
    logger2.info(f"Tool 'get_top_n_orders' called called for: {user_id}")
    dataf = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
    try:
        relevant_cols = ['customId_customId', 'customer_name', 'totalAmount', 'totalQuantity']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing expected column in DataFrame: {e}. Available columns: {dataf.columns.tolist()}"

    # Determine the sort column
    if by_type == 'revenue':
        sort_column = 'totalAmount'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    else:
        return "Invalid 'by_type' parameter. Please choose 'revenue' or 'totalQuantity'."

    # Sort the DataFrame and get the top N
    top_n_df = df_copy.sort_values(by=sort_column, ascending=False).head(n)

    # Fill potential NaN values in text fields for clean printing
    top_n_df_filled = top_n_df.fillna({'customId_customId': 'N/A', 'customer_name': 'N/A'})

    # Format the output string
    output_strings = []
    output_strings.append(f"--- Top {n} Orders by {by_type.capitalize()} ---")
    output_strings.append("\nCustom ID - Customer Name - Revenue - Total Quantity")
    output_strings.append("-" * 60) # Separator line

    for index, row in top_n_df_filled.iterrows():
        # Format totalAmount as currency and totalQuantity as an integer
        formatted_row = (
            f"{row['customId_customId']} - "
            f"{row['customer_name']} - "
            f"${row['totalAmount']:.2f} - "
            f"{int(row['totalQuantity'])}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_top_n_customers(user_id:str, n : int, by_type:str)-> str:
    """
    Gets the top N customers from the DataFrame based on aggregated 
    revenue or quantity.

    Args:
        user_id str: Use given user id.
        n (int): The number of top customers to return.
        by_type (str): The criteria to sort by. 
                       Must be 'revenue', 'totalQuantity', or 'orderCount'.

    Returns:
        str: A formatted string of the top N customers.
    """
    
    # Select and copy the relevant columns
    logger2.info(f"Tool 'get_top_n_customers' called called for: {user_id}")
    dataf = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
    try:
        relevant_cols = ['customer_name', 'totalAmount', 'totalQuantity', 'id']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing expected column in DataFrame: {e}. Available columns: {dataf.columns.tolist()}"

    # Aggregate data by customer
    customer_agg = df_copy.groupby('customer_name').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('totalQuantity', 'sum'),
        orderCount=('id', 'count')
    ).reset_index()

    # Calculate average order value
    customer_agg['averageOrderValue'] = customer_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # Determine the sort column
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return ("Invalid 'by_type' parameter. Please choose 'revenue', "
                "'totalQuantity', or 'orderCount'.")

    # Sort the DataFrame and get the top N
    top_n_df = customer_agg.sort_values(by=sort_column, ascending=False).head(n)

    # Fill potential NaN values in text fields for clean printing
    top_n_df_filled = top_n_df.fillna({'customer_name': 'N/A'})

    # Format the output string
    output_strings = []
    output_strings.append(f"--- Top {n} Customers by {by_type.capitalize()} ---")
    output_strings.append("\nCustomer Name - Total Revenue - Total Quantity - Order Count - Avg. Order Value")
    output_strings.append("-" * 80) # Separator line

    for index, row in top_n_df_filled.iterrows():
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
def get_top_n_products(user_id:str, n : int, by_type:str)-> str:
    """
    Gets the top N products from the DataFrame based on aggregated 
    revenue or quantity.

    Args:
        user_id str: Use given user id.
        n (int): The number of top customers to return.
        by_type (str): The criteria to sort by. 
                       Must be 'revenue', 'totalQuantity', or 'orderCount'.

    Returns:
        str: A formatted string of the top N customers.
    """
    
    # Select and copy the relevant columns
    logger2.info(f"Tool 'get_top_n_products' called called for: {user_id}")
    dataf = pd.read_csv(os.path.join("data", user_id, "pproducts.csv"))
    # Select and copy the relevant columns
    try:
        relevant_cols = ['product_variant', 'totalAmount', 'quantity', 'orderId', 'customer_name']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing expected column in DataFrame: {e}. Available columns: {dataf.columns.tolist()}"

    # Aggregate data by product_variant
    product_agg = df_copy.groupby('product_variant').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('quantity', 'sum'),
        orderCount=('orderId', 'nunique'), # Count distinct orders
        customerCount=('customer_name', 'nunique') # Count distinct customers
    ).reset_index()

    # Calculate average revenue per order
    product_agg['avgRevenuePerOrder'] = product_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )
    
    # Calculate average quantity per order
    product_agg['avgQuantityPerOrder'] = product_agg.apply(
        lambda row: row['totalQuantity'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )


    # Determine the sort column
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return ("Invalid 'by_type' parameter. Please choose 'revenue', "
                "'totalQuantity', or 'orderCount'.")

    # Sort the DataFrame and get the top N
    top_n_df = product_agg.sort_values(by=sort_column, ascending=False).head(n)

    # Fill potential NaN values in text fields for clean printing
    top_n_df_filled = top_n_df.fillna({'product_variant': 'N/A'})

    # Format the output string
    output_strings = []
    output_strings.append(f"--- Top {n} Products by {by_type.capitalize()} ---")
    output_strings.append("\nProduct - Total Revenue - Total Quantity - Order Count - Unique Customers - Avg. Qty/Order")
    output_strings.append("-" * 100) # Separator line

    for index, row in top_n_df_filled.iterrows():
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
    logger2.info(f"Tool 'get_product_details' called called for: {user_id}")
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
def get_orders_by_customer(user_id:str, customer_id:str)-> str:
    """
    Returns a DataFrame in md format with specific order details for a given customer_id - use get_customers tool before.
    """
    logger2.info(f"Tool 'get_orders_by_customer' called called for: {user_id}")
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
            get_orders_by_customer]

        )
        session = session_db
        print("👍 New create_Ask_ai_many_c_agent and Session are ready.")
    except Exception as e:
        logger2.error(f"error creating agent: {e}")
        agent, session = None
    return agent, session