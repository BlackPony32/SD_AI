import os
import asyncio
import httpx
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

import numpy as np

async def download_file(client, url: str, entity: str):
    """Download individual file and return entity-content pair"""
    response = await client.get(url)
    if response.status_code == 200:
        return (entity, response.content)
    raise Exception(f"Failed to download {entity} file")

async def get_distributor_data(
    distributor_id: str, 
    entity: str
):
    """
    Fetch data for multiple customers and entities in a single request as one file
    """
    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
        
    SD_API_URL = "https://simply-depo-api-2y3qx63wua-uc.a.run.app/api/data-exports/ai-tool/distributor-data"
    
    body = {
        "distributor_id": distributor_id,
        "entity": entity
    }

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "x-api-key": os.getenv('X_API_KEY')
    }

    # Set a generous timeout config (5s to connect, 120s to read the data stream)
    timeout_config = httpx.Timeout(5.0, read=120.0)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        response = None  # Initialize to prevent NameError in except block
        
        for attempt in range(3):
            try:
                response = await client.post(SD_API_URL, json=body, headers=headers)
                
                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    # Exponential backoff for rate limits: 2s, 4s, 8s
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue

                else:
                    # Explicitly handle non-200 / non-429 statuses (like the 404 Prisma error)
                    raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            except Exception as e:
                if attempt == 2:
                    # Safely log response details only if the response object actually exists
                    if response is not None:
                        try:
                            print(f"Final attempt failed body: {response.text}")
                        except Exception:
                            pass
                    raise Exception(f"Failed after retries: {e}")
                
                # Backoff between standard retries (e.g., timeouts/network drops): 5s, 10s
                await asyncio.sleep(5 * (attempt + 1))
                
import time
from AI.utils import get_logger

from AI.utils import (
    _process_and_save_file_data, read_dataframe_async, save_dataframe_async, combine_sections
)

logger2 = get_logger("logger2", "project_log_many.log", False)

import pandas as pd

import aiohttp
import aiofiles


from AI.group_customer_analyze.preprocess_data_group_c import (
    save_df, prepared_big_data
)

async def _download_and_save_file(session: aiohttp.ClientSession, file_url: str, file_path: str):
    """Downloads a file from a URL using a shared session and saves it asynchronously."""
    try:
        async with session.get(file_url) as response:
            if response.status == 200:
                # Read the file data and save it
                content = await response.read()
                async with aiofiles.open(file_path, mode='wb') as f:
                    await f.write(content)
                print(f"Successfully saved: {file_path}")
            else:
                print(f"Failed to download {file_url}. HTTP Status: {response.status}")
    except Exception as e:
        print(f"Error downloading {file_url}: {e}")

async def handle_distributor_data(api_response: dict, requested_entity: str, user_uuid: str):
    """
    Parses the API response, creates the user folder, and concurrently downloads files.
    """
    distributor_name = api_response.get("distributorName", "Unknown_Distributor")
    
    # 1. Define and create the specific user folder
    user_folder = os.path.join('data', str(user_uuid), 'work_data_folder')
    await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)
    
    tasks = []
    
    # 2. Open a single session for all concurrent downloads
    async with aiohttp.ClientSession() as session:
        
        # CASE 1: Multiple files (e.g., customers, orders, order_products)
        if "fileUrls" in api_response:
            for file_type, file_url in api_response["fileUrls"].items():
                # Dynamically creates: raw_file_orders.csv, raw_file_customers.csv, etc.
                filename = f"raw_file_{file_type}.csv"
                file_path = os.path.join(user_folder, filename)
                
                tasks.append(_download_and_save_file(session, file_url, file_path))
                
        # CASE 2: Single file (e.g., catalog)
        elif "fileUrl" in api_response:
            file_url = api_response["fileUrl"]
            # Uses the requested_entity to create: raw_file_catalog.csv
            filename = f"raw_file_{requested_entity}.csv"
            file_path = os.path.join(user_folder, filename)
            
            tasks.append(_download_and_save_file(session, file_url, file_path))

        # 3. Run all gathered tasks concurrently
        if tasks:
            print(f"Starting {len(tasks)} download tasks for {distributor_name}...")
            print(f"Target directory: {user_folder}")
            await asyncio.gather(*tasks)
            print("All files processed successfully!")
        else:
            print("No file URLs found in the response.")

from concurrent.futures import ThreadPoolExecutor


def clean_csv_logic(filepath: str) -> str:
    """
    Clean CSV logic:
    1. Removes 'INACTIVE' status rows.
    2. Drops technical/internal IDs.
    3. Drops columns with > 75% missing values.
    4. Fills inventory NaNs with 0.
    5. Saves to 'cleane_catalog.csv'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
        
    df = pd.read_csv(filepath)
    original_cols = set(df.columns)
    original_len = len(df)
    # 1. Filter out 'INACTIVE' status rows
    if 'status' in df.columns:
        # We use .str.upper() to ensure we catch 'inactive', 'Inactive', etc.
        df = df[df['status'].str.upper() != 'INACTIVE'].copy()
    
    # 2. Identify and drop sparse columns (> 95% empty)
    limit = len(df) * 0.05
    df_clean = df.dropna(thresh=limit, axis=1).copy()
    
    # 3. Drop specific 'useless' technical columns if they exist
    technical_cols = ['status', 'requiredFieldsMissing', 'description', 'hasColorVariation', 'hasSizeVariation', 'tags_tag_tag','type']
    cols_to_drop = [col for col in technical_cols if col in df_clean.columns]
    df_clean.drop(columns=cols_to_drop, inplace=True)
    
    # 4. Standardize inventory columns (fill NaNs with 0)
    inv_cols = [c for c in df_clean.columns if 'inventory' in c.lower()]
    for col in inv_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Identify all dropped columns
    dropped_cols = list(original_cols - set(df_clean.columns))

    # 5. Save to the requested filename
    file_path_catalog = os.path.join('data', 'FULL_DIST_TEST', 'cleaned_catalog.csv')

    df_clean.to_csv(file_path_catalog, index=False)
    
    return {
        "file_path": file_path_catalog,
        "dropped_columns": dropped_cols,
        "rows_remaining": f'{len(df_clean)} from {original_len}'
    }

async def get_cleaned_csv(filepath: str) -> str:
    """Async wrapper to process the CSV without blocking the event loop."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, clean_csv_logic, filepath)
    return result



async def main():
    start_time = time.perf_counter()
    user_folder = 'FULL_DIST_TEST'
    data = await get_distributor_data("f70070d6-6869-4544-99d7-539f40d7c70b", "customers")
    print(f"Step 1 - Data fetch completed: {time.perf_counter() - start_time:.2f}s")
    await handle_distributor_data(data, requested_entity="customers", user_uuid=user_folder)
    print(f"Step 2 - Data fetch completed: {time.perf_counter() - start_time:.2f}s")
    
    file_path_orders = os.path.join('data', user_folder, 'work_data_folder','raw_file_orders.csv')
    file_path_products = os.path.join('data', user_folder, 'work_data_folder','raw_file_order_products.csv')
    file_path_customers = os.path.join('data', user_folder,'work_data_folder', 'raw_file_customers.csv')
    file_path_catalog = os.path.join('data', user_folder,'work_data_folder', 'raw_file_catalog.csv')

    # Preprocess data
    try:
        full_cleaned_orders, full_cleaned_products = await prepared_big_data(
            str(file_path_orders), 
            str(file_path_products)
        )

        print(f"Step 2 - Data preprocessing completed: {time.perf_counter() - start_time:.2f}s")

        path = await get_cleaned_csv(file_path_catalog)
        print(path['dropped_columns'], path['rows_remaining'])
        # Save cleaned data concurrently
        cleaned_orders_path =  os.path.join('data', user_folder,  'cleaned_real_big_orders.csv') 
        cleaned_products_path =  os.path.join('data', user_folder,  'cleaned_real_big_products.csv')

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
                
        except Exception as e:
            logger2.warning(f"Can not check if customers orders are empty: {e}")
        # Read dataframes concurrently
        try:
            orders_df, products_df, customer_df = await asyncio.gather(
                read_dataframe_async(str(cleaned_orders_path)),
                read_dataframe_async(str(cleaned_products_path)),
                read_dataframe_async(str(file_path_customers))
            )
            print(f"Step 3 - Data loading completed: {time.perf_counter() - start_time:.2f}s")
            # Clean column names
            #merged_orders, products_df = await asyncio.to_thread(
            #    _sync_process_merge_logic, 
            #    orders_df, 
            #    customer_df, 
            #    products_df
            #)
  #
            #print(f"Step 3 - Data cleaning completed: {time.perf_counter() - start_time:.2f}s")
        except Exception as e:
            orders_df, products_df, customer_df = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
            logger2.error(f"main read_dataframe_async error: {e}")
    except Exception as e:
        logger2.error(f"Data cleaning error: {e}")




if __name__ == "__main__":
    asyncio.run(main())