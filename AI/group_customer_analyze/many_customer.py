from fastapi import FastAPI, Body
from typing import Literal, Dict
from pydantic import BaseModel
import os
import logging
import requests
import asyncio
import httpx
from uuid import uuid4
import aiofiles
from dotenv import load_dotenv
import os
import asyncio
import csv
import io
import aiofiles
from typing import Dict, List
import random
import pandas as pd
from AI.group_customer_analyze.create_report_group_c import generate_analytics_report
from AI.utils import get_logger
import json
load_dotenv()
app = FastAPI()

logger2 = get_logger("logger2", "project_log_many.log", False)


async def old_get_exported_data_many(customer_id: str, entity: str) -> dict:
    """
    Asynchronously connects to the API endpoint to export customer profile data and returns the file content.
    Includes retry logic for HTTP 429 errors.
    """
    allowed_entities = ["orders", "order_products", 'customer', "notes", "tasks", "activities"]
    if entity not in allowed_entities:
        raise ValueError(f"Invalid entity: {entity}. Must be one of {allowed_entities}")

    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
    url = SD_API_URL

    params = {"customer_id": customer_id, "entity": entity}
    headers = {"x-api-key": os.getenv('X_API_KEY')}

    async with httpx.AsyncClient() as client:
        for attempt in range(3):  # Try up to 3 times
            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    exported_url = data.get("fileUrl")
                    customer_name = data.get("customerName", "Unknown")
                    if not exported_url:
                        raise Exception("No 'fileUrl' found in response")
                    file_response = await client.get(exported_url)
                    if file_response.status_code == 200:
                        return {"file": file_response.content, "customer_name": customer_name}
                    else:
                        raise Exception(f"Failed to download file: HTTP {file_response.status_code}")
                elif response.status_code == 429:
                    if attempt < 4:
                        backoff = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
                        jitter = random.uniform(0, 5)  # Random 0-5s
                        sleep_time = backoff + jitter
                        logger2.warning(f"429 for customer {customer_id}, entity {entity}. Retrying in {sleep_time:.1f}s...")
                        await asyncio.sleep(sleep_time)
                    else:
                        raise Exception("Too many retries")
                else:
                    raise Exception(f"Error: HTTP {response.status_code}")
            except Exception as e:
                if attempt == 4:
                    raise Exception(f"Failed after retries: {e}")
                # Sleep on other exceptions too, with backoff
                await asyncio.sleep(5 * (2 ** attempt) + random.uniform(0, 5))

AllowedEntity = Literal["orders", "order_products", "notes", "tasks", "activities"]

async def download_file(client, url: str, entity: str):
    """Download individual file and return entity-content pair"""
    response = await client.get(url)
    if response.status_code == 200:
        return (entity, response.content)
    raise Exception(f"Failed to download {entity} file")

async def get_exported_data_many(customer_id: str, entities: List[str]) -> dict:
    """
    Fetch data for multiple entities in a single request
    """
    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
    
    params = {
        "customer_id": customer_id,
        "entities": json.dumps(entities)
    }
    headers = {"x-api-key": os.getenv('X_API_KEY')}

    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                response = await client.get(SD_API_URL, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    customer_name = data.get("customerName", "Unknown")
                    file_urls = data.get("fileUrls", {})
                    
                    # Download all files concurrently
                    download_tasks = [
                        download_file(client, url, entity)
                        for entity, url in file_urls.items()
                    ]
                    file_contents = await asyncio.gather(*download_tasks)
                    
                    return {
                        "customer_name": customer_name,
                        "files": dict(file_contents)
                    }
                    
                elif response.status_code == 429:
                    await asyncio.sleep(1 * (2 ** attempt))  # Reduced retry delay
                else:
                    raise Exception(f"HTTP Error {response.status_code}")
                    
            except Exception as e:
                if attempt == 2:
                    raise Exception(f"Failed after retries: {e}")
                await asyncio.sleep(5 * (2 ** attempt))

async def save_customer_data(
    customer_ids: List[str],
    entity: AllowedEntity,
    data: Dict[str, bytes],
    uuid: str,
    customer_names: Dict[str, str]  # Add customer names dictionary
) -> Dict[str, str]:
    """
    Saves data for each customer ID to a file, adding customer name as a new column.
    
    Args:
        customer_ids: List of customer IDs
        entity: Entity type (e.g., 'orders')
        data: Dictionary mapping customer IDs to data bytes
        uuid: UUID for the folder structure
        customer_names: Dict mapping customer IDs to names

    Returns:
        Dictionary mapping customer IDs to saved file paths or errors
    """
    results = {}
    
    async def save_single_customer(customer_id: str):
        # Get customer name (default to UNKNOWN if not found)
        customer_name = customer_names.get(customer_id, "UNKNOWN_CUSTOMER")

        # Create directory path
        save_dir = os.path.join("data", uuid, "raw_data", customer_id, entity)
        await asyncio.to_thread(os.makedirs, save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{entity}.csv")

        # Handle missing data
        if customer_id not in data or not data[customer_id]:
            results[customer_id] = f"Error: No data for {customer_id}"
            return

        try:
            # Process CSV data to add customer name column
            csv_bytes = data[customer_id]
            #csv_text = csv_bytes.decode('utf-8')

            def process_csv_sync():
                df = pd.read_csv(io.BytesIO(csv_bytes))
                df['customer_name'] = customer_name
                return df.to_csv(index=False).encode('utf-8')

            modified_csv = await asyncio.to_thread(process_csv_sync)

            # Write to file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(modified_csv)

            results[customer_id] = file_path

        except Exception as e:
            error_msg = f"Error processing {customer_id}: {str(e)}"
            results[customer_id] = error_msg

    # Execute all saves concurrently
    tasks = [save_single_customer(cid) for cid in customer_ids]
    await asyncio.gather(*tasks)
    return results

    

async def get_exported_data_one_file(customer_ids: List[str], entities: List[str]) -> dict:
    """
    Fetch data for multiple customers and entities in a single request as one file
    """
    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
    
    params = {
        "customer_ids": json.dumps(customer_ids),
        "entities": json.dumps(entities),
        "one_file": "true"
    }
    headers = {"x-api-key": os.getenv('X_API_KEY')}

    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                response = await client.get(SD_API_URL, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    file_url = data.get("fileUrl")
                    if not file_url:
                        raise Exception("No fileUrl in response")
                    
                    # Download the single file
                    file_content = await download_file(client, file_url, "combined")
                    
                    return {
                        "files": {"combined": file_content}
                    }
                    
                elif response.status_code == 429:
                    await asyncio.sleep(1 * (2 ** attempt))  # Reduced retry delay
                else:
                    raise Exception(f"HTTP Error {response.status_code}")
                    
            except Exception as e:
                if attempt == 2:
                    raise Exception(f"Failed after retries: {e}")
                await asyncio.sleep(5 * (2 ** attempt))    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')