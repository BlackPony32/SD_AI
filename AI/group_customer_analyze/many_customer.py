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

from AI.group_customer_analyze.create_report_group_c import generate_analytics_report
from AI.utils import get_logger

load_dotenv()
app = FastAPI()

logger2 = get_logger("logger2", "project_log_many.log", False)


async def get_exported_data_many(customer_id: str, entity: str) -> dict:
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
                    if attempt < 2:  # Retry if not the last attempt
                        logger2.warning(f"429 for customer {customer_id}, entity {entity}. Retrying in 5s...")
                        await asyncio.sleep(10)
                    else:
                        raise Exception("Too many retries")
                else:
                    raise Exception(f"Error: HTTP {response.status_code}")
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    raise Exception(f"Failed after retries: {e}")
                await asyncio.sleep(10)  # Wait before retrying

AllowedEntity = Literal["orders", "order_products", "notes", "tasks", "activities"]



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
        save_dir = os.path.join("data", uuid, customer_id, entity)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{entity}.csv")
        
        # Handle missing data
        if customer_id not in data or not data[customer_id]:
            results[customer_id] = f"Error: No data for {customer_id}"
            return
        
        try:
            # Process CSV data to add customer name column
            csv_bytes = data[customer_id]
            csv_text = csv_bytes.decode('utf-8')
            
            # Use StringIO for efficient text processing
            input_stream = io.StringIO(csv_text)
            output_stream = io.StringIO()
            
            reader = csv.reader(input_stream)
            writer = csv.writer(output_stream)
            
            # Process header row
            headers = next(reader, None)
            if headers is not None:
                headers.append("customer_name")
                writer.writerow(headers)
                
                # Process data rows
                for row in reader:
                    row.append(customer_name)
                    writer.writerow(row)
            else:
                # Empty file - write header only
                writer.writerow(["customer_name"])
            
            # Get modified CSV content
            modified_csv = output_stream.getvalue().encode('utf-8')
            
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

    
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')