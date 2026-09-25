import os
import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional, Union
import time
import aiohttp
import aiofiles

from AI.utils import get_logger

logger2 = get_logger("logger2", "project_log_many.log", False)

from dotenv import load_dotenv
load_dotenv()


async def get_distributor_data(
    distributor_id: str, 
    entities: Optional[List[str]] = None,
    client: Optional[httpx.AsyncClient] = None
):
    """
    Fetch data for multiple customers and entities in a single request.
    """
    SD_DISTRIBUTOR_API_URL = os.getenv('SD_DISTRIBUTOR_API_URL')
    if not SD_DISTRIBUTOR_API_URL:
        raise ValueError("SD_DISTRIBUTOR_API_URL environment variable is not set")
        
    body = {
        "distributor_id": distributor_id,
        "entities": entities
    }

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "x-api-key": os.getenv('X_API_KEY')
    }
    
    # Grab the name of the entity for our logs (e.g., "customers", "orders")
    log_name = entities[0] if entities else "unknown_entity"

    async def _fetch(active_client: httpx.AsyncClient):
        # 1. Start the stopwatch for this specific API call
        start_time = time.perf_counter()
        
        for attempt in range(3):
            try:
                response = await active_client.post(SD_DISTRIBUTOR_API_URL, json=body, headers=headers)
                
                if response.status_code == 200:
                    # 2. Stop the stopwatch on success
                    elapsed_time = time.perf_counter() - start_time
                    logger2.info(f"API Fetch {str(distributor_id)} OK: '{log_name}' took {elapsed_time:.2f}s")
                    return response.json()
                
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                if attempt == 2 or e.response.status_code < 500:
                    elapsed_time = time.perf_counter() - start_time
                    logger2.error(f"API Fetch HTTP Error: '{log_name}' failed after {elapsed_time:.2f}s | Status: {e.response.status_code}")
                    raise RuntimeError(f"HTTP Error {e.response.status_code}: {e.response.text}") from e
                
                await asyncio.sleep(5 * (attempt + 1))

            except httpx.RequestError as e:
                if attempt == 2:
                    elapsed_time = time.perf_counter() - start_time
                    logger2.error(f"API Fetch {str(distributor_id)} Network Error: '{log_name}' failed after {elapsed_time:.2f}s | Error: {str(e)}")
                    raise RuntimeError(f"Network Error: Failed after retries: {e}") from e
                
                await asyncio.sleep(5 * (attempt + 1))

    if client:
        return await _fetch(client)
    else:
        timeout_config = httpx.Timeout(5.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout_config) as temp_client:
            return await _fetch(temp_client)

    # Use the shared client if provided, otherwise create a temporary one
    if client:
        return await _fetch(client)
    else:
        timeout_config = httpx.Timeout(5.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout_config) as temp_client:
            return await _fetch(temp_client)
               
async def _download_and_save_file(session: aiohttp.ClientSession, file_url: str, file_path: str):
    """Downloads a file from a URL using a shared session and saves it asynchronously."""
    try:
        async with session.get(file_url) as response:
            if response.status == 200:
                # Read the file data and save it
                content = await response.read()
                async with aiofiles.open(file_path, mode='wb') as f:
                    await f.write(content)
                logger2.info(f"Successfully saved: {file_path}")
            else:
                logger2.error(f"Failed to download {file_url}. HTTP Status: {response.status}")
    except Exception as e:
        logger2.error(f"Error downloading {file_url}: {e}")

async def handle_distributor_data(
    api_response: dict, 
    requested_entity: str, 
    user_uuid: str, 
    session: aiohttp.ClientSession  # <-- Pass the session in here
):
    """
    Parses the API response, creates the user folder, and queues downloads.
    """
    distributor_name = api_response.get("distributorName", "Unknown_Distributor")
    user_folder = os.path.join('data', str(user_uuid), 'work_data_folder')
    await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)
    
    tasks = []
    
    # CASE 1: Multiple files
    if "fileUrls" in api_response:
        for file_type, file_url in api_response["fileUrls"].items():
            filename = f"raw_file_{file_type}.csv"
            file_path = os.path.join(user_folder, filename)
            tasks.append(_download_and_save_file(session, file_url, file_path))
            
    # CASE 2: Single file
    elif "fileUrl" in api_response:
        file_url = api_response["fileUrl"]
        filename = f"raw_file_{requested_entity}.csv"
        file_path = os.path.join(user_folder, filename)
        tasks.append(_download_and_save_file(session, file_url, file_path))

    # Run the tasks for THIS specific entity
    if tasks:
        await asyncio.gather(*tasks)
        logger2.info(f"Successfully saved {len(tasks)} file(s) for {requested_entity}.")

import re

def _extract_url(raw: str) -> str:
    """
    The distributor API sometimes returns fields as markdown links
    `[text](url)` instead of a plain URL. Pull out the real URL
    (inside the parentheses) if that's the case.
    """
    if not raw:
        return raw
    match = re.match(r'^\[.*\]\((.*)\)$', raw.strip())
    return match.group(1) if match else raw.strip()


async def handle_activities_data(
    api_response: Union[dict, list],
    requested_entity: str,
    user_uuid: str,
    session: aiohttp.ClientSession
):
    """
    Parses the API response (dict OR list of dicts), creates the user
    folder, and queues downloads.
    """
    # Normalize: API can return either a single object or a list of them
    records = api_response if isinstance(api_response, list) else [api_response]

    user_folder = os.path.join('data', str(user_uuid), 'work_data_folder')
    await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)

    tasks = []

    for record in records:
        distributor_name = record.get("distributorName", "Unknown_Distributor")

        # CASE 1: Multiple files
        if "fileUrls" in record:
            for file_type, raw_url in record["fileUrls"].items():
                file_url = _extract_url(raw_url)
                filename = f"raw_file_{file_type}.csv"
                file_path = os.path.join(user_folder, filename)
                tasks.append(_download_and_save_file(session, file_url, file_path))

        # CASE 2: Single file
        elif "fileUrl" in record:
            file_url = _extract_url(record["fileUrl"])
            filename = f"raw_file_{requested_entity}.csv"
            file_path = os.path.join(user_folder, filename)
            tasks.append(_download_and_save_file(session, file_url, file_path))

    if tasks:
        await asyncio.gather(*tasks)
        logger2.info(f"Successfully saved {len(tasks)} file(s) for {requested_entity}.")

