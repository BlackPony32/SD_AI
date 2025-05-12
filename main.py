from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi import  Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from fastapi.responses import PlainTextResponse
import pandas as pd
import os
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.concurrency import run_in_threadpool
import uuid
from pathlib import Path
from urllib.parse import urlparse

#from db.database_reports import generate_customer_reports
from AI.Activities_AI import process_ai_activities_request
from AI.Order_analytics import generate_sales_report
from AI.create_process_data import create_user_data
from AI.Activities_analytics import analyze_activities
from AI.Tasks_analytics import tasks_report
from AI.Notes_analytics import notes_report

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
from typing import Literal
import shutil

from concurrent.futures import ProcessPoolExecutor
load_dotenv()
    


app = FastAPI()
AllowedEntity = Literal["orders", "order_products", "notes", "tasks", "activities"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_file_path = "project_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor()

import requests

def get_exported_data(customer_id, entity):
    """
    Connects to the API endpoint to export customer profile data and returns the file content.

    Args:
        customer_id (str): The customer's UUID.
        entity (str): The type of data to export ('orders', 'order_products', 'notes', 'tasks', or 'activities').

    Returns:
        bytes: The content of the exported data file.

    Raises:
        ValueError: If the entity value is invalid.
        Exception: If the API request or file download fails, including detailed error information.
    """
    # Define allowed entity values
    allowed_entities = ["orders", "order_products", "notes", "tasks", "activities"]
    if entity not in allowed_entities:
        raise ValueError(f"Invalid entity: {entity}. Must be one of {allowed_entities}")

    SD_API_URL = os.getenv('SD_API_URL')
    if not SD_API_URL:
        raise Exception("SD_API_URL environment variable is not set")
    url = SD_API_URL

    # Query parameters
    params = {
        "customer_id": customer_id,
        "entity": entity
    }

    x_api_key = os.getenv('X_API_KEY')
    if not x_api_key:
        raise Exception("X_API_KEY environment variable is not set")
    # Authentication header
    headers = {
        "x-api-key": x_api_key
    }

    def _get_error_detail(resp: requests.Response) -> str:
        """
        Extracts detailed error message from the response, preferring JSON fields.
        """
        try:
            err_json = resp.json()
            for key in ("error", "message", "detail"):  # common keys
                if key in err_json:
                    return f"{key}: {err_json[key]}"
            return str(err_json)
        except ValueError:
            return resp.text or "<no response body>"

    # Make the GET request to the API
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print("Content-Type:", response.headers.get('Content-Type', 'Not specified'))
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    # Handle the response
    if response.status_code == 200:
        # Parse the response as JSON
        try:
            data = response.json()
            print("Parsed JSON data:", data)
            exported_url = data.get("fileUrl")
            if not exported_url:
                raise Exception("No 'fileUrl' found in response")
        except ValueError:
            detail = _get_error_detail(response)
            raise Exception(f"Response is not valid JSON — {detail}")

        # Download the file from the exported URL
        try:
            file_response = requests.get(exported_url, timeout=10)
            if file_response.status_code == 200:
                return file_response.content
            else:
                detail = _get_error_detail(file_response)
                raise Exception(f"Failed to download file: HTTP {file_response.status_code} — {detail}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"File download failed: {e}")
    elif response.status_code == 401:
        detail = _get_error_detail(response)
        raise Exception(f"Authentication error (401): {detail}")
    else:
        detail = _get_error_detail(response)
        raise Exception(f"Error: HTTP {response.status_code} — {detail}")

# Initialize at app startup
@app.on_event("startup")
async def startup_event():
    app.state.process_executor = ProcessPoolExecutor(max_workers=4)  # Adjust based on CPU cores

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    app.state.process_executor.shutdown(wait=True)

@app.post("/generate-reports/{customer_id}")
async def create_reports(customer_id: str, entity: AllowedEntity):
    """
    Endpoint to generate and save a report for a given customer and entity.
    
    - For 'orders', it generates two files (orders and order_products) and then creates a sales report.
    - For other entities, it saves the corresponding file.
    """
    try:
        if entity == "orders":
            # Retrieve file contents for orders and order_products 
            try:
                orders_file_content = get_exported_data(customer_id, "orders")
            except Exception as e:
                logger.error(f"Error in get_exported_data orders: {e}")
            
            try:
                products_file_content = get_exported_data(customer_id, "order_products")
            except Exception as e:
                logger.error(f"Error in get_exported_data order_products: {e}")
            # Define directory and file paths for orders and order_products
            orders_dir = os.path.join("data", customer_id, "orders")
            os.makedirs(orders_dir, exist_ok=True)
            orders_path = os.path.join(orders_dir, "orders.csv")
            products_path = os.path.join(orders_dir, "order_products.csv")
            #
            logger.info(f"Saving orders report for customer '{customer_id}' at {orders_path}")
            async with aiofiles.open(orders_path, "wb") as f:
                await f.write(orders_file_content)
            
            logger.info(f"Saving order products report for customer '{customer_id}' at {products_path}")
            async with aiofiles.open(products_path, "wb") as f:
                await f.write(products_file_content)
            
            # Generate the sales report using the saved orders and products file paths
            try:
                result  = await generate_sales_report(orders_path, products_path, customer_id)
                report = result["full_report"]
                #print(report)
                report_sections = result["sections"] 
            except Exception as e:
                logger.error(f"Error in generate_sales_report: {e}")
            
            
            report_dir = os.path.join("data", customer_id)
            path_for_report = os.path.join(report_dir, "report.md")
            async with aiofiles.open(path_for_report, "w") as f:
                await f.write(report)
                

            return JSONResponse(status_code=200, content={"message": "Sales report generated successfully",
                                                          'file_path_product':products_path,
                                                          'file_path_orders':orders_path,
                                                          "report": report,
                                                          "sections": report_sections})
        
        elif entity == "activities":
            #TODO For 'activities', 'notes', or 'tasks' different report func
            try:
                notes = get_exported_data(customer_id, "notes")
                tasks = get_exported_data(customer_id, "tasks")
                activities = get_exported_data(customer_id, "activities")
            except Exception as e:
                logger.error(f"Error in get_exported_data activities: {e}")
            
            
            # Build the directory path and file name for non-orders entities
            dir_path = os.path.join("data", customer_id, 'activities')
            os.makedirs(dir_path, exist_ok=True)
            
            file_path_notes = os.path.join(dir_path, "notes.csv")
            file_path_tasks = os.path.join(dir_path, "tasks.csv")
            file_path_activities = os.path.join(dir_path, "activities.csv")
            
            logger.info(f"Saving report for customer '{customer_id}', entity 'notes' at {file_path_notes}")
            async with aiofiles.open(file_path_notes, "wb") as f:
                await f.write(notes)
            
            logger.info(f"Saving report for customer '{customer_id}', entity 'tasks' at {file_path_tasks}")
            async with aiofiles.open(file_path_tasks, "wb") as f:
                await f.write(tasks)
            
            logger.info(f"Saving report for customer '{customer_id}', entity 'activities' at {file_path_activities}")
            async with aiofiles.open(file_path_activities, "wb") as f:
                await f.write(activities)
            
            
            ## Generate the sales report using the saved orders and products file paths
            try:
                report_activities = await analyze_activities(file_path_notes, file_path_tasks, file_path_activities)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_activities_dir, "report_activities.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(report_activities)
            except Exception as e:
                logger.error(f"Error in analyze_activities: {e}")
                
            try:
                report_task = tasks_report(file_path_tasks)
                #print(report_task)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_activities_dir, "report_task.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(report_task)
            except Exception as e:
                logger.error(f"Error in tasks_report: {e}")
            try:    
                report_notes = notes_report(file_path_notes)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_notes = os.path.join(report_activities_dir, "report_notes.md")
                async with aiofiles.open(path_for_notes, "w") as f:
                    await f.write(str(report_notes))
            except Exception as e:
                logger.error(f"Error in notes_report: {e}")
                
            try:
                report_text = await process_ai_activities_request(customer_id)
                #print(report_text.get('model_answer'))
                if report_text['model_answer'] == 'Could not analyze the activity of your customer.':
                    print("Analysis failed - check logs for details")
                else:
                    print("Analysis succeeded:")
            except Exception as e:
                logger.error(f"Error in process_ai_activities_request: {e}")
            
            
            return JSONResponse(status_code=200, content={"message": f"Report generated successfully", 
                                                          "path_for_report": 'path_for_report',
                                                          "report": report_text.get('model_answer')})

        else:
            raise HTTPException(status_code=406, detail='Incorrect entity: use "activities" or "orders" ')

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=406, detail=f"Error generating report due to incorrect customer id")
    

from functools import partial

#TODO rename Ask_AI to Ask_AI_orders
async def Ask_AI(prompt: str, file_path_product, file_path_orders, customer_id):
    
    try:
        process_func = partial(
            _process_ai_request,
            prompt=prompt,
            customer_id = customer_id,
            file_path_product=file_path_product,
            file_path_orders=file_path_orders
        )
        
        # Run in process pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app.state.process_executor,
            process_func
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def _process_ai_request(prompt, file_path_product, file_path_orders, customer_id):
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df1 = pd.read_csv(file_path_product, encoding=encoding, low_memory=False)
                df2 = pd.read_csv(file_path_orders, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed decoding attempt with encoding: {encoding}")
        #df = df1.merge(df2, on="orderId", how="left")
        #df.to_csv('data.csv',index=False)
        llm = ChatOpenAI(model='o3-mini') #model='o3-mini'
        
        agent = create_pandas_dataframe_agent(
            llm,
            [df1, df2],
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True,
            number_of_head_rows=5,
            max_iterations=5
        )
         
        report_path = f'data//{customer_id}//report.md'
        with open(report_path, "r") as file:
            report = file.read()
        
        additional_info_path = f'data//{customer_id}//additional_info.md'
        with open(additional_info_path, "r") as file:
            additional_info = file.read()
        
        reorder_path = f'data//{customer_id}//reorder.md'
        with open(reorder_path, "r") as file:
            reorder = file.read()
        
        formatted_prompt = f"""
        You are an AI assistant providing business insights based on two related datasets:  
        - **df1** (Orders) contains critical order-related data.  
        - **df2** (Products) contains details about products within each order.  

        Also you can use some report that i gain from data: {report}
        Useful additional data: {additional_info}
        And useful additional reorder data: {reorder}
        
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

        **Question:**  
        {prompt}"""

        logger.info("\n===== Metadata =====")
        with get_openai_callback() as cb:
            agent.agent.stream_runnable = False
            start_time = time.time()
            result = agent.invoke({"input": formatted_prompt})
            execution_time = time.time() - start_time
            
            result['metadata'] = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'execution_time': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }

        
        for k, v in result['metadata'].items():
            logger.info(f"{k.replace('_', ' ').title()}: {v}")
            
        input_tokens = result['metadata'].get('prompt_tokens')
        output_tokens = result['metadata'].get('completion_tokens')
        input_cost = (input_tokens / 1000) * 0.0025
        output_cost = (output_tokens / 1000) * 0.01
        total_cost = input_cost + output_cost
        return {"output": result.get('output'), "cost": total_cost}

    except Exception as e:
        logger.error(f"Error in AI processing: {str(e)}")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/Ask_ai")
async def ask_ai_endpoint(
    request: ChatRequest,
    file_path_product: str = Query(...),
    file_path_orders: str = Query(...),
    customer_id: str = Query(...)  # Add customer ID parameter
):
    prompt = request.prompt
    try:
        # Pass customer_id to Ask_AI function
        response = await Ask_AI(prompt, file_path_product, file_path_orders, customer_id)
        answer = response.get('output')
        cost = response.get('cost', 0.0)
    except Exception as e:
        logger.error(f"Error executing LLM: {str(e)}")
        answer = "Cannot answer this question"
        cost = 0.0
    
    return {"response": answer, "prompt": prompt, "cost": cost}

@app.get("/logs/last/{num_lines}", response_class=PlainTextResponse)
async def get_last_n_log_lines(num_lines: int):
    """
    Return the last `num_lines` from the log file.

    Raises 404 if the file is missing or empty, 422 if `num_lines` is invalid, and 500 on unexpected errors.
    """
    LOG_FILE = "project_log.log"
    # Validate input
    if num_lines <= 0:
        raise HTTPException(status_code=422, detail="`num_lines` must be a positive integer.")
    try:
        with open(LOG_FILE, "r") as file:
            lines = file.readlines()

        if not lines:
            # No content in the log
            raise HTTPException(status_code=404, detail="Log file is empty.")

        # Safely slice the last lines (if num_lines exceeds total lines, return all)
        last_lines = lines[-num_lines:]
        return "".join(last_lines)

    except HTTPException:
        # Re-raise HTTP errors so FastAPI handles them correctly
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found")
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    


async def clean_directories(customer_id: str):
    """
    Remove the data directory for the given customer_id if it exists.
    """
    data_folder = os.path.join("data", customer_id)
    # Only attempt removal if the folder exists
    if os.path.isdir(data_folder):
        loop = asyncio.get_event_loop()
        try:
            # Run shutil.rmtree in thread pool to avoid blocking
            await loop.run_in_executor(executor, shutil.rmtree, data_folder)
            logger.info(f"Deleted user folder: {data_folder}")
        except Exception as e:
            logger.error(f"Failed to delete {data_folder}: {e}")
    else:
        logger.info(f"No directory to delete for user {customer_id}")

@app.get("/clean_chat/")
async def clean_chat(customer_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint to schedule directory cleanup for a given customer_id.

    This runs clean_directories in the background and immediately returns a success message.
    """
    background_tasks.add_task(clean_directories, customer_id)
    return {"response": "Chat is cleaned successfully"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000, host='0.0.0.0')