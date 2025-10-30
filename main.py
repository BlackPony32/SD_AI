from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi import  Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi import Body

import pandas as pd
import os
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from pathlib import Path
from urllib.parse import urlparse

#from db.database_reports import generate_customer_reports 
from AI.single_customer_analyze.Activities_AI import process_ai_activities_request
from AI.single_customer_analyze.Order_analytics import generate_sales_report
from AI.single_customer_analyze.create_process_data import create_user_data
from AI.single_customer_analyze.Activities_analytics import analyze_activities
from AI.single_customer_analyze.Tasks_analytics import tasks_report
from AI.single_customer_analyze.Notes_analytics import notes_report
from AI.group_customer_analyze.many_customer import save_customer_data, get_exported_data_many
from AI.group_customer_analyze.create_report_group_c import generate_analytics_report
from AI.group_customer_analyze.preprocess_data_group_c import create_group_user_data
from AI.group_customer_analyze.Ask_ai_many_customers import Ask_ai_many_customers

from AI.group_customer_analyze.many_customer import get_exported_data_one_file
from AI.utils import (
    _process_and_save_file_data, read_dataframe_async, save_dataframe_async
)

from AI.group_customer_analyze.preprocess_data_group_c import (
    save_df, prepared_big_data
)
from AI.utils import get_logger, extract_customer_id, process_fetch_results, validate_save_results, generate_file_paths, create_response, \
    analyze_customer_orders_async

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


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
from typing import List
from concurrent.futures import ProcessPoolExecutor
load_dotenv()
    


app = FastAPI()
AllowedEntity = Literal["orders", "activities"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



logger1 = get_logger("logger1", "project_log.log", False)
logger2 = get_logger("logger2", "project_log_many.log", False)

executor = ThreadPoolExecutor()

import requests

def get_exported_data(customer_id, entity):
    """
    Connects to the API endpoint to export customer profile data and returns the file content.

    Args:
        customer_id (str): The customer's UUID.
        entity (str): The type of data to export ('orders', 'order_products', 'notes', 'tasks', 'customer', or 'activities').

    Returns:
        bytes: The content of the exported data file.

    Raises:
        ValueError: If the entity value is invalid.
        Exception: If the API request or file download fails, including detailed error information.
    """
    # Define allowed entity values
    allowed_entities = ["orders", "order_products", 'customer', "notes", "tasks", "activities"]
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
    app.state.process_executor = ProcessPoolExecutor(max_workers=6)  # Adjust based on CPU cores

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    app.state.process_executor.shutdown(wait=True)

from pydantic import BaseModel

class ReportRequest(BaseModel):
    entity: AllowedEntity


@app.post("/generate-reports/{customer_id}")
async def create_reports(customer_id: str, request: ReportRequest):
    entity = request.entity
    """
    Endpoint to generate and save a report for a given customer and entity.
    
    - For 'orders', it generates two files (orders and order_products) and then creates a sales report.
    - For 'activities' it generates activities (+task, notes) report.
    """
    try:
        if entity == "orders":
            # Retrieve file contents for orders and order_products 
            try:
                orders_file_content = get_exported_data(customer_id, "orders")
            except Exception as e:
                logger1.error(f"Error in get_exported_data orders: {e}")
             
            try:
                products_file_content = get_exported_data(customer_id, "order_products")
            except Exception as e:
                logger1.error(f"Error in get_exported_data order_products: {e}")
            # Define directory and file paths for orders and order_products
            orders_dir = os.path.join("data", customer_id, "orders")
            os.makedirs(orders_dir, exist_ok=True)
            orders_path = os.path.join(orders_dir, "orders.csv")
            products_path = os.path.join(orders_dir, "order_products.csv")
            #
            logger1.info(f"Saving orders report for customer '{customer_id}' at {orders_path}")
            async with aiofiles.open(orders_path, "wb") as f:
                await f.write(orders_file_content)
            
            logger1.info(f"Saving order products report for customer '{customer_id}' at {products_path}")
            async with aiofiles.open(products_path, "wb") as f:
                await f.write(products_file_content)
            
            # Generate the sales report using the saved orders and products file paths
            try:
                result  = await generate_sales_report(orders_path, products_path, customer_id)
                report = result["full_report"]
                #print(report)
                report_sections = result["sections"] 
            except Exception as e:
                logger1.error(f"Error in generate_sales_report: {e}")
            
            
            report_dir = os.path.join("data", customer_id)
            path_for_report = os.path.join(report_dir, "report.md")
            async with aiofiles.open(path_for_report, "w") as f:
                await f.write(report)
                
            #print(report)
            return JSONResponse(status_code=200, content={"message": "Sales report generated successfully",
                                                          'file_path_product':products_path,
                                                          'file_path_orders':orders_path,
                                                          "report": report,
                                                          "sections": report_sections})
        
        elif entity == "activities":
            try:
                notes = get_exported_data(customer_id, "notes")
                tasks = get_exported_data(customer_id, "tasks")
                activities = get_exported_data(customer_id, "activities")
            except Exception as e:
                logger1.error(f"Error in get_exported_data activities: {e}")
            
            
            # Build the directory path and file name for non-orders entities
            dir_path = os.path.join("data", customer_id, 'activities')
            os.makedirs(dir_path, exist_ok=True)
            
            file_path_notes = os.path.join(dir_path, "notes.csv")
            file_path_tasks = os.path.join(dir_path, "tasks.csv")
            file_path_activities = os.path.join(dir_path, "activities.csv")
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'notes' at {file_path_notes}")
            async with aiofiles.open(file_path_notes, "wb") as f:
                await f.write(notes)
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'tasks' at {file_path_tasks}")
            async with aiofiles.open(file_path_tasks, "wb") as f:
                await f.write(tasks)
            
            logger1.info(f"Saving report for customer '{customer_id}', entity 'activities' at {file_path_activities}")
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
                logger1.error(f"Error in analyze_activities: {e}")
                
            try:
                report_task = tasks_report(file_path_tasks)
                #print(report_task)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_activities_dir, "report_task.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(report_task)
            except Exception as e:
                logger1.error(f"Error in tasks_report: {e}")
            try:    
                report_notes = notes_report(file_path_notes)
                report_activities_dir = os.path.join("data", customer_id)
                path_for_notes = os.path.join(report_activities_dir, "report_notes.md")
                async with aiofiles.open(path_for_notes, "w") as f:
                    await f.write(str(report_notes))
            except Exception as e:
                logger1.error(f"Error in notes_report: {e}")
                
            try:
                report_text, section_report = await process_ai_activities_request(customer_id)
                try:
                    full_report = report_text.get('model_answer')
                except Exception as e:
                    full_report = "Could not analyze the activity of your customer"
                    
                #print(report_text.get('model_answer'))
                if report_text['model_answer'] == 'Could not analyze the activity of your customer.':
                    print("Analysis failed - check logs for details")
                else:
                    print("Analysis succeeded:")
            except Exception as e:
                logger1.error(f"Error in process_ai_activities_request: {e}")
            
            
            return JSONResponse(status_code=200, content={"message": f"Report generated successfully", 
                                                          "file_path_activities": file_path_activities,
                                                          "file_path_tasks": file_path_tasks,
                                                          "report": full_report,
                                                          "sections": section_report})

        else:
            raise HTTPException(status_code=406, detail='Incorrect entity: use "activities" or "orders" ')

    except Exception as e:
        logger1.error(f"Error generating report: {e}")
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
        logger1.info(f"User prompt: {prompt}")
        return result
    
    except Exception as e:
        logger1.error(f"Analysis failed: {e}")
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
                logger1.warning(f"Failed decoding attempt with encoding: {encoding}")
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
         
        #report_path = f'data//{customer_id}//report.md'
        try:
            report_path = os.path.join('data', customer_id, 'report.md')
            with open(report_path, "r") as file:
                report = file.read()
        except Exception as e:
            report = 'No data given'
            logger1.warning(f"Can not read report.md due to {e} ")
        
        try:
            additional_info_path = os.path.join('data', customer_id, 'additional_info.md')
            with open(additional_info_path, "r") as file:
                additional_info = file.read()
        except Exception as e:
            additional_info = 'No data given'
            logger1.warning(f"Can not read additional_info.md due to {e} ")
        
        try:
            reorder_path = os.path.join('data', customer_id, 'reorder.md')
            with open(reorder_path, "r") as file:
                reorder = file.read()
        except Exception as e:
            reorder = 'No data given'
            logger1.warning(f"Can not read reorder.md due to {e} ")
         
        
        try:
            report_activities_path = os.path.join('data', customer_id, 'report_activities.md')
            with open(report_activities_path, "r") as file:
                report_activities = file.read()
        except Exception as e:
            report_activities = 'No data given'
            logger1.warning(f"Can not read report_activities.md due to {e} ")
        
        formatted_prompt = f"""
        You are an AI assistant providing business insights based on two related datasets:  
        - **df1** (Orders) contains critical order-related data or can be user activities data.  
        - **df2** (Products) contains details about products within each order or can be user tasks data.  

        Also you can use some report that i gain from data: {report}
        Useful additional data: {additional_info}
        Report activities additional data: {report_activities}
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

        
        logger1.info("\n===== Metadata =====")
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
        
            logger1.info(f"Input Cost:  ${in_cost:.6f}")
            logger1.info(f"Output Cost: ${out_cost:.6f}")
            logger1.info(f"Total Cost:  ${cost:.6f}")
        
            result['metadata'] = {
                'total_tokens': in_toks+out_toks,
                'prompt_tokens': in_toks,
                'completion_tokens': out_toks,
                'execution_time': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }

        
        for k, v in result['metadata'].items(): 
            logger1.info(f"{k.replace('_', ' ').title()}: {v}")
        return {"output": result.get('output'), "cost": cost}

    except Exception as e:
        logger1.error(f"Error in AI processing: {str(e)}")

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
        logger1.error(f"Error executing LLM: {str(e)}")
        answer = "Cannot answer this question"
        cost = 0.0
    
    return {"data": answer, "prompt": prompt, "cost": cost}

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
            logger1.info(f"Deleted user folder: {data_folder}")
        except Exception as e:
            logger1.error(f"Failed to delete {data_folder}: {e}")
    else:
        logger1.info(f"No directory to delete for user {customer_id}")

@app.get("/clean_chat/")
async def clean_chat(customer_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint to schedule directory cleanup for a given customer_id.

    This runs clean_directories in the background and immediately returns a success message.
    """
    background_tasks.add_task(clean_directories, customer_id)
    return {"response": "Chat is cleaned successfully"}


class ReportRequest(BaseModel):
    customer_ids: list[str]  # List of customer IDs
    entity: AllowedEntity    # Single entity, restricted to AllowedEntity values

sem = asyncio.Semaphore(15)

async def fetch_customer_data(customer_id: str, entities: List[str]):
    """
    Fetch data for multiple entities with semaphore control
    """
    async with sem:
        try:
            result = await get_exported_data_many(customer_id, entities)
            return (customer_id, result)
        except Exception as e:
            logger2.error(f"Error fetching data for customer {customer_id}: {e}")
            return (customer_id, None)


@app.post("/generate-reports-group-old")
async def create_group_reports(request: ReportRequest = Body(...)):
    """Generate group reports for multiple customers with improved asynchrony."""
    start = time.perf_counter()
    customer_ids = request.customer_ids
    uuid = str(uuid4())
    
    # Create directory structure
    folder_path = os.path.join('data', uuid, 'work_data_folder')
    await asyncio.to_thread(os.makedirs, folder_path, exist_ok=True)
    
    # Fetch data for all customers
    entities = ['orders', 'order_products', 'customer']
    tasks = [fetch_customer_data(customer_id, entities) for customer_id in customer_ids]
    results = await asyncio.gather(*tasks)
    print("Step 0 - fetch data:", time.perf_counter() - start)
    
    # Process fetched data
    data_orders, data_products, data_customer, customer_names = await process_fetch_results(results, customer_ids, entities)
    
    # Save data for all entities
    save_tasks = [
        save_customer_data(customer_ids, 'orders', data_orders, uuid, customer_names),
        save_customer_data(customer_ids, 'order_products', data_products, uuid, customer_names),
        save_customer_data(customer_ids, 'customer', data_customer, uuid, customer_names)
    ]
    save_results = await asyncio.gather(*save_tasks)
    print("Step 1 - saved customer data:", time.perf_counter() - start)
    
    # Validate save results
    success_count, failed_customer_names = await validate_save_results(save_results, customer_ids, customer_names)
    
    logger2.info(f"Saved data for: {uuid}")
    
    # Generate file paths
    ord_path, prod_path, customer_path = await generate_file_paths(customer_ids, uuid)
    print("Step 2 - generated file paths:", time.perf_counter() - start)
    
    try:
        # Create user data
        data_created, empty_files = await create_group_user_data(ord_path, prod_path, "test", uuid)
        if not data_created:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": [
                        {
                            "loc": ["body", "data_creation"],
                            "msg": "Report cannot be generated due to data validation errors",
                            "type": "data_validation_error"
                        }
                    ]
                }
            )
        
        # Process empty files
        customer_names_empty = [
            customer_names.get(extract_customer_id(file_path), f"Unknown ({extract_customer_id(file_path)})")
            for file_path in empty_files
        ] if empty_files else []
        
        # Generate analytics reports
        print("Step 3 - before generate report:", time.perf_counter() - start)
        full_report, sectioned_report = await generate_analytics_report(f'data/{uuid}', uuid)
        
        # Save full report
        async with aiofiles.open(f"data/{uuid}/full_report.txt", "w", encoding="utf-8") as f:
            await f.write(full_report)
        print("Step 4 - after generate report:", time.perf_counter() - start)
        
        # Create and return response
        return await create_response(success_count, len(customer_ids), failed_customer_names, customer_names_empty, sectioned_report, full_report, uuid)
    
    except Exception as e:
        logger2.error(f"Report generation failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {
                        "loc": ["server", "report_generation"],
                        "msg": "Internal server error during report generation",
                        "type": "internal_server_error"
                    }
                ]
            }
        )




class AI_Request(BaseModel):
    uuid: str
    prompt: str
    

@app.post("/Ask_ai_many_customers")
async def Ask_ai_many_customers_endpoint(request: AI_Request = Body(...)):
    user_uuid = request.uuid
    prompt = request.prompt

    try:
        # Use AI function to get response
        response = await Ask_ai_many_customers(prompt, user_uuid)
        
        # Check if response indicates an invalid UUID
        if response.get('error') == 'invalid_uuid':
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "detail": [
                        {
                            "loc": ["body", "uuid"],
                            "msg": "Invalid UUID provided",
                            "type": "value_error"
                        }
                    ]
                }
            )
        
        answer = response.get('output')
        cost = response.get('cost', 0.0)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "AI_answer": answer,
                "cost": cost
            }
        )
        
    except Exception as e:
        logger2.error(f"Error executing LLM: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {
                        "loc": ["server", "llm_processing"],
                        "msg": "Internal server error while processing request",
                        "type": "internal_server_error"
                    }
                ]
            }
        )


#____
@app.post("/analyze_routes/{user_id}")
async def analyze_routes(user_id: str):
    user_data_path = f"data/{user_id}"
    orders_csv_path = f"{user_data_path}/oorders.csv"
    customers_csv_path = f"{user_data_path}/raw_data/concatenated_customers.csv"
    
    # Check if user data directory exists
    if not os.path.exists(user_data_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incorrect user ID: {user_id}"
        )
            
    # Check if files exist
    if not os.path.exists(orders_csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orders file not found."
        )
    
    if not os.path.exists(customers_csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customers file not found."
        )
    
    # Analyze customer orders
    result = await analyze_customer_orders_async(orders_csv_path, customers_csv_path)
    
    # Check if result is empty
    if not result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No data available after analysis"
        )
    
    return result


class AI_Request(BaseModel):
    uuid: str
    prompt: str
    

@app.post("/Ask_ai_many_customers")
async def Ask_ai_many_customers_endpoint(request: AI_Request = Body(...)):
    user_uuid = request.uuid
    prompt = request.prompt

    try:
        # Use AI function to get response
        response = await Ask_ai_many_customers(prompt, user_uuid)
        
        # Check if response indicates an invalid UUID
        if response.get('error') == 'invalid_uuid':
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "detail": [
                        {
                            "loc": ["body", "uuid"],
                            "msg": "Invalid UUID provided",
                            "type": "value_error"
                        }
                    ]
                }
            )
        
        answer = response.get('output')
        cost = response.get('cost', 0.0)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "AI_answer": answer,
                "cost": cost
            }
        )
        
    except Exception as e:
        logger2.error(f"Error executing LLM: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {
                        "loc": ["server", "llm_processing"],
                        "msg": "Internal server error while processing request",
                        "type": "internal_server_error"
                    }
                ]
            }
        )


#____updated version
def _sync_comparison_logic(df_1: pd.DataFrame, 
                           df_2: pd.DataFrame, 
                           customer_id_s):
    """
    Internal synchronous function to perform blocking Pandas operations.
    Returns a dictionary with the results.
    
    Note:
    - df_1 is assumed to be the 'orders' DataFrame.
    - df_2 is assumed to be the 'customers' DataFrame.
    """
    
    # 1. Check for required columns
    required_cols_df1 = ['customerId']
    required_cols_df2 = ['combinedid', 'name']
    
    if not all(col in df_1.columns for col in required_cols_df1):
        return {"error": f"df_1 is missing required columns. Needed: {required_cols_df1}"}
    if not all(col in df_2.columns for col in required_cols_df2):
        return {"error": f"df_2 is missing required columns. Needed: {required_cols_df2}"}

    # 2. Get unique IDs from both DataFrames.
    ids_in_df2 = set(df_2['combinedid'])
    ids_in_df1 = set(df_1['customerId'])
    ids_from_list = set(customer_id_s)

    # --- "Empty customers" (names from df_2) ---
    
    # Find IDs that are in df_2 (customers) but not in df_1 (orders)
    missing_in_df1_ids = ids_in_df2 - ids_in_df1
    
    empty_customer_names = []
    if missing_in_df1_ids:
        # Filter df_2 to get rows with these IDs
        missing_df = df_2[df_2['combinedid'].isin(missing_in_df1_ids)]
        # Get unique names
        empty_customer_names = missing_df['name'].unique().tolist()

    # --- "Invalid IDs" (IDs from the input list) ---
    
    # Find IDs from the list that are not in df_2 (customers)
    missing_in_df2_ids = ids_from_list - ids_in_df2
    invalid_ids_list = list(missing_in_df2_ids)

    # --- Format the result ---
    result = {
        "empty_customers": empty_customer_names,
        "invalid_ids": invalid_ids_list
    }
    
    return result

async def check_customer_ids(df_1: pd.DataFrame, 
                           df_2: pd.DataFrame, 
                           customer_id_s):
    """
    Asynchronous wrapper for checking IDs in DataFrames.
    
    Note: df_1 represents orders, and df_2 represents customers.
    
    Executes blocking Pandas logic in a separate thread and
    returns a dictionary (dict) ready for JSON serialization.
    """
    # Running the heavy synchronous function in a separate thread
    result_dict = await asyncio.to_thread(_sync_comparison_logic, df_1, df_2, customer_id_s)
    
    return result_dict

@app.post("/generate-reports-group")
async def create_group_reports_new(request: ReportRequest = Body(...)):
    """Generate group reports for multiple customers with improved asynchrony."""
    start_time = time.perf_counter()
    customer_ids = request.customer_ids
    uuid = str(uuid4())
    
    # Create directory structure
    user_folder = os.path.join('data', uuid, 'work_data_folder')
    await asyncio.to_thread(os.makedirs, user_folder, exist_ok=True)
    
    # Fetch data for all customers
    entities = ['orders', 'order_products', 'customer']
    # File names
    filename_orders = 'one_file_orders.csv'
    filename_products = 'one_file_products.csv'
    filename_customers = 'one_file_customers.csv'
    
    file_path_orders = os.path.join(user_folder,  filename_orders)
    file_path_products = os.path.join(user_folder,  filename_products) 
    file_path_customers = os.path.join(user_folder,  filename_customers) 
    
    print(f"Step 0 - Starting data fetch for id {uuid}: {time.perf_counter() - start_time:.2f}s")
    
    # Fetch data concurrently
    entities_orders = ["orders"]
    entities_products = ["order_products"] 
    entities_customers = ["customer"]
    
    result_1, result_2, result_3 = await asyncio.gather(
        get_exported_data_one_file(customer_ids, entities_orders),
        get_exported_data_one_file(customer_ids, entities_products),
        get_exported_data_one_file(customer_ids, entities_customers)
    )
    
    print(f"Step 1 - Data fetch completed: {time.perf_counter() - start_time:.2f}s")
    
    # Process and save fetched data concurrently
    await asyncio.gather(
        _process_and_save_file_data(result_1, file_path_orders),
        _process_and_save_file_data(result_2, file_path_products),
        _process_and_save_file_data(result_3, file_path_customers)
    )
    
    # Preprocess data
    full_cleaned_orders, full_cleaned_products = await prepared_big_data(
        str(file_path_orders), 
        str(file_path_products)
    )
    
    print(f"Step 2 - Data preprocessing completed: {time.perf_counter() - start_time:.2f}s")
    

    # Save cleaned data concurrently
    cleaned_orders_path =  os.path.join(user_folder,  'cleaned_real_big_orders.csv') 
    cleaned_products_path =  os.path.join(user_folder,  'cleaned_real_big_products.csv')
    
    await asyncio.gather(
        save_df(full_cleaned_orders, str(cleaned_orders_path)),
        save_df(full_cleaned_products, str(cleaned_products_path))
    )
    
    # Read dataframes concurrently
    orders_df, products_df, customer_df = await asyncio.gather(
        read_dataframe_async(str(cleaned_orders_path)),
        read_dataframe_async(str(cleaned_products_path)),
        read_dataframe_async(str(file_path_customers))
    )
    
    # Clean column names
    orders_df.columns = orders_df.columns.str.strip().str.replace('\ufeff', '')
    customer_df.columns = customer_df.columns.str.strip().str.replace('\ufeff', '')
    
    print(f"Step 3 - Data cleaning completed: {time.perf_counter() - start_time:.2f}s")

    # Merge data orders
    merged = orders_df.merge(
        customer_df[['combinedid', 'displayedName']], 
        left_on='customerId', 
        right_on='combinedid', 
        how='left'
    ).rename(columns={'displayedName': 'customer_name', 'customerId_x': 'id'})
    merged = merged.drop('combinedid_y', axis=1).rename(columns={'combinedid_x': 'id'})
    # Save merged data
    merged_path = os.path.join(user_folder,  'file_a_updated.csv') #orders
    await save_df(merged, str(merged_path))

    #products
    customer_map = merged.set_index('id')['customer_name']

    # Додаємо нову колонку в df_B.
    # .map() шукає кожне значення з 'order_id' у 'customer_map'
    # і повертає відповідне 'customer_name'.
    # Якщо відповідника немає (як для 105), він ставить NaN.
    products_df['customer_name'] = products_df['orderId'].map(customer_map)
    # Видалення рядків, де 'customer_name' є NaN (тобто ордера, що були видалені через archived ot canceled status)
    products_df.dropna(subset=['customer_name'], inplace=True)
    products_df.drop(columns=['id'], inplace=True)
    try:

        # Generate analytics reports
        print("Step 4 - before generate report:", time.perf_counter() - start_time)
        from AI.group_customer_analyze.create_report_group_c import new_generate_analytics_report
        full_report, sectioned_report = await new_generate_analytics_report(merged, products_df, customer_df, uuid)
        
        # Save full report
        async with aiofiles.open(f"data/{uuid}/full_report.txt", "w", encoding="utf-8") as f:
            await f.write(full_report)
        print("Step 5 - after generate report:", time.perf_counter() - start_time)
        
        incorrect_ids = await check_customer_ids(merged, customer_df, customer_ids)
        # Create and return response
        return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "incorrect_ids" : incorrect_ids,
            "sectioned_report": sectioned_report,
            "full_report": full_report,
            "uuid": uuid
        }
    )
        #return await create_response('success_count', len(customer_ids), 'failed_customer_names', 'customer_names_empty', sectioned_report, full_report, uuid)
    
    except Exception as e:
        logger2.error(f"Report generation failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": [
                    {
                        "loc": ["server", "report_generation"],
                        "msg": "Internal server error during report generation",
                        "type": "internal_server_error"
                    }
                ]
            }
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000, host='0.0.0.0')