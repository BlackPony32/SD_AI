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
#from AI.First_summary__old import generate_statistics
from AI.Order_analytics import generate_sales_report
from AI.create_process_data import create_user_data

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import time

from concurrent.futures import ProcessPoolExecutor



app = FastAPI()

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

# Initialize at app startup
@app.on_event("startup")
async def startup_event():
    app.state.process_executor = ProcessPoolExecutor(max_workers=4)  # Adjust based on CPU cores

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    app.state.process_executor.shutdown(wait=True)

@app.post("/generate-reports/{customer_id}")
async def create_reports(customer_id: str):
    try:
        output_dir = Path("data")

        try:
            data_dir = Path("data") / customer_id
        except Exception as e:
            logger.error("Error generating customer report:", e)    
        
        orders_path = data_dir / "orders.csv"
        products_path = data_dir / "orderProducts.csv"

        # Generate statistics
        #report = await generate_statistics(orders_path, products_path) #NOTE old
        
        await create_user_data(orders_path, products_path, data_dir)
        report = await generate_sales_report(orders_path, products_path)
        
        #print(report) #NOTE can be a flag how to return status (if bad report status bad)
        return {
            "status": "success",
            "customer_id": customer_id,
            "file_path_product": products_path,
            "file_path_orders": orders_path,
            "report": report
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

from functools import partial

async def Ask_AI(prompt: str, file_path_product, file_path_orders):
    load_dotenv()
    
    try:
        process_func = partial(
            _process_ai_request,
            prompt=prompt,
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


def _process_ai_request(prompt, file_path_product, file_path_orders):
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
            number_of_head_rows=10,
            max_iterations=10
        )
        
        formatted_prompt = f"""
        You are an AI assistant providing business insights based on two related datasets:  
        - **df1** (Orders) contains critical order-related data.  
        - **df2** (Products) contains details about products within each order.  

        **Important Rules to Follow:**  
        - **Unique Values:** When answering questions about orders or products, always consider unique values.  
        - **Neutral Wording:** Do not mention "df1" or "df2" in your response. Instead, phrase answers as "According to the user's data."  
        - **No Column/File References:** Do not refer to specific file names or column names—focus on insights and conclusions.  
        - **Well-Structured Markdown Formatting:** Ensure responses are clear and organized using appropriate Markdown formatting.  
        - **No Code or Visualizations:** Do not include Python code or suggest data visualizations in your answers.  

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

        logger.info("\n===== Metadata =====")
        for k, v in result['metadata'].items():
            logger.info(f"{k.replace('_', ' ').title()}: {v}")
            
        input_tokens = result['metadata'].get('prompt_tokens')
        output_tokens = result['metadata'].get('completion_tokens')
        input_cost = (input_tokens / 1000) * 0.0025
        output_cost = (output_tokens / 1000) * 0.01
        total_cost = input_cost + output_cost
        return {"output": result, "cost": total_cost}

    except Exception as e:
        logger.error(f"Error in AI processing: {str(e)}")


class ChatRequest(BaseModel):
    prompt: str

@app.post("/Ask_ai")
async def ask_ai(request: ChatRequest, file_path_product, file_path_orders):
    prompt = request.prompt

    try:
        response = await Ask_AI(prompt, file_path_product, file_path_orders)
        answer = response.get('output')
        cost = response.get('cost', 0.0)
    except Exception as e:
        logger.error(f"Error executing LLM: {str(e)}")
        answer = "Cannot answer this question"
        cost = 0.0
    
    return {"response": answer, "prompt": prompt, "cost": cost}

@app.get("/logs/last/{num_lines}", response_class=PlainTextResponse)
async def get_last_n_log_lines(num_lines: int):
    try:
        LOG_FILE = "project_log.log"
        with open(LOG_FILE, "r") as file:
            lines = file.readlines()

        if not lines:
            raise HTTPException(status_code=404, detail="Log file is empty.")

        # Get the last `num_lines` lines
        last_lines = lines[-num_lines:]
        return "".join(last_lines)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')