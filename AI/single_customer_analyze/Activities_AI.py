import pandas as pd
import os
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.concurrency import run_in_threadpool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
import time
from typing import Literal

load_dotenv()

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

def parse_report(report_text):
    sections = report_text.strip().split('\n---\n')
    report_dict = {}

    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        # Extract and clean the header
        header = lines[0].replace('### ', '').strip().lower()
        content = '\n'.join(lines[1:]).strip()
        
        # Map headers to keys based on keywords
        key = None
        if 'task' in header:
            key = 'task'
        elif 'note' in header:
            key = 'note'
        elif 'activit' in header:  # Covers "activity" or "activities"
            key = 'activities'
        elif 'conclusion' in header or 'recommendation' in header:
            key = 'recommendations'
        
        if key:
            report_dict[key] = content
    
    return report_dict

async def read_file_async(file_path: str) -> str:
    try:
        async with aiofiles.open(file_path, 'r') as file:
            return await file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


# New Activities AI block
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession
from AI.group_customer_analyze.Agents_rules.prompts import prompt_activities_single
llm_model = OpenAIResponsesModel(model='gpt-4.1-mini', openai_client=AsyncOpenAI()) 

async def create_agent_activities_single(USER_ID, notes, tasks, activities) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        instructions = await prompt_activities_single(USER_ID, notes, tasks, activities)

        agent = Agent(
            name="Customer_Orders_Assistant",
            instructions=instructions,
            model=llm_model
        )
        print(" New create_agent_suggestions are ready.")
    except Exception as e:
        print("create_agent_suggestions error: ", e)
    return agent


async def run_agent_activities(uuid, notes, tasks, activities):
    """Logic for standard analysis topics."""
    try:
        agent = await create_agent_activities_single(uuid,notes, tasks, activities)

        runner = await Runner.run(
            agent, 
            input="Based on the data return response"
        )

        answer = runner.final_output
        return answer

    except Exception as e:
        print(f"Error in run_agent_activities topic : {e}")
        return None


async def process_ai_activities_request(customer_id: str) -> dict:
    """Get AI analyzing of notes, tasks, activities"""
    error_response = {
        'model_answer': 'Could not analyze the activity of your customer.',
        'total_tokens': None,
        'prompt_tokens': None,
        'completion_tokens': None,
        'execution_time': None,
        'model': None
    }


    start_time = time.time()
    base_path = Path(f"data/{customer_id}")
    
    try:
        # Read files concurrently
        file_paths = [
            base_path / "report_activities.md",
            base_path / "report_notes.md",
            base_path / "report_task.md"
        ]
        
        # Create tasks for parallel file reading
        read_tasks = [read_file_async(path) for path in file_paths]
        report_activities, report_notes, report_task = await asyncio.gather(*read_tasks)

        error_message = "Sorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
        if not report_activities.strip() or report_activities.strip() == error_message.strip():
            response = {
            'model_answer': "Sorry, we were unable to analyze the activity due to not enough data."
            }
            return response, {}

        # Process report_notes
        if not report_notes.strip() or report_notes.strip() == error_message.strip():
            report_notes = "No notes available."

        if not report_task.strip() or report_task.strip() == error_message.strip():
            report_task = "No tasks available."
        
        answer = await run_agent_activities(customer_id, report_notes, report_task, report_activities)


    except Exception as e:
        logger.error(f"File read error: {str(e)}")
        return error_response, {}


    try:
        execution_time = time.time() - start_time

        
        response = {
            #'prompt_tokens': answer.usage_metadata.get('input_tokens', 0),
            #'completion_tokens': answer.usage_metadata.get('output_tokens', 0),
            'execution_time': f"{execution_time:.2f} seconds",
            'model': 'gpt-4.1-mini',
            'model_answer': answer
        }
        #print(answer.content)
        section_report = parse_report(response.get('model_answer'))
        return response, section_report
        
    except AttributeError as e:
        logger.error(f"Response parsing error: {str(e)}")
        return error_response, {}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return error_response, {}