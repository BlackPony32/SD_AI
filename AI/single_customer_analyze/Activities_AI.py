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

async def process_ai_activities_request(customer_id: str) -> dict:
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
        # Else, keep report_notes as is

        # Process report_task
        if not report_task.strip() or report_task.strip() == error_message.strip():
            report_task = "No tasks available."
        
    except Exception as e:
        logger.error(f"File read error: {str(e)}")
        return error_response, {}

    try:
        llm = ChatOpenAI(model='o3-mini', verbose=False)
    except Exception as e:
        logger.error(f"LLM initialization error: {str(e)}")
        return error_response, {}

    formatted_prompt = f"""
        Use these 3 files and build a conclusion useful for your sales business based on them.

        Data from notes: {report_notes}
        Data from tasks: {report_task}
        Data from activities: {report_activities}
        If “No notes available.” or “No tasks available.” then skip analyzing it and write “Not enough data to analyze”.


        The answer format is a text in markdown format that highlights everything important, so do not make long response.
        Use the data provided to you to make all the analytics important for your business by extracting conclusions from the statistics, assigning tasks to the blocks: 
        
	----
        Analyze the customer's task:
        -analysis of them, everything that is useful to the business without overloading it.
        -Information from the file itself by task.
        ----
        Analyze the customer's Notes:
        -analysis of them, everything that is useful to the business without overloading it.
        -Information from the file itself on the tasks.
        ----
        An overview of the created activities:
        - focus on the creation and execution of orders
        ----
        Conclusion on when and to whom it is better to go to the store to create a re-order
        ---
	
        Have to understand -
        **Important Rules to Follow:**  
        - **Unique Values:** When answering questions about orders or products, always consider unique values.  
	- **No Column/File References:** Do not refer to specific file names or column names—focus on insights and conclusions.  
        - **Well-Structured Markdown Formatting:** Ensure responses are clear and organized using appropriate Markdown formatting.  
        - **No Code or Visualizations:** Do not include Python code or suggest data visualizations in your answers.  
        - If you are sure that the question has nothing to do with the data, answer - "Your question is not related to the analysis of your data, please ask another question."
        - Use am/pm time format.
        - Don't use the terms burnout or similar and do not focus on overworking people, often the work is done by one person, the main thing is to analyze the data and the work done.
        - Distributor and representatives within the team, it can be one person, but only one person goes to the stores, one from the office (replace the distributor with an Admin but don't mention this change).

    The time to visit should be prioritized:
    1) The most recent notes that indicate this
    2) Specific notes.
    3) Calculated data based on created orders.
    4) Other material details

	GOOD structure example(only as example to look):
        ### Analyze the Customer's Task  
        **Key Insights:**
        - There is only 1 task in the system, and it is currently pending and overdue.
        - The task, titled "Follow Up about new orders Call/Visit," has a due date of October 10, 2024, at 3:59 AM.    
        - It carries a MEDIUM priority and remains unassigned for both the representative and distributor, highlighting a gap in task delegation and a potential delay in following up on new orders.

        **Actionable Data:**
        - Immediate attention is warranted to reassign and complete this order follow-up to ensure outreach does not fall further behind.

        ---

        ### Analyze the Customer's Notes
        Not enough data to analyze.

        ---

        ### Overview of Created Activities
        **Key Insights:**
        - A total of 38 activities have been recorded, with 34 orders created, indicating a strong focus on order activities (approximately 89.5%).      
        - Danny Williams is the sole representative involved, responsible for check-ins and some unknown activity types, which shows high dependency on a single individual.
        - Activity trends over the months illustrate consistent order creation, with order activities observed in each month from September 2024 through April 2025.
        - Peak activity hours occur between 8:00 PM and 9:00 PM UTC, aligning with typical business operations when converted to American time zones (around 3:00 PM to 4:00 PM Eastern Time).

        **Actionable Data:**
        - The concentration of activities in order creation suggests that efforts should continue to streamline order management.
        - Expanding task assignments to include additional representatives or distributors could help balance the workload currently shouldered by Danny Williams.

        ---

        ### Conclusion: Optimal Timing for Re-Order Store Visits
        **When to Visit:**
        - Considering the peak activity window, it is advisable to schedule re-order visits around 3:00 PM to 4:00 PM Eastern Time to align with high order processing periods.
        - Addressing the overdue task promptly is critical; thus, visit timing should be adjusted to incorporate immediate follow-ups as early as possible.

        **Who to Engage:**
        - Currently, Danny Williams is the only active representative. However, due to his heavy involvement, incorporating additional team members or distributors could prevent bottlenecks and ensure a more efficient follow-up process.
        - Assigning the overdue "Follow Up about new orders Call/Visit" task to a suitable team member or distributor can expedite the re-order process and optimize customer engagement.

        **Recommendations:**
        - Reassign the overdue task promptly to either a distributor or another available representative to ensure timely follow-up.
        - Schedule store visits during the identified peak period (early to mid-afternoon Eastern Time) and ensure that multiple team members are engaged to reduce reliance on a single representative. 
        """

    try:
        # Run LLM in thread pool to avoid blocking event loop
        answer = await run_in_threadpool(llm.invoke, formatted_prompt)
    except Exception as e:
        logger.error(f"LLM invocation error: {str(e)}")
        return error_response, {}

    try:
        execution_time = time.time() - start_time
        total_tokens = answer.usage_metadata.get('total_tokens', 0)
        
        response = {
            'total_tokens': total_tokens,
            'prompt_tokens': answer.usage_metadata.get('input_tokens', 0),
            'completion_tokens': answer.usage_metadata.get('output_tokens', 0),
            'execution_time': f"{execution_time:.2f} seconds",
            'model': llm.model_name,
            'model_answer': answer.content
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