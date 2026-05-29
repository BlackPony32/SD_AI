import pandas as pd
import numpy as np
from datetime import datetime
import os
import asyncio
import aiofiles
from pathlib import Path
from pprint import pprint
from functools import partial
import time 
import re

from AI.utils import get_logger, combine_sections, calculate_cost
from AI.MCP_tools.additional_functions import generate_analytics_report_sectioned
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_suggestions, prompt_mcp_topics_customer_agent, prompt_mcp_suggestions, prompt_mcp_topics_orders_agent, prompt_mcp_topics_catalog_agent

from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession

logger2 = get_logger("logger2", "project_log_many.log", False)


from dotenv import load_dotenv
load_dotenv()

model = 'gpt-5.4-mini' #'gpt-5.4-mini'
llm_model = OpenAIResponsesModel(model=model, openai_client=AsyncOpenAI()) 

@function_tool
def get_prepared_statistics(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'mcp_get_prepared_statistics' called ")
    data_path = f"data/{user_id}/full_report_test.md"
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


async def create_agent_sectioned(USER_ID, topic, statistics, agent) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        PROMPT_DEFAULT = "You are a helpful assistant. Answer the user's query based on the data provided."
        topic_instruction_map = {
            "customers_agent": prompt_mcp_topics_customer_agent,
            "orders_agent": prompt_mcp_topics_orders_agent,
            "catalog_agent": prompt_mcp_topics_catalog_agent,
            #"billing": PROMPT_BILLING,     # maps to prompt3
        }

        # 3. Select the correct instruction
        selected_instruction = topic_instruction_map.get(agent, PROMPT_DEFAULT)
        statistics_topic = statistics.get(topic, "No statistics available for this topic.")

        instructions = await selected_instruction(USER_ID, topic, statistics_topic)

        agent = Agent(
            name="Customer_Orders_Assistant",
            instructions=instructions,
            model=llm_model
        )
        print(" New create_agent_sectioned are ready.")
    except Exception as e:
        print("create_agent_sectioned error: ", e)
    return agent

async def create_agent_suggestions(USER_ID) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        instructions = await prompt_mcp_suggestions(USER_ID)

        agent = Agent(
            name="Customer_Orders_Assistant",
            instructions=instructions,
            model=llm_model,
            tools=[get_prepared_statistics]
        )
        print(" New create_agent_suggestions are ready.")
    except Exception as e:
        print("create_agent_suggestions error: ", e)
    return agent

async def process_standard_topic(topic, orders_path, products_path, customers_path, catalog_path, uuid, agent):
    """Logic for standard analysis topics."""
    try:
        start = time.perf_counter()
        
        # 1. Generate Statistics
        statistics_dict = await generate_analytics_report_sectioned(
            orders_path, products_path, customers_path, catalog_path, agent_type=agent, report_type=topic
        )
        
        # Safely extract the section
        statistics = statistics_dict.get('sections', {})
        if isinstance(statistics, str):
            statistics_topic = statistics
        else:
            statistics_topic = statistics.get(topic, "No statistics available for this topic.")
        
        # 2. CHECK FOR BROKEN/EMPTY STATISTICS
        error_phrases = [
            "currently unavailable",
            "not enough data",
            "can not generate report",
            "no statistics generated",
            "no statistics available"
        ]
        
        is_broken = False
        if isinstance(statistics_topic, str):
            stat_lower = statistics_topic.lower()
            if any(phrase in stat_lower for phrase in error_phrases):
                is_broken = True

        # 3. Conditional AI Analysis & Early Exit
        if is_broken:
            print(f"Topic '{topic}': Skipping AI analysis. Returning error message directly to user.")
            # Return the predefined error message directly so the front-end displays it
            return statistics_topic

        # --- Everything below this line ONLY runs if statistics are valid ---

        # 4. Generate AI Analysis
        agent = await create_agent_sectioned(uuid, topic, statistics, agent)
        runner = await Runner.run(
            agent, 
            input="Analyze my data and write useful tips for business"
        )
        answer = runner.final_output
        calculate_cost(runner, model=model)

        # 5. Combine Sections
        sectioned_answer = await combine_sections(topic, statistics_topic, answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        
        if isinstance(sectioned_answer, dict):
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        
        return sectioned_answer

    except Exception as e:
        logger2.error(f"Error in standard topic '{topic}': {e}")
        return None

async def process_suggestions_topic(topic, orders_path, products_path, customers_path, catalog_path, uuid, agent):
    """Logic for suggestions analysis topics."""
    try:
        start = time.perf_counter()

        # 1. Generate Statistics
        statistics_of_topic = await generate_analytics_report_sectioned(
            orders_path, products_path, customers_path, catalog_path,  agent_type=agent, report_type='full_report'
        )

        async with aiofiles.open(f"data/{uuid}/full_report_test.md", "w", encoding="utf-8") as f:
                        await f.write(statistics_of_topic.get('full_report', '') )

        agent = await create_agent_suggestions(uuid)
        runner = await Runner.run(
            agent, 
            input="Analyze my data and write useful tips for business"
        )

        answer = runner.final_output
        answer = f"<div id=\"suggestions-block\">\n\n{answer}\n</div>"
        #print(answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        calculate_cost(runner, model=model)

        sectioned_answer = {'suggestions_div' : answer}
        if isinstance(sectioned_answer, dict):
            # Try to get the value using the topic as key, otherwise take the first value
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        return sectioned_answer

    except Exception as e:
        logger2.error(f"Error in suggestions topic '{topic}': {e}")
        return None


async def worker(semaphore, topic, orders_path, products_path, customers_path, catalog_path, uuid, agent):
    """
    Router function: Decides which logic to run based on the topic name,
    constrained by the semaphore.
    """
    async with semaphore:
        #print(f"Processing: {topic}")
        
        if topic == "suggestions_div":
            return await process_suggestions_topic(topic, orders_path, products_path, customers_path, catalog_path, uuid, agent)
        else:
            return await process_standard_topic(topic, orders_path, products_path, customers_path, catalog_path, uuid, agent)

import asyncio

import asyncio
# Assuming logger2 is imported here or passed globally

async def main_batch_process(
    orders_path, 
    products_path, 
    customers_path, 
    catalog_path, 
    uuid, 
    agent, 
    specific_topic=None
):
    try:
        # Changed print to logger for consistency
        logger2.info(f"Starting batch process: {orders_path}, {products_path}, {customers_path}, {catalog_path}, {uuid}, {agent}, {specific_topic}")
        
        TOPIC_CONFIG = {
            "customers_agent": [
                "churn_report",
                "refined_opportunity_report",
                "top_customers_report",
                "visits_report",
                "suggestions_div"
            ],
            "catalog_agent": [
                "functional_product_analysis",
                "product_performance",
                "sales_trends_report",
                "bundle_performance_report",
                "top_3_sales_breakdown",
                "time_based_product_report",
                "suggestions_div"
            ],
            "orders_agent": [
                "key_metrics_report",
                "sales_performance_report",
                "discount_report",
                "payment_status_report",
                "fulfillment_report",
                "sales_trends_report",
                "suggestions_div"
            ] 
        }
        
        # 1. Override the topics list if a specific topic is requested
        all_agent_topics = TOPIC_CONFIG.get(agent, [])
        
        if specific_topic:
            if specific_topic in all_agent_topics:
                topics = [specific_topic]
            else:
                logger2.warning(f"Topic '{specific_topic}' is not valid for agent '{agent}'.")
                friendly_msg = f"The requested report topic '{specific_topic}' is currently unavailable for this agent."
                return friendly_msg, {"error": friendly_msg}
        else:
            topics = all_agent_topics

        if not topics:
            logger2.warning(f"No topics found for agent '{agent}'.")
            return "No reports available for this configuration.", {}

        # Limit concurrency to 10
        sem = asyncio.Semaphore(10)
        tasks = []
        for topic in topics:
            task = asyncio.create_task(
                worker(sem, topic, orders_path, products_path, customers_path, catalog_path, uuid, agent)
            )
            tasks.append(task)

        logger2.info(f"Starting {len(topics)} topics...")
        
        # return_exceptions=True prevents gather from crashing if a single worker fails
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sectioned_report = {}
        for topic, result in zip(topics, results):
            if isinstance(result, Exception):
                # Log the actual technical error with full traceback
                logger2.error(f"Worker failed for topic '{topic}' (Agent: {agent}): {str(result)}", exc_info=result)
                
                # Assign a user-friendly message for this specific section
                clean_topic_name = topic.replace('_', ' ').title()
                sectioned_report[topic] = f"\n> **Notice:** We encountered an issue while generating the {clean_topic_name}. Please try again later.\n"
            else:
                #print(f"Worker succeeded for topic '{topic}' (Agent: {agent}): {str(result)}")
                sectioned_report[topic] = result
        
        # Compile the Report
        report_parts = []
        for key in topics:
            content = sectioned_report.get(key)
            if content:
                report_parts.append(str(content))
        
        def clean_markdown(text: str) -> str:
            if not text:
                return ""
            return text.replace('\n---\n', '\n\n').replace('\n---', '')

        raw_full_report = "\n".join(report_parts).strip()
        final_clean_report = await asyncio.to_thread(clean_markdown, raw_full_report)
        
        # Clean sections, ensuring we convert to string just in case
        clean_sections = await asyncio.to_thread(
            lambda: {k: clean_markdown(str(v)) for k, v in sectioned_report.items()}
        )

        return final_clean_report, clean_sections

    except Exception as e:
        # Global fallback for unexpected errors (e.g., missing files, memory issues, dict errors)
        logger2.error(f"Critical error in main_batch_process for UUID {uuid}: {str(e)}", exc_info=True)
        
        user_friendly_error = (
            "We encountered an unexpected error while generating your complete report. "
            "Our team has been notified. Please try again shortly."
        )
        return user_friendly_error, {"error": user_friendly_error}

if __name__ == "__main__":
    #report = await generate_analytics_report_sectioned(
    #    orders_path="data\\FULL_DIST_TEST\\cleaned_real_big_orders.csv",
    #    products_path="data\\FULL_DIST_TEST\\cleaned_real_big_products.csv",
    #    customers_path="data\\FULL_DIST_TEST\\work_data_folder\\raw_file_customers.csv",
    #    catalog_path="data\\FULL_DIST_TEST\\cleaned_catalog.csv",
    #    agent_type="catalog_agent",
    #    report_type="full_report"
    #)

    uuid = "FULL_DIST_TEST"
    agent = "orders_agent"

    base_dir = Path("data") / uuid

    orders_path = base_dir / "cleaned_orders.csv"
    products_path = base_dir / "cleaned_products.csv"
    customers_path = base_dir / "cleaned_customers.csv"
    catalog_path = base_dir / "cleaned_catalog.csv"
    specific_topic='sales_trends_report'

    report_sectioned_ai = asyncio.run(main_batch_process(orders_path, products_path, customers_path, catalog_path, uuid, agent))
    #print(report_sectioned_ai[0])