import asyncio
import aiofiles
import os
import pandas as pd
import time

from AI.utils import get_logger

from langchain.agents import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

logger2 = get_logger("logger2", "project_log_many.log", False)

def _create_advice_tool(user_uuid: str):
    """ More detailed information about each customer"""
    try:
       path_for_customers_Overall_report = os.path.join("data", user_uuid, "overall_report.txt")
       
       with open(path_for_customers_Overall_report, "r", encoding="utf-8") as f:
           advice_text = f.read()
    except FileNotFoundError:
        logger2.error("overall_report file not found")
        return None
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(advice_text)
    
    # Create vector store
    emb = OpenAIEmbeddings()
    index = FAISS.from_texts(chunks, emb)
    
    # Define advice retrieval tool
    @tool("AdviceTool")
    def get_data(query: str) -> str:
        '''Use this tool to fetch detailed customer statistics and insights from the overall report. Input should be a specific topic or question.'''
        docs = index.similarity_search(query, k=12)
        return "\n".join(d.page_content for d in docs)
        
    return get_data


async def Ask_ai_many_customers(prompt: str, user_uuid: str):
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        file_path_product = os.path.join('data', user_uuid, 'oorders.csv')
        file_path_orders = os.path.join('data', user_uuid, 'pproducts.csv')
        for encoding in encodings:
            try:
                df1 = pd.read_csv(file_path_product, encoding=encoding, low_memory=False)
                df2 = pd.read_csv(file_path_orders, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                logger2.warning(f"Failed decoding attempt with encoding: {encoding}")

        llm = ChatOpenAI(model='gpt-4.1-mini') #model='o3-mini'  gpt-4.1-mini

        # Create advice tool
        advice_tool = _create_advice_tool(user_uuid)

        # Create agent with extra tool
        agent = create_pandas_dataframe_agent(
            llm,
            [df1, df2],
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True,
            number_of_head_rows=5,
            max_iterations=5,
            extra_tools=[advice_tool] if advice_tool else []  # Add tool if available
        )
         

        try:
            full_report_path = os.path.join('data', user_uuid, 'full_report.md')
            with open(full_report_path, "r") as file:
                full_report = file.read()
        except Exception as e:
            full_report = 'No data given'
            logger2.warning(f"Can not read additional_info.md due to {e} ")
        
        try:
            promo_rules_path = os.path.join('AI', 'group_customer_analyze', 'promo_rules.txt')
            with open(promo_rules_path, "r") as file:
                recommendations = file.read()
        except Exception as e:
            recommendations = 'No data given'
            logger2.warning(f"Can not read additional_info.md due to {e} ")


        formatted_prompt = f"""
        You are an AI assistant which answers users' questions about the data you have 
        providing business insights based on two related datasets and files:  
        - **df1** (Orders) contains critical order-related data or can be user activities data.  
        - **df2** (Products) contains details about products within each order or can be user tasks data.  

        Calculated data for all customers in a joint report: {full_report}

        some additional data that you can use to make recommendations to the customer: {recommendations}
        
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

        -  **Use AdviceTool**: It may already contain the answer to the question. or if the question concerns more detailed data on customers or products. 
        This tool searches the txt file for data that matches the question.

        **Question:**  

        {prompt}"""

        logger2.info("\n===== Metadata =====")
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
            "gpt-5":        {"prompt": 1.25,   "completion": 10.00},
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
        
            logger2.info("Agent for func:  Ask_ai_many_customers")
            logger2.info(f"Input Cost:  ${in_cost:.6f}")
            logger2.info(f"Output Cost: ${out_cost:.6f}")
            logger2.info(f"Total Cost:  ${cost:.6f}")
        
            result['metadata'] = {
                'total_tokens': in_toks+out_toks,
                'prompt_tokens': in_toks,
                'completion_tokens': out_toks,
                'execution_time': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }

        
        for k, v in result['metadata'].items():
            logger2.info(f"{k.replace('_', ' ').title()}: {v}")

        return {"output": result.get('output'), "cost": cost}

    except Exception as e:
        logger2.error(f"Error in AI processing: {str(e)}")
        return {"error": 'invalid_uuid', "cost": 0}