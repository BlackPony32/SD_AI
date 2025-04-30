#from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain_openai import ChatOpenAI
#import pandas as pd
#from dotenv import load_dotenv
#load_dotenv()
#
#
#df1 = pd.read_csv('products.csv')
#df2 = pd.read_csv('orders.csv')
#llm = ChatOpenAI(model = 'gpt-4o')
#
#agent = create_pandas_dataframe_agent(
#    llm, [df1, df2], agent_type="openai-tools", verbose=True, allow_dangerous_code=True
#)
#result = agent.invoke(
#    {
#        "input": """
#            Analyze all sales trends and product performance using the following dataset.
#            Provide insights on revenue by product/category, order frequency, customer behavior, and inventory value.
#            Identify best/worst-selling products and any patterns in sales over time.
#            """
#    }
#)
#print(result.get('output'))

#test = llm.invoke("hello ai")
#print(test)


from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('project_log.log'), logging.StreamHandler()]
)

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data with validation and error handling"""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise


def Ask_AI():
    load_dotenv()
    
    try:
        # Load and preprocess data
        logging.info("Loading data...")
        df1 = load_data('products.csv')
        df2 = load_data('orders.csv')
        df = df1.merge(df2, on="orderId", how="left")
        df.to_csv('data.csv',index=False)
        # Initialize LLM with configurable parameters
        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        
        # Create agent with enhanced configuration
        agent = create_pandas_dataframe_agent(
            llm,
            [df1, df2],
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=16
        )
        
        # Structured prompt template
        prompt = """
            It is important to remember:
            - Don't write from df1 or df2, but rather use the form according to the user's data.
            Provide useful business analytics from the received files.
            - Do not make references to meta-information such as file names or columns - the result is a full-fledged data conclusion that should answer the following questions:
            - Provide insights on revenue by product/category, order frequency, customer behavior, and inventory value.
            - Identify best/worst-selling products and any patterns in sales over time.
                        No visualizations or table output, only markdown text.

            An example of the basis of a good answer:
            ### Revenue by Product: ...
            ### Order Frequency:...
            ### Inventory Value:...
            ### Best and Worst-Selling Products:...

            ### Sales Trends Over Time:

            ### Patterns and Conclusions:
            """
        
        prompt = """
        df1 about order(more important data) df2 about product inside order.
        It is important to remember:
            - Try not to make your answer too long.
            - Don't write from df1 or df2, but rather use the form: according to the user's data.
            - Do not make references to meta-information such as file names or columns - the result is a full-fledged data
            conclusion that should answer the following questions:
            
        
        Question is:
        Write a business analysis about Vitamin C"""
        # Track tokens and execution time
        with get_openai_callback() as cb:
            agent.agent.stream_runnable = False
            start_time = time.time()
            result = agent.invoke({"input": prompt})
            execution_time = time.time() - start_time
            
            # Add metadata to result
            result['metadata'] = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'execution_time': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }
        
        logging.info("\n===== Analysis Result =====")
        print(result.get('output'))
        
        logging.info("\n===== Metadata =====")
        for k, v in result['metadata'].items():
            print(f"{k.replace('_', ' ').title()}: {v}")
            
        return result
    
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

#Ask_AI()