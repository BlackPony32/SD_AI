import asyncio
import csv
from collections import defaultdict


from functools import partial
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import time


import os
import numpy as np
from AI.utils import get_logger

import pandas as pd
from collections import defaultdict


logger2 = get_logger("logger2", "project_log_many.log", False)



def process_data(uuid):
    """Process customer, order, and product data, saving the result to 'final_data.csv' with left joins to retain all product rows."""
    
    # Load datasets with selected columns "data/{uuid}/raw_data/concatenated_customers.csv"
    customers = pd.read_csv(f"data/{uuid}/work_data_folder/one_file_customers.csv", usecols=[
        'name', 'status',
        'billingAddress_formatted_address', 'billingAddress_street', 'billingAddress_appartement',
        'billingAddress_city', 'billingAddress_state', 'billingAddress_zip',
        'billingAddress_lat', 'billingAddress_lng',
        'shippingAddress_formatted_address', 'shippingAddress_street', 'shippingAddress_appartement',
        'shippingAddress_city', 'shippingAddress_state', 'shippingAddress_zip',
        'shippingAddress_lat', 'shippingAddress_lng',
        'territory_name', 'tags_tag_tag', 'totalOrdersVolumes'
    ]).rename(columns={
        'name': 'customer_name'
    })

    orders = pd.read_csv(f"data/{uuid}/oorders.csv", usecols=[
        'id', 'createdAt', 'orderStatus', 'paymentStatus',
        'deliveryStatus', 'totalAmount', 'customer_name'
    ]).rename(columns={
        'id': 'order_id',
        'createdAt': 'order_date',
        'totalAmount': 'order_total'
    })

    products = pd.read_csv(f"data/{uuid}/pproducts.csv", usecols=[
        'orderId', 'name', 'sku', 'manufacturerName',
        'productCategoryName', 'quantity', 'price',
        'itemDiscountAmount', 'amount', 'totalAmount',
        'product_variant'
    ]).rename(columns={
        'name': 'product_name',
        'amount': 'line_item_total',
        'totalAmount': 'Full_cost_withDisc'
    })

    # Clean merge keys by stripping whitespace
    products['orderId'] = products['orderId'].str.strip()
    orders['order_id'] = orders['order_id'].str.strip()
    orders['customer_name'] = orders['customer_name'].str.strip()
    customers['customer_name'] = customers['customer_name'].str.strip()

    # Create new product identifier
    products['product_identifier'] = products['product_name'].astype(str) + ' - ' + products['sku'].astype(str)

    # Create product analysis columns
    products['net_unit_price'] = products['price'] - products['itemDiscountAmount']
    products['discount_pct'] = (products['itemDiscountAmount'] / products['price']).replace([np.inf, -np.inf], 0) * 100
    products['is_discounted'] = products['itemDiscountAmount'] > 0

    # Diagnostic: Print initial row counts
    #print(f"Products rows: {len(products)}")
    #print(f"Orders rows: {len(orders)}")
    #print(f"Customers rows: {len(customers)}")

    # First merge: Left join to retain all products
    merged = pd.merge(
        products,
        orders,
        left_on=['orderId'],
        right_on=['order_id'],
        how='left'  # Retain all products, even without matching orders
    )
    #print(f"Rows after products-orders merge: {len(merged)}")

    # Diagnostic: Check for unmatched customer_name in merged dataframe
    #merged_customers = set(merged['customer_name'])
    #customer_names = set(customers['customer_name'])
    #missing_customers = merged_customers - customer_names
    #if missing_customers:
    #    print(f"Customers not in customers df: {len(missing_customers)}", missing_customers)

    # Second merge: Left join to retain all rows from merged dataframe
    final_df = pd.merge(
        merged,
        customers,
        on='customer_name',
        how='left'  # Retain all rows, even without matching customers
    )
    #print(f"Final rows: {len(final_df)}")

    # Convert date column
    final_df['order_date'] = pd.to_datetime(final_df['order_date'], errors='coerce')  # Handle missing dates

    # Cleanup
    final_df.drop(columns=['orderId'], inplace=True, errors='ignore')  # Drop redundant column if exists

    # Save to CSV
    final_df.to_csv(f'data/{uuid}/final_data.csv', index=False)

    # Optional: Print total revenue and order count for validation
    total_revenue = final_df['line_item_total'].sum()
    order_count = final_df['order_id'].nunique()
    #print(f"Total Revenue (sum of line_item_total): {total_revenue}")
    #print(f"Order Count (unique order_id): {order_count}")


def generate_report(uuid):
    """Generate a sales analysis report using pandas with correct revenue calculation"""
    def format_currency(amount):
        return f"${amount:,.2f}"

    # Read and preprocess data
    df = pd.read_csv(f'data/{uuid}/final_data.csv')
    
    # Clean data - focus on the columns we need
    df['Full_cost_withDisc'] = pd.to_numeric(df['Full_cost_withDisc'], errors='coerce').fillna(0)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['manufacturerName'] = df['manufacturerName'].fillna('Unknown')
    
    # Create state-level metrics using actual revenue (Full_cost_withDisc)
    # ------------------------------------------------------------
    # Get unique orders for order-level metrics
    unique_orders = df[['order_id', 'shippingAddress_state', 'orderStatus']].drop_duplicates()
    
    # State aggregations using actual revenue
    state_agg = df.groupby('shippingAddress_state').agg(
        total_sales=('Full_cost_withDisc', 'sum'),
        num_orders=('order_id', pd.Series.nunique),
    ).reset_index()
    
    # Order status counts
    status_counts = unique_orders.groupby(['shippingAddress_state', 'orderStatus']).size().unstack(fill_value=0)
    
    # Customer-product relationships
    customer_data = df.groupby(['shippingAddress_state', 'customer_name']).agg(
        customer_products=('product_name', set),
        total_spent=('Full_cost_withDisc', 'sum')
    ).reset_index()

    # Product revenues per state
    product_revenues_per_state = df.groupby(['shippingAddress_state', 'product_name'])['Full_cost_withDisc'].sum()

    # Rebuild state_data dictionary
    state_data = {}
    for state in state_agg['shippingAddress_state'].unique():
        state_row = state_agg[state_agg['shippingAddress_state'] == state].iloc[0]
        state_dict = {
            'total_sales': state_row['total_sales'],
            'order_ids': set(unique_orders[unique_orders['shippingAddress_state'] == state]['order_id']),
            'order_statuses': status_counts.loc[state].to_dict() if state in status_counts.index else {},
            'products': defaultdict(float),
            'customers': {}
        }
        
        # Add product revenues for this state
        if state in product_revenues_per_state.index:
            products_for_state = product_revenues_per_state.loc[state]
            if isinstance(products_for_state, pd.Series):
                for product, revenue in products_for_state.items():
                    state_dict['products'][product] += revenue
            else:  # Single product case
                state_dict['products'][products_for_state.index[0]] = products_for_state.iloc[0]
        
        # Add customer information
        state_customers = customer_data[customer_data['shippingAddress_state'] == state]
        for _, row in state_customers.iterrows():
            state_dict['customers'][row['customer_name']] = {
                'products': row['customer_products'],
                'total_spent': row['total_spent']
            }
        
        state_data[state] = state_dict

    # Create manufacturer metrics using actual revenue
    # ------------------------------------------------------------
    manufacturer_agg = df.groupby('manufacturerName').agg(
        total_sales=('Full_cost_withDisc', 'sum'),
        total_units=('quantity', 'sum')
    ).reset_index()
    
    manufacturer_data = {}
    for _, row in manufacturer_agg.iterrows():
        manufacturer_data[row['manufacturerName']] = {
            'total_sales': row['total_sales'],
            'total_units': row['total_units']
        }

    # Generate report (same structure as original)
    # ------------------------------------------------------------
    report_lines = ["Sales Analysis Report", "="*40, ""]
    
    # Diagnostic output
    total_revenue = df['Full_cost_withDisc'].sum()
    report_lines.append(f"Diagnostics: Total revenue = {format_currency(total_revenue)}")
    report_lines.append(f"Based on {len(df)} line items and {df['order_id'].nunique()} orders\n")
    
    # 1. State analysis
    report_lines.append("Sales by State:")
    report_lines.append("-"*40)
    customer_recommendations = []

    for state, data in sorted(state_data.items()):
        total_sales = data['total_sales']
        num_orders = len(data['order_ids'])
        avg_order_value = total_sales / num_orders if num_orders else 0

        # Status analysis
        status_counts = data['order_statuses']
        status_notes = []
        if num_orders > 0:
            pending_ratio = status_counts.get('PENDING', 0) / num_orders
            unfulfilled_ratio = status_counts.get('UNFULFILLED', 0) / num_orders
            if pending_ratio > 0.4:
                status_notes.append(f"high pending orders ({pending_ratio:.0%})")
            if unfulfilled_ratio > 0.3:
                status_notes.append(f"fulfillment delays ({unfulfilled_ratio:.0%})")
        note = f" ({', '.join(status_notes)})" if status_notes else ""

        # State summary
        report_lines.append(
            f"{state}: {format_currency(total_sales)} total sales | "
            f"{num_orders} orders | {format_currency(avg_order_value)} avg. order{note}"
        )

        # Top products (up to 3)
        products = sorted(data['products'].items(), key=lambda x: x[1], reverse=True)[:3]
        if products:
            top_products_str = ", ".join([f"{prod} ({format_currency(rev)})" for prod, rev in products])
            report_lines.append(f"  Top Products: {top_products_str}")
            
            # Identify customers missing any top product
            for product, _ in products:
                for customer, cdata in data['customers'].items():
                    if product not in cdata['products']:
                        customer_recommendations.append({
                            'customer': customer,
                            'state': state,
                            'product': product,
                            'potential': 0.15 * cdata['total_spent'],
                            'spent': cdata['total_spent']
                        })

    # 2. Product category analysis
    report_lines.extend(["", "Product Category Performance:", "-"*40])
    sorted_manufacturers = sorted(manufacturer_data.items(), key=lambda x: x[1]['total_sales'], reverse=True)

    # Top manufacturers
    for i, (manufacturer, data) in enumerate(sorted_manufacturers[:2]):
        total_sales = data['total_sales']
        total_units = data['total_units']
        avg_price = total_sales / total_units if total_units else 0
        report_lines.append(
            f"{manufacturer}: {format_currency(total_sales)} sales | "
            f"{total_units} units | {format_currency(avg_price)} avg price"
        )

    # Niche products
    niche_sales = sum(data['total_sales'] for _, data in sorted_manufacturers[2:])
    niche_units = sum(data['total_units'] for _, data in sorted_manufacturers[2:])
    niche_avg = niche_sales / niche_units if niche_units else 0
    report_lines.append(
        f"Niche Products: {format_currency(niche_sales)} sales | "
        f"{niche_units} units | {format_currency(niche_avg)} avg price"
    )

    # 3. Customer targeting recommendations (Top 5)
    report_lines.extend(["", "Top 5 Customer Recommendations:", "-"*40])
    customer_recommendations.sort(key=lambda x: x['potential'], reverse=True)
    
    if not customer_recommendations:
        report_lines.append("No recommendations generated. All customers may have purchased top products.")
    else:
        unique_recs = []
        seen = set()
        for rec in customer_recommendations:
            key = (rec['customer'], rec['state'], rec['product'])
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        
        for i, rec in enumerate(unique_recs[:5], 1):
            report_lines.append(
                f"{i}. {rec['customer']} ({rec['state']}): "
                f"Recommend {rec['product']} | "
                f"Potential: {format_currency(rec['potential'])} | "
                f"Past spend: {format_currency(rec['spent'])}"
            )
        if len(unique_recs) < 5:
            report_lines.append(f"Note: Only {len(unique_recs)} recommendation(s) generated.")

    # 4. Key insights
    report_lines.extend(["", "Strategic Insights:", "-"*40])
    top_state = max(state_data.items(), key=lambda x: x[1]['total_sales'])[0]
    top_manufacturer = sorted_manufacturers[0][0]
    report_lines.append(f"• {top_state} leads in sales volume and order value")
    report_lines.append(f"• {top_manufacturer} dominates product sales")
    
    if customer_recommendations:
        report_lines.append(f"• Top recommendation: {unique_recs[0]['customer']} in {unique_recs[0]['state']} for {unique_recs[0]['product']}")

    # Print final report
    with open(f"data/{uuid}/products_state.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    #print("\n".join(report_lines))

# Asynchronous wrapper functions
async def async_process_data(uuid):
    """Asynchronously process data by running process_data in a separate thread."""
    await asyncio.to_thread(process_data, uuid)

async def async_generate_report(uuid):
    """Asynchronously generate report by running generate_report in a separate thread."""
    report = await asyncio.to_thread(generate_report, uuid)
    return report

# Main asynchronous function to orchestrate execution
async def make_product_per_state_analysis(uuid):
    """Execute data processing followed by report generation."""
    await async_process_data(uuid)
    await async_generate_report(uuid)
    ans = await analyze_final_data(f'data/{uuid}/final_data.csv',uuid)
    raw = str(ans.get('output') or "")
    return raw


async def analyze_final_data(final_data, uuid):
    
    try:
        #system prompt
        prompt = 'Analyze my state - product sales data and make report  from those analysis'
        result = _process_ai_request(prompt=prompt,
            file_path_final_data=final_data,
            uuid=uuid)
        #logger2.info(f"User prompt: {prompt}")
        return result
    
    except Exception as e:
        logger2.error(f"Analysis AI report for customers group failed: {e}")
        raise

def _process_ai_request(prompt, file_path_final_data, uuid):
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df1 = pd.read_csv(file_path_final_data, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                logger2.warning(f"Failed decoding attempt with encoding: {encoding}")
        #df = df1.merge(df2, on="orderId", how="left")
        #df.to_csv('data.csv',index=False)
        llm = ChatOpenAI(model='gpt-4o') #model='o3-mini'  gpt-4.1-mini
        
        agent = create_pandas_dataframe_agent(
            llm,
            [df1],
            agent_type="openai-tools",
            verbose=False,
            allow_dangerous_code=True,
            number_of_head_rows=5,
            max_iterations=5
        )
        
        try:
            full_report_path = os.path.join('data',uuid, 'products_state.txt')
            with open(full_report_path, "r") as file:
                full_report = file.read()
        except Exception as e:
            full_report = 'No data given'
            logger2.warning(f"Can not read products_state.txt due to {e} ")
        
        try:
            promo_path = os.path.join('AI','group_customer_analyze', 'Agents_rules', 'promo_rules.txt')
            with open(promo_path, "r") as file:
                recommendations = file.read()
        except Exception as e:
            recommendations = 'No data given'
            logger2.warning(f"Can not read additional_info.md due to {e} ")
        
        
        formatted_prompt = f"""
        You are an AI assistant providing business insights based on  related dataset:  
        - **df1** (My data) contains critical order-related data or can be user activities data.  

        More detailed statistics are calculated for each state. 
        Some information has already been calculated, use it to provide basic information useful for business: {full_report}

        some additional data that you can use to make recommendations to the customer: {recommendations}
        
        **Important Rules to Follow:**  
        - **Unique Values:** When answering questions about orders or products, always consider unique values.  
        - **Neutral Wording:** Do not mention "df1" or "df2" in your response. Instead, phrase answers as "According to the user's data."  
        - **No Column/File References:** Do not refer to specific file names or column names—focus on insights and conclusions.  
        - **Well-Structured Markdown Formatting:** Ensure responses are clear and organized using appropriate Markdown formatting.  
        - **No Code or Visualizations:** Do not include Python code or suggest data visualizations in your answers.  
        - Make an analysis for each statistical block in the report - it should be a couple of sentences according to the result.
        - At the end, make recommendations to the business according to the data analysis - each block should be separated by '---'.
        - If you are sure that the question has nothing to do with the data, answer - "Your question is not related to the analysis of your data, please ask another question."

        The main task is to make recommendations to customers on what products to order based on state data!
        Insights are some small recommendation how to make it better (1-3)
        Don't focus only on the good, bad results should also be noticed.
        Use the names of customers in your report.
        
        Response format is:
        ---
        ## Section Title
        Content...
        
        **Insights**
         -
        ---
        ## Next Section...
        
        
        **Example Suggestions:**  
        
        **What can we improve in the products?**  
        1. **Focus on Resolving Pending Orders:**  
           – Implement automated notifications and customer engagement strategies to reduce pending order rates across all states. 
        2. **Product Promotion:**  
           – For high-sales states like CA, FL, and NY, leverage successful products such as Coca Cola to drive cross-selling opportunities with other products.
        3. **Targeted Marketing:**  
           – Use insights from best-selling products to create targeted marketing campaigns that attract potential customers, particularly in states with lower sales performance like MI and TX.
        4. **Customer Recommendations**:
            - Strengthen relationships with top customers by offering personalized promotions and loyalty incentives, such as those observed in PA and NY.
        **Task is:**  
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
            #logger2.info("here 3 ")
            result = agent.invoke({"input": formatted_prompt})
            #logger2.info("here 4 ")
            execution_time = time.time() - start_time
        
            in_toks, out_toks = cb.prompt_tokens, cb.completion_tokens
            cost, in_cost, out_cost = calculate_cost(llm.model_name, in_toks, out_toks)
        
            logger2.info("Agent for func:  orders_state")
            logger2.info(f"Input Cost orders_state:  ${in_cost:.6f}")
            logger2.info(f"Output Cost orders_state: ${out_cost:.6f}")
            logger2.info(f"Total Cost orders_state:  ${cost:.6f}")
        
            result['metadata'] = {
                'total_tokens orders_state': in_toks+out_toks,
                'prompt_tokens orders_state': in_toks,
                'completion_tokens orders_state': out_toks,
                'execution_time orders_state': f"{execution_time:.2f} seconds",
                'model orders_state': llm.model_name,
            }

        
        for k, v in result['metadata'].items():
            logger2.info(f"{k.replace('_', ' ').title()}: {v}")

        return {"output": result.get('output')}

    except Exception as e:
        logger2.error(f"Error in AI processing: {str(e)}")


# Entry point
if __name__ == "__main__":
    asyncio.run(async_generate_report()) #make_product_per_state_analysis
    
    