import pandas as pd
import numpy as np
from datetime import datetime
import os
import asyncio
import aiofiles
from pathlib import Path
from pprint import pprint

from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_suggestions, prompt_for_state_agent, prompt_agent_create_sectioned
from AI.group_customer_analyze.orders_state import async_generate_report, async_process_data
from AI.group_customer_analyze.preprocess_data_group_c import load_data, concat_customer_csv
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage
  
from AI.utils import get_logger, combine_sections, calculate_cost
logger2 = get_logger("logger2", "project_log_many.log", False)

from dotenv import load_dotenv
load_dotenv()

from functools import partial
import time 
import re

from typing import List, AsyncGenerator, Tuple, Any
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession
llm_model = OpenAIResponsesModel(model='gpt-4.1-mini', openai_client=AsyncOpenAI()) 

def _calculate_key_metrics(orders: pd.DataFrame) -> list:
    """Calculates overall Key Metrics."""
    try:
        total_delivery_fees = orders['deliveryFee'].sum()
        num_orders_with_delivery = (orders['deliveryFee'] > 0).sum()
        avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0
    
        total_sales = orders['totalAmount'].sum()
        total_orders = len(orders)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        total_discount_amount = orders['totalDiscountValue'].sum()
    
        std_order_value = orders['totalAmount'].std() if total_orders > 1 else 0
        std_fees_value = orders['deliveryFee'].std() if total_orders > 1 else 0
    
        return [
            "## Key Metrics",
            f"- **Total Sales:** {usd(total_sales)}",
            f"- **Total Orders:** {total_orders}",
            f"- **Average Order Value:** {usd(avg_order_value)}",
            f"- **Standard Deviation of Order Value:** {usd(std_order_value)}",
            f"- **Total Discounts Given:** {usd(total_discount_amount)}",
            f"- **Total Delivery Fees:** {usd(total_delivery_fees)}",
            f"- **Orders with Delivery Fees:** {num_orders_with_delivery} "
            f"({format_percentage(num_orders_with_delivery/total_orders*100)})",
            f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}",
            f"- **Standard Deviation of Delivery Fee:** {usd(std_fees_value)}",
        ]
    except Exception as e:
        logger2.warning(f"Error in calculation of key metrics: {e}")
        return ["## Key Metrics", f"Error generating section: {e}"]

def _calculate_discount_distribution(orders: pd.DataFrame) -> list:
    """Calculates overall Discount Distribution."""
    try:
        orders_copy = orders.copy() # Avoid SettingWithCopyWarning
        orders_copy['discount_category'] = orders_copy['appliedDiscountsType'].fillna('NONE')
        orders_copy.loc[(orders_copy['discount_category'] == 'NONE') & (orders_copy['totalDiscountValue'] > 0), 'discount_category'] = 'Other Discount'
 
        num_orders_with_discounts = (orders_copy['totalDiscountValue'] > 0).sum()
        total_orders = len(orders_copy)
        percentage_orders_with_discounts = (num_orders_with_discounts / total_orders * 100) if total_orders > 0 else 0
 
        discount_distribution = orders_copy.groupby('discount_category').agg(
            num_orders=('id', 'count'),
            total_discount=('totalDiscountValue', 'sum')
        ).reset_index()
 
        dd_combined = discount_distribution.copy()
        dd_combined['discount_category'] = dd_combined['discount_category'].replace('NONE', 'No Discount')
 
        lines_combined = [
            "## Overall Discount Distribution",
            f"- **Orders with Discounts:** {num_orders_with_discounts} ({format_percentage(percentage_orders_with_discounts)})",
            "",
            "| Discount Type | Number of Orders | Total Discount |",
            "|---------------|------------------|----------------|",
        ]
        for _, row in dd_combined.iterrows():
            lines_combined.append(f"| {format_status(row['discount_category'])} | {row['num_orders']} | {usd(row['total_discount'])} |")
 
        return lines_combined
    except Exception as e:
        logger2.warning(f"Can not count discount statistics due to: {e}")
        return ["## Overall Discount Distribution", f"Error generating section: {e}"]

def _calculate_sales_by_status(orders: pd.DataFrame) -> list:
    """Calculates overall sales by payment and delivery status."""
    try:
        orders_copy = orders.copy()
        orders_copy['paymentStatus'] = orders_copy['paymentStatus'].replace({"PENDING": "UNPAID"})

        total_sales_by_status_overall = (
            orders_copy.groupby(['paymentStatus', 'deliveryStatus'])['totalAmount']
            .sum()
            .reset_index()
        )

        lines_overall = [
            "## Overall Total Sales by Payment and Delivery Status",
            "| Payment Status | Delivery Status | Total Sales |",
            "|----------------|-----------------|-------------|",
        ]

        for _, row in total_sales_by_status_overall.iterrows():
            lines_overall.append(
                f"| {format_status(row['paymentStatus'])} | {format_status(row['deliveryStatus'])} | {usd(row['totalAmount'])} |"
            )
        return lines_overall
    except Exception as e:
        logger2.warning(f"Error in calculation of Sales by Status: {e}")
        return ["## Overall Total Sales by Payment and Delivery Status", f"Error generating section: {e}"]

def _calculate_payment_status(orders: pd.DataFrame) -> list:
    """Calculates overall payment status analysis."""
    try:
        orders_copy = orders.copy()
        orders_copy['paymentStatus'] = orders_copy['paymentStatus'].replace({"PENDING": "UNPAID"})
        
        existing_payment_statuses = orders_copy['paymentStatus'].unique()
        
        payment_status_counts = orders_copy['paymentStatus'].value_counts()
        total_orders = len(orders_copy)
        payment_status_percent = (payment_status_counts / total_orders * 100).round(1)
        
        lines_payment = ["## Payment Status Analysis"]
        for status in existing_payment_statuses:
            if status in payment_status_counts:
                lines_payment.append(
                    f"- **{format_status(status)}:** {payment_status_counts[status]} "
                    f"orders ({format_percentage(payment_status_percent[status])})"
                )
 
        return lines_payment
    except Exception as e:
        logger2.warning(f"Error in calculation of Payment Status: {e}")
        return ["## Payment Status Analysis", f"Error generating section: {e}"]

def _calculate_delivery_fees(orders: pd.DataFrame) -> list:
    """Calculates overall delivery fee analysis."""
    try:
        total_orders = len(orders)
        total_delivery_fees = orders['deliveryFee'].sum()
        num_orders_with_delivery = (orders['deliveryFee'] > 0).sum()
        avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0
        std_fees_value = orders['deliveryFee'].std() if total_orders > 1 else 0

        return [
            "## Delivery Fees Analysis",
            f"- **Total Delivery Fees:** {usd(total_delivery_fees)}",
            f"- **Orders with Delivery Fees:** {num_orders_with_delivery}",
            f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}",
            f"- **Standard Deviation of Delivery Fee:** {usd(std_fees_value)}"
        ]
    except Exception as e:
        logger2.warning(f"Error in calculation of Delivery Fees: {e}")
        return ["## Delivery Fees Analysis", f"Error generating section: {e}"]

def _calculate_fulfillment(orders: pd.DataFrame) -> list:
    """Calculates overall fulfillment analysis."""
    try:
        orders_copy = orders.copy()
        total_orders = len(orders_copy)
        orders_copy['deliveryStatus'] = orders_copy['deliveryStatus'].fillna('Unknown')
        fulfillment_counts = orders_copy['deliveryStatus'].value_counts()
        fulfillment_percent = (fulfillment_counts / total_orders * 100).round(1)

        lines_fulfillment_combined = ["## Fulfillment Analysis"]
        for status in fulfillment_counts.index:
            lines_fulfillment_combined.append(
                f"- **{format_status(status)}:** {fulfillment_counts[status]} orders ({format_percentage(fulfillment_percent[status])})"
            )
        return lines_fulfillment_combined
    except Exception as e:
        logger2.warning(f"Error in calculation of Fulfillment Analysis: {e}")
        return ["## Fulfillment Analysis", f"Error generating section: {e}"]

def _calculate_sales_performance(orders: pd.DataFrame) -> list:
    """Calculates overall sales performance overview."""
    try:
        total_revenue_all = orders['totalAmount'].sum()
        avg_order_value_all = orders['totalAmount'].mean() if not orders.empty else 0
        std_order_value = orders['totalAmount'].std() if len(orders) > 1 else 0

        monthly_sales_all = orders.groupby('month').agg(
            total_sales=('totalAmount', 'sum'),
            order_count=('id', 'nunique')
        ).reset_index()

        monthly_sales_all['month_dt'] = pd.to_datetime(monthly_sales_all['month'], format='%m/%Y')
        monthly_sales_all = monthly_sales_all.sort_values('month_dt')
        monthly_sales_all['pct_change'] = monthly_sales_all['total_sales'].pct_change() * 100
        monthly_sales_all = monthly_sales_all.sort_values('month_dt', ascending=False)

        lines_monthly_all = [
            "| Month     | Total Sales | Orders | Avg Sales/Order | % Change |",
            "|-----------|-------------|--------|-----------------|----------|",
        ]
        for _, row in monthly_sales_all.iterrows():
            avg_sales = row['total_sales'] / row['order_count'] if row['order_count'] else 0
            formatted_month = format_month(row['month'])
            pct_change = f"{row['pct_change']:.1f}%" if not pd.isna(row['pct_change']) else "-"
            lines_monthly_all.append(
                f"| {formatted_month} | {usd(row['total_sales'])} | "
                f"{row['order_count']} | {usd(avg_sales)} | {pct_change} |"
            )

        lines_all = [
            "## Sales Performance Overview - All Customers",
            f"- **Total Revenue:** {usd(total_revenue_all)}",
            f"- **Average Order Value:** {usd(avg_order_value_all)}",
            f"- **Standard Deviation of Order Value:** {usd(std_order_value)}",
            "",
            "### Monthly Sales Trends",
            ""
        ] + lines_monthly_all
        return lines_all
    except Exception as e:
        logger2.warning(f"Error in calculation of Sales Performance: {e}")
        return ["## Sales Performance Overview - All Customers", f"Error generating section: {e}"]

def _calculate_top_worst_products(orders: pd.DataFrame, products: pd.DataFrame) -> list:
    """Calculates overall top/worst selling products."""
    try:
        total_revenue = products['totalAmount'].sum()
        all_customers = orders['customer_name'].unique()
        
        customer_revenue = products.groupby('customer_name')['totalAmount'].sum().reset_index()
        
        category_map = products.drop_duplicates('product_variant')[['product_variant', 'productCategoryName']]
        category_map = dict(zip(category_map['product_variant'], category_map['productCategoryName']))
        
        product_group = products.groupby('product_variant').agg(
            units_sold=('quantity', 'sum'),
            revenue=('totalAmount', 'sum'),
            customer_list=('customer_name', lambda x: list(x.unique()))
        ).reset_index()
        
        product_group['revenue_percentage'] = (product_group['revenue'] / total_revenue) * 100 if total_revenue > 0 else 0
        product_group['customer_count'] = product_group['customer_list'].apply(len)
        
        sorted_products = product_group.sort_values('revenue', ascending=False)
        
        lines = []
        
        if len(sorted_products) > 0:
            # --- Overall Best Performer ---
            best = sorted_products.iloc[0]
            best_customers = ", ".join(best['customer_list'])
            non_buyers = [c for c in all_customers if c not in best['customer_list']]
            best_category = category_map.get(best['product_variant'], "Unknown")
            
            lines.append("## Top Selling Product")
            lines.append(f"- **Best Performer**: \"{best['product_variant']}\"")
            lines.append(f"  - **Units Sold**: {best['units_sold']} across all customers")
            lines.append(f"  - **Revenue Contribution**: {usd(best['revenue'])} ({best['revenue_percentage']:.1f}% of total revenue)")
            lines.append(f"  - **Customer Reach**: Sold to {best['customer_count']} customers: {best_customers}")
            
            similar_products = sorted_products[
                (sorted_products['product_variant'] != best['product_variant']) &
                (sorted_products['product_variant'].map(category_map) == best_category)
            ].head(2)
            
            lines.append("- **Opportunity**:")
            if non_buyers:
                non_buyers_revenue = customer_revenue[customer_revenue['customer_name'].isin(non_buyers)]
                top_non_buyers = non_buyers_revenue.sort_values('totalAmount', ascending=False).head(5)
                top_non_buyers_str = ", ".join(top_non_buyers['customer_name'].tolist())
                if top_non_buyers_str:
                    lines.append(f"  - Introduce to top non-buyers by revenue: {top_non_buyers_str}")
            if not similar_products.empty:
                similar_list = ", ".join([f"\"{p}\"" for p in similar_products['product_variant']])
                lines.append(f"  - Cross-sell similar products: {similar_list} ({best_category})")
            elif not non_buyers:
                lines.append("  - No obvious opportunities for this product.")

            # --- Overall Worst Performer ---
            worst = sorted_products.iloc[-1]
            worst_customers = ", ".join(worst['customer_list']) if worst['customer_list'] else "None"
            
            lines.append("")
            lines.append("## Worst Selling Product")
            lines.append(f"- **Worst Performer**: \"{worst['product_variant']}\"")
            lines.append(f"  - **Units Sold**: {worst['units_sold']} across all customers")
            lines.append(f"  - **Revenue Contribution**: {usd(worst['revenue'])} ({worst['revenue_percentage']:.1f}% of total revenue)")
            lines.append(f"  - **Customer Reach**: Sold to {worst['customer_count']} customers: {worst_customers}")
            lines.append("- **Consideration**:")
            lines.append("  - Evaluate market fit and consider targeted promotions or discontinuation.")

        return lines
    except Exception as e:
        logger2.warning(f"Error in calculation of Top-Worst Selling Products: {e}")
        return ["## Top-Worst Selling Product Analysis", f"Error generating section: {e}"]

async def generate_analytics_report_sectioned(
    orders: pd.DataFrame, 
    products: pd.DataFrame, 
    customer_df: pd.DataFrame, 
    uuid: str, 
    report_type: str = "Full Report"
):
    """
    Asynchronously generates analytics reports by running synchronous
    Pandas calculations in a thread pool.
    
    Can generate a "Full Report" (running all sections in parallel)
    or a single, specific report section.
    """
    
    # --- 1. Synchronous Pre-processing ---
    products['product_variant'] = products['name'].astype(str) + ' - ' + products['sku'].astype(str)
    
    # --- 2. Asynchronous I/O (Saving CSVs) ---
    try:
        # Run CSV saving in parallel threads
        await asyncio.gather(
            asyncio.to_thread(orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
            asyncio.to_thread(products.to_csv, f'data/{uuid}/pproducts.csv', index=False)
        )
    except Exception as e:
        logger2.warning(f"Error saving debug CSVs for {uuid}: {e}")

    # --- 3. Handle Single Section Requests ---
    if report_type == "key_metrics":
        lines = await asyncio.to_thread(_calculate_key_metrics, orders)
        return "\n".join(lines)

    if report_type == "discount_distribution":
        lines = await asyncio.to_thread(_calculate_discount_distribution, orders)
        return "\n".join(lines)

    if report_type == "overall_total_sales_by_payment_and_delivery_status":
        lines = await asyncio.to_thread(_calculate_sales_by_status, orders)
        return "\n".join(lines)

    if report_type == "payment_status_analysis":
        lines = await asyncio.to_thread(_calculate_payment_status, orders)
        return "\n".join(lines)

    if report_type == "delivery_fees_analysis":
        lines = await asyncio.to_thread(_calculate_delivery_fees, orders)
        return "\n".join(lines)

    if report_type == "fulfillment_analysis":
        lines = await asyncio.to_thread(_calculate_fulfillment, orders)
        return "\n".join(lines)

    if report_type == "sales_performance_overview":
        lines = await asyncio.to_thread(_calculate_sales_performance, orders)
        return "\n".join(lines)

    if report_type == "top_worst_selling_product":
        lines = await asyncio.to_thread(_calculate_top_worst_products, orders, products)
        return "\n".join(lines)

    # --- 4. Handle Full Report Request ---
    if report_type == "Full Report":
        # wrap each sync function in asyncio.to_thread
        tasks = [
            asyncio.to_thread(_calculate_key_metrics, orders),
            asyncio.to_thread(_calculate_discount_distribution, orders),
            asyncio.to_thread(_calculate_sales_by_status, orders),
            asyncio.to_thread(_calculate_payment_status, orders),
            asyncio.to_thread(_calculate_delivery_fees, orders),
            asyncio.to_thread(_calculate_fulfillment, orders),
            asyncio.to_thread(_calculate_sales_performance, orders),
            asyncio.to_thread(_calculate_top_worst_products, orders, products)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Unpack results (must be in the same order as tasks)
        (
            key_metrics_lines,
            discount_lines,
            sales_by_status_lines,
            payment_status_lines,
            delivery_fees_lines,
            fulfillment_lines,
            sales_perf_lines,
            top_worst_lines
        ) = results

        # Build the final report objects
        full_report_list = []
        sections_main = {}

        # Helper to add sections
        def add_to_report(name, lines):
            if lines: # Only add if calculation was successful
                section_text = "\n".join(lines)
                sections_main[name] = section_text
                full_report_list.extend(lines)
                full_report_list.append("") # Add a newline

        # Add all our parallel results
        add_to_report("key_metrics", key_metrics_lines)
        add_to_report("discount_distribution", discount_lines)
        add_to_report("overall_total_sales_by_payment_and_delivery_status", sales_by_status_lines)
        add_to_report("payment_status_analysis", payment_status_lines)
        add_to_report("delivery_fees_analysis", delivery_fees_lines)
        add_to_report("fulfillment_analysis", fulfillment_lines)
        add_to_report("sales_performance_overview", sales_perf_lines)
        add_to_report("top_worst_selling_product", top_worst_lines)

        # Add the final AI-generated suggestions placeholder
        add_to_report("suggestions_div", ["## Suggestions"])
        
        def clean_markdown(text: str) -> str:
            if not text:
                return ""
            # Replace the separator with a standard blank line for readability
            # We do two passes to catch '---' surrounded by newlines and '---' at end of strings
            return text.replace('\n---\n', '\n\n').replace('\n---', '').replace('---', '')

        # 1. Join the list first (Fastest way to build the blob)
        raw_full_report = "\n".join(full_report_list).strip()

        # 2. Run cleaning asynchronously (Non-blocking)
        # This prevents the string operation from freezing your FastApi/Server loop
        final_clean_report = await asyncio.to_thread(clean_markdown, raw_full_report)

        # 3. Clean the individual sections dictionary as well
        # We can do this in the same async thread or just list comp if data is small
        clean_sections = await asyncio.to_thread(
            lambda: {k: clean_markdown(v) for k, v in sections_main.items()}
        )

        # Return the dict structure your endpoint expects
        return {
            "full_report": final_clean_report,
            "sections": clean_sections
        }

    # Fallback for an unknown report type
    logger2.warning(f"Unknown report type requested: {report_type}")
    return {"error": f"Unknown report type: {report_type}"}

### Analysis Function
def format_month(month: str) -> str:
        """Convert MM/YYYY to MM/YY"""
        parts = month.split('/')
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1][-2:]}"
        return month

def generate_report(orders: pd.DataFrame, products: pd.DataFrame, customer_df: pd.DataFrame, uuid) -> str:
    """Function prepare all the statistics for ai 

    Args:
        orders (pd.DataFrame): orders dataframe
        products (pd.DataFrame): products dataframe
        customer_df (pd.DataFrame): some customer data (customer name)
        uuid : customer group uuid
    Returns:
        Return 3 type of reports. \n
        str: **full_report** -> full main report without sections an for all customers \n
        str: **sections** -> main report splitted by sections an for all customers \n
        str: **overall_report** -> full main report without sections and for each customer
    """
    
    #orders.to_csv('orders_many.csv', index=False)
    #products.to_csv('products_many.csv', index=False)
    products['product_variant'] = products['name'].astype(str) + ' - ' + products['sku'].astype(str)
    orders.to_csv(f'data/{uuid}/oorders.csv', index=False)
    products.to_csv(f'data/{uuid}/pproducts.csv', index=False)
    
    full_report = []
    sections_main = {}
    full_report_sectioned = {}
    
    customers_Overall_report = []
    customers_Overall_report_sectioned = {}
    agent_report = {}
    
    def add_section(name, lines, flag = False):
        if flag:
          full_report.extend(lines)
          full_report.append("")                        # full main report without sections and for each customer
          full_report_sectioned[name] = "\n".join(lines) # full main report with sections and for each customer
          sections_main[name] = "\n".join(lines)
        else:
          customers_Overall_report_sectioned[name] = "\n".join(lines)   # main report splitted by sections an for all customers
          customers_Overall_report.extend(lines)                        # full main report without sections an for all customers
          customers_Overall_report.append("")
    
    
    # Key Metrics
    try:
        # Calculate combined statistics for all customers
        total_delivery_fees = orders['deliveryFee'].sum()
        num_orders_with_delivery = (orders['deliveryFee'] > 0).sum()
        avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0
    
        total_sales = orders['totalAmount'].sum()
        total_orders = len(orders)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        total_discount_amount = orders['totalDiscountValue'].sum()
    
        # Metric: Standard Deviation of Order Values
        std_order_value = orders['totalAmount'].std() if total_orders > 1 else 0
        std_fees_value = orders['deliveryFee'].std() if total_orders > 1 else 0
    
        # Formatting lines for overall output
        lines_combined = [
            "## Key Metrics",
            f"- **Total Sales:** {usd(total_sales)}",
            f"- **Total Orders:** {total_orders}",
            f"- **Average Order Value:** {usd(avg_order_value)}",
            f"- **Standard Deviation of Order Value:** {usd(std_order_value)}",
            f"- **Total Discounts Given:** {usd(total_discount_amount)}",
            f"- **Total Delivery Fees:** {usd(total_delivery_fees)}",
            f"- **Orders with Delivery Fees:** {num_orders_with_delivery} "
            f"({format_percentage(num_orders_with_delivery/total_orders*100)})",
            f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}",
            f"- **Standard Deviation of Delivery Fee:** {usd(std_fees_value)}",
        ]
    
        add_section("key_metrics", lines_combined, True)
    
        # Calculate per-customer metrics
        customer_metrics = {}
        for customer_name, customer_orders in orders.groupby('customer_name'):
            # Calculate metrics for each customer
            cust_total_delivery = customer_orders['deliveryFee'].sum()
            cust_num_delivery = (customer_orders['deliveryFee'] > 0).sum()
            cust_avg_delivery = cust_total_delivery / cust_num_delivery if cust_num_delivery > 0 else 0
    
            cust_total_sales = customer_orders['totalAmount'].sum()
            cust_total_orders = len(customer_orders)
            cust_avg_order = cust_total_sales / cust_total_orders if cust_total_orders > 0 else 0
            cust_total_discount = customer_orders['totalDiscountValue'].sum()
    
            cust_std_order = customer_orders['totalAmount'].std() if cust_total_orders > 1 else 0
            cust_std_fees = customer_orders['deliveryFee'].std() if cust_total_orders > 1 else 0
    
            # Store customer metrics
            customer_metrics[customer_name] = {
                'total_sales': cust_total_sales,
                'total_orders': cust_total_orders,
                'avg_order_value': cust_avg_order,
                'std_order_value': cust_std_order,
                'total_discount': cust_total_discount,
                'total_delivery_fees': cust_total_delivery,
                'num_orders_with_delivery': cust_num_delivery,
                'avg_delivery_fee': cust_avg_delivery,
                'std_delivery_fee': cust_std_fees
            }
    
        # Generate individual customer reports
        for customer_name, metrics in customer_metrics.items():
            lines_customer = [
                f"## Key Metrics for customer: - {customer_name}",
                f"- **Total Sales:** {usd(metrics['total_sales'])}",
                f"- **Total Orders:** {metrics['total_orders']}",
                f"- **Average Order Value:** {usd(metrics['avg_order_value'])}",
                f"- **Standard Deviation of Order Value:** {usd(metrics['std_order_value'])}",
                f"- **Total Discounts Given:** {usd(metrics['total_discount'])}",
                f"- **Total Delivery Fees:** {usd(metrics['total_delivery_fees'])}",
                f"- **Orders with Delivery Fees:** {metrics['num_orders_with_delivery']} "
                + (f"({format_percentage(metrics['num_orders_with_delivery']/metrics['total_orders']*100)})" if metrics['total_orders'] > 0 else "0%"),
                f"- **Average Delivery Fee:** {usd(metrics['avg_delivery_fee'])}",
                f"- **Standard Deviation of Delivery Fee:** {usd(metrics['std_delivery_fee'])}",
            ]
            add_section(f"Key Metrics - {customer_name}", lines_customer)
    except Exception as e:
        logger2.warning(f"Error in the calculation of key metrics: {e}")
    
    # Discount analyze
    try:
        orders['discount_category'] = orders['appliedDiscountsType'].fillna('NONE')
        orders.loc[(orders['discount_category'] == 'NONE') & (orders['totalDiscountValue'] > 0), 'discount_category'] = 'Other Discount'
  
        # Calculate combined statistics for all customers
        num_orders_with_discounts = (orders['totalDiscountValue'] > 0).sum()
        total_orders = len(orders)
        percentage_orders_with_discounts = (num_orders_with_discounts / total_orders * 100) if total_orders > 0 else 0
  
        # Calculate overall discount distribution
        discount_distribution = orders.groupby('discount_category').agg(
            num_orders=('id', 'count'),
            total_discount=('totalDiscountValue', 'sum')
        ).reset_index()
  
        # Prepare the display dataframe for combined statistics
        dd_combined = discount_distribution.copy()
        dd_combined['discount_category'] = dd_combined['discount_category'].replace('NONE', 'No Discount')
  
        # Generate markdown lines for the combined section
        lines_combined = [
            "## Overall Discount Distribution",
            f"- **Orders with Discounts:** {num_orders_with_discounts} ({format_percentage(percentage_orders_with_discounts)})",
            "",
            "| Discount Type | Number of Orders | Total Discount |",
            "|---------------|------------------|----------------|",
        ]
        for _, row in dd_combined.iterrows():
            lines_combined.append(f"| {format_status(row['discount_category'])} | {row['num_orders']} | {usd(row['total_discount'])} |")
  
        add_section("discount_distribution", lines_combined, True)
  
  
        agent_report = {}
  
        # Group orders by customer_name
        for customer_name, customer_orders in orders.groupby('customer_name'):
            # Calculate statistics for each customer
            num_orders_with_discounts = (customer_orders['totalDiscountValue'] > 0).sum()
            total_orders = len(customer_orders)
            percentage_orders_with_discounts = (num_orders_with_discounts / total_orders * 100) if total_orders > 0 else 0
  
            # Calculate discount distribution for the customer
            discount_distribution = customer_orders.groupby('discount_category').agg(
                num_orders=('id', 'count'),
                total_discount=('totalDiscountValue', 'sum')
            ).reset_index()
  
            # Store in agent_report
            agent_report[customer_name] = {
                'num_orders_with_discounts': num_orders_with_discounts,
                'percentage_orders_with_discounts': percentage_orders_with_discounts,
                'discount_distribution': discount_distribution
            }
  
        # Generate and add per-customer sections
        for customer_name, data in agent_report.items():
            num_orders_with_discounts = data['num_orders_with_discounts']
            percentage_orders_with_discounts = data['percentage_orders_with_discounts']
            discount_distribution = data['discount_distribution']
  
            # Prepare discount distribution for display
            dd = discount_distribution.copy()
            dd['discount_category'] = dd['discount_category'].replace('NONE', 'No Discount')
  
            # Create markdown lines for the customer
            lines = [
                f"## Discount Distribution - {customer_name}",
                f"- **Orders with Discounts:** {num_orders_with_discounts} ({format_percentage(percentage_orders_with_discounts)})",
                "",
                "| Discount Type | Number of Orders | Total Discount |",
                "|---------------|------------------|----------------|",
            ]
            for _, row in dd.iterrows():
                lines.append(f"| {format_status(row['discount_category'])} | {row['num_orders']} | {usd(row['total_discount'])} |")
  
            # Add to sections and full_md using add_section
            add_section(f"Discount Distribution - {customer_name}", lines)
    except Exception as e:
        logger2.warning("Can not count discount statistics due to:", e)
    
    # Payment and Delivery Status
    try:
        # Add overall total sales by status
        orders = orders.copy()  # optional, to avoid modifying the original dataframe
        orders['paymentStatus'] = orders['paymentStatus'].replace({"PENDING": "UNPAID"})

        total_sales_by_status_overall = (
            orders.groupby(['paymentStatus', 'deliveryStatus'])['totalAmount']
            .sum()
            .reset_index()
        )

        lines_overall = [
            "## Overall Total Sales by Payment and Delivery Status",
            "| Payment Status | Delivery Status | Total Sales |",
            "|----------------|-----------------|-------------|",
        ]

        for _, row in total_sales_by_status_overall.iterrows():
            lines_overall.append(
                f"| {format_status(row['paymentStatus'])} | {format_status(row['deliveryStatus'])} | {usd(row['totalAmount'])} |"
            )
        add_section("overall_total_sales_by_payment_and_delivery_status", lines_overall, True)
    
        # Add per-customer total sales by status
        for customer_name, customer_orders in orders.groupby('customer_name'):
            total_sales_by_status = customer_orders.groupby(['paymentStatus', 'deliveryStatus'])['totalAmount'].sum().reset_index()
            lines_customer = [
                f"## Total Sales by Payment and Delivery Status - {customer_name}",
                "| Payment Status | Delivery Status | Total Sales |",
                "|----------------|-----------------|-------------|",
            ]
            for _, row in total_sales_by_status.iterrows():
                lines_customer.append(f"| {format_status(row['paymentStatus'])} | {format_status(row['deliveryStatus'])} | {usd(row['totalAmount'])} |")
            add_section(f"total_sales_by_status_{customer_name}", lines_customer)
    
        # Add overall payment status analysis
        existing_payment_statuses = orders['paymentStatus'].unique()
        existing_payment_statuses = [
            "UNPAID" if status == "PENDING" else status
            for status in existing_payment_statuses
        ]
        
        payment_status_counts = orders['paymentStatus'].replace({"PENDING": "UNPAID"}).value_counts()
        total_orders = len(orders)
        payment_status_percent = (payment_status_counts / total_orders * 100).round(1)
        
        lines_payment = ["## Payment Status Analysis"]
        for status in existing_payment_statuses:
            lines_payment.append(
                f"- **{format_status(status)}:** {payment_status_counts[status]} "
                f"orders ({format_percentage(payment_status_percent[status])})"
            )
  
        add_section("Payment Status Analysis", lines_payment)
        add_section("payment_status_analysis", lines_payment, True)
    except Exception as e:
        logger2.warning(f"Error in the calculation of Payment and Delivery Status: {e}")
    
    # Delivery Fees - Fulfillment Analysis
    try:
        # Calculate overall statistics for all customers
        total_orders = len(orders)
        total_delivery_fees = orders['deliveryFee'].sum()
        num_orders_with_delivery = (orders['deliveryFee'] > 0).sum()
        avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0
        std_fees_value = orders['deliveryFee'].std() if total_orders > 1 else 0

        # Fulfillment analysis for all customers
        orders['deliveryStatus'] = orders['deliveryStatus'].fillna('Unknown')
        fulfillment_counts = orders['deliveryStatus'].value_counts()
        fulfillment_percent = (fulfillment_counts / total_orders * 100).round(1)

        # Generate overall delivery fees section
        lines_delivery_combined = [
            "## Delivery Fees Analysis",
            f"- **Total Delivery Fees:** {usd(total_delivery_fees)}",
            f"- **Orders with Delivery Fees:** {num_orders_with_delivery}",
            f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}",
            f"- **Standard Deviation of Delivery Fee:** {usd(std_fees_value)}"
        ]

        # Generate overall fulfillment analysis section
        lines_fulfillment_combined = ["## Fulfillment Analysis"]
        for status in fulfillment_counts.index:
            lines_fulfillment_combined.append(
                f"- **{format_status(status)}:** {fulfillment_counts[status]} orders ({format_percentage(fulfillment_percent[status])})"
            )

        # Add overall sections
        add_section("delivery_fees_analysis", lines_delivery_combined, True)
        add_section("fulfillment_analysis", lines_fulfillment_combined, True)

        # Calculate per-customer statistics
        customer_delivery_data = {}

        for customer_name, customer_orders in orders.groupby('customer_name'):
            # Delivery fee statistics for customer
            cust_total_orders = len(customer_orders)
            cust_total_delivery = customer_orders['deliveryFee'].sum()
            cust_num_delivery = (customer_orders['deliveryFee'] > 0).sum()
            cust_avg_delivery = cust_total_delivery / cust_num_delivery if cust_num_delivery > 0 else 0
            cust_std_fees = customer_orders['deliveryFee'].std() if cust_total_orders > 1 else 0

            # Fulfillment analysis for customer
            cust_fulfillment_counts = customer_orders['deliveryStatus'].value_counts()
            cust_fulfillment_percent = (cust_fulfillment_counts / cust_total_orders * 100).round(1)

            # Store customer data
            customer_delivery_data[customer_name] = {
                'total_orders': cust_total_orders,
                'total_delivery': cust_total_delivery,
                'num_delivery': cust_num_delivery,
                'avg_delivery': cust_avg_delivery,
                'std_fees': cust_std_fees,
                'fulfillment_counts': cust_fulfillment_counts,
                'fulfillment_percent': cust_fulfillment_percent
            }

        # Generate individual customer reports
        for customer_name, data in customer_delivery_data.items():
            # Delivery fees section for customer
            lines_delivery_customer = [
                f"## Delivery Fees Analysis - {customer_name}",
                f"- **Total Delivery Fees:** {usd(data['total_delivery'])}",
                f"- **Orders with Delivery Fees:** {data['num_delivery']}",
                f"- **Average Delivery Fee:** {usd(data['avg_delivery'])}",
                f"- **Standard Deviation of Delivery Fee:** {usd(data['std_fees'])}"
            ]

            # Fulfillment analysis section for customer
            lines_fulfillment_customer = [f"## Fulfillment Analysis - {customer_name}"]
            for status in data['fulfillment_counts'].index:
                lines_fulfillment_customer.append(
                    f"- **{format_status(status)}:** {data['fulfillment_counts'][status]} orders ({format_percentage(data['fulfillment_percent'][status])})"
                )

            # Add customer sections
            add_section(f"Delivery Fees Analysis - {customer_name}", lines_delivery_customer)
            add_section(f"Fulfillment Analysis - {customer_name}", lines_fulfillment_customer)

    except Exception as e:
        logger2.warning(f"Error in the calculation of Delivery Fees - Fulfillment Analysis: {e}")
    
    # Sales Performance Overview per month
    try:
        # 1. Sales Performance Overview - All Customers
        total_revenue_all = orders['totalAmount'].sum()
        avg_order_value_all = orders['totalAmount'].mean() if not orders.empty else 0

        # Monthly sales trends for all customers
        monthly_sales_all = orders.groupby('month').agg(
            total_sales=('totalAmount', 'sum'),
            order_count=('id', 'nunique')
        ).reset_index()

        # Convert to datetime for proper chronological sorting
        monthly_sales_all['month_dt'] = pd.to_datetime(monthly_sales_all['month'], format='%m/%Y')

        # Sort ascending to calculate pct_change correctly (oldest to newest)
        monthly_sales_all = monthly_sales_all.sort_values('month_dt')

        # Add percentage change calculation
        monthly_sales_all['pct_change'] = monthly_sales_all['total_sales'].pct_change() * 100

        # Sort descending for display (newest to oldest)
        monthly_sales_all = monthly_sales_all.sort_values('month_dt', ascending=False)

        lines_monthly_all = [
            "| Month     | Total Sales | Orders | Avg Sales/Order | % Change |",
            "|-----------|-------------|--------|-----------------|----------|",
        ]
        for _, row in monthly_sales_all.iterrows():
            avg_sales = row['total_sales'] / row['order_count'] if row['order_count'] else 0
            formatted_month = format_month(row['month'])
            pct_change = f"{row['pct_change']:.1f}%" if not pd.isna(row['pct_change']) else "-"

            lines_monthly_all.append(
                f"| {formatted_month} | {usd(row['total_sales'])} | "
                f"{row['order_count']} | {usd(avg_sales)} | {pct_change} |"
            )

        # Generate report for all customers
        lines_all = [
            "## Sales Performance Overview - All Customers",
            f"- **Total Revenue:** {usd(total_revenue_all)}",
            f"- **Average Order Value:** {usd(avg_order_value_all)}",
            f"- **Standard Deviation of Order Value:** {usd(std_order_value)}",  # Added metric
            "",
            "### Monthly Sales Trends",
            ""
        ] + lines_monthly_all
        add_section("sales_performance_overview", lines_all, True)

        # 2. Sales Performance Overview - Each Customer
        for customer_name, customer_orders in orders.groupby('customer_name'):
            total_revenue_cust = customer_orders['totalAmount'].sum()
            avg_order_value_cust = customer_orders['totalAmount'].mean() if not customer_orders.empty else 0

            # Monthly sales trends for the customer
            monthly_sales_cust = customer_orders.groupby('month').agg(
                total_sales=('totalAmount', 'sum'),
                order_count=('id', 'nunique')
            ).reset_index()

            # Convert to datetime for proper chronological sorting
            monthly_sales_cust['month_dt'] = pd.to_datetime(monthly_sales_cust['month'], format='%m/%Y')
            monthly_sales_cust = monthly_sales_cust.sort_values('month_dt')  # Oldest to newest

            # Add percentage change calculation
            monthly_sales_cust['pct_change'] = monthly_sales_cust['total_sales'].pct_change() * 100

            lines_monthly_cust = [
                "| Month     | Total Sales | Orders | Avg Sales/Order | % Change |",
                "|-----------|-------------|--------|-----------------|----------|",
            ]
            for _, row in monthly_sales_cust.iterrows():
                avg_sales = row['total_sales'] / row['order_count'] if row['order_count'] else 0
                formatted_month = format_month(row['month'])
                pct_change = f"{row['pct_change']:.1f}%" if not pd.isna(row['pct_change']) else "-"

                lines_monthly_cust.append(
                    f"| {formatted_month} | {usd(row['total_sales'])} | "
                    f"{row['order_count']} | {usd(avg_sales)} | {pct_change} |"
                )

            # Generate report for the customer
            lines_cust = [
                f"## Sales Performance Overview - {customer_name}",
                f"- **Total Revenue:** {usd(total_revenue_cust)}",
                f"- **Average Order Value:** {usd(avg_order_value_cust)}",
                "",
                "### Monthly Sales Trends",
                ""
            ] + lines_monthly_cust
            add_section(f"sales_performance_overview_{customer_name}", lines_cust)
    except Exception as e:
        logger2.warning(f"Error in the calculation of Sales Performance Overview per month: {e}")
    
    # Top-Worst Selling Products
    try:
        total_revenue = products['totalAmount'].sum()
        all_customers = orders['customer_name'].unique()
        total_customer_count = len(all_customers)
        
        # Calculate total revenue per customer
        customer_revenue = products.groupby('customer_name')['totalAmount'].sum().reset_index()
        
        # Create product category mapping
        category_map = products.drop_duplicates('product_variant')[['product_variant', 'productCategoryName']]
        category_map = dict(zip(category_map['product_variant'], category_map['productCategoryName']))
        
        # Group products by variant
        product_group = products.groupby('product_variant').agg(
            units_sold=('quantity', 'sum'),
            revenue=('totalAmount', 'sum'),
            customer_list=('customer_name', lambda x: list(x.unique()))
        ).reset_index()
        
        # Calculate metrics
        product_group['revenue_percentage'] = (product_group['revenue'] / total_revenue) * 100
        product_group['customer_count'] = product_group['customer_list'].apply(len)
        
        # Sort by performance
        sorted_products = product_group.sort_values('revenue', ascending=False)
        
        lines = []
        
        if len(sorted_products) > 0:
            # --- Overall Best Performer ---
            best = sorted_products.iloc[0]
            best_customers = ", ".join(best['customer_list'])
            non_buyers = [c for c in all_customers if c not in best['customer_list']]
            best_category = category_map.get(best['product_variant'], "Unknown")
            
            lines.append("## Top Selling Product")
            lines.append(f"- **Best Performer**: \"{best['product_variant']}\"")
            lines.append(f"  - **Units Sold**: {best['units_sold']} across all customers")
            lines.append(f"  - **Revenue Contribution**: {usd(best['revenue'])} ({best['revenue_percentage']:.1f}% of total revenue)")
            lines.append(f"  - **Customer Reach**: Sold to {best['customer_count']} customers: {best_customers}")
            
            # Find similar products in same category
            similar_products = sorted_products[
                (sorted_products['product_variant'] != best['product_variant']) &
                (sorted_products['product_variant'].map(category_map) == best_category)
            ]
            similar_products = similar_products.head(2)
            
            lines.append("- **Opportunity**:")
            if non_buyers:
                # Filter non-buyers' revenue, sort by revenue, and take top 5
                non_buyers_revenue = customer_revenue[customer_revenue['customer_name'].isin(non_buyers)]
                top_non_buyers = non_buyers_revenue.sort_values('totalAmount', ascending=False).head(5)
                top_non_buyers_list = top_non_buyers['customer_name'].tolist()
                top_non_buyers_str = ", ".join(top_non_buyers_list)
                lines.append(f"  - Introduce to top non-buyers by revenue: {top_non_buyers_str}")
            if not similar_products.empty:
                similar_list = ", ".join([f"\"{p}\"" for p in similar_products['product_variant']])
                lines.append(f"  - Cross-sell similar products: {similar_list} ({best_category})")
            else:
                lines.append("  - No similar products found for cross-selling")
            
            # --- Overall Worst Performer ---
            worst = sorted_products.iloc[-1]
            worst_customers = ", ".join(worst['customer_list']) if worst['customer_list'] else "None"
            
            lines.append("")
            lines.append("## Worst Selling Product")
            lines.append(f"- **Worst Performer**: \"{worst['product_variant']}\"")
            lines.append(f"  - **Units Sold**: {worst['units_sold']} across all customers")
            lines.append(f"  - **Revenue Contribution**: {usd(worst['revenue'])} ({worst['revenue_percentage']:.1f}% of total revenue)")
            lines.append(f"  - **Customer Reach**: Sold to {worst['customer_count']} customers: {worst_customers}")
            lines.append("- **Consideration**:")
            if worst['units_sold'] == 0:
                lines.append("  - No sales recorded - recommend discontinuation")
            else:
                lines.append("  - Evaluate market fit and consider targeted promotions")

        # Add overall analysis to both reports
        add_section("top_worst_selling_product", lines, True)

        # --- Per-Customer Analysis ---
        customer_lines = ["## Per-Customer Product Performance"]

        for customer in all_customers:
            # Get customer-specific products
            cust_products = products[products['customer_name'] == customer]
            if cust_products.empty:
                continue

            # Group and sort customer products
            cust_group = cust_products.groupby('product_variant').agg(
                units_sold=('quantity', 'sum'),
                revenue=('totalAmount', 'sum')
            ).reset_index().sort_values('revenue', ascending=False)

            # Get top and bottom products
            top_product = cust_group.iloc[0]
            bottom_product = cust_group.iloc[-1]

            customer_lines.append(f"### {customer}")
            customer_lines.append(f"- **Top Product**: \"{top_product['product_variant']}\"")
            customer_lines.append(f"  - Units: {top_product['units_sold']}, Revenue: {usd(top_product['revenue'])}")

            # Only show worst product if different from top
            if top_product['product_variant'] != bottom_product['product_variant']:
                customer_lines.append(f"- **Worst Product**: \"{bottom_product['product_variant']}\"")
                customer_lines.append(f"  - Units: {bottom_product['units_sold']}, Revenue: {usd(bottom_product['revenue'])}")

            # Opportunity analysis
            top_cat = category_map.get(top_product['product_variant'], "")
            if top_cat:
                similar = cust_group[
                    (cust_group['product_variant'] != top_product['product_variant']) &
                    (cust_group['product_variant'].map(category_map) == top_cat)
                ]
                if not similar.empty:
                    rec_product = similar.iloc[0]['product_variant']
                    customer_lines.append(f"- **Opportunity**: Recommend similar product \"{rec_product}\" ({top_cat})")

            customer_lines.append("")

        # Add per-customer analysis to main report
        add_section("Product Performance Analysis", customer_lines)
    except Exception as e:
        logger2.warning(f"Error in the calculation of Top-Worst Selling Products: {e}")
    
    # New product recommendation
    try:
        pass
    except Exception as e:
        logger2.warning(f"Error in the calculation of New product recommendation: {e}")
    
    
    # Suggestions - full AI generate
    add_section("suggestions_div", ["## Suggestions"], True)
    add_section("Suggestions", ["## Suggestions"])
    
    customers_Overall_report = "\n".join(customers_Overall_report).strip()
    full_report = "\n".join(full_report).strip()

    return {
        "full_report": full_report,
        "sections_main": sections_main,
        "customers_Overall_report": customers_Overall_report
    }

async def combine_dicts_async(A, B):
    """Parse key to get value from statistics and AI dict to get one final result.

    Args:
        A (dict): section_name + Ai dict answer
        B (dict): section_name + Statistics 

    Returns:
        - new_dict (dict): dict that section_name + result of connected statistics and AI response.
        - result_string (str): full report with statistics and AI.
    """
    order = [
        'key_metrics',
        'discount_distribution',
        'overall_total_sales_by_payment_and_delivery_status',
        'payment_status_analysis',
        'delivery_fees_analysis',
        'fulfillment_analysis',
        'sales_performance_overview',
        'top_worst_selling_product', #top_products
        'product_per_state_analysis',
        'suggestions_div'
    ]
    
    new_dict = {}
    parts = []
    
    for key in order:
        if key not in A:
            content = B[key]
        else:    
            if key in B:
                content = f"\n{B[key]} \n\n {A[key]} \n"
            else:
                content = A[key]
        
        # Special handling for 'Suggestions'
        if key == 'suggestions_div':
            content = f"<div id=\"suggestions-block\">\n{content}\n</div>"
        
        new_dict[key] = content
        parts.append(content)
    
    result_string = " ".join(parts)
    return new_dict, result_string

def parse_analysis_response(response: str) -> dict:
    """
    Parses agent response in the format:
        ---
        ## Section Title
        Content...
        ---
        ## Next Section...
    Returns a dictionary {section_title: content}
    """

    # Split on lines that consist of only '---'
    sections = re.split(r'^\s*---\s*$', response, flags=re.MULTILINE)

    parsed = {}
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Look for the first markdown header in the section
        # (we use search, not match, so we can find it even if there's
        #  leading blank lines)
        m = re.search(r'^(#+)\s*(.+)$', section, flags=re.MULTILINE)
        if not m:
            continue

        _, raw_title = m.groups()
        title = raw_title.strip()

        # Everything after that header line is the content
        content = section[m.end():].strip()

        # (optional) special handling for a “Top-Level Recommendations” block
        if title == "Top-Level Recommendations":
            # Strip off any trailing boilerplate
            boiler = "\n\nThese actions will help"
            if boiler in content:
                content = content.split(boiler)[0]

        parsed[title] = content

    return parsed


@function_tool
def get_prepared_statistics(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'get_prepared_statistics' called ")
    data_path = f"data/{user_id}/full_report.md"
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

@function_tool
def get_recommendation(Topic: str) -> str:
    """Get 2-3 relevant predefined recommendations for chosen a business topic."""
    logger2.info(f"Tool 'get_recommendation' called for: {Topic}")
    filepath = "Ai/group_customer_analyze/Agents_rules/FAQ_SD.txt"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            advice_text =  f.read()
        logger2.info(f"Successfully read recommendations from {filepath}")
        return advice_text
    except FileNotFoundError:
        logger2.error(f"Recommendations file not found at: {filepath}")
        return "Error: Recommendations file not found."
    except Exception as e:
        logger2.error(f"Error reading {filepath}: {e}")
        return f"Error: {e}"

async def statistics_many_c_agent(USER_ID:str) -> Agent:
    """Initializes a new Inventory agent and session."""

    from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_create_full_report

    try:
        instructions = await prompt_agent_create_full_report(USER_ID)
        agent = Agent(
            name="Technical Specialist in Data Analysis Agent",
            instructions=instructions,
            model=llm_model,
            tools=[get_prepared_statistics]
        )

        print(" New statistics_many_c_agent  ready.")
    except Exception as e:
        logger2.error(f"error creating agent: {e}")
        agent = None
    return agent

async def statistics_runner(uuid:str):

    agent = await statistics_many_c_agent(uuid)

    runner = await Runner.run(
        agent, 
        input="Follow the instructions carefully—build the report according to my data.."
    )
    answer = runner.final_output 
    calculate_cost(runner, model="gpt-4.1-mini")
    #for i in range(len(runner.raw_responses)):
    #    print("Token usage : ", runner.raw_responses[i].usage, '')
    #print(answer)
    return answer


async def create_agent_sectioned(USER_ID, topic, statistics) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        instructions = await prompt_agent_create_sectioned(USER_ID, topic, statistics)

        agent = Agent(
            name="Customer_Orders_Assistant",
            instructions=instructions,
            model=llm_model
        )
        print(" New create_agent_sectioned are ready.")
    except Exception as e:
        print("create_agent_sectioned error: ", e)
    return agent

async def create_agent_products_state_analysis(USER_ID) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        #logger.info("✅ Agent run .")
        instructions = await prompt_for_state_agent(USER_ID)

        agent = Agent(
            name="Customer_product_state_Assistant",
            instructions=instructions
        )
        print(" New create_agent_products_state_analysis are ready.")
    except Exception as e:
        print(e)
    return agent

async def state_runner(orders, products_df, uuid):
    from AI.group_customer_analyze.orders_state import async_generate_report, async_process_data

    try:
        # Run CSV saving in parallel threads
        products_df['product_variant'] = products_df['name'].astype(str) + ' - ' + products_df['sku'].astype(str)
        await asyncio.gather(
            asyncio.to_thread(orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
            asyncio.to_thread(products_df.to_csv, f'data/{uuid}/pproducts.csv', index=False)
        )
    except Exception as e:
        logger2.warning(f"Error saving debug CSVs for {uuid}: {e}")
    
    await async_process_data(uuid)
    await async_generate_report(uuid)
    
    agent = await create_agent_products_state_analysis((uuid))
    try:
        runner = await Runner.run(
            agent, 
            input="Based on the data return response"#,  session=session
        )
        answer = runner.final_output 
        from pprint import pprint
        #print(answer)
        calculate_cost(runner, model="gpt-4.1-mini")
        #for i in range(len(runner.raw_responses)):
        #    print("Token usage : ", runner.raw_responses[i].usage, '')
    except Exception as e:
        print(f"Error in product_per_state_analysis runner: {e}")

    return answer

async def new_generate_analytics_report_(orders_df, products_df, customer_df, uuid):
    import time
    start = time.perf_counter()

        
    if orders_df.empty or products_df.empty:
        logger2.info("create_report_group_c data is empty so report is none")
        return '', {}
    
    answer = generate_report(orders_df, products_df, customer_df, uuid)
    print("Step 3.2 - concat data and generate report:", time.perf_counter() - start)
    full_report = answer['full_report']
    overall_report = answer['customers_Overall_report']
    sections = answer['sections_main']
    
    try:
        report_activities_dir = os.path.join("data", uuid)
        
        # save statistics report to md file for ai analyze
        path_for_overall_report = os.path.join(report_activities_dir, "overall_report.txt")
        path_for_full_report = os.path.join(report_activities_dir, "full_report.md")
        
        async with aiofiles.open(path_for_overall_report, mode="w", encoding="utf-8") as f:
            await f.write(overall_report)
    
        async with aiofiles.open(path_for_full_report, mode="w", encoding="utf-8") as f:
            await f.write(full_report)
    except Exception as e:
        logger2.error(f"Error saving group customer report result to file: {e}")
    
    # Run independent async tasks concurrently
    try:

        ai_task = asyncio.create_task(
            statistics_runner(uuid)
        )
        state_analysis_task = asyncio.create_task(
            state_runner(orders_df, products_df, uuid)
        )

        # Wait for both tasks to complete
        ans, product_per_state_analysis = await asyncio.gather(ai_task, state_analysis_task)
        #print(ans)
        print("Step 3.3 & 3.4 - Concurrent AI and state analysis:", time.perf_counter() - start)
    except Exception as e:
        logger2.error("Error in creating report state or statistic report: ", e)
        return '', {}
    
    raw = str(ans or "")              # answer of model for full report
    sections_answer = parse_analysis_response(raw)  # func that parse answer to section
    #pprint(sections_answer)
    items = list(sections_answer.items())                                       # tuple of sectioned report
    items.insert(8, ('product_per_state_analysis', product_per_state_analysis)) # add state analysis to sectioned report
    #print(items)
    new_dict = dict(items)                              # dict in format {section : ai_text}
    #pprint(new_dict)
    items2 = list(sections.items())
    items2.insert(8, ('product_per_state_analysis', ''))

    new_dict2 = dict(items2)                            # dict in format {section : calculated statistics}
    sectioned_report, full_report = await combine_dicts_async(new_dict, new_dict2)
    print("Step 3.5 - combine_dicts_async:", time.perf_counter() - start)
    return full_report, sectioned_report



#___ Block of full report by async agents call (test)

async def create_agent_suggestions(USER_ID) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        instructions = await prompt_agent_suggestions(USER_ID)

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

async def process_standard_topic(topic, merged_orders, products_df, customer_df, uuid):
    """Logic for standard analysis topics."""
    try:
        import time
        start = time.perf_counter()
        # 1. Generate Statistics
        statistics_of_topic = await generate_analytics_report_sectioned(
            merged_orders, products_df, customer_df, uuid, report_type=topic
        )

        # 2. Create Agent
        agent = await create_agent_sectioned(uuid, topic, statistics_of_topic)

        # 3. Run Agent
        runner = await Runner.run(
            agent, 
            input="Based on the data return response"
        )

        answer = runner.final_output
        
        # 4. Combine Sections
        # Assuming combine_sections is available in your scope
        sectioned_answer = await combine_sections(topic, statistics_of_topic, answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        calculate_cost(runner, model="gpt-4.1-mini")
        #for i in range(len(runner.raw_responses)):
        #    print("Token usage : ", runner.raw_responses[i].usage, '')
        if isinstance(sectioned_answer, dict):
            # Try to get the value using the topic as key, otherwise take the first value
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        return sectioned_answer

    except Exception as e:
        print(f"Error in standard topic '{topic}': {e}")
        return None

async def process_suggestions_topic(topic, merged_orders, products_df, customer_df, uuid):
    """Logic for suggestions analysis topics."""
    try:
        # 2. Create Agent
        import time
        start = time.perf_counter()
        statistics = generate_report(merged_orders, products_df, customer_df, uuid)
        async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                        await f.write(statistics.get('full_report') )

        agent = await create_agent_suggestions(uuid)

        # 3. Run Agent
        runner = await Runner.run(
            agent, 
            input="Based on the data return response"
        )

        answer = runner.final_output
        answer = f"<div id=\"suggestions-block\">\n\n{answer}\n</div>"
        #print(answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        calculate_cost(runner, model="gpt-4.1-mini")

        sectioned_answer = {'suggestions_div' : answer}
        if isinstance(sectioned_answer, dict):
            # Try to get the value using the topic as key, otherwise take the first value
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        return sectioned_answer

    except Exception as e:
        print(f"Error in standard topic '{topic}': {e}")
        return None

async def process_state_analysis(topic, merged_orders, products_df, customer_df, uuid):
    """Special logic for 'product_per_state_analysis'."""
    try:
        # 1. Prepare Data & Save CSVs (Threaded)
        # Note: Modifying DF here. If multiple tasks read this DF, ensure this doesn't conflict.
        # Since we are adding a column, it is generally safe but better done once globally if possible.
        import time
        start = time.perf_counter()
        products_df['product_variant'] = products_df['name'].astype(str) + ' - ' + products_df['sku'].astype(str)
        
        try:
            await asyncio.gather(
                asyncio.to_thread(merged_orders.to_csv, f'data/{uuid}/oorders.csv', index=False),
                asyncio.to_thread(products_df.to_csv, f'data/{uuid}/pproducts.csv', index=False)
            )
        except Exception as e:
            print(f"Warning: Error saving debug CSVs for {uuid}: {e}")

        print(f"Topic process_state_analysis 1 {topic}", time.perf_counter() - start)
        # 2. Process Data & Generate Report
        await async_process_data(uuid)
        await async_generate_report(uuid)
        print(f"Topic process_state_analysis 2 {topic}", time.perf_counter() - start)
        # 3. Create Specific Agent
        agent = await create_agent_products_state_analysis(uuid)

        # 4. Run Agent
        runner = await Runner.run(
            agent, 
            input="Based on the data return response"
        )

        answer = runner.final_output
        print(f"Topic process_state_analysis 3 {topic}", time.perf_counter() - start)
        return answer

    except Exception as e:
        print(f"Error in special topic '{topic}': {e}")
        return None

async def worker(semaphore, topic, merged_orders, products_df, customer_df, uuid):
    """
    Router function: Decides which logic to run based on the topic name,
    constrained by the semaphore.
    """
    async with semaphore:
        print(f"Processing: {topic}")
        
        if topic == "product_per_state_analysis":
            return await process_state_analysis(topic, merged_orders, products_df, customer_df, uuid)
        elif topic == "suggestions_div":
            return await process_suggestions_topic(topic, merged_orders, products_df, customer_df, uuid)
        else:
            return await process_standard_topic(topic, merged_orders, products_df, customer_df, uuid)

async def main_batch_process(merged_orders, products_df, customer_df, uuid):
    topics = [
        "key_metrics", 
        "payment_status_analysis", 
        "fulfillment_analysis",
        "discount_distribution", 
        #"product_per_state_analysis", 
        "sales_performance_overview", 
        "delivery_fees_analysis", 
        "overall_total_sales_by_payment_and_delivery_status", 
        "top_worst_selling_product",
        "suggestions_div"
    ] 

    # Limit concurrency to 10
    sem = asyncio.Semaphore(10)
    
    tasks = []
    for topic in topics:
        task = asyncio.create_task(
            worker(sem, topic, merged_orders, products_df, customer_df, uuid)
        )
        tasks.append(task)

    print(f"Starting {len(topics)} topics...")
    results = await asyncio.gather(*tasks)
    sectioned_report = dict(zip(topics, results))
    # 4. Compile the Big Report
    report_parts = []
    for key in topics:
        # Use .get() to avoid crashing if 'suggestions_div' or others are missing/None
        content = sectioned_report.get(key)

        if content:
            report_parts.append(str(content))
    
    def clean_markdown(text: str) -> str:
        if not text:
            return ""

        return text.replace('\n---\n', '\n\n').replace('\n---', '')


    raw_full_report = "\n".join(report_parts).strip()
    final_clean_report = await asyncio.to_thread(clean_markdown, raw_full_report)
    
    clean_sections = await asyncio.to_thread(
        lambda: {k: clean_markdown(v) for k, v in sectioned_report.items()}
    )

    return final_clean_report, clean_sections


if __name__ == "__main__":
    pass
