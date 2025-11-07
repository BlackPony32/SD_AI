import pandas as pd
import numpy as np
from datetime import datetime
import os
import asyncio
import aiofiles
from pathlib import Path
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import AsyncOpenAIEmbeddings, ChatOpenAI

from AI.group_customer_analyze.preprocess_data_group_c import load_data, concat_customer_csv
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage
  
from AI.group_customer_analyze.orders_state import make_product_per_state_analysis  
from AI.utils import get_logger
logger2 = get_logger("logger2", "project_log_many.log", False)

from dotenv import load_dotenv
load_dotenv()

from functools import partial
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import time 
from fastapi import FastAPI
app = FastAPI()
from langchain.agents import tool
from langchain_community.vectorstores import FAISS
#rom langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


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

async def generate_analytics_report(directory, uuid):
    import time
    start = time.perf_counter()

    try:
        orders, products = await load_data(directory)
    except Exception as e:
        logger2.error("Can not create concatenated orders and products due:", e)
    print("Step 3.1 - load data:", time.perf_counter() - start)
    
    try:
        concat_customer_path = await concat_customer_csv(f"data/{uuid}/raw_data")
        customer_df = pd.read_csv(concat_customer_path)
    except Exception as e:
        logger2.error("Can not create customer data df:", e)
        
    if orders.empty or products.empty:
        logger2.info("create_report_group_c data is empty so report is none")
        return '', {}
    
    answer = generate_report(orders, products, customer_df, uuid)
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
            analyze_orders_and_products(f'data/{uuid}/pproducts.csv', f'data/{uuid}/oorders.csv', uuid)
        )
        state_analysis_task = asyncio.create_task(
            make_product_per_state_analysis(uuid)
        )

        # Wait for both tasks to complete
        ans, product_per_state_analysis = await asyncio.gather(ai_task, state_analysis_task)
        print("Step 3.3 & 3.4 - Concurrent AI and state analysis:", time.perf_counter() - start)
    except Exception as e:
        logger2.error("Error in creating report state or statistic report: ", e)
        return '', {}
    
    raw = str(ans.get('output') or "")              # answer of model for full report
    sections_answer = parse_analysis_response(raw)  # func that parse answer to section
    
    items = list(sections_answer.items())                                       # tuple of sectioned report
    items.insert(8, ('product_per_state_analysis', product_per_state_analysis)) # add state analysis to sectioned report
    new_dict = dict(items)                              # dict in format {section : ai_text}

    items2 = list(sections.items())
    items2.insert(8, ('product_per_state_analysis', ''))

    new_dict2 = dict(items2)                            # dict in format {section : calculated statistics}
    sectioned_report, full_report = await combine_dicts_async(new_dict, new_dict2)
    print("Step 3.5 - combine_dicts_async:", time.perf_counter() - start)
    return full_report, sectioned_report
    

async def _create_advice_tool():
    # Load predefined recommendations asynchronously
    try:
        async with aiofiles.open("Ai/group_customer_analyze/FAQ_SD.txt", "r", encoding="utf-8") as f:
            advice_text = await f.read()
    except FileNotFoundError:
        logger2.error("Advice file not found")
        return None
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(advice_text)
    
    # Create vector store (synchronous, consider precomputing)
    emb = OpenAIEmbeddings()  # Use async embeddings if available
    index = FAISS.from_texts(chunks, emb)  # FAISS is sync; precompute if slow
    
    # Define advice retrieval tool
    @tool("AdviceTool")
    async def get_advice(topic: str) -> str:
        '''Retrieves 2-3 predefined recommendations for a business topic.'''
        docs = index.similarity_search(topic, k=3)  # FAISS is sync
        return "\n".join(d.page_content for d in docs)
        
    return get_advice
 
async def analyze_orders_and_products(file_path_product, file_path_orders, customer_id):
    try:
        # System prompt
        prompt = 'Make small analysis from my data and give some suggestions - only main info that I needed'
        result = await _process_ai_request(
            prompt=prompt,
            customer_id=customer_id,
            file_path_product=file_path_product,
            file_path_orders=file_path_orders
        )
        return result
    
    except Exception as e:
        logger2.error(f"Analysis AI report for customers group failed: {e}")
        raise
    
    except Exception as e:
        logger2.error(f"Analysis AI report for customers group failed: {e}")
        raise

async def _process_ai_request(prompt, file_path_product, file_path_orders, customer_id):
    try:
        # Async file reading for CSVs
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df1 = pd.read_csv(file_path_product, encoding=encoding, low_memory=False)
                df2 = pd.read_csv(file_path_orders, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                logger2.warning(f"Failed decoding attempt with encoding: {encoding}")
        
        llm = ChatOpenAI(model='gpt-4.1-mini')  # Ensure async support #model='o3-mini'  gpt-4.1-mini
        
        # Create advice tool
        import time
        start = time.perf_counter()
        print("Step 3.1.1 - before _create_advice_tool:", time.perf_counter() - start)
        advice_tool =await _create_advice_tool()
        print("Step 3.1.2 - after _create_advice_tool:", time.perf_counter() - start)
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
            async with aiofiles.open(os.path.join('data', customer_id, 'full_report.md'), mode='r') as file:
                full_report = await file.read()
        except Exception as e:
            full_report = 'No data given'
            logger2.warning(f"Can not read full_report.md due to {e}")
        
        try:
            async with aiofiles.open('AI/group_customer_analyze/promo_rules.txt', mode='r') as file:
                recommendations = await file.read()
        except Exception as e:
            recommendations = 'No data given'
            logger2.warning(f"Can not read promo_rules.txt due to {e}")
        
        
        formatted_prompt = f"""
        You are an AI assistant providing business insights based on two related datasets:  
        - **df1** (Orders) contains critical order-related data or can be user activities data.  
        - **df2** (Products) contains details about products within each order or can be user tasks data.  

        Calculated statistics that the user sees: {full_report}

        And some additional data that you can use to make recommendations to the customer: {recommendations}
        
        **Important Rules to Follow:**  
        - **Unique Values:** When answering questions about orders or products, always consider unique values.  
        - **Neutral Wording:** Do not mention "df1" or "df2" in your response. Instead, phrase answers as "According to the user's data."  
        - **No Column/File References:** Do not refer to specific file names or column names—focus on insights and conclusions.  
        - **Well-Structured Markdown Formatting:** Ensure responses are clear and organized using appropriate Markdown formatting.  
        - **No Code or Visualizations:** Do not include Python code or suggest data visualizations in your answers.  
        - Make an analysis for each statistical block in the report - it should be a couple of sentences according to the result.
        - At the end, make recommendations to the business according to the data analysis - each block should be separated by '---'.
        - If you are sure that the question has nothing to do with the data, answer - "Your question is not related to the analysis of your data, please ask another question."

        Section headings should be in accordance with the data in the report and named accordingly: 
        ['key_metrics', 'discount_distribution',
        'overall_total_sales_by_payment_and_delivery_status',
        'payment_status_analysis',
        'delivery_fees_analysis',
        'fulfillment_analysis',
        'sales_performance_overview',
        'top_worst_selling_product'] + "suggestions_div" for your final recommendations
        Note do not skip any title - if no info - write 'Not enough info to analyze' to content
        **Critical Instructions for Insights:**
        - For **Insights** in each section, ALWAYS use the `AdviceTool` with the section title as input
        - Use EXACTLY 2-3 recommendations from the tool output
        - Make **Insights** based on the notes you receive in accordance with the data.

        Example:
        ## sales_performance_overview
        [Your analysis...]

        **Insights**
        {{{{ADVICE FROM TOOL FOR "Sales Performance Overview"}}}}
        Response format is:
        ---
        ## Section Title
        Content...
        
        **Insights**
        ---
        ## Next Section...
        
        
        **Example Suggestions:**  
        
        **Top-Level Recommendations**  
        1. **Leverage Tiered Incentives:**  
           – Introduce small invoice-level or item-level discounts for high-margin lines to boost adoption, especially during slower months.          
           – Pilot “buy-more-save-more” bundles featuring best-sellers plus slow movers.  

        2. **Convert Pending Orders:**  
           – Implement gentle reminders or time-limited incentives (e.g., free shipping) to nudge pending transactions to completion.  

        3. **Optimize Delivery Fees:**  
           – Offer free delivery thresholds (e.g., orders >$300) to increase average cart size while preserving margin on smaller orders.  

        4. **Seasonal Promotion Planning:**  
           – Capitalize on the strong early-year momentum by aligning marketing pushes in Jan–Mar; bolster mid-year demand with targeted campaigns.   

        5. **Refine Assortment:**  
           – Reevaluate underperforming SKUs for promotional clearance or phased-out stocking.  
           – Expand cross-sell recommendations around “Diet Coke” to zero- and vanilla-flavored extensions—leveraging proven customer interest.

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
            logger2.info("here 1 ")
            result = await agent.ainvoke({"input": formatted_prompt})
            logger2.info("here 2 ")
            execution_time = time.time() - start_time
        
            in_toks, out_toks = cb.prompt_tokens, cb.completion_tokens
            cost, in_cost, out_cost = calculate_cost(llm.model_name, in_toks, out_toks)
        
            logger2.info("Agent for func: create_report_group")
            logger2.info(f"Input Cost create_report_group: ${in_cost:.6f}")
            logger2.info(f"Output Cost create_report_group: ${out_cost:.6f}")
            logger2.info(f"Total Cost create_report_group: ${cost:.6f}")
        
            result['metadata'] = {
                'total_tokens create_report_group': in_toks + out_toks,
                'prompt_tokens create_report_group': in_toks,
                'completion_tokens create_report_group': out_toks,
                'execution_time create_report_group': f"{execution_time:.2f} seconds",
                'model': llm.model_name,
            }

        
        for k, v in result['metadata'].items():
            logger2.info(f"{k.replace('_', ' ').title()}: {v}")

        return {"output": result.get('output')}

    except Exception as e:
        logger2.error(f"Error in AI processing: {str(e)}")

import re

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


async def new_generate_analytics_report(orders_df, products_df, customer_df, uuid):
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
            analyze_orders_and_products(f'data/{uuid}/pproducts.csv', f'data/{uuid}/oorders.csv', uuid)
        )
        state_analysis_task = asyncio.create_task(
            make_product_per_state_analysis(uuid)
        )

        # Wait for both tasks to complete
        ans, product_per_state_analysis = await asyncio.gather(ai_task, state_analysis_task)
    
        print("Step 3.3 & 3.4 - Concurrent AI and state analysis:", time.perf_counter() - start)
    except Exception as e:
        logger2.error("Error in creating report state or statistic report: ", e)
        return '', {}
    
    raw = str(ans.get('output') or "")              # answer of model for full report
    sections_answer = parse_analysis_response(raw)  # func that parse answer to section
    #pprint(sections_answer)
    items = list(sections_answer.items())                                       # tuple of sectioned report
    items.insert(8, ('product_per_state_analysis', product_per_state_analysis)) # add state analysis to sectioned report
    new_dict = dict(items)                              # dict in format {section : ai_text}

    items2 = list(sections.items())
    items2.insert(8, ('product_per_state_analysis', ''))

    new_dict2 = dict(items2)                            # dict in format {section : calculated statistics}
    sectioned_report, full_report = await combine_dicts_async(new_dict, new_dict2)
    print("Step 3.5 - combine_dicts_async:", time.perf_counter() - start)
    return full_report, sectioned_report

if __name__ == "__main__":
    #asyncio.run(main())
    from pprint import pprint
    #full_report = asyncio.run(Ask_AI_group_orders('data/uuid/pproducts.csv','data/uuid/oorders.csv', 'uuid'))
    sectioned_report, full_report = asyncio.run(generate_analytics_report('data/uuid'))
    #print(report['full_report'])
    with open("full_report_test.txt", "w", encoding="utf-8") as f:
                f.write(sectioned_report)
    #print(report['overall_report'])
    print(sectioned_report) #.get('Delivery and Fulfillment Report') .get('output')