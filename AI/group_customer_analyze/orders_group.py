import pandas as pd
import numpy as np
import datetime
import os
import asyncio
import aiofiles
from pathlib import Path
from pprint import pprint
from functools import partial
import time 
import re
from itertools import combinations
from collections import Counter
import traceback
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage

from AI.utils import get_logger, combine_sections, calculate_cost
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_suggestions, prompt_mcp_topics_customer_agent, prompt_mcp_suggestions, prompt_mcp_topics_orders_agent, prompt_mcp_topics_catalog_agent

logger2 = get_logger("logger2", "project_log_many.log", False)


from dotenv import load_dotenv
load_dotenv()

model = 'gpt-5.4-mini' #'gpt-5.4-mini'
llm_model = OpenAIResponsesModel(model=model, openai_client=AsyncOpenAI()) 

@function_tool
def get_prepared_statistics(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'mcp_get_prepared_statistics' called ")
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

ORDER_COLUMN_ALIASES = {
    "customerId": "customer_id",
}

def _load_orders_csv(orders_path) -> pd.DataFrame:
    orders = pd.read_csv(orders_path, encoding='utf-8-sig')
    orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
    orders = orders.rename(columns={
        k: v for k, v in ORDER_COLUMN_ALIASES.items() if k in orders.columns
    })
    return orders

def _calculate_key_metrics_orders_g(orders_path) -> list:
    """Calculates Key Metrics including Standard Deviation."""
    try:
        orders = _load_orders_csv(orders_path)
        total = len(orders)
        orders['deliveryStatus'] = orders['deliveryStatus'].fillna('Unknown')
        
        # Financials
        total_revenue = orders['totalAmount'].sum()
        total_discounts = orders['totalDiscountValue'].sum()
        total_delivery = orders['deliveryFee'].sum()
        
        # Counts
        total_orders = len(orders)
        orders_with_delivery = (orders['deliveryFee'] > 0).sum()
        
        # Averages & Std Dev
        avg_order_value = orders['totalAmount'].mean() if total_orders > 0 else 0
        std_order_value = orders['totalAmount'].std() if total_orders > 1 else 0
        
        avg_delivery = orders.loc[orders['deliveryFee'] > 0, 'deliveryFee'].mean() if orders_with_delivery > 0 else 0
        std_delivery = orders['deliveryFee'].std() if total_orders > 1 else 0

        lines = [
            "## Executive Key Metrics",
            "*Overview of financial performance and order variability.*",
            "",
            "### Sales & Orders",
            f"- **Total Sales:** {usd(total_revenue)}",
            f"- **Total Orders:** {total_orders}",
            f"- **Average Order Value:** {usd(avg_order_value)}",
            f"- **Order Value Standard Deviation:** {usd(std_order_value)}",
            "> *Standard Deviation measures consistency. A high number means order sizes vary wildly; a low number means most orders are around the average.*",
            "",
            "### Fees & Discounts",
            f"- **Total Discounts Given:** {usd(total_discounts)}",
            f"- **Total Delivery Fees Collected:** {usd(total_delivery)}",
            f"- **Orders with Delivery Fee:** {orders_with_delivery} ({format_percentage(orders_with_delivery/total_orders*100)})",
            f"- **Average Delivery Fee:** {usd(avg_delivery)}",
            f"- **Delivery Fee Standard Deviation:** {usd(std_delivery)}"
        ]
        return lines
    except Exception as e:
        logger2.error(f"Error calculating key metrics in _calculate_key_metrics_orders: {e}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _calculate_sales_orders_performance_g(orders_path) -> list:
    """Calculates Monthly Sales with % Change."""
    try:
        orders = _load_orders_csv(orders_path)
        total = len(orders)
        orders['deliveryStatus'] = orders['deliveryStatus'].fillna('Unknown')
        
        monthly = orders.groupby('month').agg(
            total_sales=('totalAmount', 'sum'),
            order_count=('customer_id', 'count')
        ).reset_index()
        
        monthly['month_dt'] = pd.to_datetime(monthly['month'], format='%m/%Y', errors='coerce')
        monthly = monthly.sort_values('month_dt')
        
        monthly['pct_change'] = monthly['total_sales'].pct_change() * 100
        
        lines = [
            "## Sales Performance & Trends",
            "*Monthly breakdown of revenue and order volume.*",
            "",
            "| Month | Total Sales | Orders | Avg Sales/Order | % Change (MoM) |",
            "|---|---|---|---|---|",
        ]
        
        for _, row in monthly.sort_values('month_dt', ascending=False).iterrows():
            avg = row['total_sales'] / row['order_count'] if row['order_count'] else 0
            change = f"{row['pct_change']:+.1f}%" if pd.notna(row['pct_change']) else "-"
            
            lines.append(f"| {row['month']} | {usd(row['total_sales'])} | {row['order_count']} | {usd(avg)} | {change} |")
            
        return lines
    except Exception as e:
        logger2.error(f"Error calculating sales performance in _calculate_sales_orders_performance: {e}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _calculate_discount_distribution_g(orders_path) -> list:
    """
    Calculates advanced Global Discount Performance with optimized vectorized
    categorization, refund filtering, and strict financial metrics.
    """
    try:
        # Load and clean headers
        orders = _load_orders_csv(orders_path)
        
        # 1. OPTIMIZATION: Filter out Refunds immediately to ensure clean net data
        if 'paymentStatus' in orders.columns:
            orders = orders[orders['paymentStatus'].fillna('').str.upper() != 'REFUNDED'].copy()
        
        # 2. Determine individual discount presence mathematically
        orders['has_cust'] = orders['customerDiscountValue'].fillna(0) > 0
        orders['has_mfg'] = orders['manufacturerDiscountValue'].fillna(0) > 0
        orders['has_invoice'] = orders['totalOrderDiscountValue'].fillna(0) > 0
        
        orders['remainder'] = orders['totalDiscountValue'].fillna(0) - (
            orders['customerDiscountValue'].fillna(0) + 
            orders['manufacturerDiscountValue'].fillna(0) + 
            orders['totalOrderDiscountValue'].fillna(0)
        )
        orders['has_item_or_slot'] = orders['remainder'] > 0.01
        
        # Count concurrent discount types
        orders['active_discount_types_count'] = (
            orders['has_cust'].astype(int) + 
            orders['has_mfg'].astype(int) + 
            orders['has_invoice'].astype(int) + 
            orders['has_item_or_slot'].astype(int)
        )
        
        if 'appliedDiscountsType' not in orders.columns:
            orders['appliedDiscountsType'] = 'NONE'

        # 3. OPTIMIZATION: Vectorized Categorization using np.select
        conditions = [
            orders['totalDiscountValue'].fillna(0) <= 0,
            orders['active_discount_types_count'] > 1,
            orders['has_cust'],
            orders['has_mfg'],
            orders['has_invoice'],
            orders['appliedDiscountsType'].fillna('').str.upper() == 'ITEM_DISCOUNT'
        ]
        
        choices = [
            'No Discount (Baseline)',
            'Stacked / Mixed Discounts',
            'Customer Discount',
            'Manufacturer Discount',
            'Invoice Total Discount',
            'Discount on Selected Entities'
        ]
        
        orders['DiscountCategory'] = np.select(conditions, choices, default='Slotting')

        # 4. OPTIMIZATION: Calculate Global Metrics using totalAmountWithoutDelivery
        total_revenue = orders['totalAmountWithoutDelivery'].sum()
        total_orders = len(orders)
        num_with_disc = (orders['totalDiscountValue'] > 0).sum()
        
        stats = orders.groupby('DiscountCategory').agg(
            Count=('id', 'count'),
            TotalDiscount=('totalDiscountValue', 'sum'),
            NetSales=('totalAmountWithoutDelivery', 'sum') 
        ).reset_index()
        
        # Restore Gross Sales for accurate AOV scaling
        stats['GrossSales'] = stats['NetSales'] + stats['TotalDiscount']
        
        # 5. Calculate Core Performance Metrics with Divide-by-Zero Safeties
        stats['AvgOrderValue'] = np.where(
            stats['Count'] > 0, 
            stats['GrossSales'] / stats['Count'], 
            0
        )
        
        stats['EffectiveDiscountRate'] = np.where(
            stats['GrossSales'] > 0, 
            (stats['TotalDiscount'] / stats['GrossSales']) * 100, 
            0
        )
        
        stats['RevenueShare'] = np.where(
            total_revenue > 0, 
            (stats['NetSales'] / total_revenue) * 100, 
            0
        )
        
        # Extract baseline AOV
        baseline_row = stats[stats['DiscountCategory'] == 'No Discount (Baseline)']
        baseline_aov = baseline_row['AvgOrderValue'].values[0] if not baseline_row.empty else 0
        
        # Advanced Metrics with Safety Overrides
        stats['AOVLift'] = np.where(
            baseline_aov > 0, 
            ((stats['AvgOrderValue'] - baseline_aov) / baseline_aov) * 100, 
            0
        )
        
        stats['DiscountEfficiency'] = np.where(
            stats['TotalDiscount'] > 0, 
            stats['GrossSales'] / stats['TotalDiscount'], 
            0
        )
        
        # Dynamic health status
        def assign_status(row):
            if row['DiscountCategory'] == 'No Discount (Baseline)':
                return 'Healthy Baseline'
            if row['EffectiveDiscountRate'] > 40:
                return 'Critical Margin Loss'
            if row['EffectiveDiscountRate'] > 20 or (row['DiscountEfficiency'] > 0 and row['DiscountEfficiency'] < 5):
                return 'Review (Leakage Risk)'
            if row['DiscountEfficiency'] >= 15:
                return 'Highly Efficient'
            return 'Stable / Moderate'

        stats['Status'] = stats.apply(assign_status, axis=1)
        
        # Sort output
        stats['is_baseline'] = stats['DiscountCategory'] == 'No Discount (Baseline)'
        stats = stats.sort_values(by=['is_baseline', 'NetSales'], ascending=[False, False]).drop(columns=['is_baseline'])
        
        # 6. Format Output Report
        lines = [
            "## Global Discount Performance Report",
            f"**Total Orders:** {total_orders} | **Orders with Discounts:** {num_with_disc} ({format_percentage(num_with_disc/total_orders*100) if total_orders > 0 else '0.0%' })",
            "",
            "| Discount Category | Orders | % of Total Revenue | EDR (Margin Cut) | Avg Order Value | AOV Lift | Return on $1 Discount | Performance Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        
        for _, row in stats.iterrows():
            name = row['DiscountCategory']
            count = row['Count']
            share = format_percentage(row['RevenueShare'])
            edr = format_percentage(row['EffectiveDiscountRate']) if row['TotalDiscount'] > 0 else "0.0%"
            aov = usd(row['AvgOrderValue'])
            lift = f"{row['AOVLift']:+.1f}%" if name != 'No Discount (Baseline)' else "0.0%"
            
            efficiency = f"{usd(row['DiscountEfficiency'])} generated" if row['TotalDiscount'] > 0 else "-"
            status = row['Status']
            
            lines.append(f"| {name} | {count} | {share} | {edr} | {aov} | {lift} | {efficiency} | {status} |")
            
        lines.extend([
            "",
            "### Key Performance Metric Explanations",
            "- **% of Total Revenue:** Indicates the concentration of incoming cash flow tied to that specific discount bucket.",
            "  * *Formula:* `(Net Sales of Category / Total Global Net Sales) * 100`",
            "- **Effective Discount Rate (EDR):** The actual percentage removed from the gross pricing of those orders. This reveals your true margin cut.",
            "  * *Formula:* `(Total Discount Given / Gross Sales of Category) * 100`",
            "- **Avg Order Value (AOV):** The mean dollar size per purchase inside that specific bucket, excluding delivery fees to show true retail volume.",
            "  * *Formula:* `Gross Sales of Category / Order Count of Category`",
            "- **AOV Lift:** Compares shopping cart performance directly against full-price orders. A *negative lift* indicates promotions are applied to smaller cart sizes rather than scaling up basket depth.",
            "  * *Formula:* `((Category AOV - Baseline AOV) / Baseline AOV) * 100`",
            "- **Return on $1 Discount (Discount Efficiency):** Measures promotional efficiency. It tracks how many gross dollars of retail volume were moved for every single dollar given away in promotions.",
            "  * *Formula:* `Gross Sales of Category / Total Discount Given`",
            "- **Performance Status:** Automated health grading lanes based on discount margins (EDR) and financial returns. Flags code stacking combinations or legacy slots that drain business profit margins."
        ])
            
        return lines

    except Exception as e:
        logger2.error(f"Error calculating discount distribution in calculate_discount_performance_report: {e}")
        return ["This report is currently unavailable due to a temporary structural update. Please try again or contact support if the issue persists."]

def _calculate_orders_fulfillment_g(orders_path) -> list:
    """Calculates Fulfillment Breakdown with Revenue Column."""
    try:
        orders = _load_orders_csv(orders_path)
        total = len(orders)
        orders['deliveryStatus'] = orders['deliveryStatus'].fillna('Unknown')
        
        stats = orders.groupby('deliveryStatus').agg(
            Count=('customer_id', 'count'),
            Revenue=('totalAmount', 'sum')
        ).sort_values('Count', ascending=False)
        
        lines = [
            "## Fulfillment Analysis",
            "*Breakdown of orders and revenue by delivery status.*",
            "",
            "| Delivery Status | Orders | Percentage | Revenue |",
            "|---|---|---|---|",
        ]
        
        for status, row in stats.iterrows():
            lines.append(f"| {format_status(status)} | {row['Count']} | {format_percentage(row['Count']/total*100)} | {usd(row['Revenue'])} |")
            
        return lines
    except Exception as e:
        logger2.error(f"Error calculating fulfillment breakdown in _calculate_orders_fulfillment: {e}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _calculate_payment_status_g(orders_path) -> list:
    """Calculates Payment Status with Revenue Column."""
    try:
        orders = _load_orders_csv(orders_path)
        total = len(orders)
        orders['paymentStatus'] = orders['paymentStatus'].fillna('Unknown')
        
        stats = orders.groupby('paymentStatus').agg(
            Count=('customer_id', 'count'),
            Revenue=('totalAmount', 'sum')
        ).sort_values('Count', ascending=False)
        
        lines = [
            "## Payment Status Analysis",
            "| Payment Status | Orders | Percentage | Revenue |",
            "|---|---|---|---|",
        ]
        
        for status, row in stats.iterrows():
            lines.append(f"| {format_status(status)} | {row['Count']} | {format_percentage(row['Count']/total*100)} | {usd(row['Revenue'])} |")
            
        return lines
    except Exception as e:
        logger2.error(f"Error calculating payment status in _calculate_payment_status: {e}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _sales_trends_orders_report_g(orders_path) -> list:
    try:
        # --- 1. Load Data ---
        try:
            if not os.path.exists(orders_path):
                return ["Error: File not found."]
            
            df = _load_orders_csv(orders_path)
            
            if df.empty:
                return ["Can not generate report: there is not enough data."]
                
        except Exception as e:
            logger2.error(f"Data Loading Error in Sales Trends Report: {str(e)}")
            return ["This report is currently unavailable due to a data loading update. Please check back later or contact support."]

        # --- 2. Preprocess Dates ---
        try:
            # Ensure UTC and valid dates
            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
            df = df.dropna(subset=['createdAt'])
            
            if df.empty:
                return ["Can not generate report: there is not enough data after parsing dates."]
                
            reference_date = df['createdAt'].max()

            # FIX: Convert to timezone-naive before converting to Period to silence the UserWarning
            df_tz_naive = df['createdAt'].dt.tz_localize(None)
            df['YearMonth'] = df_tz_naive.dt.to_period('M')
            df['Quarter'] = df_tz_naive.dt.to_period('Q')
            
            df['DayOfWeek'] = df['createdAt'].dt.day_name()
            
        except Exception as e:
            logger2.error(f"Date Preprocessing Error in Sales Trends Report: {str(e)}")
            return ["This report is currently unavailable due to a temporary date format issue. Please check back later."]

        # --- 3. Sales Trends Analysis ---
        try:
            # A. Monthly Trends
            monthly_stats = df.groupby('YearMonth').agg({
                'totalAmount': 'sum',
                'customer_id': 'count'
            }).rename(columns={'totalAmount': 'Revenue', 'customer_id': 'Orders'})
            
            monthly_stats['AOV'] = monthly_stats['Revenue'] / monthly_stats['Orders']
            monthly_stats['Rev_Growth'] = monthly_stats['Revenue'].pct_change() * 100
            monthly_report = monthly_stats.sort_index(ascending=False)

            # B. Quarterly Summary
            quarterly_stats = df.groupby('Quarter').agg({
                'totalAmount': 'sum',
                'customer_id': 'count'
            }).rename(columns={'totalAmount': 'Revenue', 'customer_id': 'Orders'})
            quarterly_stats = quarterly_stats.sort_index(ascending=False)

            # C. Day of Week Analysis
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)
            
            # FIX: Pass observed=False to silence the FutureWarning
            dow_stats = df.groupby('DayOfWeek', observed=False).agg({
                'customer_id': 'count',
                'totalAmount': 'mean'
            }).rename(columns={'customer_id': 'TotalOrders', 'totalAmount': 'AvgOrderValue'})

        except Exception as e:
            logger2.error(f"Trend Analysis Error in Sales Trends Report: {str(e)}")
            return ["This report is currently unavailable due to a calculation update. Please contact support."]

        # --- 4. Customer Quality (Cohort Analysis) ---
        try:
            # Group by Customer to get first order date and lifetime value
            cust_stats = df.groupby('customer_id').agg({
                'createdAt': 'min',          # First Order Date
                'totalAmount': 'sum',        # Lifetime Revenue
            }).rename(columns={'createdAt': 'FirstOrder', 'totalAmount': 'LifetimeRevenue'})
            
            # Extract Join Year
            cust_stats['JoinYear'] = cust_stats['FirstOrder'].dt.year
            
            # Aggregate by Cohort (Join Year)
            cohort_stats = cust_stats.groupby('JoinYear').agg({
                'LifetimeRevenue': ['count', 'mean']
            })
            cohort_stats.columns = ['NewCustomers', 'AvgLifetimeValue']
            cohort_stats = cohort_stats.sort_index(ascending=False)
            
        except Exception as e:
            logger2.error(f"Cohort Analysis Error in Sales Trends Report: {str(e)}")
            return ["This report is currently unavailable due to an issue analyzing customer cohorts."]

        # --- 5. Generate Markdown Report ---
        try:
            lines = []
            lines.append("# Comprehensive Sales & Customer Quality Report")
            lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
            lines.append(f"**Data Range:** {df['createdAt'].min().strftime('%m/%d/%Y')} to {reference_date.strftime('%m/%d/%Y')}")
            lines.append("")
            
            # Executive Summary
            lines.append("## 1. Executive Trend Summary")
            if len(monthly_stats) >= 2:
                curr_m = monthly_stats.iloc[-1]
                prev_m = monthly_stats.iloc[-2]
                trend_icon = "📈" if curr_m['Revenue'] > prev_m['Revenue'] else "📉"
                lines.append(f"- **Latest Month ({curr_m.name}):** ${curr_m['Revenue']:,.2f} ({curr_m['Rev_Growth']:+.1f}% vs prev) {trend_icon}")
            else:
                lines.append("- Not enough data for trend analysis.")
                
            lines.append(f"- **Overall Average Order Value (AOV):** ${df['totalAmount'].mean():,.2f}")
            lines.append("")

            # Quarterly
            lines.append("## 2. Quarterly Performance")
            lines.append("| Quarter | Revenue | Orders | Avg Revenue/Order |")
            lines.append("|---|---|---|---|")
            for q, row in quarterly_stats.iterrows():
                avg = row['Revenue'] / row['Orders'] if row['Orders'] > 0 else 0
                lines.append(f"| {q} | ${row['Revenue']:,.2f} | {int(row['Orders'])} | ${avg:,.2f} |")
                
            lines.append("")
            
            # Monthly
            lines.append("## 3. Monthly Sales History (Last 12 Months)")
            lines.append("| Month | Revenue | Growth | Orders | AOV |")
            lines.append("|---|---|---|---|---|")
            for period, row in monthly_report.head(12).iterrows():
                growth = f"{row['Rev_Growth']:+.1f}%" if pd.notna(row['Rev_Growth']) else "-"
                aov_val = row['AOV'] if pd.notna(row['AOV']) else 0
                lines.append(f"| {period} | ${row['Revenue']:,.2f} | {growth} | {int(row['Orders'])} | ${aov_val:,.2f} |")

            lines.append("")

            # Operational
            lines.append("## 4. Operational Insights: Day of Week")
            lines.append("| Day | Total Orders | Avg Order Size |")
            lines.append("|---|---|---|")
            for day, row in dow_stats.iterrows():
                avg_ord_val = row['AvgOrderValue'] if pd.notna(row['AvgOrderValue']) else 0
                lines.append(f"| {day} | {int(row['TotalOrders'])} | ${avg_ord_val:,.2f} |")

            lines.append("")
            
            # Customer Quality
            lines.append("## 5. Customer Quality (Cohort Analysis)")
            lines.append("Analyzing the value of new customers acquired each year.")
            lines.append("| Join Year | New Customers Acquired | Avg Lifetime Value (CLV) |")
            lines.append("|---|---|---|")
            for year, row in cohort_stats.iterrows():
                avg_ltv = row['AvgLifetimeValue'] if pd.notna(row['AvgLifetimeValue']) else 0
                lines.append(f"| {year} | {int(row['NewCustomers'])} | ${avg_ltv:,.2f} |")

            return lines

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Sales Trends Report: {str(e)}")
            return ["An error occurred while compiling the final report text. Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Sales Trends Orders Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]


AGENT_CONFIG = {
    "orders_agent": {
        "key_metrics_report": _calculate_key_metrics_orders_g,
        "sales_performance_report": _calculate_sales_orders_performance_g,
        "discount_report": _calculate_discount_distribution_g,
        "payment_status_report": _calculate_payment_status_g,
        "fulfillment_report": _calculate_orders_fulfillment_g,
        "sales_trends_report": _sales_trends_orders_report_g
    }
}

import inspect
# Main Async Report Generator
async def group_orders_statistics(
    orders_path, 
    products_path,
    agent_type: str, 
    report_type: str = "full_report"
):
    # --- 1. Validation ---
    if agent_type not in AGENT_CONFIG:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_functions = AGENT_CONFIG[agent_type]

    # --- 2. Determine Scope (Full vs Single) ---

    target_reports = {}
    
    if report_type == "full_report":
        target_reports = agent_functions
    else:
        if report_type not in agent_functions:
             raise ValueError(f"Report '{report_type}' not found for agent '{agent_type}'")
        target_reports = {report_type: agent_functions[report_type]}

    # --- 4. Dynamic Task Creation ---
    tasks = []
    report_names = [] # Keep track of order

    for name, func in target_reports.items():
        sig = inspect.signature(func)
        kwargs = {}
        if 'orders_path' in sig.parameters: kwargs['orders_path'] = orders_path
        if 'products_path' in sig.parameters: kwargs['products_path'] = products_path
  
        # Create the thread task
        tasks.append(asyncio.to_thread(func, **kwargs))
        report_names.append(name)

    
    results_list = await asyncio.gather(*tasks)

    full_report_list = []
    sections_main = {}

    # Zip allows us to pair the Report Name with its Result dynamically
    for name, lines in zip(report_names, results_list):
        if lines:
            section_text = "\n".join(lines)
            sections_main[name] = section_text
            full_report_list.extend(lines)
            full_report_list.append("") # Add newline

    # Add Suggestions Only for Full Report 
    if report_type == "Full Report":
        sections_main["suggestions_div"] = "## Suggestions"
        full_report_list.append("## Suggestions")


    raw_full_report = "\n".join(full_report_list).strip()
    
    return {
        "full_report": raw_full_report,
        "sections": sections_main
    }


# Agent analysis statistics
async def create_agent_sectioned(USER_ID, topic, statistics, agent) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        PROMPT_DEFAULT = "You are a helpful assistant. Answer the user's query based on the data provided."
        topic_instruction_map = {
            "orders_agent": prompt_mcp_topics_orders_agent,
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

async def process_standard_topic(topic, orders_path, products_path, uuid, agent):
    """Logic for standard analysis topics."""
    try:
        start = time.perf_counter()
        
        # 1. Generate Statistics
        statistics_dict = await group_orders_statistics(
            orders_path, products_path, agent_type=agent, report_type=topic
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

async def process_suggestions_topic(topic, orders_path, products_path, uuid, agent):
    """Logic for suggestions analysis topics."""
    try:
        start = time.perf_counter()

        # 1. Generate Statistics
        statistics_of_topic = await group_orders_statistics(
            orders_path, products_path, agent_type=agent, report_type='full_report'
        )

        async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
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


async def worker(semaphore, topic, orders_path, products_path, uuid, agent):
    """
    Router function: Decides which logic to run based on the topic name,
    constrained by the semaphore.
    """
    async with semaphore:
        #print(f"Processing: {topic}")
        
        if topic == "suggestions_div":
            return await process_suggestions_topic(topic, orders_path, products_path, uuid, agent)
        else:
            return await process_standard_topic(topic, orders_path, products_path, uuid, agent)


async def main_batch_orders_process(
    orders_path, 
    products_path, 
    uuid, 
    agent, 
    specific_topic=None
):
    try:
        # Changed print to logger for consistency
        logger2.info(f"Starting batch process: {orders_path}, {products_path}, {uuid}, {agent}, {specific_topic}")
        
        TOPIC_CONFIG = {
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
                worker(sem, topic, orders_path, products_path, uuid, agent)
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
        #print(final_clean_report)
        return final_clean_report, clean_sections

    except Exception as e:
        # Global fallback for unexpected errors (e.g., missing files, memory issues, dict errors)
        logger2.error(f"Critical error in main_batch_orders_process for UUID {uuid}: {str(e)}", exc_info=True)
        
        user_friendly_error = (
            "We encountered an unexpected error while generating your complete report. "
            "Our team has been notified. Please try again shortly."
        )
        return user_friendly_error, {"error": user_friendly_error}






async def main():
    #report = await group_orders_statistics(
    #    orders_path="data\\testing_2\\work_data_folder\\cleaned_real_big_orders.csv",
    #    products_path="data\\testing_2\\work_data_folder\\cleaned_real_big_products.csv",
    #    agent_type="orders_agent",
    #    report_type="full_report"
    #)
    #print(report.get("sections", "No full report generated.").get("sales_trends_report", "Report section not found."))
    report, sections = await main_batch_orders_process(
        orders_path="data\\testing_2\\work_data_folder\\cleaned_real_big_orders.csv",
        products_path="data\\testing_2\\work_data_folder\\cleaned_real_big_products.csv",
        uuid="testing_2",
        agent="orders_agent",
        specific_topic=None) #ange to None for full report
    print(report)

if __name__ == "__main__":
    asyncio.run(main())