import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import asyncio
import aiofiles
from pathlib import Path
from AI.single_customer_analyze.create_process_data import create_user_data
from AI.single_customer_analyze.Activities_analytics import analyze_activities

# Set up logging
logger = logging.getLogger(__name__)
log_file_path = "project_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

#____start of additional func
def get_top_products_for_contact(contact_name, orders_df, products_df, top_n=3):
    """Helper function to get top products for a specific contact"""
    contact_orders = orders_df[orders_df['contactDuplicate_name'] == contact_name]
    merged = pd.merge(contact_orders, products_df, left_on='id', right_on='orderId')
    product_counts = merged['manufacturerName'].value_counts().head(top_n)
    return product_counts.index.tolist()

def top_new_contact(ord_df, products_df):
    #ord_df = pd.read_csv(file_path)
    #products_df = pd.read_csv(products_file_path)
    
    successful_orders = ord_df[(ord_df['orderStatus'] == 'COMPLETED') & 
                              (ord_df['paymentStatus'] == 'PAID') & 
                              (ord_df['deliveryStatus'] == 'FULFILLED')]
    
    if successful_orders.empty:
        return ("No data", [])
    
    best_contact = successful_orders['contactDuplicate_name'].value_counts().idxmax()
    top_products = get_top_products_for_contact(best_contact, ord_df, products_df)
    
    return (best_contact, top_products)

def top_reorder_contact(ord_df, products_df):
    #ord_df = pd.read_csv(file_path)
    #products_df = pd.read_csv(products_file_path)
    
    repeat_contacts = ord_df.groupby('contactDuplicate_name').filter(lambda x: len(x) > 1)
    
    if repeat_contacts.empty:
        return ("No data", [])
    
    best_contact = repeat_contacts['contactDuplicate_name'].value_counts().idxmax()
    top_products = get_top_products_for_contact(best_contact, ord_df, products_df)
    
    return (best_contact, top_products)

def format_am_pm(hour):
    """Convert 24-hour format to 12-hour AM/PM format"""
    if hour == 0:
        return "12:00 AM"
    elif 1 <= hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"

def peak_visit_time(df):
    #df = pd.read_csv(file_path)
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    
    # Calculate best day
    df['weekday'] = df['createdAt'].dt.day_name()
    day_counts = df['weekday'].value_counts()
    best_day = day_counts.idxmax() if not day_counts.empty else "No data"
    
    # Calculate peak time window
    df['hour'] = df['createdAt'].dt.hour
    hour_counts = df['hour'].value_counts()
    
    if not hour_counts.empty:
        peak_hour = hour_counts.idxmax()
        end_hour = (peak_hour + 1) % 24
        time_window = f"{format_am_pm(peak_hour)} - {format_am_pm(end_hour)}"
    else:
        time_window = "No data"
    
    return (best_day, time_window)

def get_contact_peak_time(contact_name, orders_df):
    """Calculate peak visit time for individual contact"""
    contact_orders = orders_df[orders_df['contactDuplicate_name'] == contact_name]
    
    if contact_orders.empty:
        return ("No data", "No data")
    
    # Convert to datetime and extract time features
    contact_orders = contact_orders.copy()
    contact_orders['createdAt'] = pd.to_datetime(contact_orders['createdAt'])
    contact_orders['weekday'] = contact_orders['createdAt'].dt.day_name()
    contact_orders['hour'] = contact_orders['createdAt'].dt.hour
    
    # Find most common weekday
    day_counts = contact_orders['weekday'].value_counts()
    best_day = day_counts.idxmax() if not day_counts.empty else "No pattern"
    
    # Find most common hour
    hour_counts = contact_orders['hour'].value_counts()
    if not hour_counts.empty:
        peak_hour = hour_counts.idxmax()
        end_hour = (peak_hour + 1) % 24
        time_window = f"{format_am_pm(peak_hour)} - {format_am_pm(end_hour)}"
    else:
        time_window = "No pattern"
    
    return (best_day, time_window)

def customer_insights(ord_df, products_df):
    #ord_df = pd.read_csv(file_path)
    #products_df = pd.read_csv(products_file_path)
    
    # Get unique contacts with their order counts
    contacts = ord_df['contactDuplicate_name'].value_counts().reset_index()
    contacts.columns = ['contact', 'order_count']
    
    results = []
    for _, row in contacts.iterrows():
        contact = row['contact']
        count = row['order_count']
        
        # Get peak time
        best_day, time_window = get_contact_peak_time(contact, ord_df)
        
        # Get top products
        top_products = get_top_products_for_contact(contact, ord_df, products_df)
        
        results.append({
            'Contact': contact,
            'Total Orders': count,
            'Peak Day': best_day,
            'Peak Time': time_window,
            'Top Products': ', '.join(top_products) if top_products else 'N/A'
        })
    
    return pd.DataFrame(results)
#_____end of additional report part____

### Preprocessing Functions
def preprocess_orders(orders_df: pd.DataFrame):
    """Clean and transform orders data, returning processed DataFrame and an optional error message."""
    # Check for required columns
    required_columns = ['createdAt', 'deliveryStatus', 'paymentStatus', 'orderStatus', 'archived', 
                        'totalAmount', 'totalDiscountValue', 'deliveryFee']
    missing_cols = [col for col in required_columns if col not in orders_df.columns]
    if missing_cols:
        return pd.DataFrame(), f"The orders file is missing these required columns: {', '.join(missing_cols)}. Please check the file."

    try:
        # Convert and filter dates
        try:
            orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], utc=True)
            # Changed from .dt.to_period('M') to .dt.strftime('%m/%Y')
            orders_df['month'] = orders_df['createdAt'].dt.strftime('%m/%Y')
            logger.info("Dates processed successfully")
        except Exception as e:
            logger.error(f"Error processing dates: {str(e)}")
            return pd.DataFrame(), "There was an issue with the date format in the orders file. Please ensure the 'createdAt' column is in a valid date format."

        # Standardize statuses
        try:
            orders_df['deliveryStatus'] = orders_df['deliveryStatus'].str.strip().str.upper().fillna('UNKNOWN')
            orders_df['paymentStatus'] = orders_df['paymentStatus'].str.strip().str.upper().fillna('UNKNOWN')
            logger.info("Statuses standardized successfully")
        except Exception as e:
            logger.error(f"Error standardizing statuses: {str(e)}")
            return pd.DataFrame(), "There was an issue processing the status columns in the orders file. Please check the data."

        # Filter out 'CANCELED' orders and keep archived == False
        try:
            orders_df = orders_df[
                (orders_df['orderStatus'] != 'CANCELED') &
                (orders_df['archived'] == False)
            ]
            if orders_df.empty:
                return pd.DataFrame(), "No valid orders found after filtering. All orders are either canceled or archived. Please try a different file."
            logger.info(f"Filtered orders, remaining: {len(orders_df)}")
        except Exception as e:
            logger.error(f"Error filtering orders: {str(e)}")
            return pd.DataFrame(), "There was an issue filtering the orders. Please check the 'orderStatus' and 'archived' columns."

        # Clean financial columns
        try:
            money_cols = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
            orders_df[money_cols] = orders_df[money_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            logger.info("Financial columns cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning financial columns: {str(e)}")
            return pd.DataFrame(), "There was an issue processing the financial columns in the orders file. Please ensure they contain numeric data."

        orders_df = orders_df.round({col: 2 for col in money_cols})
        return orders_df

    except Exception as e:
        logger.error(f"Order preprocessing failed: {str(e)}")
        return pd.DataFrame(), "Something went wrong while processing the orders data. Please try another file."

def preprocess_products(products_df: pd.DataFrame):
    """Clean and transform products data, returning processed DataFrame and an optional error message."""
    # Check for required columns
    required_columns = ['sku', 'quantity', 'paidQuantity', 'price', 'itemDiscountAmount', 'orderId']
    missing_cols = [col for col in required_columns if col not in products_df.columns]
    if missing_cols:
        return pd.DataFrame(), f"The products file is missing these required columns: {', '.join(missing_cols)}. Please check the file."

    try:
        # Clean SKUs and quantities
        try:
            products_df['sku'] = products_df['sku'].fillna('MISSING_SKU')
            products_df['quantity'] = pd.to_numeric(products_df['quantity'], errors='coerce').fillna(0)
            products_df['paidQuantity'] = pd.to_numeric(products_df['paidQuantity'], errors='coerce').fillna(0)
            logger.info("SKUs and quantities cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning SKUs and quantities: {str(e)}")
            return pd.DataFrame(), "There was an issue processing the SKU or quantity columns in the products file. Please check the data."

        # Clean prices
        try:
            price_cols = ['price', 'itemDiscountAmount']
            products_df[price_cols] = products_df[price_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            logger.info("Prices cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning prices: {str(e)}")
            return pd.DataFrame(), "There was an issue processing the price columns in the products file. Please ensure they contain numeric data."

        return products_df

    except Exception as e:
        logger.error(f"Product preprocessing failed: {str(e)}")
        return pd.DataFrame(), "Something went wrong while processing the products data. Please try another file."

### Analysis Function
async def generate_sales_report(orders_path: str, products_path: str, customer_id: str) -> str:
    """Generate comprehensive sales report in markdown format with soft error handling."""
    try:
        # Step 0 - process data
        try:
            await create_user_data(orders_path, products_path, customer_id)
            orders = pd.read_csv(f'data/{customer_id}/work_ord.csv')
            products = pd.read_csv(f'data/{customer_id}/work_prod.csv')
        except Exception as e:
            logger.error(f"create work data error: {e}")

        # Step 1: Load data
        try:
            if orders.empty or products.empty:
                return {
                    "full_report": "There is not enough information to generate the metrics",
                    "sections": {}
                }
            else:
                logger.info(f"Loaded {len(orders)} orders and {len(products)} products")
        except FileNotFoundError as e:
            logger.warning(f"File not found: {str(e)}")
            return {
                "full_report": "**Oops!** We were unsuccessful in processing your data.",
                "sections": {}
            }
        except pd.errors.ParserError as e:
            logger.warning(f"Error parsing CSV: {str(e)}")
            return {
                "full_report": "**Hmm,** there was a problem reading orders or products info. Please check the data and try again",
                "sections": {}
            }
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            return {
                "full_report": "**Hmm,** there was a problem reading orders or products info. Please check the data and try again.",
                "sections": {}
            }

        # Step 2: Preprocess data
        orders = preprocess_orders(orders)
        products = preprocess_products(products)

        # Step 3: Merge datasets
        try:
            merged_df = products.merge(
                orders[['id', 'salesDuplicate_name', 'contactDuplicate_name', 
                        'month', 'totalAmount', 'deliveryFee', 'deliveryStatus', 'paymentStatus']],
                left_on='orderId', 
                right_on='id',
                how='inner'
            )
            if merged_df.empty:
                return "**Hmm,** the orders and products data didn’t match up. Please ensure there are corresponding 'orderId' and 'id' values in both files."
            logger.info(f"Merged DataFrame has {len(merged_df)} rows")
        except Exception as e:
            logger.error(f"Error merging dataframes: {str(e)}")
            return {
                "full_report": "**Hmm,** we couldn’t combine the orders and products data. Please make sure the data match up correctly",
                "sections": {}
            }

        # Helper formatting functions
        def usd(value: float) -> str:
            if float(value).is_integer():
                return f"${value:,.0f}"
            return f"${value:,.2f}"
        

        def format_status(status: str) -> str:
            return status.replace('_', ' ')

        def format_percentage(value: float) -> str:
            if float(value).is_integer():
                return f"{int(value)}%"
            return f"{value:.1f}%"
        
        # Step 4: Calculate metrics
        try:
            # Total sales by payment and delivery status
            total_sales_by_status = orders.groupby(['paymentStatus', 'deliveryStatus'])['totalAmount'].sum().reset_index()

            # Discount Analysis
            total_discount_amount = orders['totalDiscountValue'].sum()
            num_orders_with_discounts = (orders['totalDiscountValue'] > 0).sum()
            orders['discount_category'] = orders['appliedDiscountsType'].fillna('NONE')
            orders.loc[(orders['discount_category'] == 'NONE') & (orders['totalDiscountValue'] > 0), 'discount_category'] = 'Other Discount'
            percentage_orders_with_discounts = num_orders_with_discounts / len(orders) * 100
            discount_distribution = orders.groupby('discount_category').agg(
                num_orders=('id', 'count'),
                total_discount=('totalDiscountValue', 'sum')
            ).reset_index()

            # Delivery Analysis
            total_delivery_fees = orders['deliveryFee'].sum()
            num_orders_with_delivery = (orders['deliveryFee'] > 0).sum()
            avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0

            # Key Metrics
            total_sales = orders['totalAmount'].sum()
            total_orders = len(orders)
            avg_order_value = total_sales / total_orders if total_orders > 0 else 0

            # Product Analysis
            merged_df['item_revenue'] = (merged_df['price'] - merged_df['itemDiscountAmount']) * merged_df['quantity']
            product_stats = merged_df.groupby('sku').agg(
                total_quantity=('quantity', 'sum'),
                total_revenue=('item_revenue', 'sum')
            ).sort_values('total_revenue', ascending=False)
            product_stats['avg_selling_price'] = product_stats['total_revenue'] / product_stats['total_quantity']

            # Fulfillment Analysis
            existing_delivery_statuses = orders['deliveryStatus'].unique()
            fulfillment_counts = orders['deliveryStatus'].value_counts()
            fulfillment_percent = (fulfillment_counts / total_orders * 100).round(1)

            # Payment Status Analysis
            existing_payment_statuses = orders['paymentStatus'].unique()
            payment_status_counts = orders['paymentStatus'].value_counts()
            payment_status_percent = (payment_status_counts / total_orders * 100).round(1)

            # Salesperson Performance
            salesperson_stats = orders.groupby('salesDuplicate_name').agg(
                total_sales=('totalAmount', 'sum'),
                order_count=('id', 'nunique'),
                avg_order_value=('totalAmount', 'mean')
            ).sort_values('total_sales', ascending=False)

            # Customer Purchase Analysis
            customer_stats = orders.groupby('contactDuplicate_name').agg(
                total_purchases=('totalAmount', 'sum'),
                order_count=('id', 'nunique'),
                avg_order_value=('totalAmount', 'mean')
            ).sort_values('total_purchases', ascending=False)

            # Monthly Trends
            merged_df['product'] = merged_df.get('name', 'Product') + ' - ' + merged_df['sku']
            product_sales = merged_df.groupby(['month', 'product'])['item_revenue'].sum().reset_index()
            top_products = product_sales.loc[product_sales.groupby('month')['item_revenue'].idxmax()]
            monthly_sales = orders.groupby('month').agg(
                total_sales=('totalAmount', 'sum'),
                order_count=('id', 'nunique')
            ).reset_index()

            logger.info("Metrics calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "full_report": "**Oh no!** We hit a snag while calculating the sales stats. This might be due to missing or incorrect data.",
                "sections": {}
            }

        # Step 5: Generate recommendations
        try:
            # Top Products
            top_skus = product_stats.head(3).index
            product_names = merged_df.groupby('sku')['name'].first() if 'name' in merged_df.columns else pd.Series(index=top_skus, data='Unknown Product')
            top_products_list = [f"{product_names.get(sku, 'Unknown Product')} - {sku}" for sku in top_skus]

            # Sales Team
            if not salesperson_stats.empty:
                top_salesperson = salesperson_stats.iloc[0].name
                bottom_salespeople = salesperson_stats.tail(3).index.tolist() if len(salesperson_stats) > 3 else []
            else:
                top_salesperson = None
                bottom_salespeople = []

            # Customers
            top_customers = customer_stats.head(3).index.tolist() if not customer_stats.empty else []

            # Monthly Trends
            if len(monthly_sales) > 1:
                monthly_sales = monthly_sales.sort_values('month')
                trends = monthly_sales['total_sales'].diff().dropna()
                trend = "increasing" if all(trends > 0) else "decreasing" if all(trends < 0) else "fluctuating"
            else:
                trend = "insufficient data"

            # Fulfillment Issues
            fulfillment_rec = None
            if 'UNFULFILLED' in fulfillment_percent and fulfillment_percent['UNFULFILLED'] > 10:
                fulfillment_rec = f"Improve the fulfillment process to reduce unfulfilled orders (currently {format_percentage(fulfillment_percent['UNFULFILLED'])})."

            # Payment Issues
            payment_rec = None
            if 'UNPAID' in payment_status_percent and payment_status_percent['UNPAID'] > 10:
                payment_rec = f"Enhance payment collection to address unpaid orders (currently {format_percentage(payment_status_percent['UNPAID'])})."
            
            logger.info("Recommendations generated successfully")
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "full_report": "**Hmm,** we couldn’t generate suggestions this time. There might not be enough data or the data might be incomplete. Please check your data and try again.",
                "sections": {}
            }

        # Step 6: Build Markdown Report
        try:
            full_md = []
            sections = {}

            def add_section(name, lines):
                sections[name] = "\n".join(lines)
                full_md.extend(lines)
                full_md.append("")

            def clean_number(n):
                return int(n) if float(n).is_integer() else round(n, 2)

            def format_month(month: str) -> str:
                """Convert MM/YYYY to MM/YY"""
                parts = month.split('/')
                if len(parts) == 2:
                    return f"{parts[0]}/{parts[1][-2:]}"
                return month
    
            # 1) Total Sales by Status
            lines = [
                "# Sales Performance Report",
                "",
                "## Total Sales by Payment and Delivery Status",
                "| Payment Status | Delivery Status | Total Sales |",
                "|----------------|-----------------|-------------|",
            ]
            for _, row in total_sales_by_status.iterrows():
                lines.append(f"| {format_status(row['paymentStatus'])} | {format_status(row['deliveryStatus'])} | {usd(row['totalAmount'])} |")
            add_section("total_sales_by_status", lines)

            # 2) Key Metrics
            lines = [
                "## Key Metrics",
                f"- **Total Sales:** {usd(total_sales)}",
                f"- **Total Orders:** {total_orders}",
                f"- **Average Order Value:** {usd(avg_order_value)}",
                f"- **Total Discounts Given:** {usd(total_discount_amount)}",
                f"- **Total Delivery Fees:** {usd(total_delivery_fees)}",
                f"- **Orders with Delivery Fees:** {num_orders_with_delivery} "
                f"({format_percentage(num_orders_with_delivery/total_orders*100)})",
                f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}",
            ]
            add_section("key_metrics", lines)

            # 3) Payment Status Analysis
            lines = ["## Payment Status Analysis"]
            for status in existing_payment_statuses:
                lines.append(f"- **{format_status(status)}:** {payment_status_counts[status]} "
                             f"orders ({format_percentage(payment_status_percent[status])})")
            
            add_section("payment_status_analysis", lines)

            # 4) Discount Distribution
            dd = discount_distribution.copy()
            dd['discount_category'] = dd['discount_category'].replace('NONE', 'No Discount')
            lines = [
                "## Discount Distribution",
                f"- **Orders with Discounts:** {num_orders_with_discounts} "
                f"({format_percentage(percentage_orders_with_discounts)})",
                "",
                "| Discount Type | Number of Orders | Total Discount |",
                "|---------------|------------------|----------------|",
            ]
            for _, row in dd.iterrows():
                lines.append(f"| {format_status(row['discount_category'])} | {row['num_orders']} | {usd(row['total_discount'])} |")
            add_section("discount_distribution", lines)

            # 5) Fulfillment Analysis
            lines = ["## Fulfillment Analysis"]
            for status in existing_delivery_statuses:
                lines.append(f"- **{format_status(status)}:** {fulfillment_counts[status]} "
                             f"orders ({format_percentage(fulfillment_percent[status])})")
            add_section("fulfillment_analysis", lines)

            # 6) Top Performing Products
            lines = [
                "## Top 10 Performing Products",
                "| SKU | Quantity Sold | Avg Selling Price | Total Revenue |",
                "|-----|---------------|-------------------|---------------|",
            ]
            for sku, row in product_stats.head(10).iterrows():
                lines.append(f"| {sku} | {clean_number(row['total_quantity'])} | "
                             f"{usd(row['avg_selling_price'])} | {usd(row['total_revenue'])} |")
            add_section("top_products", lines)

            # 7) Sales Team Performance
            lines = [
                "## Sales Team Performance",
                "| Salesperson | Total Sales | Orders | Avg Order Value |",
                "|-------------|-------------|--------|-----------------|",
            ]
            for name, row in salesperson_stats.iterrows():
                lines.append(f"| {name} | {usd(row['total_sales'])} | "
                             f"{clean_number(row['order_count'])} | {usd(row['avg_order_value'])} |")
            add_section("sales_team_performance", lines)

            # 8) Customer Purchase Analysis
            lines = [
                "## Customer Purchase Analysis",
                "| Customer | Total Purchases | Orders | Avg Order Value |",
                "|----------|-----------------|--------|-----------------|",
            ]
            for name, row in customer_stats.head(10).iterrows():
                lines.append(f"| {name} | {usd(row['total_purchases'])} | "
                             f"{clean_number(row['order_count'])} | {usd(row['avg_order_value'])} |")
            add_section("customer_purchase_analysis", lines)

            # 9) Monthly Sales Trends
            lines = [
                "## Monthly Sales Trends",
                "| Month | Total Sales | Orders | Avg Sales/Order | Top Product |",
                "|-------|-------------|--------|-----------------|-------------|",
            ]
            for _, row in monthly_sales.iterrows():
                avg = row['total_sales'] / row['order_count'] if row['order_count'] else 0
                top_product = top_products[top_products['month'] == row['month']]['product'].values[0]
                lines.append(f"| {format_month(row['month'])} | {usd(row['total_sales'])} | "
                             f"{row['order_count']} | {usd(avg)} | {top_product} |")
            add_section("monthly_sales_trends", lines)

            # 10) Suggestions
            lines = ["## Suggestions"]
            if top_products_list:
                lines.append(f"- **Focus marketing efforts** on top products: {', '.join(top_products_list)}.")
            if top_salesperson:
                lines.append(f"- **Leverage success** of {top_salesperson}.")
            if bottom_salespeople:
                lines.append(f"- **Support underperformers**: training for {', '.join(bottom_salespeople)}.")
            if top_customers:
                lines.append(f"- **Reward loyalty**: deals for {', '.join(top_customers)}.")
            trend_map = {
                "increasing": "- **Sales trending up**: increase inventory or staff.",
                "decreasing": "- **Sales trending down**: investigate causes; consider promotions.",
                "fluctuating": "- **Fluctuating sales**: analyze external factors.",
            }
            lines.append(trend_map.get(trend, "- **Trend analysis**: insufficient data; keep monitoring."))
            if fulfillment_rec:
                lines.append(f"- {fulfillment_rec}")
            if payment_rec:
                lines.append(f"- {payment_rec}")
            lines.append("- **Seasonal analysis**: review patterns over time.")
            #add_section("suggestions", lines)

            # 11) Suggestions (as HTML)
            
            content = "\n".join(lines)

            html_suggestions = (
                "<div id=\"suggestions-block\">\n\n"
                f"{content}\n"
                "\n</div>"
            )
            add_section("suggestions", [html_suggestions])
            # Pass this to your response logic
            result = sections['suggestions_div'] = "".join(html_suggestions)
            #print(result)

            # Additional reports
            try:
                contact_new, products_new = top_new_contact(orders, products)
                contact_re, products_re = top_reorder_contact(orders, products)
                visit_day, visit_time = peak_visit_time(orders)
                result = f"""
                # Sales Optimization Insights
                1.  **Top New Contact**: `{contact_new}`  
                    *Top Products*: {', '.join(products_new) if products_new else 'N/A'}  
                    *Rationale*: Highest number of successfully completed orders with consistent product preferences
                2.  **Top Reorder Contact**: `{contact_re}`  
                    *Top Products*: {', '.join(products_re) if products_re else 'N/A'}  
                    *Rationale*: Most frequent repeat purchases with predictable ordering patterns
                3.  **Peak Visit Time**: `{visit_day}, {visit_time}`  
                    *Rationale*: Historical data shows maximum order creation during this timeframe
                """
                report_dir = os.path.join("data", customer_id)
                path_for_report = os.path.join(report_dir, "additional_info.md")
                async with aiofiles.open(path_for_report, "w") as f:
                    await f.write(result)
                df_insights = customer_insights(orders, products)
                reorder = f"""Top visit time: {df_insights.to_markdown(index=False)}"""
                path_for_reorder = os.path.join(report_dir, "reorder.md")
                async with aiofiles.open(path_for_reorder, "w") as f:
                    await f.write(reorder)
            except Exception as e:
                logger.error(f"Error generating additional report: {str(e)}")

            # Assemble
            full_report = "\n".join(full_md).strip()
            return {
                "full_report": full_report,
                "sections": sections
            }

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "full_report": "**Oh no!** Something went wrong while creating the report.",
                "sections": {}
            }

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return {
            "full_report": "**Oh no!** Something unexpected happened while generating the report. Please try again or contact support.",
            "sections": {}
        }
        
### Main Execution

async def main():
    report = await generate_sales_report("data/aaea5465-b48e-429b-946a-0f6ddc3dd66d/work_ord.csv", "data/aaea5465-b48e-429b-946a-0f6ddc3dd66d/work_prod.csv", 'aaea5465-b48e-429b-946a-0f6ddc3dd66d')
    #print(report["full_report"])           #  entire report
    #print(report["sections"])  # just the Key Metrics part
    print(report["full_report"])

if __name__ == "__main__":
    asyncio.run(main())