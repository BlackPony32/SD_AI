import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import aiofiles

# Set up logging
logger = logging.getLogger(__name__)

# Define required columns for orders and products
order_headers = [
    "orderId", "customerId", "order_createdAt", "order_updatedAt", "orderStatus",
    "paymentStatus", "deliveryStatus", "archived", "totalAmount",
    "totalOrderDiscountValue", "customerDiscountValue",
    "deliveryFee", "type", "createdType"
]

product_headers = [
    "orderId", "id", "name", "product_createdAt", "product_updatedAt", "status", "type",
    "minOrderQTY", "sku", "wholesalePrice", "retailPrice",
    "distributorPrice", "sellingOutOfStock"
]

### Preprocessing Functions

def ensure_columns(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """
    Ensure all required columns are present in the dataframe.
    If a column is missing, add it with a default value of 'no_data_' + column_name.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        required_columns (list): List of column names that should be present.
    
    Returns:
        pd.DataFrame: The dataframe with all required columns.
    """
    for col in required_columns:
        if col not in df.columns:
            df[col] = f"no_data_{col}"
    return df

def preprocess_orders(orders: pd.DataFrame, order_headers: list) -> pd.DataFrame:
    """
    Preprocess the orders dataframe:
    - Ensure all required columns are present with 'no_data_' + column_name for missing ones.
    - Convert numeric columns to proper types, filling non-numeric values with 0.
    
    Args:
        orders (pd.DataFrame): The orders dataframe.
        order_headers (list): List of required column names.
    
    Returns:
        pd.DataFrame: The preprocessed orders dataframe.
    """
    # Ensure all required columns are present
    orders = ensure_columns(orders, order_headers)
    
    # Define numeric columns that should be converted
    numeric_columns = ['totalAmount', 'totalOrderDiscountValue', 'deliveryFee']
    for col in numeric_columns:
        if col in orders.columns:
            # Convert to numeric, coercing errors to NaN, then fill NaN with 0
            orders[col] = pd.to_numeric(orders[col], errors='coerce').fillna(0)
    
    return orders

def preprocess_products(products: pd.DataFrame, product_headers: list) -> pd.DataFrame:
    """
    Preprocess the products dataframe:
    - Ensure all required columns are present with 'no_data_' + column_name for missing ones.
    - Handle specific columns with appropriate default values.
    - Convert numeric columns to proper types.
    
    Args:
        products (pd.DataFrame): The products dataframe.
        product_headers (list): List of required column names.
    
    Returns:
        pd.DataFrame: The preprocessed products dataframe.
    """
    # Ensure all required columns are present
    products = ensure_columns(products, product_headers)
    
    # Handle specific columns with custom defaults
    if 'name' in products.columns:
        products['name'] = products['name'].fillna('no_name')
    if 'sku' in products.columns:
        products['sku'] = products['sku'].fillna('No_given_sku')
    
    # Define numeric columns and convert them
    numeric_columns = ['minOrderQTY', 'wholesalePrice', 'retailPrice']
    for col in numeric_columns:
        if col in products.columns:
            products[col] = pd.to_numeric(products[col], errors='coerce').fillna(0)
    
    return products

### Main Statistics Generation Function

async def generate_statistics(orders_path: str, products_path: str) -> str:
    """
    Generate sales and revenue statistics from orders and products CSV files.
    Returns a markdown-formatted report.
    
    Args:
        orders_path (str): Path to the orders CSV file.
        products_path (str): Path to the products CSV file.
    
    Returns:
        str: A markdown-formatted report with sales and revenue statistics.
    """
    try:
        logger.info(f"Starting statistics generation for orders: {orders_path}, products: {products_path}")

        # Check file existence
        if not Path(orders_path).exists():
            logger.warning(f"File not found {orders_path}")
            return "### AI report function is not available for this user"
        if not Path(products_path).exists():
            logger.warning(f"File not found {orders_path}")
            return "### AI report function is not available for this user"

        # Load data
        try:
            orders = pd.read_csv(orders_path, low_memory=False)
            products = pd.read_csv(products_path, low_memory=False)
        except FileNotFoundError as e:
            logger.warning(f"File not found: {str(e)}")
            return f"### File not found: {str(e)}"
        except pd.errors.ParserError as e:
            logger.warning(f"Error parsing CSV: {str(e)}")
            return f"### Error parsing data: {str(e)}"
        except Exception as e:
            logger.warning("No orders or products data found")
            info_message = f"Error generating report: {str(e)}"
            logger.warning(info_message)
            return f"### Statistics cannot be generated because there is not enough data"
        # Preprocess the dataframes
        orders = preprocess_orders(orders, order_headers)
        products = preprocess_products(products, product_headers)

        # Handle date and time
        if 'order_createdAt_date' in orders.columns and 'order_createdAt_time' in orders.columns:
            orders['order_createdAt'] = pd.to_datetime(
                orders['order_createdAt_date'] + ' ' + orders['order_createdAt_time'],
                format='%m-%d-%Y %H:%M:%S',
                errors='coerce'
            )
            orders = orders.drop(columns=['order_createdAt_date', 'order_createdAt_time'])
        else:
            logger.warning("order_createdAt_date or order_createdAt_time columns not found in orders data")

        # Check for empty datasets
        if orders.empty:
            logger.warning("No orders data found")
            return "### Insufficient Data\nNo meaningful insights can be generated due to empty orders."
        if products.empty:
            logger.warning("No products data found")
            return "### Insufficient Data\nNo meaningful insights can be generated due to empty products data."

        # Check critical columns before merging
        required_columns = ['orderId']
        for col in required_columns:
            if col not in orders.columns or col not in products.columns:
                return f"### Error\nRequired column '{col}' missing in one or both datasets."

        # Merge data
        merged = pd.merge(orders, products, on='orderId', how='left')

        # Revenue Analysis
        product_stats = merged.groupby('sku').agg(
            units_sold=('sku', 'count'),
            total_revenue=('retailPrice', 'sum'),
            total_profit=('retailPrice', lambda x: sum(x) - sum(merged.loc[x.index, 'wholesalePrice']))
        ).sort_values('total_revenue', ascending=False)
        product_stats = product_stats.fillna(0)

        # Inventory Valuation
        total_wholesale = products['wholesalePrice'].sum()
        potential_retail_value = products['retailPrice'].sum()

        # Temporal Patterns
        if 'order_createdAt' in merged.columns:
            merged['month'] = merged['order_createdAt'].dt.strftime('%Y-%m')
            monthly_sales = merged[merged['orderStatus'] == 'COMPLETED'].groupby('month')['totalAmount'].sum()
            monthly_sales = monthly_sales.fillna(0)
            if len(monthly_sales) > 2:
                sales_fluctuation = monthly_sales.pct_change().std() * 100
            else:
                sales_fluctuation = 0
        else:
            monthly_sales = pd.Series()
            sales_fluctuation = 0

        # Order and Payment Status
        status_breakdown = orders['orderStatus'].value_counts(normalize=True).mul(100).fillna(0)
        payment_health = orders.groupby('orderStatus')['paymentStatus'].value_counts(normalize=True).mul(100).fillna(0)

        # Profitability Analysis
        total_revenue = product_stats['total_revenue'].sum()
        total_profit = product_stats['total_profit'].sum()
        total_discounts = orders['totalOrderDiscountValue'].sum()

        ### Generate Markdown Report
        md = []
        md.append("# 📊 **Sales & Revenue Analysis**\n")
        md.append("## 🛍️ **Revenue by Product**")
        md.append(f"Total sales generated **${total_revenue:,.2f}** across **{len(product_stats)} products**:\n")
        md.append("| Product Name        | Revenue  | Units Sold | Profit Margin |")
        md.append("|--------------------|----------|------------|--------------|")
        for sku, row in product_stats.iterrows():
            profit_margin = (row['total_profit'] / row['total_revenue']) * 100 if row['total_revenue'] > 0 else 0
            md.append(f"| **{sku}**   | **${row['total_revenue']:,.2f}**  | **{row['units_sold']}** | **{profit_margin:.1f}%**  |")
        md.append("\n---\n")

        md.append("## 🏆 **Best and Worst-Selling Products**")
        md.append("### 🔝 **Top-Performing Products:**")
        top3 = product_stats.head(3)
        for i, (sku, row) in enumerate(top3.iterrows(), 1):
            medal = "🏅" if i == 1 else "🥈" if i == 2 else "🥉"
            md.append(f"{i}. {medal} **{sku}** – **${row['total_revenue']:,.2f}** revenue (**{row['units_sold']} units sold**)")
        if len(product_stats) > 3:
            md.append("\n### ⚠️ **Underperformer:**")
            md.append(f"🚨 **{product_stats.index[-1]}** – **Only ${product_stats.iloc[-1]['total_revenue']:,.2f}** revenue (**{product_stats.iloc[-1]['units_sold']} units sold**)")
        md.append("\n---\n")

        md.append("## 📦 **Inventory Value**")
        md.append(f"- **Current stock wholesale value:** **${total_wholesale:,.2f}**")
        md.append(f"- **Potential retail value:** **${potential_retail_value:,.2f}**")
        md.append(f"- **Value multiplier:** **{potential_retail_value/total_wholesale:.1f}× markup**")
        md.append("\n---\n")

        md.append("## 📈 **Sales Trends**")
        if not monthly_sales.empty:
            md.append(f"- 📅 **Peak sales month:** **{monthly_sales.idxmax()}** (**${monthly_sales.max():,.2f}**)")
            md.append(f"- 📉 **Lowest activity:** **{monthly_sales.idxmin()}** (**${monthly_sales.min():,.2f}**)")
            if np.isnan(sales_fluctuation):
                md.append(f"- 📊 **Monthly sales fluctuation:** **not enough month data**")
            else:
                md.append(f"- 📊 **Monthly sales fluctuation:** **±{sales_fluctuation:.1f}%**")
        else:
            md.append("- No completed orders or date data available for trend analysis.")
        md.append("\n---\n")

        md.append("## 🛒 **Customer Behavior**")
        md.append(f"- ✅ **Order completion rate:** **{status_breakdown.get('COMPLETED', 0):.1f}%**")
        md.append(f"- ❌ **Abandoned cart rate:** **{status_breakdown.get('PENDING', 0):.1f}%**")
        md.append(f"- 💳 **Payment success in pending orders:** **{payment_health.get(('PENDING', 'PAID'), 0):.1f}%**")
        md.append("\n---\n")

        md.append("## 💰 **Profit Analysis**")
        if total_revenue > 0:
            md.append(f"- **Total profit margin:** **{(total_profit / total_revenue) * 100:.1f}%**")
        else:
            md.append("- **Total profit margin:** **0.0%** (no revenue generated)")
        md.append(f"- **Total discounts given:** **${total_discounts:,.2f}** _(Impact on revenue: minimal)_")
        md.append("\n---\n")

        md.append("### 🔎 **Key Insights & Recommendations:**")
        if not product_stats.empty:
            top_skus = ', '.join(map(str, product_stats.head(2).index))
            md.append(f"✔️ **Top sellers**: {top_skus} should be prioritized in promotions.")
        md.append("✔️ **Inventory strategy**: Consider adjusting stock levels based on sales trends.")
        md.append(f"⚠️ **Abandoned cart rate** is **high ({status_breakdown.get('PENDING', 0):.1f}%)**—optimize checkout flow to reduce drop-offs.")
        if not monthly_sales.empty:
            md.append(f"📅 **Sales peak in {monthly_sales.idxmax()}**—capitalize on seasonal trends with targeted marketing.")
        md.append("\n---\n")

        logger.info("Successfully generated statistics report")
        return '\n'.join(md)

    except Exception as e:
        error_message = f"Error generating statistics: {str(e)}"
        logger.error(error_message, exc_info=True)
        return f"### Error\n{error_message}"

### Example Usage
# Uncomment the following lines to test the program
# async def main():
#     report = await generate_statistics("orders.csv", "products.csv")
#     print(report)
#
# if __name__ == "__main__":
#     asyncio.run(main())