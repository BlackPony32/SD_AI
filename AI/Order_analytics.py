import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import asyncio

# Set up logging
logger = logging.getLogger(__name__)

### Preprocessing Functions
def preprocess_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform orders data"""
    try:
        # Convert and filter dates
        orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], utc=True)
        orders_df['month'] = orders_df['createdAt'].dt.to_period('M')
        
        # Standardize statuses
        orders_df['deliveryStatus'] = orders_df['deliveryStatus'].str.strip().str.upper().fillna('UNKNOWN')
        orders_df['paymentStatus'] = orders_df['paymentStatus'].str.strip().str.upper().fillna('UNKNOWN')
        
        # Filter out only 'CANCELED' orders, keep 'PENDING' and archived == False
        orders_df = orders_df[
            (orders_df['orderStatus'] != 'CANCELED') &
            (orders_df['archived'] == False)
        ]
        
        # Clean financial columns
        money_cols = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
        orders_df[money_cols] = orders_df[money_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return orders_df.round({col: 2 for col in money_cols})
    
    except Exception as e:
        logger.error(f"Order preprocessing failed: {str(e)}")
        return pd.DataFrame()

def preprocess_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform products data"""
    try:
        # Clean SKUs and quantities
        products_df['sku'] = products_df['sku'].fillna('MISSING_SKU')
        products_df['quantity'] = pd.to_numeric(products_df['quantity'], errors='coerce').fillna(0)
        products_df['paidQuantity'] = pd.to_numeric(products_df['paidQuantity'], errors='coerce').fillna(0)
        
        # Clean prices
        price_cols = ['price', 'itemDiscountAmount']
        products_df[price_cols] = products_df[price_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return products_df
    except Exception as e:
        logger.error(f"Product preprocessing failed: {str(e)}")
        return pd.DataFrame()

### Analysis Function
async def generate_sales_report(orders_path: str, products_path: str) -> str:
    """Generate comprehensive sales report in markdown format"""
    try:
        
        # Load data
        orders = pd.read_csv(orders_path)
        products = pd.read_csv(products_path)
        
        # Preprocess
        orders = preprocess_orders(orders)
        products = preprocess_products(products)
        
        if orders.empty or products.empty:
            return "## ❌ Error\nNo valid data to generate report"
            
        # Merge datasets
        merged_df = products.merge(
            orders[['id', 'salesDuplicate_name', 'contactDuplicate_name', 
                   'month', 'totalAmount', 'deliveryFee', 'deliveryStatus', 'paymentStatus']],
            left_on='orderId', 
            right_on='id',
            how='inner'
        )
        
        # Helper formatting function
        def usd(value: float) -> str:
            return f"${value:,.2f}"
        
        # Calculate metrics
        # Basic Metrics
        total_sales = orders['totalAmount'].sum()
        total_orders = orders['id'].nunique()
        avg_order_value = total_sales / total_orders if total_orders else 0
        
        # Delivery Metrics
        orders_with_delivery = orders[orders['deliveryFee'] > 0]
        num_orders_with_delivery = orders_with_delivery['id'].nunique()
        total_delivery_fees = orders['deliveryFee'].sum()
        avg_delivery_fee = total_delivery_fees / num_orders_with_delivery if num_orders_with_delivery > 0 else 0
        
        # Discount Analysis
        discount_stats = orders.agg({
            'totalDiscountValue': 'sum',
            'id': lambda x: x[orders['totalDiscountValue'] > 0].nunique()
        })
        
        # Product Analysis with correct revenue calculation
        merged_df['item_revenue'] = (merged_df['price'] - merged_df['itemDiscountAmount']) * merged_df['quantity']
        product_stats = merged_df.groupby('sku').agg(
            total_quantity=('quantity', 'sum'),
            total_revenue=('item_revenue', 'sum')
        ).sort_values('total_revenue', ascending=False)
        product_stats['avg_selling_price'] = product_stats['total_revenue'] / product_stats['total_quantity']
        
        # Fulfillment Analysis
        possible_delivery_statuses = ['FULFILLED', 'PARTIALLY_FULFILLED', 'UNFULFILLED', 'UNKNOWN']
        fulfillment_counts = orders['deliveryStatus'].value_counts().reindex(possible_delivery_statuses, fill_value=0)
        fulfillment_percent = (fulfillment_counts / total_orders * 100).round(1)
        
        # Payment Status Analysis
        possible_payment_statuses = ['PAID', 'PARTIALLY_PAID', 'UNPAID', 'PENDING', 'UNKNOWN']
        payment_status_counts = orders['paymentStatus'].value_counts().reindex(possible_payment_statuses, fill_value=0)
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
        monthly_sales = orders.groupby('month').agg(
            total_sales=('totalAmount', 'sum'),
            order_count=('id', 'nunique')
        ).reset_index()
        
        # Build Markdown Report
        md = []
        md.append("# Sales Performance Report\n")
        
        # Key Metrics
        md.append("##  Key Metrics")
        md.append(f"- **Total Sales:** {usd(total_sales)}")
        md.append(f"- **Total Orders:** {total_orders}")
        md.append(f"- **Average Order Value:** {usd(avg_order_value)}")
        md.append(f"- **Total Discounts Given:** {usd(discount_stats['totalDiscountValue'])}")
        md.append(f"- **Orders with Discounts:** {discount_stats['id']} ({discount_stats['id']/total_orders:.1%})")
        md.append(f"- **Total Delivery Fees:** {usd(total_delivery_fees)}")
        md.append(f"- **Orders with Delivery Fees:** {num_orders_with_delivery} ({num_orders_with_delivery/total_orders:.1%})")
        md.append(f"- **Average Delivery Fee:** {usd(avg_delivery_fee)}")
        md.append("---\n")
        
        # Payment Status Analysis
        md.append("## Payment Status Analysis")
        for status in possible_payment_statuses:
            count = payment_status_counts[status]
            percent = payment_status_percent[status]
            md.append(f"- **{status.title()}:** {count} orders ({percent:.1f}%)")
        md.append("---\n")
        
        # Fulfillment Analysis
        md.append("## Fulfillment Analysis")
        for status in possible_delivery_statuses:
            count = fulfillment_counts[status]
            percent = fulfillment_percent[status]
            md.append(f"- **{status.title()}:** {count} orders ({percent:.1f}%)")
        if len(fulfillment_counts[fulfillment_counts > 0]) == 1:
            md.append("\n*Note: All orders have the same delivery status due to filtering conditions.*")
        md.append("---\n")
        
        # Top Products
        md.append("## Top Performing Products")
        md.append("| SKU | Quantity Sold | Avg Selling Price | Total Revenue |")
        md.append("|-----|---------------|-------------------|---------------|")
        for sku, row in product_stats.head(10).iterrows():
            md.append(f"| {sku} | {row['total_quantity']} | {usd(row['avg_selling_price'])} | {usd(row['total_revenue'])} |")
        md.append("\n---\n")
        
        # Sales Team Performance
        md.append("## Sales Team Performance")
        md.append("| Salesperson | Total Sales | Orders | Avg Order Value |")
        md.append("|-------------|-------------|--------|-----------------|")
        for name, row in salesperson_stats.iterrows():
            md.append(f"| {name} | {usd(row['total_sales'])} | {row['order_count']} | {usd(row['avg_order_value'])} |")
        md.append("\n---\n")
        
        # Customer Purchase Analysis
        md.append("## Customer Purchase Analysis")
        md.append("| Customer | Total Purchases | Orders | Avg Order Value |")
        md.append("|----------|-----------------|--------|-----------------|")
        for name, row in customer_stats.head(10).iterrows():
            md.append(f"| {name} | {usd(row['total_purchases'])} | {row['order_count']} | {usd(row['avg_order_value'])} |")
        md.append("\n---\n")
        
        # Monthly Trends
        md.append("## Monthly Sales Trends")
        md.append("| Month | Total Sales | Orders | Avg Sales/Order |")
        md.append("|-------|-------------|--------|-----------------|")
        for _, row in monthly_sales.iterrows():
            avg = row['total_sales'] / row['order_count'] if row['order_count'] else 0
            md.append(f"| {row['month']} | {usd(row['total_sales'])} | {row['order_count']} | {usd(avg)} |")
        md.append("\n---")
        
        # Recommendations
        md.append("## Recommendations")
        md.append("- **Focus** on top performing products from the SKU list")
        md.append("- **Investigate** reasons for partial/unfulfilled orders")
        md.append("- **Create incentives** for sales team members with lowest performance")
        md.append("- **Analyze** monthly trends for seasonal patterns")
        
        return "\n".join(md)
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return f"## ❌ Error\nReport generation failed: {str(e)}"

#async def main():
#    report = await generate_sales_report("ord.csv", "prod.csv")
#    print(report)
#
#if __name__ == "__main__":
#    asyncio.run(main())