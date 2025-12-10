import asyncio
import csv
from collections import defaultdict
from functools import partial
import time
import os

import numpy as np
import pandas as pd
from collections import defaultdict

from AI.utils import get_logger
logger2 = get_logger("logger2", "project_log_many.log", False)


def process_data(uuid):
    """Process customer, order, and product data, saving the result to 'customers_state_final.csv' with left joins to retain all product rows."""
    
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

    # First merge: Left join to retain all products
    merged = pd.merge(
        products,
        orders,
        left_on=['orderId'],
        right_on=['order_id'],
        how='left'  # Retain all products, even without matching orders
    )
    
    # Second merge: Left join to retain all rows from merged dataframe
    final_df = pd.merge(
        merged,
        customers,
        on='customer_name',
        how='left'  # Retain all rows, even without matching customers
    )

    final_df['order_date'] = pd.to_datetime(final_df['order_date'], errors='coerce')  # Handle missing dates
    final_df.drop(columns=['orderId'], inplace=True, errors='ignore')  # Drop redundant column if exists

    # Save to CSV
    final_df.to_csv(f'data/{uuid}/customers_state_final.csv', index=False)


import pandas as pd
from collections import defaultdict

def generate_report(uuid):
    """Generate a sales analysis report using pandas with correct revenue calculation"""
    def format_currency(amount):
        return f"${amount:,.2f}"

    # Read and preprocess data
    try:
        df = pd.read_csv(f'data/{uuid}/customers_state_final.csv')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Clean data
    df['Full_cost_withDisc'] = pd.to_numeric(df['Full_cost_withDisc'], errors='coerce').fillna(0)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    
    if 'name' not in df.columns and 'product_name' in df.columns:
        df['name'] = df['product_name'] # Fallback
        
    df['name'] = df['name'].fillna('Unknown Product')
    df['sku'] = df['sku'].fillna('No SKU')
    df['product_variant'] = df['name'].astype(str) + ' - ' + df['sku'].astype(str)
    
    unique_orders = df[['order_id', 'shippingAddress_state', 'orderStatus']].drop_duplicates()
    
    # State aggregations using actual revenue
    state_agg = df.groupby('shippingAddress_state').agg(
        total_sales=('Full_cost_withDisc', 'sum'),
        num_orders=('order_id', pd.Series.nunique),
    ).reset_index()
    
    # Order status counts
    status_counts = unique_orders.groupby(['shippingAddress_state', 'orderStatus']).size().unstack(fill_value=0)
    
    customer_data = df.groupby(['shippingAddress_state', 'customer_name']).agg(
        customer_products=('product_variant', set),
        total_spent=('Full_cost_withDisc', 'sum')
    ).reset_index()

    # Доходи по продуктах у штатах тепер використовують product_variant
    product_revenues_per_state = df.groupby(['shippingAddress_state', 'product_variant'])['Full_cost_withDisc'].sum()

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

    product_variant_agg = df.groupby('product_variant').agg(
        total_sales=('Full_cost_withDisc', 'sum'),
        total_units=('quantity', 'sum')
    ).reset_index()
    
    product_variant_data = {}
    for _, row in product_variant_agg.iterrows():
        product_variant_data[row['product_variant']] = {
            'total_sales': row['total_sales'],
            'total_units': row['total_units']
        }

    # Generate report 
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
        status_counts_dict = data['order_statuses']
        status_notes = []
        if num_orders > 0:
            pending_ratio = status_counts_dict.get('PENDING', 0) / num_orders
            unfulfilled_ratio = status_counts_dict.get('UNFULFILLED', 0) / num_orders
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

        # Top products (up to 3) - тут вже автоматично використовуються product_variant
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

    # 2. Product Performance (Замінено з Product Category Performance)
    report_lines.extend(["", "Product Performance (Variant):", "-"*40])
    sorted_products = sorted(product_variant_data.items(), key=lambda x: x[1]['total_sales'], reverse=True)

    # Top products (Show top 5 variants)
    top_n = 5
    for i, (variant, data) in enumerate(sorted_products[:top_n]):
        total_sales = data['total_sales']
        total_units = data['total_units']
        avg_price = total_sales / total_units if total_units else 0
        report_lines.append(
            f"{variant}: {format_currency(total_sales)} sales | "
            f"{total_units} units | {format_currency(avg_price)} avg price"
        )

    # Other products
    if len(sorted_products) > top_n:
        niche_sales = sum(data['total_sales'] for _, data in sorted_products[top_n:])
        niche_units = sum(data['total_units'] for _, data in sorted_products[top_n:])
        niche_avg = niche_sales / niche_units if niche_units else 0
        report_lines.append(
            f"Other Products ({len(sorted_products) - top_n} variants): {format_currency(niche_sales)} sales | "
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
    if state_data:
        top_state = max(state_data.items(), key=lambda x: x[1]['total_sales'])[0]
        report_lines.append(f"• {top_state} leads in sales volume and order value")
    
    if sorted_products:
        top_product = sorted_products[0][0]
        report_lines.append(f"• {top_product} is the best-selling product variant")
    
    if customer_recommendations and unique_recs:
        report_lines.append(f"• Top recommendation: {unique_recs[0]['customer']} in {unique_recs[0]['state']} for {unique_recs[0]['product']}")

    # Print final report
    output_path = f"data/{uuid}/products_state.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        # print(f"Report generated at {output_path}") # Optional logging
    except Exception as e:
        print(f"Error writing report: {e}")

    # Optional: print to console for debugging if needed
    # print("\n".join(report_lines))

# Asynchronous wrapper functions
async def async_process_data(uuid):
    """Asynchronously process data by running process_data in a separate thread."""
    await asyncio.to_thread(process_data, uuid)

async def async_generate_report(uuid):
    """Asynchronously generate report by running generate_report in a separate thread."""
    report = await asyncio.to_thread(generate_report, uuid)
    return report



# Entry point
if __name__ == "__main__":
    asyncio.run(async_generate_report()) #make_product_per_state_analysis
    
    