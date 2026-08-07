import asyncio
import pandas as pd
import numpy as np
import datetime
import itertools
import os
from itertools import combinations
from collections import Counter
import traceback
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage

from AI.utils import get_logger, combine_sections, calculate_cost
logger2 = get_logger("logger2", "project_log_many.log", False)

def format_month(month: str) -> str:
        """Convert MM/YYYY to MM/YY"""
        parts = month.split('/')
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1][-2:]}"
        return month

# Customer agent predefined functions

def _stopped_ordering_report(orders_path: str, customers_path: str, churn_threshold_days: int = 90) -> list:
    """
    Generates a churn report identifying customers who haven't ordered recently.
    Logs technical errors to 'logger2' and returns a clean, user-friendly report.
    """
    lines = []
    lines.append("# Customer Inactivity Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
    lines.append("")

    try:
        # --- 1. Validation & Loading ---
        if not (os.path.exists(orders_path) and os.path.exists(customers_path)):
             logger2.error(f"Files not found: {orders_path}, {customers_path}")
             return ["Can not generate report: there is not enough data."]

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        
        # Clean column names
        df_orders.columns = df_orders.columns.str.strip().str.replace('\ufeff', '')
        df_customers.columns = df_customers.columns.str.strip().str.replace('\ufeff', '')
        
        # --- 2. Process Orders ---
        # Convert dates
        if 'createdAt' not in df_orders.columns:
            logger2.error("Orders file missing 'createdAt' column.")
             
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])
        
        if df_orders.empty:
            logger2.error("Orders file contains no valid dates.")
            return ["Can not generate report: there is no valid order data."]

        # Reference Date: Max date in the orders file (to handle historical data correctly)
        reference_date = df_orders['createdAt'].max()
        lines.append(f"**Data Reference Date:** {reference_date.strftime('%m/%d/%Y')}")
        lines.append(f"**Threshold:** Customers inactive for > {churn_threshold_days} days")
        lines.append("")
        
        # Aggregate Order Stats by Customer ID
        # Check required columns
        req_cols = ['customer_id', 'totalAmount']
        if not all(col in df_orders.columns for col in req_cols):
            logger2.error(f"Orders file missing required columns: {', '.join(req_cols)}")

        order_stats = df_orders.groupby('customer_id').agg({
            'createdAt': 'max',       # Last order date
            'totalAmount': 'sum',     # Lifetime spend
            'id': 'count'     # Total order count
        }).rename(columns={
            'createdAt': 'LastOrderDate',
            'totalAmount': 'TotalSpend',
            'id': 'OrderCount'
        })
        
        # --- 3. Process Customers (Name Resolution) ---
        # Identify key columns. Usually 'combinedid' matches 'customer_id' in orders.
        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        
        # Find the best name column: 'name' -> 'displayedName' -> 'displayName'
        name_col = 'name'
        if name_col not in df_customers.columns:
            if 'displayedName' in df_customers.columns: name_col = 'displayedName'
            elif 'displayName' in df_customers.columns: name_col = 'displayName'
            else: 
                 # If no name column, use ID as fallback but log warning
                 logger2.warning("No name column found in customers file. Using ID.")
                 name_col = cust_id_col
            
        # --- 4. Merge Data ---
        # Left join: Keep all ordering customers, add names where available
        # Reset index on order_stats to make 'customer_id' a column for merging
        order_stats_reset = order_stats.reset_index()
        
        merged_stats = order_stats_reset.merge(
            df_customers[[cust_id_col, name_col]], 
            left_on='customer_id', 
            right_on=cust_id_col, 
            how='left'
        )
        
        # Fallback: If name is missing in customer file, use the ID
        merged_stats['FinalName'] = merged_stats[name_col].fillna(merged_stats['customer_id'])
        
        # --- 5. Calculate Churn ---
        merged_stats['DaysSinceLastOrder'] = (reference_date - merged_stats['LastOrderDate']).dt.days
        
        # Filter: Days > Threshold
        churned = merged_stats[merged_stats['DaysSinceLastOrder'] > churn_threshold_days].copy()
        
        # Sort: Highest Spend First (High Value Churn)
        churned = churned.sort_values(by='TotalSpend', ascending=False)
        
        # --- 6. Generate Report Content ---
        # Metrics
        total_customers = len(order_stats)
        total_churned = len(churned)
        churn_rate = (total_churned / total_customers * 100) if total_customers > 0 else 0
        lost_revenue = churned['TotalSpend'].sum()
        
        lines.append("## Executive Summary")
        lines.append(f"- **Total Active Customers:** {total_customers}")
        lines.append(f"- **Inactive (Churned) Customers:** {total_churned} ({churn_rate:.1f}%)")
        lines.append(f"- **Total Lifetime Value of Inactive Customers:** ${lost_revenue:,.2f}")
        lines.append("")
        
        lines.append("## Inactive Customers List - Top 20 by revenue")
        lines.append("This list helps you identify high-value customers who have stopped ordering. Prioritize contacting those at the top.")
        lines.append("- **Days Inactive:** Days since their last purchase.")
        lines.append("- **Total Spend:** Total amount they have spent with you historically (High value = Priority).")
        lines.append("")
        
        headers = ["Customer Name", "Last Order Date", "Days Inactive", "Total Spend", "Orders"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        if churned.empty:
             lines.append("| No customers found matching criteria | - | - | - | - |")
        else:
            for _, row in churned.head(20).iterrows():
                name = str(row['FinalName']).replace("|", "-") # Clean pipe characters for markdown table
                last = row['LastOrderDate'].strftime('%m/%d/%Y')
                days = str(int(row['DaysSinceLastOrder']))
                spend = f"${row['TotalSpend']:,.2f}"
                count = str(row['OrderCount'])
                lines.append(f"| {name} | {last} | {days} | {spend} | {count} |")
            
        return lines

    except Exception as e:
        logger2.error(f"Critical Error in Stopped Ordering Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _opportunity_report(orders_path, customers_path, products_path) -> list:
    """
    Generates an opportunity report.
    Logs errors to 'logger2' and includes professional, concise table guides.
    """
    lines = []
    lines.append("# Opportunity Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
    lines.append("")

    # --- 1. Load & Clean Data ---
    try:
        if not (os.path.exists(orders_path) and os.path.exists(customers_path) and os.path.exists(products_path)):
            logger2.error(f"Files not found: {orders_path}, {customers_path}, {products_path}")
            return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')

        # Clean Columns
        for df in [df_orders, df_customers, df_products]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # Parse Dates
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])
        if df_orders.empty: logger2.error("No valid orders found.")

        # Customer Name Logic
        if 'combinedid' in df_customers.columns: cust_id_col = 'combinedid'
        elif 'id' in df_customers.columns: cust_id_col = 'id'
        else: logger2.error("Customer ID column missing.")

        name_col = 'name'
        if name_col not in df_customers.columns:
             if 'displayedName' in df_customers.columns: name_col = 'displayedName'
             elif 'displayName' in df_customers.columns: name_col = 'displayName'
             else: name_col = cust_id_col

        # Merge Customers
        cust_subset = df_customers[[cust_id_col, name_col]].rename(columns={cust_id_col: 'CustomerPK', name_col: 'CustomerNameVal'})
        df_orders_merged = df_orders.merge(cust_subset, left_on='customer_id', right_on='CustomerPK', how='left')
        df_orders_merged['CustomerName'] = df_orders_merged['CustomerNameVal'].fillna(df_orders_merged['customer_id'])

        # Product Display Names
        if 'sku' not in df_products.columns: df_products['sku'] = ''
        df_products['sku'] = df_products['sku'].fillna('')
        df_products['DisplayName'] = df_products.apply(
            lambda x: f"{x['name']} ({x['sku']})" if x['sku'] else x['name'], axis=1
        )

    except Exception as e:
        logger2.error(f"Data Load Error: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

    # --- 2. Top Products ---
    try:
        lines.append("## 1. Top Performing Products")
        lines.append("*Identifies key drivers of revenue and order volume.*")
        lines.append("- **Role:** 'Stars' excel in both revenue and volume; 'Cash Cows' drive revenue; 'Traffic Builders' drive frequency.")
        lines.append("- **Revenue & Orders:** Total sales value and unique order count.")
        lines.append("")
        
        # Safe aggregation
        cols = df_products.columns
        if 'totalAmount' not in cols or 'orderId' not in cols: raise ValueError("Missing required product columns.")
        
        prod_stats = df_products.groupby('DisplayName').agg({
            'totalAmount': 'sum',
            'orderId': 'nunique',
            'price': 'mean' 
        }).rename(columns={'totalAmount': 'Revenue', 'orderId': 'Freq'})
        
        top_rev = prod_stats.sort_values(by='Revenue', ascending=False).head(15)
        top_freq = prod_stats.sort_values(by='Freq', ascending=False).head(15)
        combined_index = list(set(top_rev.index) | set(top_freq.index))
        combined_stats = prod_stats.loc[combined_index].copy()
        
        combined_stats['Role'] = combined_stats.apply(
            lambda r: "Star" if (r.name in top_rev.index and r.name in top_freq.index) else 
                      ("Cash Cow" if r.name in top_rev.index else "Traffic Builder"), axis=1
        )
        combined_stats = combined_stats.sort_values('Revenue', ascending=False)
        
        lines.append("| Product | Role | Revenue | Orders | Avg Price |")
        lines.append("|---|---|---|---|---|")
        for name, row in combined_stats.iterrows():
            lines.append(f"| {name} | {row['Role']} | ${row['Revenue']:,.2f} | {row['Freq']} | ${row['price']:,.2f} |")

    except Exception as e:
        logger2.error(f"Top Products Error: {str(e)}")
        lines.append("*(Section unavailable)*")

    lines.append("")

    # --- 3. Bundle Opportunities ---
    try:
        lines.append("## 2. Bundle Opportunities & Cross-Selling")
        lines.append("*Highlights high-frequency product pairings and calculates unrealized revenue from partial-bundle buyers.*")
        lines.append("- **Potential Revenue:** Estimated value of cross-selling the missing item to target customers.")
        lines.append("- **Missed Opportunity (Targets):** Specific customers who buy one item of the bundle but not the other.")
        lines.append("")
        
        basket = df_products.groupby('orderId')['DisplayName'].apply(set)
        
        pair_counts = Counter()
        for items in basket:
            items_list = sorted(list(items))
            if len(items_list) > 1:
                pair_counts.update(combinations(items_list, 2))
                
        # Scan Top 30 frequent bundles to find those with REAL opportunities
        candidates = pair_counts.most_common(30)
        
        valid_opps = []
        
        order_cust_map = df_orders_merged.set_index('id')['CustomerName'].to_dict()
        # Cache customer inventories
        cust_inv = {}
        for oid, items in basket.items():
            c = order_cust_map.get(oid)
            if c:
                if c not in cust_inv: cust_inv[c] = set()
                cust_inv[c].update(items)

        for (prod_a, prod_b), freq in candidates:
            # 1. Calculate Value
            price_a = prod_stats.loc[prod_a, 'price'] if prod_a in prod_stats.index else 0
            price_b = prod_stats.loc[prod_b, 'price'] if prod_b in prod_stats.index else 0
            bundle_val = price_a + price_b
            
            # 2. Find Targets (Symmetric Difference: Has A XOR Has B)
            # Potential Revenue =(Price of Item A + Price of Item B) \times Number of Target Customers
            targets = []
            for c, inv in cust_inv.items():
                if (prod_a in inv) and (prod_b not in inv):
                     targets.append(f"{str(c)} (Needs {prod_b})")
                elif (prod_b in inv) and (prod_a not in inv):
                     targets.append(f"{str(c)} (Needs {prod_a})")
            
            # 3. Calculate Potential Revenue
            potential_rev = len(targets) * bundle_val
            
            # Only keep if there is money to be made
            if len(targets) > 0:
                valid_opps.append({
                    'Bundle': f"{prod_a} + {prod_b}",
                    'Freq': freq,
                    'UnitValue': bundle_val,
                    'PotentialRev': potential_rev,
                    'TargetCount': len(targets),
                    'TargetList': ", ".join(targets[:3]) + (f" (+{len(targets)-3} more)" if len(targets)>3 else "")
                })

        # Sort by Potential Revenue (Money Maker Logic) instead of just Frequency
        valid_opps.sort(key=lambda x: x['PotentialRev'], reverse=True)
        
        lines.append("| Bundle Pair | Common Orders | Bundle Price | Potential Revenue | Missed Opportunity (Targets) |")
        lines.append("|---|---|---|---|---|")
        
        if valid_opps:
            for op in valid_opps[:5]: # Show top 5 money makers
                lines.append(f"| {op['Bundle']} | {op['Freq']} | ${op['UnitValue']:,.2f} | **${op['PotentialRev']:,.2f}** | {op['TargetList']} |")
        else:
            lines.append("| Market saturation reached for top bundles (no targeted cross-sell opportunities identified). | - | - | - | - |")

    except Exception as e:
        logger2.error(f"Bundle Error: {str(e)}")
        lines.append("*(Section unavailable)*")

    return lines

def _top_customers_report(orders_path, customers_path, products_path) -> list:
    """
    Generates a report on top customers.
    Logs errors to 'logger2' and provides user-friendly context for each table.
    """
    lines = []
    lines.append("# Top Customer Intelligence Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
    lines.append("")

    try:
        # --- 1. Load & Clean Data ---
        if not (os.path.exists(orders_path) and os.path.exists(customers_path) and os.path.exists(products_path)):
             logger2.error(f"Files not found: {orders_path}, {customers_path}, {products_path}")
             return ["Can not generate report: there is not enough data."]

        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')

        # Clean Columns (Strip whitespace & remove BOM)
        for df in [df_orders, df_customers, df_products]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # Parse Order Dates
        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        df_orders = df_orders.dropna(subset=['createdAt'])
        if df_orders.empty:
            logger2.error("No valid orders found after date parsing.")
            return ["Can not generate report: there is not enough data."]
        reference_date = df_orders['createdAt'].max()
        lines.append(f"**Data Reference:** {reference_date.strftime('%m/%d/%Y')}")
        lines.append("")

        # Customer Name Logic (Resolve names robustly)
        if 'combinedid' in df_customers.columns: cust_id_col = 'combinedid'
        elif 'id' in df_customers.columns: cust_id_col = 'id'
        else: 
            logger2.error("Customer ID column missing.")
            return ["Can not generate report: there is not enough data."]

        name_col = 'name'
        if name_col not in df_customers.columns:
             if 'displayedName' in df_customers.columns: name_col = 'displayedName'
             elif 'displayName' in df_customers.columns: name_col = 'displayName'
             else: name_col = cust_id_col

        # Merge Customer Names (Customers -> Orders)
        cust_subset = df_customers[[cust_id_col, name_col]].rename(columns={cust_id_col: 'CustomerPK', name_col: 'CustomerNameVal'})
        
        df_orders_merged = df_orders.merge(
            cust_subset,
            left_on='customer_id',
            right_on='CustomerPK',
            how='left'
        )
        df_orders_merged['CustomerName'] = df_orders_merged['CustomerNameVal'].fillna(df_orders_merged['customer_id'])

        # Product Display Name (Name + SKU)
        if 'sku' not in df_products.columns: df_products['sku'] = ''
        df_products['sku'] = df_products['sku'].fillna('')
        df_products['DisplayName'] = df_products.apply(
            lambda x: f"{x['name']} ({x['sku']})" if x['sku'] else x['name'], axis=1
        )

        # Merge Orders -> Products (Link Products to Customer Names)
        # Ensure orderId is present in products
        if 'orderId' not in df_products.columns:
            logger2.error("Products file missing 'orderId' column.")
            return ["Can not generate report: there is not enough data."]
             
        df_products_linked = df_products.merge(
            df_orders_merged[['id', 'CustomerName']],   # 'id' = order's own PK
            left_on='orderId', 
            right_on='id', 
            how='left'
        )

        # --- 2. Calculate Customer Metrics ---
        customer_stats = df_orders_merged.groupby('CustomerName').agg({
            'totalAmount': 'sum',
            'customer_id': 'count',
            'createdAt': 'max'
        }).rename(columns={
            'totalAmount': 'TotalRevenue', 
            'customer_id': 'OrderCount', 
            'createdAt': 'LastOrderDate'
        })
        
        customer_stats['AOV'] = customer_stats['TotalRevenue'] / customer_stats['OrderCount']
        customer_stats['DaysSinceLastOrder'] = (reference_date - customer_stats['LastOrderDate']).dt.days
        
        # Rankings
        top_revenue = customer_stats.sort_values(by='TotalRevenue', ascending=False).head(10)
        
        # Min 2 orders for AOV rankings to avoid one-off outliers
        top_aov = customer_stats[customer_stats['OrderCount'] > 1].sort_values(by='AOV', ascending=False).head(10)

        # --- 3. VIP Deep Dive (Favorite Products) ---
        vip_names = top_revenue.head(5).index.tolist()
        
        vip_profiles = []
        for cust in vip_names:
            stats = customer_stats.loc[cust]
            
            # Find Favorites
            cust_prods = df_products_linked[df_products_linked['CustomerName'] == cust]
            
            fav_str = "No product data"
            if not cust_prods.empty:
                # Top 3 by Quantity
                if 'quantity' in cust_prods.columns:
                     fav_prods = cust_prods.groupby('DisplayName')['quantity'].sum().sort_values(ascending=False).head(3)
                     fav_list = [f"{p_name} ({int(qty)})" for p_name, qty in fav_prods.items()]
                     fav_str = ", ".join(fav_list)
            
            vip_profiles.append({
                'Customer': cust,
                'Revenue': stats['TotalRevenue'],
                'Orders': stats['OrderCount'],
                'LastSeen': f"{int(stats['DaysSinceLastOrder'])} days ago",
                'Favorites': fav_str
            })
            
        df_vip = pd.DataFrame(vip_profiles)

        # --- 4. Pareto Analysis ---
        total_rev = customer_stats['TotalRevenue'].sum()
        top_10_pct_count = max(1, int(len(customer_stats) * 0.1))
        top_10_rev = customer_stats.sort_values(by='TotalRevenue', ascending=False).head(top_10_pct_count)['TotalRevenue'].sum()
        pareto_share = (top_10_rev / total_rev * 100) if total_rev > 0 else 0

        # --- 5. Generate Report Content ---
        lines.append("## 1. Executive Summary")
        lines.append(f"- **Total Active Customers:** {len(customer_stats)}")
        lines.append(f"- **Revenue Concentration (Pareto):** The top {top_10_pct_count} customer(s) (approx 10%) contribute **{pareto_share:.1f}%** of your total revenue.")
        if not top_revenue.empty:
             lines.append(f"- **#1 Best Customer:** {top_revenue.index[0]} (${top_revenue.iloc[0]['TotalRevenue']:,.2f})")
        lines.append("")
        
        lines.append("## 2. Leaderboards")
        lines.append("### Top Spenders (Highest Lifetime Value)")
        lines.append("*AOV - Average Order Value*")
        lines.append("| Rank | Customer | Revenue | Orders | Last Order | AOV |")
        lines.append("|---|---|---|---|---|---|")
        for i, (name, row) in enumerate(top_revenue.iterrows(), 1):
            last = row['LastOrderDate'].strftime('%m/%d/%Y')
            lines.append(f"| {i} | {name} | ${row['TotalRevenue']:,.2f} | {row['OrderCount']} | {last} | ${row['AOV']:,.2f} |")
            
        lines.append("")
        lines.append("### Big Ticket Buyers (Highest Average Order Value)")
        lines.append("*Customers who spend the most per transaction (min. 2 orders).*")
        lines.append("| Rank | Customer | AOV | Total Revenue | Orders |")
        lines.append("|---|---|---|---|---|")
        for i, (name, row) in enumerate(top_aov.iterrows(), 1):
            lines.append(f"| {i} | {name} | ${row['AOV']:,.2f} | ${row['TotalRevenue']:,.2f} | {row['OrderCount']} |")

        lines.append("")
        lines.append("## 3. VIP Deep Dive (Top 5 Spenders)")
        lines.append("*Detailed profile of your top 5 best customers to help you understand their preferences.*")
        lines.append("| Customer | Revenue | Orders | Last Seen | Top 3 Favorite Products |")
        lines.append("|---|---|---|---|---|")
        if not df_vip.empty:
            for _, r in df_vip.iterrows():
                lines.append(f"| {r['Customer']} | ${r['Revenue']:,.2f} | {r['Orders']} | {r['LastSeen']} | {r['Favorites']} |")
        else:
            lines.append("| No VIP data found | - | - | - | - |")

        return lines

    except Exception as e:
        logger2.error(f"Critical Error in Top Customers Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]
    
def _visits_report(orders_path, customers_path) -> list:
    """
    Generates a report comparing visited vs. unvisited customers.
    UPDATES:
    - Adds 'Last Rep' column to At-Risk Visits table to show who manages the account.
    - Includes robust error handling and clear definitions.
    """
    lines = []
    lines.append("# Visited vs Not Visited Report")
    lines.append(f"**Generated:** {datetime.datetime.now().strftime('%m/%d/%Y')}")
    lines.append("")

    try:
        # --- 1. Load Data ---
        if not (os.path.exists(orders_path) and os.path.exists(customers_path)):
             logger2.error(f"Files not found: {orders_path}, {customers_path}")
             return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

        # Load with proper encoding
        df_orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        df_customers = pd.read_csv(customers_path, encoding='utf-8-sig')
        
        # Clean column names
        df_orders.columns = df_orders.columns.str.strip().str.replace('\ufeff', '')
        df_customers.columns = df_customers.columns.str.strip().str.replace('\ufeff', '')

        # --- 2. Preprocess Dates ---
        # Parse dates and handle timezones
        if 'createdAt' not in df_orders.columns:
            logger2.error("Orders file missing 'createdAt' column.")
            return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

        df_orders['createdAt'] = pd.to_datetime(df_orders['createdAt'], errors='coerce')
        # Convert to naive datetime
        if df_orders['createdAt'].dt.tz is not None:
            df_orders['createdAt'] = df_orders['createdAt'].dt.tz_localize(None)
        df_orders = df_orders.dropna(subset=['createdAt'])
        
        # Reference Date (Now)
        reference_date = pd.Timestamp.now().replace(tzinfo=None)
        
        # Parse lastCheckInAt
        if 'lastCheckInAt' in df_customers.columns:
            # Try specific format first
            df_customers['lastCheckInAt'] = pd.to_datetime(
                df_customers['lastCheckInAt'], 
                errors='coerce',
                format='%a %b %d %Y GMT+0000 (Coordinated Universal Time)'
            )
            # Fallback to generic parsing
            if df_customers['lastCheckInAt'].isna().all():
                 df_customers['lastCheckInAt'] = pd.to_datetime(df_customers['lastCheckInAt'], errors='coerce')
            
            # Remove timezone
            if df_customers['lastCheckInAt'].dt.tz is not None:
                df_customers['lastCheckInAt'] = df_customers['lastCheckInAt'].dt.tz_localize(None)
        else:
            # Create empty column if missing
            df_customers['lastCheckInAt'] = pd.NaT

        # --- 3. Identify Sales Reps (Who manages the customer?) ---
        # Strategy: Get the sales rep from the most recent order for each customer
        if 'salesDuplicate_name' in df_orders.columns:
            # Sort by date descending so first item is most recent
            last_reps = df_orders.sort_values('createdAt', ascending=False).groupby('customer_id')['salesDuplicate_name'].first().reset_index()
            last_reps.columns = ['customer_id', 'LastRep']
        else:
            last_reps = pd.DataFrame(columns=['customer_id', 'LastRep'])

        # --- 4. Aggregate Revenue by Customer ---
        # Check required columns
        if 'customer_id' not in df_orders.columns or 'totalAmount' not in df_orders.columns:
             logger2.error("Orders file missing required columns (customer_id, totalAmount).")
             return ["Can not generate report: there is not enough data."]

        order_stats = df_orders.groupby('customer_id').agg({
            'totalAmount': 'sum',
            'customer_id': 'count',
            'createdAt': 'max'
        }).rename(columns={
            'totalAmount': 'TotalRevenue',
            'customer_id': 'OrderCount',
            'createdAt': 'LastOrderDate'
        }).reset_index()

        # Merge Rep info into stats
        order_stats = order_stats.merge(last_reps, on='customer_id', how='left')
        order_stats['LastRep'] = order_stats['LastRep'].fillna("Unknown")

        # --- 5. Merge Customers with Order Stats ---
        # Identify key columns
        cust_id_col = 'combinedid' if 'combinedid' in df_customers.columns else 'id'
        
        df_merged = df_customers.merge(
            order_stats,
            left_on=cust_id_col,
            right_on='customer_id',
            how='left'
        )
        
        if df_merged.empty:
            return ["Can not generate report: there is not enough data."]
        
        # Get customer name
        name_col = 'name'
        if name_col not in df_merged.columns:
             if 'displayedName' in df_merged.columns: name_col = 'displayedName'
             elif 'displayName' in df_merged.columns: name_col = 'displayName'
             else: name_col = cust_id_col
             
        df_merged['CustomerName'] = df_merged[name_col].fillna(df_merged[cust_id_col]).astype(str)

        # --- 6. Fill NaN values ---
        df_merged['TotalRevenue'] = df_merged['TotalRevenue'].fillna(0)
        df_merged['OrderCount'] = df_merged['OrderCount'].fillna(0)
        df_merged['HasOrders'] = df_merged['OrderCount'] > 0
        df_merged['LastRep'] = df_merged['LastRep'].fillna("No Orders")
        
        # Handle LastOrderDate
        df_merged['LastOrderDate'] = df_merged['LastOrderDate'].apply(
            lambda x: x.replace(tzinfo=None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo else x
        )

        # --- 7. Classify Visitation Status ---
        df_merged['IsVisited'] = df_merged['lastCheckInAt'].notna()
        
        # Calculate DaysSinceVisit
        df_merged['DaysSinceVisit'] = df_merged['lastCheckInAt'].apply(
            lambda x: (reference_date - x).days if pd.notna(x) else float('inf')
        )
        
        # Calculate DaysSinceOrder
        df_merged['DaysSinceOrder'] = df_merged['LastOrderDate'].apply(
            lambda x: (reference_date - x).days if pd.notna(x) else float('inf')
        )
        
        # Churn Status
        df_merged['IsChurned'] = (df_merged['DaysSinceOrder'] > 90) | (df_merged['OrderCount'] == 0)

        # --- 8. Group Analysis ---
        summary = df_merged.groupby('IsVisited').agg({
            'CustomerName': 'count',
            'TotalRevenue': 'sum',
            'OrderCount': 'sum',
            'HasOrders': 'sum'
        }).rename(columns={'CustomerName': 'CustomerCount'})
        
        summary['AvgRevenuePerCustomer'] = summary.apply(
            lambda row: row['TotalRevenue'] / row['CustomerCount'] if row['CustomerCount'] > 0 else 0,
            axis=1
        )
        
        summary['OrderRate'] = summary.apply(
            lambda row: row['HasOrders'] / row['CustomerCount'] * 100 if row['CustomerCount'] > 0 else 0,
            axis=1
        )
        
        # --- 9. Opportunity Lists ---
        
        # List A: Unvisited Goldmine (Never visited but have revenue)
        unvisited_gold = df_merged[
            ~df_merged['IsVisited'] & 
            (df_merged['TotalRevenue'] > 0)
        ].sort_values(by='TotalRevenue', ascending=False).head(10)
        
        # List B: Re-Visit Candidates (Visited > 60 days ago)
        revisit_candidates = df_merged[
            (df_merged['IsVisited']) & 
            (df_merged['DaysSinceVisit'] > 60) &
            (df_merged['TotalRevenue'] > 0)
        ].sort_values(by='TotalRevenue', ascending=False).head(10)
        
        # List C: At-Risk Visits (Visited <90 days ago but Churned)
        failed_visits = df_merged[
            (df_merged['IsVisited']) &
            (df_merged['DaysSinceVisit'] < 90) &
            (df_merged['IsChurned']) &
            (df_merged['TotalRevenue'] > 0)
        ].sort_values(by='TotalRevenue', ascending=False).head(10)

        # --- 10. Generate Report Content ---
        lines.append("## 1. Impact Analysis: Visited vs. Not Visited")
        lines.append("Comparing performance between customers you have visited in-person vs. those you haven't.")
        lines.append("- **Order Rate:** Percentage of customers in this group who have placed at least one order.")
        lines.append("")
        
        lines.append("| Segment | Customers | With Orders | Total Revenue | Avg Revenue/Cust | Order Rate |")
        lines.append("|---|---|---|---|---|---|")
        
        for is_vis, row in summary.iterrows():
            label = "Visited (At least once)" if is_vis else "Never Visited"
            lines.append(f"| {label} | {int(row['CustomerCount'])} | {int(row['HasOrders'])} | ${row['TotalRevenue']:,.2f} | ${row['AvgRevenuePerCustomer']:,.2f} | {row['OrderRate']:.1f}% |")
        
        # Overall totals
        total_customers = summary['CustomerCount'].sum()
        total_with_orders = summary['HasOrders'].sum()
        total_revenue = summary['TotalRevenue'].sum()
        overall_order_rate = (total_with_orders / total_customers * 100) if total_customers > 0 else 0
        
        lines.append(f"| **Overall** | **{int(total_customers)}** | **{int(total_with_orders)}** | **${total_revenue:,.2f}** | **${total_revenue/total_customers:,.2f}** | **{overall_order_rate:.1f}%** |")
            
        lines.append("")
        lines.append("## 2. Top Opportunities (Unvisited)")
        lines.append("High-potential customers who are buying from you but have **never** been visited.")
        lines.append("| # | Customer | Total Revenue | Orders | Last Order | Status |")
        lines.append("|---|---|---|---|---|---|")
        if not unvisited_gold.empty:
            for i, (_, r) in enumerate(unvisited_gold.iterrows(), 1):
                last_order = r['LastOrderDate'].strftime('%m/%d/%Y') if pd.notna(r['LastOrderDate']) else "Never"
                status = f"{int(r['OrderCount'])} orders"
                lines.append(f"| {i} | {r['CustomerName'][:50]} | ${r['TotalRevenue']:,.2f} | {int(r['OrderCount'])} | {last_order} | {status} |")
        else:
            lines.append("| - | No unvisited customers found | - | - | - | - |")
            
        lines.append("")
        lines.append("## 3. Re-Visit Candidates")
        lines.append("Top accounts visited **> 60 days ago**. You should stop by again to maintain the relationship.")
        lines.append("| # | Customer | Days Since Visit | Total Revenue | Last Order |")
        lines.append("|---|---|---|---|---|")
        if not revisit_candidates.empty:
            for i, (_, r) in enumerate(revisit_candidates.iterrows(), 1):
                days = int(r['DaysSinceVisit'])
                last_order = r['LastOrderDate'].strftime('%m/%d/%Y') if pd.notna(r['LastOrderDate']) else "Never"
                lines.append(f"| {i} | {r['CustomerName'][:50]} | {days} days | ${r['TotalRevenue']:,.2f} | {last_order} |")
        else:
            lines.append("| - | No re-visit candidates found | - | - | - |")

        lines.append("")
        lines.append("## 4. At-Risk Visits")
        lines.append("Customers you visited recently (<90 days) but who **still haven't ordered** (or stopped ordering).")
        lines.append("- **Revenue Risk:** Total historical sales from this customer that might be lost.")
        lines.append("- **Last Rep:** The sales representative associated with the customer's last order.")
        lines.append("")
        
        lines.append("| # | Customer | Visit Date | Last Order | Days Since Order | Last Rep | Revenue Risk |")
        lines.append("|---|---|---|---|---|---|---|")
        if not failed_visits.empty:
            for i, (_, r) in enumerate(failed_visits.iterrows(), 1):
                visit_date = r['lastCheckInAt'].strftime('%m/%d/%Y') if pd.notna(r['lastCheckInAt']) else "Unknown"
                last_order = r['LastOrderDate'].strftime('%m/%d/%Y') if pd.notna(r['LastOrderDate']) else "Never"
                days_since_order = int(r['DaysSinceOrder']) if r['DaysSinceOrder'] != float('inf') else "Never"
                last_rep = r['LastRep']
                lines.append(f"| {i} | {r['CustomerName'][:50]} | {visit_date} | {last_order} | {days_since_order} | {last_rep} | ${r['TotalRevenue']:,.2f} |")
        else:
            lines.append("| - | No at-risk visits found | - | - | - | - | - |")

        return lines

    except Exception as e:
        logger2.error(f"Critical Error in Visits Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]


# new fulfillment report that combines catalog and sales data for deeper insights
def load_and_clean_catalog(catalog_path: str) -> pd.DataFrame:
    """Loads the catalog, applies smart naming, and calculates financial baselines."""
    try:
        df = pd.read_csv(catalog_path)
    except Exception as e:
        logger2.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

    try:
        parent_map = df.set_index('id')['name'].to_dict() if 'parentProductId' in df.columns else {}
        df['parent_name'] = df.get('parentProductId', pd.Series()).map(parent_map)
        
        df['base_name'] = (df.get('name', pd.Series())
                           .fillna(df['parent_name'] + ' (' + df.get('sku', pd.Series()).fillna('Variant') + ')')
                           .fillna(df.get('sku', pd.Series()))
                           .fillna('Unknown Product'))
        
        def build_full_name(row):
            parts = [str(row['base_name'])]
            if 'size' in row and pd.notna(row['size']) and str(row['size']) not in parts[0]:
                parts.append(str(row['size']))
            if 'color' in row and pd.notna(row['color']) and str(row['color']) not in parts[0]:
                parts.append(str(row['color']))
            return " - ".join(parts)
            
        df['display_name'] = df.apply(build_full_name, axis=1)
        
        # Financial & Inventory Metrics
        df['inventory_onHand'] = pd.to_numeric(df.get('inventory_onHand', 0), errors='coerce').fillna(0)
        df['inventory_allocated'] = pd.to_numeric(df.get('inventory_allocated', 0), errors='coerce').fillna(0)
        df['wholesalePrice'] = pd.to_numeric(df.get('wholesalePrice', 0), errors='coerce').fillna(0)
        
        # Calculate active capital (positive stock) and backorder liabilities (negative stock)
        df['active_stock_value'] = np.where(df['inventory_onHand'] > 0, df['inventory_onHand'] * df['wholesalePrice'], 0)
        
        # True Deficit: If onHand is negative, that is the minimum deficit. 
        # If pending allocations exceed positive onHand, that creates a deficit as well.
        df['fulfillment_deficit'] = np.maximum(
            0, 
            df['inventory_allocated'] - df['inventory_onHand']
        )
        
        df['revenue_at_risk'] = df['fulfillment_deficit'] * df['wholesalePrice']
        
        # This is the column that was wrong in your table
        df['available'] = df['inventory_onHand'] - df['inventory_allocated']

        return df
    except Exception as e:
        logger2.error(f"Error during data cleaning: {e}")
        return pd.DataFrame()

def _generate_executive_inventory_report(catalog_path: str) -> list:
    """Generates a dense, business-focused markdown report of inventory health and fulfillment risks."""
    df = load_and_clean_catalog(catalog_path)
    if df.empty:
        logger2.error("Error: Unable to process the catalog data - catalog is empty or malformed.")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

    # 1. Macro Health Metrics
    #total_products = len(df)
    #missing_critical_data = df[['sku', 'barcode', 'wholesalePrice']].isna().any(axis=1).sum()
    total_capital = df['active_stock_value'].sum()
    total_liability = df['revenue_at_risk'].sum()
    overall_alloc_rate = (df['inventory_allocated'].sum() / df[df['inventory_onHand'] > 0]['inventory_onHand'].sum() * 100) if df[df['inventory_onHand'] > 0]['inventory_onHand'].sum() > 0 else 0

    # 2. Critical Fulfillment Liabilities (Items costing the business money right now)
    liabilities_df = df[df['fulfillment_deficit'] > 0].sort_values(by='revenue_at_risk', ascending=False).head(5)

    # 3. Capital Inefficiency (High stock, zero demand)
    inefficient_df = df[(df['inventory_allocated'] == 0) & (df['inventory_onHand'] > 0)].sort_values(by='active_stock_value', ascending=False).head(5)

    # 4. Build the Report
    md = [
        "# Executive Inventory & Fulfillment Report",
        "\n## Portfolio Health & Capital Allocation",
        f"**Active Capital Tied in Inventory:** ${total_capital:,.2f}",
        f"**Total Revenue at Risk (Backorders & Deficits):** ${total_liability:,.2f}",
        f"**Pipeline Utilization:** {overall_alloc_rate:.1f}% of active inventory is currently pending fulfillment.",
        
        "\n## Critical Fulfillment Liabilities",
        "*Products with the highest unfulfilled demand or negative inventory balances, ranked by total wholesale revenue at risk.*",
        "| Product | ON HAND | ALLOCATED | AVAILABLE | Revenue at Risk |",
        "|---|---|---|---|---|"
    ]

    if not liabilities_df.empty:
        for _, row in liabilities_df.iterrows():
            md.append(f"| **{row['display_name']}** | {row['inventory_onHand']:.0f} | {row['inventory_allocated']:.0f} | {row['available']:.0f} | **${row['revenue_at_risk']:,.2f}** |")
    else:
        md.append("| _No immediate fulfillment liabilities detected._ | - | - | - | - |")

    md.extend([
        "\n## Capital Inefficiency Alerts",
        "*Products tying up the most capital with zero pending orders. Consider targeted promotions or liquidation.*",
        "| Product | Current Stock | Wholesale Price | Tied Capital |",
        "|---|---|---|---|"
    ])

    if not inefficient_df.empty:
        for _, row in inefficient_df.iterrows():
            md.append(f"| **{row['display_name']}** | {row['inventory_onHand']:.0f} | ${row['wholesalePrice']:,.2f} | **${row['active_stock_value']:,.2f}** |")
    else:
        md.append("| _No highly inefficient capital allocation detected._ | - | - | - |")

    return md

def _generate_product_performance(products_path, catalog_path, orders_path) -> list:
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(catalog_path) and os.path.exists(orders_path)):
                return ["Error: One or more required data files (products, catalog, orders) are missing."]
            
            catalog_df = pd.read_csv(catalog_path)
            orders_df = pd.read_csv(orders_path)
            products_df = pd.read_csv(products_path)

            if catalog_df.empty or orders_df.empty or products_df.empty:
                return ["Can not generate report: there is not enough data in the provided files."]
                
        except Exception as e:
            logger2.error(f"Data Loading Error in Product Performance Report: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Strict Catalog Filtering & Mapping ---
        try:
            cat_dict = catalog_df.set_index('id')[['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name']].to_dict('index')
            valid_catalog_ids = set(catalog_df['id'].dropna().unique())

            def get_detailed_name(row):
                name = row.get('name', "Unknown Product")
                if pd.isna(name): 
                    name = "Unknown Product"
                
                pid = row.get('productId')
                if pid in cat_dict:
                    cat_item = cat_dict[pid]
                    cat_name = cat_item.get('name')
                    
                    if pd.notna(cat_name): 
                        name = cat_name
                        
                    size = cat_item.get('size')
                    color = cat_item.get('color')
                    
                    if pd.notna(size): 
                        return f"{name} {size}"
                    elif pd.notna(color): 
                        return f"{name} {color}"
                        
                return str(name)

            df_products_active = products_df[products_df['productId'].isin(valid_catalog_ids)].copy()
            if df_products_active.empty:
                return ["Can not generate report: No matching active products found between orders and catalog."]

            df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Product Performance Report: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Merge with Orders & Calculate Metrics ---
        try:
            # Check if required columns exist before merging
            if 'id' not in orders_df.columns or 'customer_name' not in orders_df.columns:
                return ["Can not generate report: Missing required columns in orders data."]

            merged = pd.merge(df_products_active, orders_df[['id', 'customer_name']], left_on='orderId', right_on='id', how='inner')
            
            if merged.empty:
                return ["Can not generate report: No matching order history for active products."]

            metrics = merged.groupby('detailed_name').agg(
                total_revenue=('totalAmount', 'sum'),
                total_units=('quantity', 'sum'),
                order_count=('orderId', 'nunique'),
                unique_customers=('customer_name', 'nunique'),
                avg_price=('price', 'mean')
            ).reset_index()

            # Safely build stock dictionary
            stock_dict = {}
            for d_name in metrics['detailed_name']:
                matching_rows = df_products_active[df_products_active['detailed_name'] == d_name]
                if not matching_rows.empty:
                    pid = matching_rows['productId'].iloc[0]
                    stock_val = cat_dict.get(pid, {}).get('inventory_onHand', 0)
                    stock_dict[d_name] = int(stock_val) if pd.notna(stock_val) else 0
                else:
                    stock_dict[d_name] = 0
                    
            metrics['stock'] = metrics['detailed_name'].map(stock_dict)

        except Exception as e:
            logger2.error(f"Metrics Calculation Error in Product Performance Report: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Define Categories (Top 5) ---
        try:
            # A. Top Revenue Drivers
            heavyweights = metrics.sort_values('total_revenue', ascending=False).head(5)
            top_revenue_names = heavyweights['detailed_name'].tolist()

            # B. High Engagement Opportunities
            potential_gems = metrics[~metrics['detailed_name'].isin(top_revenue_names)].copy()
            
            denom = np.log1p(potential_gems['total_revenue'].clip(lower=0))
            
            # Compute raw score and replace infinity/NaN (caused by $0 revenue) with 0
            raw_score = (potential_gems['unique_customers'] * potential_gems['order_count']) / denom
            potential_gems['engagement_score'] = raw_score.replace([np.inf, -np.inf], 0).fillna(0)

            hidden_gems = potential_gems.sort_values('engagement_score', ascending=False).head(5)

            # C. Underperforming Assets
            dead_weight = metrics.sort_values(['order_count', 'total_revenue'], ascending=[True, True]).head(5)

        except Exception as e:
            logger2.error(f"Categorization Error in Product Performance Report: {str(e)}")
            return ["This report is currently unavailable due to an issue determining product performance segments."]

        # --- 5. Build the Markdown Report ---
        try:
            md = ["# Product Performance & Portfolio Insights\n"]

            # Top Revenue Drivers
            md.append("## Top Revenue Drivers")
            md.append("*These products generate the highest gross revenue. It is critical to maintain adequate inventory levels to protect these revenue streams.*")
            md.append("| Product | Total Revenue | Units Sold | Orders | Unique Buyers | Stock |")
            md.append("|---|---|---|---|---|---|")
            for _, row in heavyweights.iterrows():
                md.append(f"| **{row['detailed_name']}** | ${row['total_revenue']:,.2f} | {int(row['total_units'])} | {int(row['order_count'])} | {int(row['unique_customers'])} | {row['stock']} |")
            md.append("\n---\n")

            # High Penetration Opportunities
            md.append("## High Penetration Opportunities")
            md.append("Engagement Score = (Unique Buyers * Total Orders) / ln(Total Revenue + 1)")
            md.append("*These products demonstrate high purchase frequency and broad customer appeal but yield lower overall revenue. **Action Item:** Consider increasing case sizes, bundling with high-margin items, or adjusting pricing to capitalize on volume.*")
            md.append("| Product | Engagement Score | Unique Buyers | Orders | Total Revenue | Avg Price | Stock |")
            md.append("|---|---|---|---|---|---|---|")
            for _, row in hidden_gems.iterrows():
                md.append(f"| **{row['detailed_name']}** | {row['engagement_score']:.2f} | {int(row['unique_customers'])} | {int(row['order_count'])} | ${row['total_revenue']:,.2f} | ${row['avg_price']:.2f} | {row['stock']} |")
            md.append("\n---\n")

            # Underperforming Assets
            md.append("## Underperforming Assets")
            md.append("*These products exhibit minimal order volume and low revenue, tying up capital and inventory space. **Action Item:** Evaluate for liquidation, promotional discounting, or removal from the active catalog.*")
            md.append("| Product | Orders | Total Revenue | Units Sold | Unique Buyers | Stock |")
            md.append("|---|---|---|---|---|---|")
            for _, row in dead_weight.iterrows():
                md.append(f"| **{row['detailed_name']}** | {int(row['order_count'])} | ${row['total_revenue']:,.2f} | {int(row['total_units'])} | {int(row['unique_customers'])} | {row['stock']} |")

            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Product Performance Report: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Product Performance Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _generate_bundle_report(products_path, catalog_path, orders_path) -> list:
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(catalog_path) and os.path.exists(orders_path)):
                return ["Error: One or more required data files (products, catalog, orders) are missing."]
            
            df_catalog = pd.read_csv(catalog_path)
            df_orders = pd.read_csv(orders_path)
            df_products = pd.read_csv(products_path)
            
            if df_catalog.empty or df_orders.empty or df_products.empty:
                return ["Can not generate report: there is not enough data in the provided files."]

        except Exception as e:
            logger2.error(f"Data Loading Error in Bundle Report: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Customer LTV & Catalog Prep ---
        try:
            # Calculate Customer Lifetime Value
            if 'customer_name' in df_orders.columns and 'totalAmount' in df_orders.columns:
                ltv_df = df_orders.groupby('customer_name')['totalAmount'].sum().to_dict()
            else:
                ltv_df = {}

            # Create a strict catalog dictionary for active items
            cat_dict = df_catalog.set_index('id')[['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name']].to_dict('index')
            valid_catalog_ids = set(df_catalog['id'].dropna().unique())

            def get_detailed_name(row):
                name = row.get('name', "Unknown Product")
                if pd.isna(name): 
                    name = "Unknown Product"
                
                pid = row.get('productId')
                if pid in cat_dict:
                    cat_item = cat_dict[pid]
                    cat_name = cat_item.get('name')
                    if pd.notna(cat_name): 
                        name = cat_name
                        
                    size = cat_item.get('size')
                    color = cat_item.get('color')
                    
                    if pd.notna(size): return f"{name} {size}"
                    elif pd.notna(color): return f"{name} {color}"
                return str(name)

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Bundle Report: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Filter Active Products & Gather Stats ---
        try:
            # Apply strict filter: Only evaluate products inside the active catalog
            df_products_active = df_products[df_products['productId'].isin(valid_catalog_ids)].copy()
            if df_products_active.empty:
                return ["Can not generate report: No matching products found between order history and active catalog."]

            df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

            # Gather Product Stats (Pricing, Stock, Category)
            prod_stats = df_products_active.groupby('detailed_name').agg({
                'price': 'mean',
                'quantity': 'mean',
                'productCategoryName': lambda x: x.mode()[0] if not x.mode().empty else 'Uncategorized',
                'productId': 'first'
            }).reset_index()

            prod_info = {}
            for _, row in prod_stats.iterrows():
                d_name = row['detailed_name']
                pid = row['productId']
                price = row['price']
                stock = 0
                
                if pid in cat_dict:
                    cat_item = cat_dict[pid]
                    stock = cat_item.get('inventory_onHand', 0)
                    ws_price = cat_item.get('wholesalePrice')
                    if pd.notna(ws_price):
                        price = ws_price
                        
                prod_info[d_name] = {
                    'price': price if pd.notna(price) else 0.0,
                    'stock': stock if pd.notna(stock) else 0,
                    'category': row['productCategoryName']
                }

        except Exception as e:
            logger2.error(f"Product Stats Error in Bundle Report: {str(e)}")
            return ["This report is currently unavailable due to an error calculating product statistics."]

        # --- 4. Find Top Product Pairs ---
        try:
            # Strictly Active Catalog Only
            order_prods_df = df_products_active.dropna(subset=['detailed_name']).groupby('orderId')['detailed_name'].unique()
            pairs = []
            
            for prods in order_prods_df:
                if len(prods) > 1:
                    valid_pairs = []
                    for p1, p2 in itertools.combinations(sorted(prods), 2):
                        # Safe dictionary access in case a product slipped through without info
                        if p1 in prod_info and p2 in prod_info:
                            if prod_info[p1]['category'] != prod_info[p2]['category']:
                                valid_pairs.append((p1, p2))
                    pairs.extend(valid_pairs)

            pair_counts = pd.Series(pairs).value_counts().reset_index()
            if pair_counts.empty:
                return ["Not enough cross-category purchase data to generate bundle recommendations."]
                
            pair_counts.columns = ['pair', 'common_orders']
            top_pairs = pair_counts.head(3)

        except Exception as e:
            logger2.error(f"Pair Calculation Error in Bundle Report: {str(e)}")
            return ["This report is currently unavailable due to an issue analyzing product purchase combinations."]

        # --- 5. Generate Targets & Markdown Report ---
        try:
            # Merge for Customer-Level Insights
            merged = pd.merge(df_products_active, df_orders[['id', 'customer_name']], left_on='orderId', right_on='id', how='left')
            
            # Pre-calculate what products each customer has EVER bought
            cust_prods = merged.groupby('customer_name')['detailed_name'].unique().to_dict()

            md_lines = ["# Strategic Bundle Recommendations\n"]
            
            for idx, row in top_pairs.iterrows():
                prod_a, prod_b = row['pair']
                common_orders = row['common_orders']
                
                info_a = prod_info.get(prod_a, {'stock': 0, 'price': 0, 'category': 'Unknown'})
                info_b = prod_info.get(prod_b, {'stock': 0, 'price': 0, 'category': 'Unknown'})
                
                # Stock Formatting
                stock_a, stock_b = int(info_a['stock']), int(info_b['stock'])
                stock_a_str = f"**In Stock ({stock_a} units)**" if stock_a > 0 else f"**Out of Stock ({stock_a} units)**"
                stock_b_str = f"**In Stock ({stock_b} units)**" if stock_b > 0 else f"**Out of Stock ({stock_b} units)**"
                
                # 1. Success Stories Extraction
                order_groups = merged.groupby('orderId').agg({'detailed_name': lambda x: set(x), 'customer_name': 'first'})
                success_orders = order_groups[order_groups['detailed_name'].apply(lambda x: prod_a in x and prod_b in x)]
                success_custs = success_orders['customer_name'].dropna().unique().tolist()
                
                # 2. Missed Opportunities (Strict Targeting)
                targets = []
                
                # Target buyers of A who haven't bought B
                buyers_a = merged[merged['detailed_name'] == prod_a]
                if not buyers_a.empty:
                    agg_a = buyers_a.groupby('customer_name').agg({'quantity': 'mean'})
                    for cust, cust_row in agg_a.iterrows():
                        if cust in cust_prods and prod_b not in cust_prods[cust]:
                            targets.append({
                                'customer': cust, 'source': prod_a, 'pitch': prod_b,
                                'qty': cust_row['quantity'], 'val': cust_row['quantity'] * info_b['price']
                            })
                            
                # Target buyers of B who haven't bought A
                buyers_b = merged[merged['detailed_name'] == prod_b]
                if not buyers_b.empty:
                    agg_b = buyers_b.groupby('customer_name').agg({'quantity': 'mean'})
                    for cust, cust_row in agg_b.iterrows():
                        if cust in cust_prods and prod_a not in cust_prods[cust]:
                            targets.append({
                                'customer': cust, 'source': prod_b, 'pitch': prod_a,
                                'qty': cust_row['quantity'], 'val': cust_row['quantity'] * info_a['price']
                            })

                targets_df = pd.DataFrame(targets)
                if not targets_df.empty:
                    targets_df = targets_df.sort_values('val', ascending=False).drop_duplicates(subset=['customer'])
                    
                total_targets = len(targets_df) if not targets_df.empty else 0
                total_missed_val = targets_df['val'].sum() if total_targets > 0 else 0

                # Build Markdown for this pair
                md_lines.append(f"## {prod_a} & {prod_b}")
                md_lines.append("\n**Stock Availability & Standalone Pricing:**")
                md_lines.append(f"- **{prod_a} ({info_a['category']}):** {stock_a_str} (Price: ${info_a['price']:.2f})")
                md_lines.append(f"- **{prod_b} ({info_b['category']}):** {stock_b_str} (Price: ${info_b['price']:.2f})\n")
                
                md_lines.append(f"**The Synergy:** Found together organically in **{common_orders}** historical order(s).")
                
                # Success Stories Display
                if success_custs:
                    success_display = ", ".join(success_custs[:3])
                    if len(success_custs) > 3:
                        success_display += f", and {len(success_custs)-3} others"
                    md_lines.append(f"- **Proven Traction:** Customers like **{success_display}** have already purchased these together, validating the cross-category demand.")
                    
                # Scaling Strategy
                md_lines.append("\n###  How to Scale This Bundle")
                if total_targets > 0:
                    discount_pct = 10
                    conversion_rate = 0.30
                    projected_rev = total_missed_val * conversion_rate * (1 - discount_pct/100)
                    
                    md_lines.append(f"You have **{total_targets} customers** who buy one of these items, but not the other. The total untapped potential is **${total_missed_val:,.2f}**.")
                    md_lines.append(f"1. **Action:** Create a **{discount_pct}% Off Bundle Discount** in SimplyDepo specifically pairing these two items.")
                    md_lines.append(f"2. **Outreach:** Pitch this new incentive to the target list below.")
                    md_lines.append(f"3. **Impact:** If just 30% of these targets convert using the discount, you generate **~${projected_rev:,.2f}** in immediate incremental revenue.")
                else:
                    md_lines.append("1. **Action:** Create a **Bundle Discount** in SimplyDepo for these items.")
                    md_lines.append("2. **Impact:** Even with a small customer base, incentivizing cross-category purchases increases Average Order Value (AOV) and establishes new, more profitable purchasing habits across your territory.")
                    
                md_lines.append("\n**Top Target Customers (Missed Opportunities):**")
                if total_targets > 0:
                    for _, t in targets_df.head(5).iterrows():
                        md_lines.append(f"- **{t['customer']}:** Buys ~{int(t['qty'])}x {t['source']} per order → **Pitch: {int(t['qty'])}x {t['pitch']}** (Upsell Value: **${t['val']:,.2f}**)")
                else:
                    md_lines.append("- *All current buyers of these products already purchase them together! This is a highly mature bundle. Expand your reach to completely new accounts.*")
                    
                md_lines.append("\n---\n")
                
            return md_lines

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Bundle Report: {str(e)}")
            return ["An error occurred while compiling the final report text. Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Bundle Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def _generate_time_based_product_report(products_path, catalog_path, orders_path) -> list:
    try:
        # --- 1 & 2. Load Data and Parse Dates ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(catalog_path) and os.path.exists(orders_path)):
                return ["Error: One or more required data files (products, catalog, orders) are missing."]
            
            catalog_df = pd.read_csv(catalog_path)
            orders_df = pd.read_csv(orders_path)
            products_df = pd.read_csv(products_path)

            if catalog_df.empty or orders_df.empty or products_df.empty:
                return ["Can not generate report: there is not enough data in the provided files."]

            # Clean and Parse Dates
            catalog_df['createdAt'] = pd.to_datetime(
                catalog_df['createdAt'].astype(str).str.replace(r' GMT\+0000 \(Coordinated Universal Time\)', '', regex=True), 
                errors='coerce'
            )
            orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], errors='coerce')
            
            # Establish dynamic boundaries
            max_order_date = orders_df['createdAt'].max()
            max_cat_date = catalog_df['createdAt'].max()
            
            # Prevent failure if all dates are NaT
            if pd.isna(max_order_date) or pd.isna(max_cat_date):
                return ["Can not generate report: unable to parse valid dates from the data to establish timelines."]
                
            recent_cutoff = max_order_date - pd.Timedelta(days=180)
            new_product_cutoff = max_cat_date - pd.Timedelta(days=180)

        except Exception as e:
            logger2.error(f"Data Loading/Date Parsing Error in Time-Based Report: {str(e)}")
            return ["This report is currently unavailable due to a data loading or date format error. Please check back later."]

        # --- 3. Build Catalog Dictionary & Filter Products ---
        try:
            # Adding inventory_allocated to dictionary mapping for the 'Available' calculation
            cat_dict = catalog_df.set_index('id')[['size', 'color', 'inventory_onHand', 'inventory_allocated', 'name', 'createdAt']].to_dict('index')
            valid_catalog_ids = set(catalog_df['id'].dropna().unique())

            def get_detailed_name(row):
                name = row.get('name', "Unknown Product")
                if pd.isna(name): 
                    name = "Unknown Product"
                
                pid = row.get('productId')
                if pid in cat_dict:
                    cat_item = cat_dict[pid]
                    cat_name = cat_item.get('name')
                    
                    if pd.notna(cat_name): 
                        name = cat_name
                        
                    size = cat_item.get('size')
                    color = cat_item.get('color')
                    
                    if pd.notna(size): return f"{name} {size}"
                    elif pd.notna(color): return f"{name} {color}"
                    
                return str(name)

            df_products_active = products_df[products_df['productId'].isin(valid_catalog_ids)].copy()
            
            if df_products_active.empty:
                return ["Can not generate report: No matching products found between orders and active catalog."]

            df_products_active['detailed_name'] = df_products_active.apply(get_detailed_name, axis=1)

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Time-Based Report: {str(e)}")
            return ["This report is currently unavailable due to an issue mapping product catalog data."]

        # --- 4 & 5. Merge and Calculate Time-Based Metrics ---
        try:
            # Merge Products with Orders
            if 'id' not in orders_df.columns or 'customer_name' not in orders_df.columns or 'createdAt' not in orders_df.columns:
                return ["Can not generate report: Missing required columns in orders data."]
                
            merged = pd.merge(df_products_active, orders_df[['id', 'customer_name', 'createdAt']], 
                              left_on='orderId', right_on='id', how='inner', suffixes=('', '_order'))
            
            if merged.empty:
                return ["Can not generate report: No matching order history found for active products."]
            
            # Group and Calculate Time-Based Metrics
            metrics = merged.groupby('detailed_name').agg(
                total_revenue=('totalAmount', 'sum'),
                total_units=('quantity', 'sum'),
                order_count=('orderId', 'nunique'),
                unique_customers=('customer_name', 'nunique'),
                last_order_date=('createdAt_order', 'max')
            ).reset_index()

            # Calculate 'Available' (onHand - allocated) and map creation dates
            available_dict = {}
            created_dict = {}
            
            for d_name in metrics['detailed_name']:
                matching_rows = df_products_active[df_products_active['detailed_name'] == d_name]
                if not matching_rows.empty:
                    pid = matching_rows['productId'].iloc[0]
                    if pid in cat_dict:
                        cat_item = cat_dict[pid]
                        on_hand = float(cat_item.get('inventory_onHand', 0)) if pd.notna(cat_item.get('inventory_onHand')) else 0.0
                        allocated = float(cat_item.get('inventory_allocated', 0)) if pd.notna(cat_item.get('inventory_allocated')) else 0.0
                        
                        # Available is strictly on_hand - allocated
                        available_dict[d_name] = int(on_hand - allocated)
                        created_dict[d_name] = cat_item.get('createdAt', pd.NaT)
                    else:
                        available_dict[d_name] = 0
                        created_dict[d_name] = pd.NaT
                else:
                    available_dict[d_name] = 0
                    created_dict[d_name] = pd.NaT

            metrics['available'] = metrics['detailed_name'].map(available_dict)
            metrics['catalog_created_at'] = metrics['detailed_name'].map(created_dict)

            # Identify Targets
            new_products = metrics[metrics['catalog_created_at'] >= new_product_cutoff].sort_values('total_revenue', ascending=False)
            
            # Isolate all stagnant products
            stopped_selling_all = metrics[(metrics['last_order_date'] < recent_cutoff) & (metrics['order_count'] > 0)].sort_values('last_order_date', ascending=False)
            
            # Keep only Top 10 for the table
            stopped_selling = stopped_selling_all.head(10)

        except Exception as e:
            logger2.error(f"Metrics Calculation Error in Time-Based Report: {str(e)}")
            return ["This report is currently unavailable due to an error calculating time-based metrics. Please contact support."]

        # --- 6. Build the Markdown Report ---
        try:
            md = ["# Time-Based Product Performance Report\n"]

            # Table 1: New Products Analysis
            md.append("## 1. New Products Analysis (Recently Added)")
            md.append(f"*Products added to the catalog since **{new_product_cutoff.strftime('%m/%d/%Y')}**. Analyzed to determine if they are gaining traction and beneficial to the business.*")
            md.append("| Product | Date Added | Total Revenue | Units Sold | Orders | Unique Buyers | Available |")
            md.append("|---|---|---|---|---|---|---|")
            
            if new_products.empty:
                md.append("| No new products found in this timeframe | - | - | - | - | - | - |")
            else:
                for _, row in new_products.iterrows():
                    date_added = row['catalog_created_at'].strftime('%m/%d/%Y') if pd.notnull(row['catalog_created_at']) else 'N/A'
                    md.append(f"| **{row['detailed_name']}** | {date_added} | ${row['total_revenue']:,.2f} | {int(row['total_units'])} | {int(row['order_count'])} | {int(row['unique_customers'])} | {row['available']} |")
            md.append("\n---\n")

            # Table 2: Stagnant Products
            md.append("## 2. Stagnant Products (Stopped Selling)")
            md.append(f"*Products with historical sales but no recent orders since **{recent_cutoff.strftime('%m/%d/%Y')}** (Last 6 Months).*")
            md.append("| Product | Last Sold Date | Lifetime Orders | Total Revenue | Available |")
            md.append("|---|---|---|---|---|")
            
            if stopped_selling.empty:
                md.append("| No stagnant products found | - | - | - | - |")
            else:
                for _, row in stopped_selling.iterrows():
                    last_sold = row['last_order_date'].strftime('%m/%d/%Y') if pd.notnull(row['last_order_date']) else 'N/A'
                    md.append(f"| **{row['detailed_name']}** | {last_sold} | {int(row['order_count'])} | ${row['total_revenue']:,.2f} | {row['available']} |")

                # Dynamic notation if there are more than 10 stagnant products
                if len(stopped_selling_all) > 10:
                    extra_count = len(stopped_selling_all) - 10
                    md.append(f"\n*Note: There are {extra_count} more stagnant products not shown here. You can ask the AI to reveal the full list if needed.*")

            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Time-Based Report: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Time-Based Product Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

# big report that combines catalog and sales data for deeper insights
def get_top_3_products_by_revenue(cat: pd.DataFrame, prod: pd.DataFrame) -> list:
    """Helper: Identifies the Top 3 parent products by total gross revenue."""
    
    # 1. Map parent names safely
    cat['resolved_parent_id'] = cat['parentProductId'].fillna(cat['id'])
    id_to_name = cat.set_index('id')['name'].to_dict()
    cat['parent_name'] = cat['resolved_parent_id'].map(id_to_name)
    
    # 2. ROBUST COLUMN CHECK: Find the Product ID column
    prod_id_col = 'productId'
    if 'productId' not in prod.columns:
        if 'product_id' in prod.columns:
            prod_id_col = 'product_id'
        elif 'id' in prod.columns and 'customer_name' not in prod.columns:
            # Fallback if it's just named 'id' (and ensure it's not the orders table)
            prod_id_col = 'id'
        else:
            raise KeyError(
                f"ERROR: Could not find 'productId' in the sales dataframe!\n"
                f"You likely passed the wrong CSV file (e.g., Orders instead of Products).\n"
                f"Available columns in the dataframe you passed: {prod.columns.tolist()}"
            )

    # 3. ROBUST COLUMN CHECK: Find the Revenue column
    revenue_col = 'totalAmount'
    if 'totalAmount' not in prod.columns:
        if 'total_amount' in prod.columns:
            revenue_col = 'total_amount'
        elif 'price' in prod.columns:
            revenue_col = 'price'

    # 4. Safe Merge (dropping duplicates in catalog IDs just in case)
    cat_unique = cat[['id', 'parent_name']].drop_duplicates(subset=['id'])
    merged_sales = pd.merge(prod, cat_unique, left_on=prod_id_col, right_on='id', how='inner')
    
    # 5. Calculate and Return
    merged_sales[revenue_col] = pd.to_numeric(merged_sales[revenue_col], errors='coerce').fillna(0)
    return merged_sales.groupby('parent_name')[revenue_col].sum().nlargest(3).index.tolist()

def generate_precise_product_analysis(product_name, catalog_df, products_df):
    """Core logic for calculating variant-level sales and inventory."""
    
    prod_name_lower = str(product_name).lower().strip()
    cat_match = catalog_df[catalog_df['name'].str.lower() == prod_name_lower].copy()
    
    if cat_match.empty:
        return f"### Analysis for '{product_name}'\nNo active products found in the catalog matching this exact name."
    
    parent_ids = cat_match['id'].tolist()
    parent_names_map = cat_match.set_index('id')['name'].to_dict()
    
    children = catalog_df[catalog_df['parentProductId'].isin(parent_ids)].copy()
    parents_with_children = children['parentProductId'].unique()
    simple_items = cat_match[~cat_match['id'].isin(parents_with_children)].copy()
    
    target_catalog_items = pd.concat([children, simple_items])
    target_product_ids = target_catalog_items['id'].dropna().unique().tolist()
    
    def get_name(row):
        if pd.isna(row['name']) or str(row['name']).strip() == '':
            return parent_names_map.get(row['parentProductId'], 'Unknown')
        return row['name']
        
    target_catalog_items['variant_name'] = target_catalog_items.apply(get_name, axis=1)
    target_catalog_items['sku'] = target_catalog_items['sku'].fillna('N/A')
    target_catalog_items['size'] = target_catalog_items['size'].fillna('-')
    target_catalog_items['color'] = target_catalog_items['color'].fillna('-')
    target_catalog_items['inventory_onHand'] = pd.to_numeric(target_catalog_items['inventory_onHand'], errors='coerce').fillna(0)
    
    inventory_by_sku = target_catalog_items.groupby(
        ['sku', 'variant_name', 'size', 'color'], dropna=False
    )['inventory_onHand'].sum().reset_index()
    inventory_by_sku.rename(columns={'inventory_onHand': 'units_left'}, inplace=True)
    
    prod_match = products_df[products_df['productId'].isin(target_product_ids)].copy()
    prod_match['quantity'] = pd.to_numeric(prod_match['quantity'], errors='coerce').fillna(0)
    prod_match['totalAmount'] = pd.to_numeric(prod_match['totalAmount'], errors='coerce').fillna(0)
    prod_match['sku'] = prod_match['sku'].fillna('N/A')
    
    sales_by_sku = prod_match.groupby('sku').agg({
        'quantity': 'sum',
        'totalAmount': 'sum'
    }).reset_index()
    sales_by_sku.rename(columns={'quantity': 'units_sold', 'totalAmount': 'revenue'}, inplace=True)
    
    merged_data = pd.merge(inventory_by_sku, sales_by_sku, on='sku', how='left').fillna(0)
    
    total_inventory = merged_data['units_left'].sum()
    total_sold = merged_data['units_sold'].sum()
    total_revenue = merged_data['revenue'].sum()
    
    variant_rows = []
    for _, row in merged_data.iterrows():
        sold = row['units_sold']
        stock = row['units_left']
        
        if stock < 0: rec = "**OVERSOLD**"
        elif stock == 0 and sold > 0: rec = "**URGENT RESTOCK**"
        elif sold > 0 and (stock / sold) < 0.2: rec = "**Top Performer**"
        elif sold > 0: rec = "**Steady Seller**"
        else: rec = "**Zero Sales**"
            
        variant_rows.append({
            'Name': row['variant_name'], 'SKU': row['sku'], 'Size': row['size'],
            'Color': row['color'], 'Units Sold': sold, 'Units Left': stock, 
            'Revenue': row['revenue']
        })
    
    variant_rows = sorted(variant_rows, key=lambda x: x['Units Sold'], reverse=True)
    
    md = f"### Deep-Dive Analysis: {product_name.title()}\n\n"
    md += "#### Overall Product Line Metrics\n"
    md += f"- **Total Current Inventory:** {total_inventory:,.0f} units\n"
    md += f"- **All-Time Units Sold:** {total_sold:,.0f} units\n"
    md += f"- **Gross Revenue Generated:** ${total_revenue:,.2f}\n\n"
    md += "#### Variation Breakdown\n"
    md += "| Variation Name | SKU | Size | Color | Units Sold | Units Left | Revenue |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    for row in variant_rows:
        md += f"| {row['Name']} | {row['SKU']} | {row['Size']} | {row['Color']} | {row['Units Sold']:,.0f} | **{row['Units Left']:,.0f}** | **${row['Revenue']:,.2f}** |\n"
        
    return md

def _calculate_top_3_sales_breakdown(catalog_path: str, products_path: str) -> list:
    """Entry point 1: Generates the variant and sales performance report."""
    try:
        # --- 1. Load Data ---
        try:
            if not (os.path.exists(catalog_path) and os.path.exists(products_path)):
                return ["Error: Required data files (catalog, products) are missing."]
            
            cat = pd.read_csv(catalog_path, low_memory=False)
            prod = pd.read_csv(products_path, low_memory=False)
            
            if cat.empty or prod.empty:
                return ["Can not generate report: there is not enough data in the provided files."]
        except Exception as e:
            logger2.error(f"Data Loading Error in Top 3 Sales Breakdown: {str(e)}")
            return ["Can not generate report: there is not enough data in the provided files."]

        # --- 2. Generate Report ---
        try:
            top_products = get_top_3_products_by_revenue(cat, prod)
            
            if not top_products:
                return ["Can not generate report: No top products could be identified from the data."]

            report_lines = ["#  Top 3 Products by Gross Revenue: Variation Breakdown\n"]
            
            for i, product_name in enumerate(top_products, 1):
                report_lines.append(f"## {i}. Top Product: {product_name}\n")
                
                # Loop-level try/except so one bad product doesn't crash the whole report
                try:
                    analysis = generate_precise_product_analysis(product_name, cat, prod)
                    report_lines.append(analysis)
                except Exception as loop_e:
                    logger2.error(f"Analysis Error for product '{product_name}': {str(loop_e)}")
                    report_lines.append(f"*Detailed breakdown for '{product_name}' could not be generated at this time.*")
                
                report_lines.append("\n---\n")
                
            return ["\n".join(report_lines)]

        except Exception as e:
            logger2.error(f"Report Generation Error in Top 3 Sales Breakdown: {str(e)}")
            return ["Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Top 3 Sales Breakdown Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]

def generate_advanced_customer_insights(product_name, catalog_df, products_df, orders_df):
    """Core logic for calculating customer distribution, retention, and cohorts."""
    prod_name_lower = str(product_name).lower().strip()
    
    cat_match = catalog_df[catalog_df['name'].str.lower() == prod_name_lower].copy()
    if cat_match.empty: return {"error": f"No active products found in the catalog matching '{product_name}'."}
    
    parent_ids = cat_match['id'].tolist()
    children = catalog_df[catalog_df['parentProductId'].isin(parent_ids)].copy()
    parents_with_children = children['parentProductId'].unique()
    simple_items = cat_match[~cat_match['id'].isin(parents_with_children)].copy()
    
    target_catalog_items = pd.concat([children, simple_items])
    target_product_ids = target_catalog_items['id'].dropna().unique().tolist()
    
    prod_match = products_df[products_df['productId'].isin(target_product_ids)].copy()
    if prod_match.empty: return {"error": f"No sales data found for '{product_name}'."}
         
    prod_match['quantity'] = pd.to_numeric(prod_match['quantity'], errors='coerce').fillna(0)
    prod_match['totalAmount'] = pd.to_numeric(prod_match['totalAmount'], errors='coerce').fillna(0)
    
    sales_orders = pd.merge(
        prod_match, 
        orders_df[['id', 'customer_name', 'createdAt', 'paymentStatus']], 
        left_on='orderId', right_on='id', how='inner', suffixes=('_prod', '_order')
    )
    
    paid_sales = sales_orders[sales_orders['paymentStatus'].isin(['PAID', 'PARTIALLY_PAID'])].copy()
    if paid_sales.empty: return {"error": f"No PAID sales data found for '{product_name}'."}

    paid_sales['createdAt_order'] = pd.to_datetime(paid_sales['createdAt_order'], utc=True)

    order_group = paid_sales.groupby('orderId').agg({'quantity': 'sum', 'totalAmount': 'sum'}).reset_index()
    total_orders = len(order_group)
    
    try:
        if len(order_group['quantity'].unique()) <= 2: raise ValueError
        bins = pd.qcut(order_group['quantity'], q=3, duplicates='drop')
        dist_counts = order_group.groupby(bins, observed=True).size()
        
        formatted_dist = {}
        for interval, count in dist_counts.items():
            if count > 0:
                lower = int(np.floor(interval.left)) + 1 if interval.left != order_group['quantity'].min() - 0.001 else int(interval.left)
                upper = int(np.floor(interval.right))
                label = f"Exactly {lower} units" if lower == upper else f"{lower} to {upper} units"
                formatted_dist[label] = int(count)
    except ValueError:
        formatted_dist = {f"{int(order_group['quantity'].iloc[0])} units": total_orders}
        
    customer_group = paid_sales.groupby('customer_name').agg({
        'orderId': 'nunique', 'quantity': 'sum', 'totalAmount': 'sum', 'createdAt_order': ['min', 'max']
    })
    customer_group.columns = ['orders_count', 'total_units', 'total_spent', 'first_order', 'last_order']
    customer_group = customer_group.reset_index()

    customer_group['customer_lifespan_days'] = (customer_group['last_order'] - customer_group['first_order']).dt.days
    customer_group['avg_days_between_orders'] = np.where(
        customer_group['orders_count'] > 1,
        customer_group['customer_lifespan_days'] / (customer_group['orders_count'] - 1),
        None 
    )
    
    top_customers = customer_group.sort_values(by='total_spent', ascending=False).head(5)

    top_cust_list = []
    for _, row in top_customers.iterrows():
        freq = f"{row['avg_days_between_orders']:.0f} days" if pd.notnull(row['avg_days_between_orders']) and row['avg_days_between_orders'] > 0 else "1st time buyer or Same Day"
        
        # Format the last order date cleanly ('%m/%d/%Y' for "10/24/2025")
        last_order_date = row['last_order'].strftime('%m/%d/%Y') if pd.notnull(row['last_order']) else "Unknown"

        top_cust_list.append({
            "Name": row['customer_name'], 
            "Orders": row['orders_count'],
            "Units": row['total_units'], 
            "Spent": row['total_spent'], 
            "Purchase Frequency": freq,
            "Last Order": last_order_date  # <--- NEW FIELD ADDED HERE
        })

    return {
        "product_name": product_name,
        "metrics": {
            "unique_buyers": int(paid_sales['customer_name'].nunique()),
            "total_orders": total_orders,
            "avg_units_per_order": float(order_group['quantity'].mean()),
            "avg_spend_per_order": float(order_group['totalAmount'].mean())
        },
        "order_distribution": formatted_dist,
        "top_customers": top_cust_list
    }

def format_customer_insights_md(data: dict) -> str:
    """Helper: Converts the raw insight dict into Markdown."""
    if "error" in data: return data["error"]
        
    m = data["metrics"]
    md = f"### Customer Insights: {data['product_name'].title()}\n\n"
    md += "#### Realized Product Popularity\n"
    md += f"- **Total Unique Buyers:** {m['unique_buyers']:,} customers\n"
    md += f"- **Total Paid Orders:** {m['total_orders']:,} orders\n"
    md += f"- **Average Purchase Size:** {m['avg_units_per_order']:,.1f} units per order\n"
    md += f"- **Average Spend Per Order:** ${m['avg_spend_per_order']:,.2f}\n"
    md += "> *Logic Note: Strictly filtered to PAID/PARTIALLY PAID orders.*\n\n"
    
    md += "#### Dynamic Order Size Distribution\n"
    md += "| Purchase Size Bracket | Total Orders | % of Total Orders |\n|---|---|---|\n"
    for label, count in data["order_distribution"].items():
        pct = (count / m['total_orders']) * 100
        md += f"| {label} | {count} | {pct:.1f}% |\n"
        
    md += "\n#### Top 5 Valuable Customers (Ranked by Revenue)\n"
    
    # NEW COLUMN ADDED TO HEADER
    md += "| Customer Name | Total Orders | Units Bought | Total Spent | Restock Frequency | Last Order Date |\n"
    md += "|---|---|---|---|---|---|\n"
    
    for c in data["top_customers"]:
        # NEW VARIABLE INJECTED INTO ROW
        md += f"| {c['Name']} | {c['Orders']} | {c['Units']:,.0f} | **${c['Spent']:,.2f}** | {c['Purchase Frequency']} | {c['Last Order']} |\n"
        
    return md

def _sales_trends_catalog_report(orders_path, catalog_path, products_path) -> list:
    try:
        # --- 1. Load Data ---
        try:
            if not (os.path.exists(orders_path) and os.path.exists(catalog_path) and os.path.exists(products_path)):
                return ["Error: Required data files (orders, catalog, products) are missing."]
            
            cat = pd.read_csv(catalog_path, low_memory=False)
            prod = pd.read_csv(products_path, low_memory=False)
            orders = pd.read_csv(orders_path, low_memory=False)
            
            if cat.empty or prod.empty or orders.empty:
                return ["Can not generate report: there is not enough data in the provided files."]
        except Exception as e:
            logger2.error(f"Data Loading Error in Sales Trends Catalog Report: {str(e)}")
            return ["Please check back later."]

        # --- 2. Generate Report ---
        try:
            top_products = get_top_3_products_by_revenue(cat, prod)
            
            if not top_products:
                return ["Can not generate report: No top products could be identified from the data."]

            report_lines = ["# Top 3 Sales Trends\n"]
            
            for i, product_name in enumerate(top_products, 1):
                report_lines.append(f"## {i}. {product_name}\n")
                
                # Loop-level try/except for safe data extraction
                try:
                    data = generate_advanced_customer_insights(product_name, cat, prod, orders)
                    
                    if "error" in data:
                        report_lines.append(data["error"])
                    else:
                        report_lines.append(format_customer_insights_md(data))
                except Exception as loop_e:
                    logger2.error(f"Customer Insights Error for product '{product_name}': {str(loop_e)}")
                    report_lines.append(f"*Insights for '{product_name}' could not be generated at this time.*")
                        
                report_lines.append("\n---\n")
                
            return ["\n".join(report_lines)]

        except Exception as e:
            logger2.error(f"Report Generation Error in Sales Trends Catalog Report: {str(e)}")
            return ["Please try again later."]

    # Fallback Catch-All
    except Exception as e:
        logger2.error(f"Critical Error in Sales Trends Catalog Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary change. Please check back later or contact support if you need assistance."]



# Orders agent predefined functions 
def _calculate_key_metrics_orders(orders_path) -> list:
    """Calculates Key Metrics including Standard Deviation."""
    try:
        orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
        
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

def _calculate_sales_orders_performance(orders_path) -> list:
    """Calculates Monthly Sales with % Change."""
    try:
        orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
        
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

def _calculate_discount_distribution(orders_path) -> list:
    """
    Calculates advanced Global Discount Performance with optimized vectorized
    categorization, refund filtering, and strict financial metrics.
    """
    try:
        # Load and clean headers
        orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
        
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

def _calculate_orders_fulfillment(orders_path) -> list:
    """Calculates Fulfillment Breakdown with Revenue Column."""
    try:
        orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
        
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

def _calculate_payment_status(orders_path) -> list:
    """Calculates Payment Status with Revenue Column."""
    try:
        orders = pd.read_csv(orders_path, encoding='utf-8-sig')
        orders.columns = orders.columns.str.strip().str.replace('\ufeff', '')
        
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

def _sales_trends_orders_report(orders_path) -> list:
    try:
        # --- 1. Load Data ---
        try:
            if not os.path.exists(orders_path):
                return ["Error: File not found."]
            
            df = pd.read_csv(orders_path, encoding='utf-8-sig')
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
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


# Map  names to the actual internal functions  
AGENT_CONFIG = {
    "customers_agent": {
        "churn_report": _stopped_ordering_report,
        "refined_opportunity_report": _opportunity_report,
        "top_customers_report": _top_customers_report,
        "visits_report": _visits_report
    },
    "catalog_agent": {
        "functional_product_analysis": _generate_executive_inventory_report,
        "product_performance":_generate_product_performance,
        "sales_trends_report": _sales_trends_catalog_report,
        "bundle_performance_report": _generate_bundle_report,
        "top_3_sales_breakdown": _calculate_top_3_sales_breakdown,
        "time_based_product_report": _generate_time_based_product_report
    },
    "orders_agent": {
        "key_metrics_report": _calculate_key_metrics_orders,
        "sales_performance_report": _calculate_sales_orders_performance,
        "discount_report": _calculate_discount_distribution,
        "payment_status_report": _calculate_payment_status,
        "fulfillment_report": _calculate_orders_fulfillment,
        "sales_trends_report": _sales_trends_orders_report
    }
}
import inspect
# Main Async Report Generator
async def generate_analytics_report_sectioned(
    orders_path, 
    products_path, 
    customers_path, 
    catalog_path,
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
        if 'customers_path' in sig.parameters: kwargs['customers_path'] = customers_path
        if 'products_path' in sig.parameters: kwargs['products_path'] = products_path
        if 'catalog_path' in sig.parameters: kwargs['catalog_path'] = catalog_path
        
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


async def main():
    # Example usage eb9985d2-3b94-4146-adb5-c0877672be72 GNGR_TEST FULL_DIST_TEST
    #report = await generate_analytics_report_sectioned(
    #    orders_path="data\\GNGR_TEST\\oorders.csv",
    #    products_path="data\\GNGR_TEST\\pproducts.csv",
    #    customers_path="data\\GNGR_TEST\\ccustomers.csv",
    #    catalog_path="data\\GNGR_TEST\\cleaned_catalog.csv",
    #    agent_type="orders_agent",
    #    report_type="full_report"
    #)
    report = await generate_analytics_report_sectioned(
        orders_path="data\\FULL_DIST_TEST\\cleaned_orders.csv",
        products_path="data\\FULL_DIST_TEST\\cleaned_products.csv",
        customers_path="data\\FULL_DIST_TEST\\cleaned_customers.csv",
        catalog_path="data\\FULL_DIST_TEST\\cleaned_catalog.csv",
        agent_type="customers_agent",
        report_type="visits_report"
    )
    print(report.get("sections", "No full report generated.").get("visits_report", "Report section not found."))


if __name__ == "__main__":
    asyncio.run(main())