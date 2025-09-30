import pandas as pd
import numpy as np
import asyncio
import aiofiles
from datetime import datetime


from AI.utils import get_logger
logger2 = get_logger("logger2", "project_log_many.log", False)


# Functions to calculate the key report metrics
def get_top_products_for_contact(contact_name, orders_df, products_df, top_n=3):
    """Helper function to get top products for a specific contact"""
    try:
        contact_orders = orders_df[orders_df['contactDuplicate_name'] == contact_name]
        merged = pd.merge(contact_orders, products_df, left_on='id', right_on='orderId')
        product_counts = merged['manufacturerName'].value_counts().head(top_n)
        return product_counts.index.tolist()
    except Exception as e:
        logger2.error("Error in get_top_products_for_contact:", e)
    
def top_new_contact(ord_df, products_df):
    """Helper function to get top_new_contact by a rules: orderStatus - COMPLETED, paymentStatus - PAID, deliveryStatus - FULFILLED"""
    try:
        successful_orders = ord_df[(ord_df['orderStatus'] == 'COMPLETED') & 
                                  (ord_df['paymentStatus'] == 'PAID') & 
                                  (ord_df['deliveryStatus'] == 'FULFILLED')]

        if successful_orders.empty:
            return ("No data", [])

        best_contact = successful_orders['contactDuplicate_name'].value_counts().idxmax()
        top_products = get_top_products_for_contact(best_contact, ord_df, products_df)

        return (best_contact, top_products)
    except Exception as e:
        logger2.error("Error in top_new_contact:", e)
        
def top_reorder_contact(ord_df, products_df):
    """Helper function to get top_reorder_contact by a rule contactDuplicate_name > 1 in data"""
    
    try:
        repeat_contacts = ord_df.groupby('contactDuplicate_name').filter(lambda x: len(x) > 1)

        if repeat_contacts.empty:
            return ("No data", [])

        best_contact = repeat_contacts['contactDuplicate_name'].value_counts().idxmax()
        top_products = get_top_products_for_contact(best_contact, ord_df, products_df)

        return (best_contact, top_products)
    except Exception as e:
        logger2.error("Error in top_reorder_contact:", e)
        
def format_am_pm(hour):
    """Convert 24-hour format to 12-hour AM/PM format"""
    try:
        if hour == 0:
            return "12:00 AM"
        elif 1 <= hour < 12:
            return f"{hour}:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{hour - 12}:00 PM"
    except Exception as e:
        logger2.error("Error in format_am_pm:", e)
    
def peak_visit_time(df):
    try:
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
    except Exception as e:
        logger2.error("Error in peak_visit_time:", e)
    
def get_contact_peak_time(contact_name, orders_df):
    """Calculate peak visit time for individual contact"""
    try:
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
    except Exception as e:
        logger2.error("Error in get_contact_peak_time:", e)
    
def customer_insights(ord_df, products_df):
    try:
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
    except Exception as e:
        logger2.error("Error in customer_insights:", e)
# Helper formatting functions block
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

