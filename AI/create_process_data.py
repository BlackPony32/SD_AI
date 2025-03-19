import pandas as pd
import asyncio

# Expected columns for validation
EXPECTED_COLUMNS = {
    "orders": [
        "id", "salesDuplicate_name", "appCustomer_name", "contactDuplicate_name", "contactDuplicate_phone", "contactDuplicate_email", "contactDuplicate_role", 
        "paymentTermsDuplicate_name", "paymentTermsDuplicate_daysInvoices", "paymentTermsDuplicate_dayOfMonthDue", "paymentTermsDuplicate_dueNextMonthDays", "paymentTermsDuplicate_type", 
        "createdType", "type", "archived", "customerDiscount", "totalOrderDiscountAmount", "totalOrderDiscountType", "manualDeliveryFee", "deliveryFee", "appliedDiscountsType", 
        "emailed", "remindersSent", "createdAt", "updatedAt", "shipEngineUpdatedAt", "cancelReason", "fulfillBy", "fulfillVia", "shippingCarrierDuplicate_name", "canceledAt", "shippedAt", "completedAt", 
        "paidAt", "partiallyPaidAt", "unpaidAt", "fulfilledAt", "paymentDue", "partiallyFulfilledAt", "unfulfilledAt", "orderStatus", "paymentStatus", "deliveryStatus", "customId_customId", 
        "note_text", "quickbooksOrderId", "createdBy", "totalRawAmount", "manufacturerDiscountValue", "totalOrderDiscountValue", "customerDiscountValue", "totalDiscountValue", "totalAmountWithoutDelivery", 
        "totalAmount", "totalQuantity"
    ],
    "products": [
        "id", "orderId", "manufacturerName", "productCategoryName", "createdAt", "name", "type", "description", "sku", "barcode", "itemsPerCase", "color", "size", "quantity", 
        "paidQuantity", "price", "itemDiscountAmount", "itemDiscountType", "itemDiscountValue", "delivered", "amount", "totalRawAmount", "totalAmount"
    ]
}

def convert_to_datetime(df, columns):
    """Converts specified columns to datetime format with UTC timezone."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    return df

def preprocess_orders(file_path):
    """Loads and cleans order data from a CSV file."""
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Check for missing columns
    missing_columns = [col for col in EXPECTED_COLUMNS["orders"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in orders: {missing_columns}")
    
    # Remove duplicate orders based on 'id'
    df.drop_duplicates(subset=["id"], inplace=True)
    
    # Convert datetime columns to UTC
    datetime_cols = [
        "createdAt", "updatedAt", "shipEngineUpdatedAt", "canceledAt", "shippedAt", 
        "completedAt", "paidAt", "partiallyPaidAt", "unpaidAt", "fulfilledAt", 
        "paymentDue", "partiallyFulfilledAt", "unfulfilledAt"
    ]
    df = convert_to_datetime(df, datetime_cols)
    
    # Extract 'month' from 'createdAt'; timezone info is dropped, but UTC ensures correctness
    df['month'] = df['createdAt'].dt.tz_localize(None).dt.to_period('M')

    
    # Convert financial columns to numeric, filling NaN with 0
    df["totalAmount"] = pd.to_numeric(df["totalAmount"], errors='coerce').fillna(0)
    df["totalRawAmount"] = pd.to_numeric(df["totalRawAmount"], errors='coerce').fillna(0)
    
    # Round financial columns to 2 decimal places for consistency
    df = df.round({'totalAmount': 2, 'totalDiscountValue': 2, 'deliveryFee': 2})
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = [
        'appCustomer_name', 'shippingCarrierDuplicate_name', 'fulfillBy', 'canceledAt', 
        'fulfillVia', 'emailed', 'cancelReason', 'note_text', 'partiallyPaidAt', 
        'quickbooksOrderId', 'remindersSent', 'paymentTermsDuplicate_dayOfMonthDue', 
        'contactDuplicate_phone', 'contactDuplicate_email', 'paymentTermsDuplicate_daysInvoices'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

def preprocess_products(file_path):
    """Loads and cleans product data from a CSV file."""
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Check for missing columns
    missing_columns = [col for col in EXPECTED_COLUMNS["products"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in products: {missing_columns}")
    
    # Remove duplicates based on 'id' and 'orderId'
    df.drop_duplicates(subset=["id", "orderId"], inplace=True)
    
    # Convert 'createdAt' to datetime with UTC
    df = convert_to_datetime(df, ["createdAt"])
    
    # Convert numeric columns, filling NaN with 0
    df["price"] = pd.to_numeric(df["price"], errors='coerce').fillna(0)
    df["totalAmount"] = pd.to_numeric(df["totalAmount"], errors='coerce').fillna(0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce').fillna(0).astype(int)
    df["paidQuantity"] = pd.to_numeric(df["paidQuantity"], errors='coerce').fillna(0).astype(int)
    
    # Round financial columns to 2 decimal places for consistency
    df = df.round({'totalAmount': 2, 'totalDiscountValue': 2, 'deliveryFee': 2})
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = ['description', 'barcode', 'color', 'size']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

async def create_user_data(orders_df_path, products_df_path, folder_to_save):
    """Processes order and product data asynchronously and saves the cleaned results."""
    # Run preprocessing functions in separate threads asynchronously
    orders_df, products_df = await asyncio.gather(
        asyncio.to_thread(preprocess_orders, orders_df_path),
        asyncio.to_thread(preprocess_products, products_df_path)
    )
    
    # Save cleaned DataFrames to CSV files asynchronously
    await asyncio.gather(
        asyncio.to_thread(orders_df.to_csv, f'{folder_to_save}/work_ord.csv', index=False),
        asyncio.to_thread(products_df.to_csv, f'{folder_to_save}/work_prod.csv', index=False)
    )