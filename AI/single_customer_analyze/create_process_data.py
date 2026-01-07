import pandas as pd
import asyncio
import logging

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
    for col in columns:
        if col in df.columns:
            original = df[col].copy()
            
            # Enhanced preprocessing
            cleaned = (
                original
                .astype(str)
                .str.split(r'\s*\(.*', n=1).str[0]  # Remove anything after (
                .str.strip()
                .str.replace(r'([+-]\d{2}):(\d{2})$', r'\1\2', regex=True)  # Fix tz format
                .str.replace(r'\b(UTC|GMT)\b', '', regex=True)  # Remove UTC/GMT prefix
                .str.replace(r'\s+', ' ', regex=True)  # Normalize spaces
            )
            
            # List of formats to try (order matters!)
            formats = [
                '%Y-%m-%d %H:%M:%S%z',          # Case: "2025-03-06 13:24:40+0000"
                '%a %b %d %Y %H:%M:%S %z',      # Case: "Fri Jul 26 2024 18:53:00 +0000"
                '%Y-%m-%d %H:%M:%S',            # Fallback for tz-naive
                '%a %b %d %Y %H:%M:%S',         # Fallback for tz-naive
            ]
            
            parsed = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
            
            # Try each format sequentially
            for fmt in formats:
                mask = parsed.isna()
                if not mask.any():
                    break
                
                # Attempt parsing with current format
                temp = pd.to_datetime(
                    cleaned[mask],
                    format=fmt,
                    errors='coerce',
                    utc=True
                )
                
                # Only keep successful parses
                parsed[mask] = temp.dropna()
            
            df[col] = parsed
            
            # Report failures
            failed = original[parsed.isna()]

                
    return df

def preprocess_orders(file_path):
    """Loads and cleans order data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded orders CSV from {file_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Error loading orders CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["orders"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["orders"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger.warning(f"Added missing column '{col}' to orders DataFrame")
    
    # Remove duplicate orders based on 'id'
    df.drop_duplicates(subset=["id"], inplace=True)
    
    # Convert datetime columns to UTC
    datetime_cols = [
        "createdAt", "updatedAt", "shipEngineUpdatedAt", "canceledAt", "shippedAt", 
        "completedAt", "paidAt", "partiallyPaidAt", "unpaidAt", "fulfilledAt", 
        "paymentDue", "partiallyFulfilledAt", "unfulfilledAt"
    ]
    df = convert_to_datetime(df, datetime_cols)
    
    # Extract 'month' from 'createdAt'; works with pd.NA, resulting in NaT
    df['month'] = df['createdAt'].dt.tz_localize(None).dt.to_period('M')
    
    # Convert financial columns to numeric, filling NaN with 0
    for col in ["totalAmount", "totalRawAmount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Round financial columns to 2 decimal places, only if present
    columns_to_round = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
    df = df.round({col: 2 for col in columns_to_round if col in df.columns})
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = [
        'appCustomer_name', 'shippingCarrierDuplicate_name', 'fulfillBy', 'canceledAt', 
        'fulfillVia', 'emailed', 'cancelReason', 'note_text', 'partiallyPaidAt', 
        'quickbooksOrderId', 'remindersSent', 'paymentTermsDuplicate_dayOfMonthDue', 
        'contactDuplicate_phone', 'contactDuplicate_email', 'paymentTermsDuplicate_daysInvoices'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df

def preprocess_products(file_path):
    """Loads and cleans product data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded products CSV from {file_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.warning(f"Error loading products CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["products"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["products"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger.warning(f"Added missing column '{col}' to products DataFrame")
    
    # Remove duplicates based on 'id' and 'orderId'
    df.drop_duplicates(subset=["id", "orderId"], inplace=True)
    
    # Convert 'createdAt' to datetime with UTC
    df = convert_to_datetime(df, ["createdAt"])
    
    # Convert numeric columns, filling NaN with 0
    for col in ["price", "totalAmount", "quantity", "paidQuantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if col in ["quantity", "paidQuantity"]:
                df[col] = df[col].astype(int)
    
    # Round financial columns to 2 decimal places, only if present
    columns_to_round = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
    df = df.round({col: 2 for col in columns_to_round if col in df.columns})
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = ['description', 'barcode']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df

async def create_user_data(orders_df_path, products_df_path, folder_to_save):
    """Processes order and product data asynchronously and saves the cleaned results."""
    logger.info("Starting create_user_data")

    # Run preprocessing functions in separate threads asynchronously
    orders_df, products_df = await asyncio.gather(
        asyncio.to_thread(preprocess_orders, orders_df_path),
        asyncio.to_thread(preprocess_products, products_df_path)
    )
    
    # Define an async function to save DataFrame with error handling
    async def save_df(df, path):
        try:
            await asyncio.to_thread(df.to_csv, path, index=False)
            logger.info(f"Saved DataFrame to {path}")
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
    
    # Save cleaned DataFrames to CSV files asynchronously
    try:
        await asyncio.gather(
            save_df(orders_df, f'data/{folder_to_save}/work_ord.csv'),
            save_df(products_df, f'data/{folder_to_save}/work_prod.csv')
        )
    except Exception as e:
        logger.error("file creating", e)
    
    logger.info("Completed create_user_data")
