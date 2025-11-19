import pandas as pd
import asyncio
import aiofiles
import glob
from pathlib import Path

from AI.utils import get_logger
logger2 = get_logger("logger2", "project_log_many.log", False)

EXPECTED_COLUMNS = {
    "orders": [
        "salesDuplicate_name", "appCustomer_name", "contactDuplicate_name", "contactDuplicate_phone", "contactDuplicate_email", "contactDuplicate_role", 
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
    try:
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
    except Exception as e:
        logger2.error("Error in convert to datetime: ",e)

def preprocess_orders(file_path):
    """Loads and cleans order data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger2.info(f"Loaded orders CSV from {file_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger2.error(f"Error loading orders CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["orders"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["orders"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger2.warning(f"Added missing column '{col}' to orders DataFrame")
    if df.empty:
        logger2.warning(f"Orders DataFrame is empty from file: {file_path}")
        return df, file_path  # Return tuple with file_path
    
    # Remove duplicate orders based on 'id'
    df.drop_duplicates(subset=["id"], inplace=True)
    
    # Convert datetime columns to UTC
    datetime_cols = [
        "createdAt", "updatedAt", "shipEngineUpdatedAt", "canceledAt", "shippedAt", 
        "completedAt", "paidAt", "partiallyPaidAt", "unpaidAt", "fulfilledAt", 
        "paymentDue", "partiallyFulfilledAt", "unfulfilledAt"
    ]
    df = convert_to_datetime(df, datetime_cols)
    
    try:
        # Extract 'month' from 'createdAt'; works with pd.NA, resulting in NaT
        df['month'] = df['createdAt'].dt.tz_localize(None).dt.to_period('M')

        # Convert financial columns to numeric, filling NaN with 0
        for col in ["totalAmount", "totalRawAmount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Round financial columns to 2 decimal places, only if present
        columns_to_round = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
        df = df.round({col: 2 for col in columns_to_round if col in df.columns})
    except Exception as e:
        logger2.error("Error in preproc orders financial columns: ", e)    
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = [
        'appCustomer_name', 'shippingCarrierDuplicate_name', 'fulfillBy', 'canceledAt', 
        'fulfillVia', 'emailed', 'cancelReason', 'note_text', 'partiallyPaidAt', 
        'quickbooksOrderId', 'remindersSent', 'paymentTermsDuplicate_dayOfMonthDue', 
        'contactDuplicate_phone', 'contactDuplicate_email', 'paymentTermsDuplicate_daysInvoices'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df, file_path

def preprocess_products(file_path):
    """Loads and cleans product data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger2.info(f"Loaded products CSV from {file_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger2.warning(f"Error loading products CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["products"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["products"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger2.warning(f"Added missing column '{col}' to products DataFrame")
    if df.empty:
        logger2.warning(f"Products DataFrame is empty from file: {file_path}")
        return df, file_path  # Return tuple with file_path
    
    # Remove duplicates based on 'id' and 'orderId'
    df.drop_duplicates(subset=["id", "orderId"], inplace=True)
    
    # Convert 'createdAt' to datetime with UTC
    df = convert_to_datetime(df, ["createdAt"])
    
    # Convert numeric columns, filling NaN with 0
    for col in ["price", "itemDiscountAmount", "totalAmount", "quantity", "paidQuantity"]:
        if col in df.columns:
            # Convert to float first (don't convert to int yet)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Round FINANCIAL columns to 2 decimal places
    monetary_cols = ['price', 'itemDiscountAmount', 'totalAmount']
    present_monetary_cols = [col for col in monetary_cols if col in df.columns]
    if present_monetary_cols:
        df = df.round({col: 2 for col in present_monetary_cols})
    
    # Convert quantities to integers AFTER rounding
    for col in ["quantity", "paidQuantity"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = ['description', 'barcode', 'color', 'size']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df, file_path


def data_clean_orders(orders_df: pd.DataFrame):
    """Clean and transform orders data, returning processed DataFrame and an optional error message."""
    # Check for required columns
    required_columns = ['createdAt', 'deliveryStatus', 'paymentStatus', 'orderStatus', 'archived', 
                        'totalAmount', 'totalDiscountValue', 'deliveryFee']
    missing_cols = [col for col in required_columns if col not in orders_df.columns]
    if missing_cols:
        logger2.warning( f"The orders file is missing these required columns: {', '.join(missing_cols)}. Please check the file.")
        return pd.DataFrame()

    try:
        # Convert and filter dates
        try:
            orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'], utc=True)
            # Changed from .dt.to_period('M') to .dt.strftime('%m/%Y')
            orders_df['month'] = orders_df['createdAt'].dt.strftime('%m/%Y')
            logger2.info("Dates processed successfully")
        except Exception as e:
            logger2.error(f"Error processing dates: {str(e)}")
            logger2.warning("There was an issue with the date format in the orders file. Please ensure the 'createdAt' column is in a valid date format.")
            return pd.DataFrame()
        # Standardize statuses
        try:
            orders_df['deliveryStatus'] = orders_df['deliveryStatus'].str.strip().str.upper().fillna('UNKNOWN')
            orders_df['paymentStatus'] = orders_df['paymentStatus'].str.strip().str.upper().fillna('UNKNOWN')
            logger2.info("Statuses standardized successfully")
        except Exception as e:
            logger2.error(f"Error standardizing statuses: {str(e)}")
            return pd.DataFrame()
        # Filter out 'CANCELED' orders and keep archived == False
        try:
            #orders_df = orders_df[
            #    (orders_df['orderStatus'] != 'CANCELED') &
            #    (orders_df['archived'] == False)
            #]
            archived_mask = orders_df['archived'].astype(str).str.lower().isin(['false', '0', 'no', 'f'])
            status_mask = orders_df['orderStatus'].str.upper() != 'CANCELED'
            orders_df = orders_df[status_mask & archived_mask]
            if orders_df.empty:
                logger2.warning("No valid orders found after filtering. All orders are either canceled or archived. Please try a different file.")
                return orders_df
            logger2.info(f"Filtered orders, remaining: {len(orders_df)}")
        except Exception as e:
            logger2.error(f"Error filtering orders: {str(e)}")
            return pd.DataFrame()
        # Clean financial columns
        try:
            money_cols = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
            orders_df.loc[:, money_cols] = (
                orders_df.loc[:, money_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
            )

            logger2.info("Financial columns cleaned successfully")
        except Exception as e:
            logger2.error(f"Error cleaning financial columns: {str(e)}")
            return pd.DataFrame()
        
        orders_df = orders_df.round({col: 2 for col in money_cols})
        return orders_df

    except Exception as e:
        logger2.error(f"Order preprocessing failed: {str(e)}")
        return pd.DataFrame()

def data_clean_products(products_df: pd.DataFrame):
    """Clean and transform products data, returning processed DataFrame and an optional error message."""
    # Check for required columns
    required_columns = ['sku', 'quantity', 'paidQuantity', 'price', 'itemDiscountAmount', 'orderId']
    missing_cols = [col for col in required_columns if col not in products_df.columns]
    if missing_cols:
        logger2.warning(f"The products file is missing these required columns: {', '.join(missing_cols)}. Please check the file.")
        return pd.DataFrame()
    try:
        # Clean SKUs and quantities
        try:
            products_df['sku'] = products_df['sku'].fillna('MISSING_SKU')
            products_df['quantity'] = pd.to_numeric(products_df['quantity'], errors='coerce').fillna(0)
            products_df['paidQuantity'] = pd.to_numeric(products_df['paidQuantity'], errors='coerce').fillna(0)
            logger2.info("SKUs and quantities cleaned successfully")
        except Exception as e:
            logger2.error(f"Error cleaning SKUs and quantities: {str(e)}")
            return pd.DataFrame()
        # Clean prices
        try:
            price_cols = ['price', 'itemDiscountAmount']
            products_df[price_cols] = products_df[price_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            logger2.info("Prices cleaned successfully")
        except Exception as e:
            logger2.error(f"Error cleaning prices: {str(e)}")
            return pd.DataFrame()
        return products_df

    except Exception as e:
        logger2.error(f"Product preprocessing failed: {str(e)}")
        return pd.DataFrame()


async def save_df(df, path):
    """Asynchronously saves a DataFrame to a CSV file with error handling."""
    try:
        await asyncio.to_thread(df.to_csv, path, index=False)
        #logger2.info(f"Saved DataFrame to {path}")
    except Exception as e:
        logger2.error(f"Error saving {path}: {e}")

async def create_group_user_data(orders_df_paths, products_df_paths, folder_to_save, uuid):
    """Processes multiple pairs of order and product data asynchronously and saves the cleaned results.
    Returns (success_status, list_of_empty_files)"""
    logger2.info("Starting create_user_data for multiple file pairs")
    
    # Validate that the number of orders and products files match
    if len(orders_df_paths) != len(products_df_paths):
        raise ValueError("Mismatched number of orders and products files")
    
    all_empty_files = []  # Track all empty files across all pairs
    
    async def process_pair(orders_path, products_path, index):
        nonlocal all_empty_files
        pair_empty_files = []  # Track empty files for this specific pair
        
        try:
            #logger2.info(f"Processing pair {index}: {orders_path} and {products_path}")
            
            # Preprocess orders and products concurrently
            (orders_df, orders_path), (products_df, products_path) = await asyncio.gather(
                asyncio.to_thread(preprocess_orders, orders_path),
                asyncio.to_thread(preprocess_products, products_path)
            )
            
            # Check for empty DataFrames
            if orders_df.empty:
                pair_empty_files.append(orders_path)
            if products_df.empty:
                pair_empty_files.append(products_path)
                
            # Track empty files for final report
            all_empty_files.extend(pair_empty_files)
            
            # Skip cleaning if either DataFrame is empty
            if not orders_df.empty and not products_df.empty:
                _orders_df, _products_df = await asyncio.gather(
                    asyncio.to_thread(data_clean_orders, orders_df),
                    asyncio.to_thread(data_clean_products, products_df)
                )
            else:
                _orders_df = orders_df
                _products_df = products_df
                logger2.warning(f"Skipping cleaning for pair {index} due to empty DataFrames")
            
            # Save results
            orders_save_path = f'data/{uuid}/work_data_folder/work_ord_{index}.csv'
            products_save_path = f'data/{uuid}/work_data_folder/work_prod_{index}.csv'
            await asyncio.gather(
                save_df(_orders_df, orders_save_path),
                save_df(_products_df, products_save_path)
            )
            
            #logger2.info(f"Completed processing pair {index}")
            return pair_empty_files
        except Exception as e:
            logger2.error(f"Error processing pair {index}: {e}")
            # Add both files to empty list on critical error
            all_empty_files.extend([orders_path, products_path])
            return [orders_path, products_path]
    
    try:
        tasks = []
        for i, (orders_path, products_path) in enumerate(zip(orders_df_paths, products_df_paths)):
            task = asyncio.create_task(process_pair(orders_path, products_path, i))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        logger2.info("Completed create_user_data for all file pairs")
        
        # Return success status and all empty files found
        #print('gere')
        #print(all_empty_files)
        return True, all_empty_files
    except Exception as e:
        logger2.error("Error in create_group_user_data: ", e)
        return False, all_empty_files

async def concat_customer_csv(uuid: str) -> str:
    """
    Asynchronously concatenate all customer.csv files within the given UUID folder.
    
    Args:
        uuid (str): The main UUID identifying the root folder.
    
    Returns:
        str: Path to the concatenated CSV file.
    
    Raises:
        ValueError: If no customer CSV files are found.
    """
    main_path = Path(uuid)
    
    # Synchronously list and sort customer UUID folders
    customer_folders = sorted([f for f in main_path.iterdir() if f.is_dir()])
    
    # Construct paths to all existing customer.csv files
    csv_paths = [customer_folder / "customer" / "customer.csv" 
                 for customer_folder in customer_folders 
                 if (customer_folder / "customer" / "customer.csv").exists()]
    
    if not csv_paths:
        raise ValueError("No customer CSV files found")
    
    # Read all CSV files concurrently
    all_lines = await asyncio.gather(*[read_csv_customer(path) for path in csv_paths])
    
    # Extract header from the first file
    header = all_lines[0][0]
    
    # Collect all data rows (excluding headers) from all files
    data_rows = [line for lines in all_lines for line in lines[1:]]
    
    # Combine header and data rows
    concatenated_content = [header] + data_rows
    
    # Define the new file path
    new_file_path = main_path / "concatenated_customers.csv"
    
    # Write the concatenated content asynchronously
    async with aiofiles.open(new_file_path, mode='w', encoding='utf-8') as f:
        await f.write('\n'.join(concatenated_content))
    
    return str(new_file_path)

async def read_csv_customer(path):
    """
    Asynchronously read a CSV file and return its lines.
    
    Args:
        path: Path to the CSV file.
    
    Returns:
        list: List of lines from the file.
    """
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        content = await f.read()
    return content.splitlines()

# Async data loading
async def read_csv_async(file_path):
    try:
        df = await asyncio.to_thread(pd.read_csv, file_path)
        return df
    except Exception as e:
        return pd.DataFrame()
    
async def load_data(directory):
    try:
        orders_files = await asyncio.to_thread(glob.glob, f"{directory}/work_data_folder/work_ord_*.csv")
        products_files = await asyncio.to_thread(glob.glob, f"{directory}/work_data_folder/work_prod_*.csv")

        orders_dfs = [df for df in await asyncio.gather(*[read_csv_async(f) for f in orders_files]) if not df.empty]
        products_dfs = [df for df in await asyncio.gather(*[read_csv_async(f) for f in products_files]) if not df.empty]

        if not orders_dfs or not products_dfs:
            return pd.DataFrame(), pd.DataFrame()

        orders = pd.concat(orders_dfs, ignore_index=True)
        products = pd.concat(products_dfs, ignore_index=True)

        return orders, products
    except Exception as e:
        logger2.error("Load data problem: ", e)
        return pd.DataFrame(), pd.DataFrame()
    
#Test zone
def one_file_preprocess_orders(file_path):
    """Loads and cleans order data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger2.info(f"Loaded orders CSV from {file_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger2.error(f"Error loading orders CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["orders"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["orders"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger2.info(f"Added missing column '{col}' to orders DataFrame")
    if df.empty:
        logger2.warning(f"Orders DataFrame is empty from file: {file_path}")
        return df, file_path  # Return tuple with file_path
    
    # Remove duplicate orders based on 'id'
    #df.drop_duplicates(subset=["id"], inplace=True)

    # Convert datetime columns to UTC
    datetime_cols = [
        "createdAt", "updatedAt", "shipEngineUpdatedAt", "canceledAt", "shippedAt", 
        "completedAt", "paidAt", "partiallyPaidAt", "unpaidAt", "fulfilledAt", 
        "paymentDue", "partiallyFulfilledAt", "unfulfilledAt"
    ]
    df = convert_to_datetime(df, datetime_cols)
    
    try:
        # Extract 'month' from 'createdAt'; works with pd.NA, resulting in NaT
        df['month'] = df['createdAt'].dt.tz_localize(None).dt.to_period('M')

        # Convert financial columns to numeric, filling NaN with 0
        for col in ["totalAmount", "totalRawAmount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Round financial columns to 2 decimal places, only if present
        columns_to_round = ['totalAmount', 'totalDiscountValue', 'deliveryFee']
        df = df.round({col: 2 for col in columns_to_round if col in df.columns})
    except Exception as e:
        logger2.error("Error in preproc orders financial columns: ", e)    
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = [
        'appCustomer_name', 'shippingCarrierDuplicate_name', 'fulfillBy', 'canceledAt', 
        'fulfillVia', 'emailed', 'cancelReason', 'note_text', 'partiallyPaidAt', 
        'quickbooksOrderId', 'remindersSent', 'paymentTermsDuplicate_dayOfMonthDue', 
        'contactDuplicate_phone', 'contactDuplicate_email', 'paymentTermsDuplicate_daysInvoices'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df, file_path

def one_file_preprocess_products(file_path):
    """Loads and cleans product data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger2.info(f"Loaded products CSV from {file_path}")
        df['sku'] = df['sku'].astype(str)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger2.warning(f"Error loading products CSV file {file_path}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS["products"])
    else:
        # Add missing columns
        for col in EXPECTED_COLUMNS["products"]:
            if col not in df.columns:
                df[col] = pd.NA
                logger2.info(f"Added missing column '{col}' to products DataFrame")
    if df.empty:
        logger2.warning(f"Products DataFrame is empty from file: {file_path}")
        return df, file_path  # Return tuple with file_path
    
    # Remove duplicates based on 'id' and 'orderId'
    #df.drop_duplicates(subset=["id", "orderId"], inplace=True)
    
    # Convert 'createdAt' to datetime with UTC
    df = convert_to_datetime(df, ["createdAt"])
    
    # Convert numeric columns, filling NaN with 0
    for col in ["price", "itemDiscountAmount", "totalAmount", "quantity", "paidQuantity"]:
        if col in df.columns:
            # Convert to float first (don't convert to int yet)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Round FINANCIAL columns to 2 decimal places
    monetary_cols = ['price', 'itemDiscountAmount', 'totalAmount']
    present_monetary_cols = [col for col in monetary_cols if col in df.columns]
    if present_monetary_cols:
        df = df.round({col: 2 for col in present_monetary_cols})
    
    # Convert quantities to integers AFTER rounding
    for col in ["quantity", "paidQuantity"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Drop unnecessary columns, ignoring errors if columns are missing
    columns_to_drop = ['description', 'barcode', 'color', 'size']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df, file_path



async def prepared_big_data(orders_path: str, products_path: str) -> tuple:
    """Make full data preprocessing asynchronously"""
    # Run CPU-intensive preprocessing in thread pool
    loop = asyncio.get_event_loop()
    
    preprocessed_orders, _o = await loop.run_in_executor(
        None, one_file_preprocess_orders, orders_path
    )
    preprocessed_products, _p = await loop.run_in_executor(
        None, one_file_preprocess_products, products_path
    )

    full_cleaned_orders = await loop.run_in_executor(
        None, data_clean_orders, preprocessed_orders
    )
    full_cleaned_products = await loop.run_in_executor(
        None, data_clean_products, preprocessed_products
    )

    return full_cleaned_orders, full_cleaned_products
