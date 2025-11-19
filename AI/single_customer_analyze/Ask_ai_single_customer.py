from typing import List, AsyncGenerator, Tuple, Any
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_create_full_report, prompt_agent_create_sectioned, prompt_for_state_agent

from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession

from agents.extensions.memory import AdvancedSQLiteSession

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from agents.extensions.memory import AdvancedSQLiteSession
import asyncio
import aiofiles
import os
import time
from dotenv import load_dotenv
load_dotenv()


from typing import List, AsyncGenerator, Tuple, Any
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_create_full_report, prompt_agent_create_sectioned, prompt_for_state_agent
import pandas as pd

from AI.utils import get_logger
logger2 = get_logger("logger2", "project_log_many.log", False)

llm_model = OpenAIResponsesModel(model='gpt-4.1', openai_client=AsyncOpenAI()) 

def get_all_data(customer_id):
    DATA_DIR = os.path.join('data', customer_id)

    activities_df = pd.read_csv(os.path.join(DATA_DIR, 'activities',"activities.csv"))
    notes_df = pd.read_csv(os.path.join(DATA_DIR, 'activities',"notes.csv"))
    order_products_df = pd.read_csv(os.path.join(DATA_DIR, 'orders',"order_products.csv"))
    order_products_df = order_products_df.where(pd.notnull(order_products_df), None)

    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'orders',"orders.csv"))
    orders_df = orders_df.where(pd.notnull(orders_df), None)
    tasks_df = pd.read_csv(os.path.join(DATA_DIR, 'activities',"tasks.csv"))
    tasks_df = tasks_df.where(pd.notnull(tasks_df), None)

    return activities_df, notes_df, order_products_df, orders_df, tasks_df

#activities_df, notes_df, order_products_df, orders_df, tasks_df = get_all_data('b17f86d1-9bf9-4d75-aedd-d25b0ddc9562')

def _tasks_to_records(df) -> List[Dict[str, Any]]:
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _get_all_tasks() -> List[Dict[str, Any]]:
    return _tasks_to_records(tasks_df)


def _find_task(task_id: str) -> Optional[Dict[str, Any]]:
    df = tasks_df
    if "id" not in df.columns:
        return None
    row = df[df["id"] == task_id].head(1)
    if row.empty:
        return None
    return _tasks_to_records(row)[0]


def _parse_created_at(value):
    if not value or not isinstance(value, str):
        return datetime.min

    s = value.strip()

    if " (" in s:
        s = s.split(" (", 1)[0].strip()

    try:
        return datetime.strptime(s, "%a %b %d %Y %H:%M:%S GMT%z")
    except ValueError:
        pass

    try:
        return datetime.strptime(s, "%a %b %d %Y %H:%M:%S")
    except ValueError:
        return datetime.min


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _filter_by_customer(customer_id: Optional[str]) -> List[Dict[str, Any]]:
    df = orders_df
    if customer_id and "customerId" in df.columns:
        df = df[df["customerId"] == customer_id]
    return _df_to_records(df)


def _find_order(
    order_id: Optional[str] = None,
    custom_order_id: Optional[str] = None,
    shopify_order_id: Optional[str] = None,
    quickbooks_order_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = orders_df

    if order_id:
        row = df[df["id"] == order_id].head(1)
        if not row.empty:
            return _df_to_records(row)[0]

    if custom_order_id and "customId_customId" in df.columns:
        row = df[df["customId_customId"] == custom_order_id].head(1)
        if not row.empty:
            return _df_to_records(row)[0]

    if shopify_order_id and "shopifyOrderId" in df.columns:
        row = df[df["shopifyOrderId"] == shopify_order_id].head(1)
        if not row.empty:
            return _df_to_records(row)[0]

    if quickbooks_order_id and "quickbooksOrderId" in df.columns:
        row = df[df["quickbooksOrderId"] == quickbooks_order_id].head(1)
        if not row.empty:
            return _df_to_records(row)[0]
    return None


def _status_view(o: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "orderId": o.get("id"),
        "customOrderId": o.get("customId_customId"),
        "shopifyOrderId": o.get("shopifyOrderId"),
        "quickbooksOrderId": o.get("quickbooksOrderId"),
        "createdAt": o.get("createdAt"),
        "updatedAt": o.get("updatedAt"),
        "orderStatus": o.get("orderStatus"),
        "paymentStatus": o.get("paymentStatus"),
        "deliveryStatus": o.get("deliveryStatus"),
        "shippedAt": o.get("shippedAt"),
        "completedAt": o.get("completedAt"),
        "canceledAt": o.get("canceledAt"),
        "fulfilledAt": o.get("fulfilledAt"),
        "unfulfilledAt": o.get("unfulfilledAt"),
        "partiallyFulfilledAt": o.get("partiallyFulfilledAt"),
        "paidAt": o.get("paidAt"),
        "partiallyPaidAt": o.get("partiallyPaidAt"),
        "unpaidAt": o.get("unpaidAt"),
        "paymentDue": o.get("paymentDue"),
        "totalAmount": o.get("totalAmount"),
        "totalAmountWithoutDelivery": o.get("totalAmountWithoutDelivery"),
        "totalQuantity": o.get("totalQuantity"),
        "note": o.get("note_text"),
    }

def _order_products_for_order(order_id: str) -> List[Dict[str, Any]]:
    df = order_products_df
    if "orderId" in df.columns:
        df = df[df["orderId"] == order_id]
    return df.to_dict(orient="records")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _fetch_activities(customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    df = activities_df

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return records


def _fetch_notes(customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    df = notes_df

    return df.where(pd.notnull(df), None).to_dict(orient="records")


@function_tool
def activities_timeline(
    customer_id: Optional[str] = None,
    limit: int = 200,
) -> str:
    """
    Return normalized timeline entries.

    Each entry:
    - ts: timestamp of the event
    - type: activity type
    - createdBy: source (DISTRIBUTOR / BACKOFFICE / etc.)
    - source_name: best available human-readable actor name
    - activity_id: UUID or id
    - id: internal numeric/id
    """
    rows = _fetch_activities(customer_id)
    rows = sorted(rows, key=lambda r: _parse_created_at(r.get("createdAt")))[:limit]

    timeline = []
    for r in rows:
        timeline.append({
            "ts": r.get("createdAt"),
            "type": r.get("type"),
            "createdBy": r.get("createdBy"),
            "source_name": (
                r.get("createdByBackOfficeRepresentative")
                or r.get("appCustomer_name")
                or r.get("distributor_name")
            ),
            "activity_id": r.get("id"),
            "id": r.get("id"),
        })

    return timeline


@function_tool
def activities_summary(
    customer_id: Optional[str] = None,
) -> str:
    """
    Summarize activities.

    Output (JSON as text):
    - hasActivities: bool
    - total: total events
    - lastActivityAt: latest createdAt (original string)
    - byType: {type: count}
    """
    rows = _fetch_activities(customer_id)

    if not rows:
        summary = {
            "hasActivities": False,
            "total": 0,
            "lastActivityAt": None,
            "byType": {},
        }
    else:
        by_type: Dict[str, int] = {}
        last_dt: Optional[datetime] = None
        last_str: Optional[str] = None

        for r in rows:
            t = str(r.get("type") or "UNKNOWN")
            by_type[t] = by_type.get(t, 0) + 1

            created_raw = r.get("createdAt")
            dt = _parse_created_at(created_raw)

            if dt and (last_dt is None or dt > last_dt):
                last_dt = dt
                last_str = created_raw

        summary = {
            "hasActivities": True,
            "total": len(rows),
            "lastActivityAt": last_str,
            "byType": by_type,
        }

    return summary

@function_tool
def notes_list(
    customer_id: Optional[str] = None,
    limit: int = 100,
) -> str:
    """
    Return raw notes.

    Args:
    - customer_id: optional; used if notes are not pre-scoped and CSV has such column.
    - limit: max number of notes, sorted by createdAt descending.
    """
    rows = _fetch_notes(customer_id)

    rows = sorted(
        rows,
        key=lambda r: _parse_created_at(r.get("createdAt")),
        reverse=True
    )[:limit]

    return rows


@function_tool
def notes_last(
    customer_id: Optional[str] = None,
) -> str:
    """
    Return the latest single note (or null if none).
    """
    rows = _fetch_notes(customer_id)

    if not rows:
        result: Dict[str, Any] = {"hasNotes": False, "lastNote": None}
    else:
        last = max(rows, key=lambda r: _parse_created_at(r.get("createdAt")))
        result = {
            "hasNotes": True,
            "lastNote": last,
        }

    return result


@function_tool
def order_items_get(
    order_id: str,
) -> str:
    """
    Return all raw item rows for the given order_id.
    """
    items = _order_products_for_order(order_id)

    return items


@function_tool
def order_items_summary(
    order_id: str,
) -> str:
    """
    Summary includes:
    - hasItems: bool
    - totalItems: number of distinct lines
    - totalQuantity: sum of quantity
    - manufacturers: unique manufacturerName values
    - categories: unique productCategoryName values
    - amountTotals:
        - totalRawAmount: sum of totalRawAmount
        - totalAmount: sum of totalAmount
    """
    items = _order_products_for_order(order_id)

    if not items:
        summary: Dict[str, Any] = {
            "hasItems": False,
            "totalItems": 0,
            "totalQuantity": 0,
            "manufacturers": [],
            "categories": [],
            "amountTotals": {
                "totalRawAmount": 0,
                "totalAmount": 0,
            },
        }
    else:
        total_items = len(items)
        total_quantity = sum((i.get("quantity") or 0) for i in items)

        manufacturers = sorted({
            _safe_str(i.get("manufacturerName"))
            for i in items
            if i.get("manufacturerName")
        })

        categories = sorted({
            _safe_str(i.get("productCategoryName"))
            for i in items
            if i.get("productCategoryName")
        })

        total_raw = sum((i.get("totalRawAmount") or 0) for i in items)
        total_amt = sum((i.get("totalAmount") or 0) for i in items)

        summary = {
            "hasItems": True,
            "totalItems": total_items,
            "totalQuantity": total_quantity,
            "manufacturers": manufacturers,
            "categories": categories,
            "amountTotals": {
                "totalRawAmount": float(total_raw),
                "totalAmount": float(total_amt),
            },
        }

    return summary


@function_tool
def orders_list(
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Returns lightweight orders list:
    - orderId
    - customOrderId
    - createdAt
    - orderStatus
    - paymentStatus
    - deliveryStatus
    - totalAmount
    - totalQuantity
    """
    rows = _filter_by_customer(customer_id)

    if status:
        s = status.upper()
        rows = [r for r in rows if str(r.get("orderStatus", "")).upper() == s]

    rows = sorted(rows, key=lambda r: _parse_created_at(r.get("createdAt")), reverse=True)[:limit]

    result = []
    for o in rows:
        result.append({
            "orderId": o.get("id"),
            "customOrderId": o.get("customId_customId"),
            "createdAt": o.get("createdAt"),
            "orderStatus": o.get("orderStatus"),
            "paymentStatus": o.get("paymentStatus"),
            "deliveryStatus": o.get("deliveryStatus"),
            "totalAmount": o.get("totalAmount"),
            "totalQuantity": o.get("totalQuantity"),
        })

    return result


@function_tool
def order_get(
    order_id: Optional[str] = None,
    custom_order_id: Optional[str] = None,
    shopify_order_id: Optional[str] = None,
    quickbooks_order_id: Optional[str] = None,
) -> str:
    """
    Returns the full row from orders.csv as JSON.
    """
    if not any([order_id, custom_order_id, shopify_order_id, quickbooks_order_id]):
        return "You must provide at least one identifier: "

    o = _find_order(
        order_id=order_id,
        custom_order_id=custom_order_id,
        shopify_order_id=shopify_order_id,
        quickbooks_order_id=quickbooks_order_id,
    )

    if not o:
        return "Order not found."

    return o

@function_tool
def order_status(
    order_id: Optional[str] = None,
    custom_order_id: Optional[str] = None,
    shopify_order_id: Optional[str] = None,
    quickbooks_order_id: Optional[str] = None,
) -> str:
    """
    Returns:
    - orderId, external ids
    - orderStatus, paymentStatus, deliveryStatus
    - key timestamps (created/shipped/completed/canceled/etc.)
    - monetary info (totalAmount, totalQuantity)
    - note (note_text)
    """
    if not any([order_id, custom_order_id, shopify_order_id, quickbooks_order_id]):
        return "You must provide at least one identifier."

    o = _find_order(
        order_id=order_id,
        custom_order_id=custom_order_id,
        shopify_order_id=shopify_order_id,
        quickbooks_order_id=quickbooks_order_id,
    )

    if not o:
        return "Order not found."

    status = _status_view(o)

    return status
    

@function_tool
def tasks_list(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 100,
) -> str:
    """
    Returns tasks with optional filters.

    Args:
    - status: optional; filter by status (e.g. 'PENDING', 'COMPLETED').
    - priority: optional; filter by priority (e.g. 'LOW', 'MEDIUM', 'HIGH').
    - limit: max number of tasks, ordered by createdAt descending (string-based).
    """
    rows = _get_all_tasks()

    if status:
        s = status.upper()
        rows = [t for t in rows if str(t.get("status", "")).upper() == s]

    if priority:
        p = priority.upper()
        rows = [t for t in rows if str(t.get("priority", "")).upper() == p]

    rows = sorted(
        rows,
        key=lambda r: _parse_created_at(r.get("createdAt")),
        reverse=True
    )[:limit]

    result = []
    for t in rows:
        result.append({
            "id": t.get("id"),
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "dueDate": t.get("dueDate"),
            "dueTime": t.get("dueTime"),
            "assignedDistributor_name": t.get("assignedDistributor_name"),
            "representative_name": t.get("representative_name"),
            "createdAt": t.get("createdAt"),
        })

    return result


@function_tool
def task_get(
    task_id: str,
) -> str:
    """
    Returns the full row from tasks.csv for the given task_id.
    """
    task = _find_task(task_id)

    if not task:
        return "Task not found."

    return task


@function_tool
def tasks_summary() -> str:
    """
    Output:
    - hasTasks: bool
    - total: total number of tasks
    - byStatus: {status: count}
    - byPriority: {priority: count}
    - pendingCount: number of non-completed tasks
    - nextDue: best-effort string with the nearest dueDate/dueTime among pending tasks
    """
    rows = _get_all_tasks()

    if not rows:
        summary: Dict[str, Any] = {
            "hasTasks": False,
            "total": 0,
            "byStatus": {},
            "byPriority": {},
            "pendingCount": 0,
            "nextDue": None,
        }
    else:
        by_status: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        pending = []

        for t in rows:
            st = str(t.get("status") or "UNKNOWN").upper()
            pr = str(t.get("priority") or "UNKNOWN").upper()

            by_status[st] = by_status.get(st, 0) + 1
            by_priority[pr] = by_priority.get(pr, 0) + 1

            if st not in ("COMPLETED", "CLOSED", "CANCELLED", "CANCELED"):
                pending.append(t)

        next_due = None
        if pending:
            with_due = [t for t in pending if t.get("dueDate")]
            if with_due:
                with_due_sorted = sorted(
                    with_due,
                    key=lambda t: str(t.get("dueDate")) + " " + str(t.get("dueTime") or "")
                )
                nd = with_due_sorted[0]
                next_due = {
                    "id": nd.get("id"),
                    "title": nd.get("title"),
                    "dueDate": nd.get("dueDate"),
                    "dueTime": nd.get("dueTime"),
                    "priority": nd.get("priority"),
                }

        summary = {
            "hasTasks": True,
            "total": len(rows),
            "byStatus": by_status,
            "byPriority": by_priority,
            "pendingCount": len(pending),
            "nextDue": next_due,
        }

    return summary
    
@function_tool
def customer_profile(
    customer_id: Optional[str] = None,
) -> str:
    """
    Wrapper around build_customer_profile.

    - If customer_id is provided and the datasets support it, the profile
      is scoped to that customer.
    - If not provided, assumes data is already scoped to the authenticated customer.
    """
    profile = 'build_customer_profile(customer_id=customer_id)'

    return profile

#___

@function_tool
def General_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'General_statistics_tool' called ")
    data_path = f"data/{user_id}/report.md"
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

@function_tool
def General_notes_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'General_notes_statistics_tool' called ")
    data_path = f"data/{user_id}/report_notes.md"
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

@function_tool
def General_tasks_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'General_tasks_statistics_tool' called ")
    data_path = f"data/{user_id}/report_task.md"
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

@function_tool
def General_activities_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'General_activities_statistics_tool' called ")
    data_path = f"data/{user_id}/report_activities.md"
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

@function_tool
def get_top_n_orders(user_id:str, n : int, by_type:str) -> str:
    """
    Gets the top N orders from the DataFrame based on revenue or quantity.

    Args:
        user_id str: Use given user id.
        n (int): The number of top orders to return.
        by_type (str): The criteria to sort by. 
                       Must be 'revenue' or 'totalQuantity'.

    Returns:
        str: A formatted string of the top N orders.
    """
    
    # Select and copy the relevant columns
    logger2.info(f"Tool 'get_top_n_orders' called called for: {user_id}")
    dataf = pd.read_csv(os.path.join("data", user_id, "work_ord.csv"))
    try:
        relevant_cols = ['customId_customId', 'totalAmount', 'totalQuantity']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing expected column in DataFrame: {e}. Available columns: {dataf.columns.tolist()}"

    # Determine the sort column
    if by_type == 'revenue':
        sort_column = 'totalAmount'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    else:
        return "Invalid 'by_type' parameter. Please choose 'revenue' or 'totalQuantity'."

    # Sort the DataFrame and get the top N
    top_n_df = df_copy.sort_values(by=sort_column, ascending=False).head(n)

    # Fill potential NaN values in text fields for clean printing
    top_n_df_filled = top_n_df.fillna({'customId_customId': 'N/A'})

    # Format the output string
    output_strings = []
    output_strings.append(f"--- Top {n} Orders by {by_type.capitalize()} ---")
    output_strings.append("\nCustom ID -  Revenue - Total Quantity")
    output_strings.append("-" * 60) # Separator line

    for index, row in top_n_df_filled.iterrows():
        # Format totalAmount as currency and totalQuantity as an integer
        formatted_row = (
            f"{row['customId_customId']} - "
            f"${row['totalAmount']:.2f} - "
            f"{int(row['totalQuantity'])}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)


@function_tool
def get_top_n_products(user_id:str, n : int, by_type:str)-> str:
    """
    Gets the top N products from the DataFrame based on aggregated 
    revenue or quantity.

    Args:
        user_id str: Use given user id.
        n (int): The number of top customers to return.
        by_type (str): The criteria to sort by. 
                       Must be 'revenue', 'totalQuantity', or 'orderCount'.

    Returns:
        str: A formatted string of the top N customers.
    """
    
    # Select and copy the relevant columns
    logger2.info(f"Tool 'get_top_n_products' called called for: {user_id}")
    dataf = pd.read_csv(os.path.join("data", user_id, "work_prod.csv"))
    dataf['product_variant'] = dataf['name'].astype(str) + ' - ' + dataf['sku'].astype(str)
    # Select and copy the relevant columns
    try:
        relevant_cols = ['product_variant', 'totalAmount', 'quantity', 'orderId']
        df_copy = dataf[relevant_cols].copy()
    except KeyError as e:
        return f"Error: Missing expected column in DataFrame: {e}. Available columns: {dataf.columns.tolist()}"

    # Aggregate data by product_variant
    product_agg = df_copy.groupby('product_variant').agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('quantity', 'sum'),
        orderCount=('orderId', 'nunique'), # Count distinct orders
    ).reset_index()

    # Calculate average revenue per order
    product_agg['avgRevenuePerOrder'] = product_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )
    
    # Calculate average quantity per order
    product_agg['avgQuantityPerOrder'] = product_agg.apply(
        lambda row: row['totalQuantity'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )


    # Determine the sort column
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return ("Invalid 'by_type' parameter. Please choose 'revenue', "
                "'totalQuantity', or 'orderCount'.")

    # Sort the DataFrame and get the top N
    top_n_df = product_agg.sort_values(by=sort_column, ascending=False).head(n)

    # Fill potential NaN values in text fields for clean printing
    top_n_df_filled = top_n_df.fillna({'product_variant': 'N/A'})

    # Format the output string
    output_strings = []
    output_strings.append(f"--- Top {n} Products by {by_type.capitalize()} ---")
    output_strings.append("\nProduct - Total Revenue - Total Quantity - Order Count -  Avg. Qty/Order")
    output_strings.append("-" * 100) # Separator line

    for index, row in top_n_df_filled.iterrows():
        formatted_row = (
            f"{row['product_variant']} - "
            f"${row['totalRevenue']:,.2f} - "
            f"{int(row['totalQuantity'])} - "
            f"{row['orderCount']} - "
            f"{row['avgQuantityPerOrder']:.1f}"
        )
        output_strings.append(formatted_row)

    return '\n'.join(output_strings)

@function_tool
def get_order_details(order_custom_id:int, user_id:str):
    """
    Gets complete information about an order by its customId_customId,
    using data from both DataFrames.

    Args:
        order_custom_id (int or str): The unique 'customId_customId' of the order.
        user_id (str): Use given user id.

    Returns:
        str: A formatted string with the complete order information.
    """
    
    output_strings = []
    # Select and copy the relevant columns
    try:
        logger2.info(f"Tool 'get_order_details' called called for: {user_id}")
        df_orders = pd.read_csv(os.path.join("data", user_id, "work_ord.csv"))
        df_products = pd.read_csv(os.path.join("data", user_id, "work_prod.csv"))
        df_orders['customId_customId'] = pd.to_numeric(df_orders['customId_customId'], errors='coerce')

    except Exception as e:
        print(f"Error loading files: {e}")
    df_products['product_variant'] = df_products['name'].astype(str) + ' - ' + df_products['sku'].astype(str)
    if 'customId_customId' not in df_orders.columns:
        print("Error: The 'customId_customId' column was not found in oorders.csv.")
        df_orders = None # Set to None to prevent errors
    # 1. Find the order in df_orders
    try:
        # Convert ID to a numeric type for comparison
        numeric_custom_id = int(order_custom_id)
    except ValueError:
        return f"Error: 'order_custom_id' must be numeric. Received: {order_custom_id}"

    # Filter df_orders
    order_details = df_orders[df_orders['customId_customId'] == numeric_custom_id]
    
    if order_details.empty:
        return f"--- Order with Custom ID: {order_custom_id} not found ---"
        
    # Get the first (and likely only) row
    order_row = order_details.iloc[0]
    
    # Get the internal ID to link with products
    internal_order_id = order_row['id']

    # --- 2. Format Main Information ---
    output_strings.append(f"--- Order Details (Custom ID: {numeric_custom_id}) ---")
    
    try:
        # Try to get just the date
        order_date = pd.to_datetime(order_row.get('createdAt')).strftime('%Y-%m-%d')
    except:
        order_date = order_row.get('createdAt', 'N/A')
        
    output_strings.append(f"Date:           {order_date}")
    output_strings.append(f"Status:         {order_row.get('orderStatus', 'N/A')}")
    output_strings.append(f"Payment:        {order_row.get('paymentStatus', 'N/A')}")
    output_strings.append(f"Delivery:       {order_row.get('deliveryStatus', 'N/A')}")
    
    # --- 3. Financial Information ---
    output_strings.append("\n--- Financials ---")
    output_strings.append(f"Subtotal (excl. delivery): ${order_row.get('totalAmountWithoutDelivery', 0):.2f}")
    output_strings.append(f"Discount:              ${order_row.get('totalDiscountValue', 0):.2f}")
    output_strings.append(f"Delivery Fee:          ${order_row.get('deliveryFee', 0):.2f}")
    output_strings.append(f"GRAND TOTAL:           ${order_row.get('totalAmount', 0):.2f}")
    output_strings.append(f"Total Units:           {int(order_row.get('totalQuantity', 0))}")

    # --- 4. Find products in df_products ---
    order_products = df_products[df_products['orderId'] == internal_order_id]
    
    output_strings.append(f"\n--- Products in Order ({len(order_products)}) ---")
    
    if order_products.empty:
        output_strings.append("Products for this order were not found.")
    else:
        # Products table header
        output_strings.append(f"{'Product':<40} | {'Qty':<5} | {'Price':<10} | {'Total':<10}")
        output_strings.append("-" * 70)
        
        for index, prod_row in order_products.iterrows():
            product_name = prod_row.get('product_variant', 'N/A')
            # Truncate the name if it's too long
            if len(product_name) > 38:
                product_name = product_name[:35] + "..."
            
            quantity = int(prod_row.get('quantity', 0))
            price = prod_row.get('price', 0)
            total_line_amount = prod_row.get('totalAmount', 0)
            
            output_strings.append(f"{product_name:<40} | {quantity:<5} | ${price:<9.2f} | ${total_line_amount:<9.2f}")
            
    return '\n'.join(output_strings)


# --- Function 1: Get Product List ---
@function_tool
def get_product_catalog(user_id:str)-> str:
    """
    Parses the products DataFrame and returns a dictionary
    of unique product attributes (variants, names, SKUs).
    
    Args:
        user_id (str): Use given user id.

    Returns:
        dict: A dictionary containing lists of unique
              product_variants, names, and skus.
              Returns None if the DataFrame is invalid.
    """
    try:
        logger2.info(f"Tool 'get_product_catalog' called called for: {user_id}")
        df_products = pd.read_csv(os.path.join("data", user_id, "work_prod.csv"))
    
        # Clean the 'combined﻿id' column name if it exists
        if 'combined\ufeffid' in df_products.columns:
            df_products.rename(columns={'combined\ufeffid': 'combined_id'}, inplace=True)
        df_products['product_variant'] = df_products['name'].astype(str) + ' - ' + df_products['sku'].astype(str)
    except Exception as e:
        print(f"Error loading pproducts.csv: {e}")

    if df_products is None:
        return None
        
    try:
        catalog = {
            "all_product_variants": df_products['product_variant'].unique().tolist(),
            "all_product_names": df_products['name'].unique().tolist(),
            "all_skus": df_products['sku'].unique().tolist(),
            # --- NEWLY ADDED ---
            "all_categories": df_products['productCategoryName'].dropna().unique().tolist()
        }
        
        # Sort lists for easier reading
        for key in catalog:
            catalog[key].sort()
            
        return catalog
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        return None
    except Exception as e:
        print(f"An error occurred in get_product_catalog: {e}")
        return None

# --- Function 2: Get Product Details ---
def _generate_product_report(df_to_report, report_title):
    """
    Helper function to generate a detailed report from a DataFrame.
    """
    output_strings = [report_title]

    # --- 1. Static Info ---
    output_strings.append("\n--- Matched Attributes ---")
    output_strings.append(f"Base Names:        {', '.join(df_to_report['name'].unique())}")
    output_strings.append(f"SKUs:              {', '.join(df_to_report['sku'].unique())}")
    output_strings.append(f"Manufacturers:     {', '.join(df_to_report['manufacturerName'].unique())}")
    
    valid_categories = df_to_report['productCategoryName'].dropna().unique()
    if len(valid_categories) > 0:
        output_strings.append(f"Categories:        {', '.join(valid_categories)}")
    else:
        output_strings.append("Categories:        N/A")
        
    items_per_case_vals = df_to_report['itemsPerCase'].unique()
    output_strings.append(f"Items per Case:    {', '.join(map(str, items_per_case_vals))}")

    # --- 2. Aggregated Sales Info ---
    output_strings.append("\n--- Lifetime Sales Summary (for this group) ---")
    total_revenue = df_to_report['totalAmount'].sum()
    total_quantity = df_to_report['quantity'].sum()
    total_orders = df_to_report['orderId'].nunique()
    total_customers = df_to_report['customer_name'].nunique()
    
    output_strings.append(f"Total Revenue:     ${total_revenue:,.2f}")
    output_strings.append(f"Total Units Sold:  {total_quantity:,}")
    output_strings.append(f"Total Orders:      {total_orders:,}")
    output_strings.append(f"Unique Customers:  {total_customers:,}")

    if total_quantity > 0:
        avg_price_per_unit = total_revenue / total_quantity
        output_strings.append(f"Avg. Price / Unit: ${avg_price_per_unit:.2f}")
    
    if total_orders > 0:
        avg_revenue_per_order = total_revenue / total_orders
        output_strings.append(f"Avg. Revenue / Order: ${avg_revenue_per_order:,.2f}")

    # --- 3. Top Customers for this group ---
    output_strings.append(f"\n--- Top 5 Customers (for this group) ---")
    customer_sales = df_to_report.groupby('customer_name')['totalAmount'].sum().sort_values(ascending=False).head(5)
    
    if customer_sales.empty:
        output_strings.append("No customer sales data available.")
    else:
        for customer, revenue in customer_sales.items():
            output_strings.append(f"{customer:<30} | ${revenue:,.2f}")

    # --- 4. Order Dates ---
    try:
        first_order = pd.to_datetime(df_to_report['createdAt']).min().strftime('%Y-%m-%d')
        last_order = pd.to_datetime(df_to_report['createdAt']).max().strftime('%Y-%m-%d')
        output_strings.append("\n--- Order History (for this group) ---")
        output_strings.append(f"First Order Date:  {first_order}")
        output_strings.append(f"Last Order Date:   {last_order}")
    except Exception:
        output_strings.append("\n--- Order History (for this group) ---")
        output_strings.append("Could not parse order dates.")

    return '\n'.join(output_strings)

@function_tool
def get_product_details(user_id:str, name:str=None, sku:str=None, category:str=None)-> str:
    """
    Call this tool after get_product_catalog, using the validated identifiers.
    Purpose: To provide the user with a detailed report based on their specific query.

    Example Scenarios for Step 2:
    Case 1 (Name Only): User asks for "all Mars products."
    Call: get_product_details(user_id, name='Mars')
    Case 2 (SKU Only): User asks about "SKU 12345."
    Call: get_product_details(user_id, sku='12345')
    Case 3 (Category Only): User asks for "everything in the Sodas category."
    Call: get_product_details(user_id, category='Sodas')
    Case 4 (Name + SKU): User asks for "Coke Original."
    Call: get_product_details(user_id, name='Coca Cola', sku='Original')
    Case 5 (Name + Category): User asks for "Coke products in the Sodas category."
    Call: get_product_details(user_id, name='Coca Cola', category='Sodas')

    Args:
        user_id (str): Use given user id.
        name (str, optional): A string to match in the 'name' column.
        sku (str, optional): A string to match in the 'sku' column.
        category (str, optional): A string to match in the 
                                  'productCategoryName' column.
    Returns:
        str: A formatted string with full product info.
    """
    logger2.info(f"Tool 'get_product_details' called called for: {user_id}")
    df_products = pd.read_csv(os.path.join("data", user_id, "pproducts.csv"))
    if df_products is None:
        return "Error: The products DataFrame is None. Please check file loading."
        
    # --- UPDATED ---
    if name is None and sku is None and category is None:
        return "Error: Please provide at least one filter (name, sku, or category)."

    filtered_df = df_products.copy()
    filters_applied = []

    # Apply filters
    if name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(name, case=False, na=False)]
        filters_applied.append(f"Name containing '{name}'")
        
    if sku:
        filtered_df = filtered_df[filtered_df['sku'].str.contains(sku, case=False, na=False)]
        filters_applied.append(f"SKU containing '{sku}'")

    # --- UPDATED ---
    if category:
        # Use str.contains for partial, case-insensitive matching
        # na=False ensures that rows with NaN categories are skipped
        filtered_df = filtered_df[filtered_df['productCategoryName'].str.contains(category, case=False, na=False)]
        filters_applied.append(f"Category containing '{category}'")
    # --- END UPDATE ---

    # Check for results
    if filtered_df.empty:
        return f"--- No products found matching: {', '.join(filters_applied)} ---"

    # --- Generate Report ---
    report_title = f"--- Product Report For: {', '.join(filters_applied)} ---"
    report_string = _generate_product_report(filtered_df, report_title)

    # --- "See Also" Logic ---
    variants_found = filtered_df['product_variant'].unique()
    
    if len(variants_found) > 1:
        report_string += "\n\n--- Specific Variants Found ---"
        report_string += "\nYour search returned multiple variants. You may be interested in a detailed report on one of these:"
        
        limit = 10
        for variant in variants_found[:limit]:
            report_string += f"\n- {variant}"
            
        if len(variants_found) > limit:
            report_string += f"\n...and {len(variants_found) - limit} more."

    return report_string


@function_tool
def get_orders_by_customer_id(user_id:str, customer_id:str)-> str:
    """
    Returns a DataFrame in md format with specific order details for a given customer_id - use get_customers tool before.
    """
    logger2.info(f"Tool 'get_orders_by_customer' called called for: {user_id} and {customer_id}")
    # Filter DataFrame by customer_id
    dataframe = pd.read_csv(os.path.join("data", user_id, "oorders.csv"))
    customer_orders_df = dataframe[dataframe['customerId'] == customer_id].copy()
    
    # Define columns requested
    requested_columns = [
        'id',
        'customId_customId',
        'totalOrderDiscountAmount',
        'totalOrderDiscountType',
        'createdAt',
        'orderStatus',
        'deliveryStatus',
        'paymentStatus',
        'totalAmount',
        'totalQuantity'
    ]
    
    # Select and rename columns
    customer_orders_summary = customer_orders_df[requested_columns]
    
    # Rename columns as per user request
    column_rename_map = {
        'id': 'order_id',
        'customId_customId': 'custom_order_id'
    }
    customer_orders_summary = customer_orders_summary.rename(columns=column_rename_map)
    customer_orders_summary_md = customer_orders_summary.to_markdown(index=False)
    return customer_orders_summary_md


async def create_Ask_ai_single_c_agent(USER_ID:str) -> Tuple[Agent, AdvancedSQLiteSession]:
    """Initializes a new Inventory agent and session."""

    try:
        from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_Ask_ai_solo

        session_db = AdvancedSQLiteSession(
            session_id=USER_ID,
            create_tables=True,
            db_path=f"data/{USER_ID}/conversations.db",
            logger=logger2
        )
    except Exception as e:
        logger2.error(f"error creating session: {e}")

    try:
        instructions = await prompt_agent_Ask_ai_solo(USER_ID)
        agent = Agent(
            name="Warehouse_Inventory_Assistant",
            instructions=instructions,
            model=llm_model,
            tools=[
            General_statistics_tool,
            get_top_n_orders,
            get_top_n_products,
            get_order_details,
            get_product_catalog,
            get_product_details,
            get_orders_by_customer_id]

        )
        session = session_db

    except Exception as e:
        logger2.error(f"error creating agent: {e}")
        agent, session = None
    return agent, session