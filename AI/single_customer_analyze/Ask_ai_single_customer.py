from typing import List, AsyncGenerator, Tuple, Any

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
logger1 = get_logger("logger1", "project_log.log", False)

llm_model = OpenAIResponsesModel(model='gpt-4.1', openai_client=AsyncOpenAI()) 

import os
import pandas as pd

def get_all_data(customer_id):
    DATA_DIR = os.path.join('data', customer_id)
    
    # Helper function to safely load CSV
    def safe_read_csv(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return None
    
    activities_path = os.path.join(DATA_DIR, 'activities', 'activities.csv')
    activities_df = safe_read_csv(activities_path)
    
    notes_path = os.path.join(DATA_DIR, 'activities', 'notes.csv')
    notes_df = safe_read_csv(notes_path)
    
    order_products_path = os.path.join(DATA_DIR, 'orders', 'order_products.csv')
    order_products_df = safe_read_csv(order_products_path)
    if order_products_df is not None:
        order_products_df = order_products_df.where(pd.notnull(order_products_df), None)
    
    orders_path = os.path.join(DATA_DIR, 'orders', 'orders.csv')
    orders_df = safe_read_csv(orders_path)
    if orders_df is not None:
        orders_df = orders_df.where(pd.notnull(orders_df), None)
    
    tasks_path = os.path.join(DATA_DIR, 'activities', 'tasks.csv')
    tasks_df = safe_read_csv(tasks_path)
    if tasks_df is not None:
        tasks_df = tasks_df.where(pd.notnull(tasks_df), None)
    

    return activities_df, notes_df, order_products_df, orders_df, tasks_df

activities_df, notes_df, order_products_df, orders_df, tasks_df = get_all_data('b17f86d1-9bf9-4d75-aedd-d25b0ddc9562')

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
from pathlib import Path

@function_tool
def General_statistics_tool(user_id: str) -> str:
    """
    Retrieves user data from 'report.md' and 'reorder.md'.
    Uses fallback encodings to prevent crashes on bad characters.
    """
    logger1.info(f"Tool 'General_statistics_tool' called for user: {user_id}")
    
    # 1. Setup Pathlib paths
    base_dir = Path("data") / user_id
    files_to_read = [
        base_dir / "report.md",
        base_dir / "reorder.md"
    ]
    
    encodings_to_try = ['utf-8', 'cp1252', 'utf-16', 'latin-1']
    collected_content = []

    # 2. Iterate through both files
    for file_path in files_to_read:
        if not file_path.exists():
            logger1.warning(f"File not found: {file_path}")
            continue

        file_content = None
        
        # 3. Attempt to read with multiple encodings
        for enc in encodings_to_try:
            try:
                with file_path.open("r", encoding=enc) as f:
                    file_content = f.read()
                logger1.info(f"Successfully read {file_path.name} using {enc}")
                break # Stop trying encodings if one works
            except UnicodeDecodeError:
                continue # Try next encoding
            except Exception as e:
                logger1.error(f"Error reading {file_path}: {e}")
                break

        if file_content:
            # Add a header so the AI knows which file this text came from
            collected_content.append(f"--- Content from {file_path.name} ---\n{file_content}")
        else:
            logger1.error(f"Failed to decode {file_path.name} with any supported encoding.")

    # 4. Return combined results
    if not collected_content:
        return "Error: No data found or readable in report/reorder files."
        
    return "\n\n".join(collected_content)

@function_tool
def General_notes_statistics_tool(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""
    
    logger1.info(f"Tool 'General_notes_statistics_tool' called")
    data_path = f"data/{user_id}/report_notes.md"
    
    # List of encodings to try, in order of preference
    # 1. utf-8: The standard.
    # 2. cp1252: The default for Windows (likely the culprit for 0xef).
    # 3. utf-16: Common if the file was created by PowerShell or Windows Notepad.
    # 4. latin-1: The "catch-all" - it rarely raises errors but might produce symbols.
    encodings_to_try = ['utf-8', 'cp1252', 'utf-16', 'latin-1']

    try:
        # Check if file exists first to avoid looping unnecessarily
        with open(data_path, "rb") as f:
            pass
    except FileNotFoundError:
        logger1.error(f"Statistics file not found at: {data_path}")
        return "Error: Statistics file not found."

    # Start the loop
    for enc in encodings_to_try:
        try:
            logger1.info(f"Attempting to read {data_path} with encoding: {enc}")
            with open(data_path, "r", encoding=enc) as f:
                statistics = f.read()
            
            # If we reach this line, it worked!
            logger1.info(f"Successfully read statistics using encoding: {enc}")
            return statistics
            
        except UnicodeDecodeError:
            # If this encoding fails, log it and loop to the next one
            logger1.warning(f"Failed to read with {enc}, trying next...")
            continue
        except Exception as e:
            # Catch other errors (like permissions) immediately
            logger1.error(f"Unexpected error reading {data_path}: {e}")
            return f"Error: {e}"

    # If the loop finishes without returning, nothing worked
    logger1.error(f"All encoding attempts failed for {data_path}")
    return "Error: Unable to decode file with standard encodings."

@function_tool
def General_tasks_statistics_tool(user_id: str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger1.info(f"Tool 'General_tasks_statistics_tool' called")
    
    # 1. Setup Path
    file_path = Path("data") / user_id / "report_task.md"
    
    # 2. Check existence
    if not file_path.exists():
        logger1.error(f"Statistics file not found at: {file_path}")
        return "Error: Statistics file not found."

    # 3. Encoding Loop
    encodings_to_try = ['utf-8', 'cp1252', 'utf-16', 'latin-1']
    
    for enc in encodings_to_try:
        try:
            with file_path.open("r", encoding=enc) as f:
                statistics = f.read()
            logger1.info(f"Successfully read statistics from {file_path} using {enc}")
            return statistics
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger1.error(f"Error reading {file_path}: {e}")
            return f"Error: {e}"

    logger1.error(f"Failed to decode {file_path} with any standard encoding.")
    return "Error: Unable to decode file."

@function_tool
def General_activities_statistics_tool(user_id: str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger1.info(f"Tool 'General_activities_statistics_tool' called")
    from main import get_exported_data
    from AI.single_customer_analyze.Activities_analytics import _calculate_metrics_sync, convert_to_datetime
    # 1. Setup Path
    file_path = Path("data") / user_id / "report_activities.md"
    
    # 2. Check existence
    if not file_path.exists():
        try:
            try:
                activities = get_exported_data(user_id, "activities")
            except Exception as e:
                logger1.error(f"Error in get_exported_data activities: {e}")


            # Build the directory path and file name for non-orders entities
            dir_path = os.path.join("data", user_id, 'activities')
            os.makedirs(dir_path, exist_ok=True)


            file_path_activities = os.path.join(dir_path, "activities.csv")

            logger1.info(f"Saving report for customer '{user_id}', entity 'activities' at {file_path_activities}")
            with open(file_path_activities, "wb") as f:
                f.write(activities)
            df = pd.read_csv(file_path_activities)
            df = convert_to_datetime(df, ["createdAt"])
            df = convert_to_datetime(df, ["updatedAt"])
            statistics = _calculate_metrics_sync(df)
        except Exception as e:
            logger1.error(f'General_activities_statistics_tool activities question problem: {e}')
            statistics = 'Can not calculate it, try again later'
        return statistics


    # 3. Encoding Loop
    encodings_to_try = ['utf-8', 'cp1252', 'utf-16', 'latin-1']
    
    for enc in encodings_to_try:
        try:
            with file_path.open("r", encoding=enc) as f:
                statistics = f.read()
            logger1.info(f"Successfully read statistics from {file_path} using {enc}")
            return statistics
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger1.error(f"Error reading {file_path}: {e}")
            return f"Error: {e}"

    logger1.error(f"Failed to decode {file_path} with any standard encoding.")
    return "Error: Unable to decode file."

@function_tool
def get_top_n_orders(
    user_id: str, 
    n: int, 
    by_type: str, 
    sort_order: str = 'desc', 
    start_date: str = None, 
    end_date: str = None,
    status_filter: str = None
) -> str:
    """
    Gets the top (or bottom) N orders based on revenue or quantity, 
    optionally filtered by a start date and order status.

    Parameters:
    - user_id: The user's ID.
    - n: Number of records to return.
    - by_type: 'revenue' or 'totalQuantity'.
    - sort_order: 'desc' (Highest first) or 'asc' (Lowest first).
    - start_date: Filter orders created ON or AFTER this date. Format: 'YYYY-MM-DD'.
    - end_date: Filter data from start day to this date (YYYY-MM-DD).
    - status_filter: Filter by specific order status. 
      VALID VALUES: orderStatus - ['COMPLETED', 'PENDING' or None].
    """
    logger1.info(f"Tool 'get_top_n_orders' called for: {user_id} order: {sort_order},     by_type: {by_type}, start_date: {start_date},  status_filter: {status_filter}, end_date: {end_date}")
    # 1. Path Setup
    csv_path = Path("data") / user_id / "oorders.csv"
    if not csv_path.exists():
        return f"Error: File not found."

    try:
        # Load specific columns including Date and Statuses
        # Adjusted to match your CSV structure
        relevant_cols = [
            'customId_customId', 'customer_name', 'totalAmount', 
            'totalQuantity', 'createdAt', 'orderStatus' 
        ]
        dataf = pd.read_csv(csv_path, usecols=relevant_cols)
        
        # Convert Date Column (Handling timezone like in your file: 2025-04-14 15:21:27+00:00)
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')

    except Exception as e:
        return f"Error processing CSV: {e}"

    # 2. Filtering Logic
    df_filtered = dataf.copy()

    # A) Time Filter (Start Date -> Now)
    if start_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            start_dt = pd.to_datetime(start_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_filtered = df_filtered[df_filtered['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_filtered = df_filtered[df_filtered['createdAt'] <= end_dt]
    
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    # B) Status Filter (Strict Matching)
    if status_filter:
        # Normalize to upper case to match CSV content
        s_filter = status_filter.upper()
        # Check if such status exists in the filtered data to avoid returning empty list silently
        if not df_filtered['orderStatus'].str.contains(s_filter, case=False, na=False).any():
             return f"Warning: No orders found with status '{status_filter}'."
        
        df_filtered = df_filtered[df_filtered['orderStatus'].str.upper() == s_filter]

    if df_filtered.empty:
        return "No orders found matching these criteria."

    # 3. Sort Logic (Standard)
    sort_col = 'totalAmount' if by_type == 'revenue' else 'totalQuantity'
    is_ascending = (sort_order == 'asc')
    
    top_n_df = df_filtered.sort_values(by=sort_col, ascending=is_ascending).head(n)

    # 4. Output Formatting
    output = [f"Found {len(top_n_df)} orders (Sorted by {by_type}, {sort_order}):"]
    for _, row in top_n_df.iterrows():
        # Clean date for display (removing time part for readability)
        d_str = row['createdAt'].strftime('%Y-%m-%d') if pd.notnull(row['createdAt']) else "N/A"
        output.append(
            f"ID: {row['customId_customId']} | Date: {d_str} | "
            f"Customer: {row['customer_name']} | ${row['totalAmount']:.2f}"
        )

    return '\n'.join(output)


@function_tool
def get_top_n_products(
    user_id: str, 
    n: int, 
    by_type: str, 
    sort_order: str = 'desc', 
    start_date: str = None,
    end_date: str = None,
    group_by: str = 'variant'
) -> str:
    """
    Gets top N products, categories, or manufacturers based on revenue/quantity.
    
    Parameters:
    - user_id: User ID.
    - n: Number of items to return.
    - by_type: 'revenue', 'totalQuantity', 'orderCount'.
    - sort_order: 'desc' (Best) or 'asc' (Worst).
    - start_date: Filter data from this date (YYYY-MM-DD).
    - end_date: Filter data from start day to this date (YYYY-MM-DD).
    - group_by: Aggregation level. Options: 
        'variant' (Specific Product), 
        'category' (Product Category), 
        'manufacturer' (Brand/Manufacturer).
    """
    logger1.info(f"Tool 'get_top_n_products' called for: {user_id} order: {sort_order},     by_type: {by_type}, start_date: {start_date},  group_by: {group_by}, end_date: {end_date}")
    
    # 1. Path Setup
    csv_path = Path("data") / user_id / "pproducts.csv"
    if not csv_path.exists():
        return f"Error: File not found."

    try:
        dataf = pd.read_csv(csv_path)
        # Ensure date column is datetime
        dataf['createdAt'] = pd.to_datetime(dataf['createdAt'], errors='coerce')
        
        relevant_cols = [
            'product_variant', 'productCategoryName', 'manufacturerName',
            'totalAmount', 'quantity', 'orderId', 'customer_name', 'createdAt'
        ]
        # Check if columns exist (dynamic check because CSVs vary)
        existing_cols = [c for c in relevant_cols if c in dataf.columns]
        df_copy = dataf[existing_cols].copy()
        
    except Exception as e:
        return f"Error reading CSV: {e}"

    # 2. Time Filter
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            df_copy = df_copy[df_copy['createdAt'] >= start_dt]
        except Exception:
            return "Error: Invalid start_date format. Use 'YYYY-MM-DD'."

    if end_date:
        try:
            # Convert input 'YYYY-MM-DD' to datetime compatible with the dataframe
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') # Assuming input is UTC or making it aware
            # Filter: Date in row must be >= start_date
            df_copy = df_copy[df_copy['createdAt'] <= end_dt]
    
        except Exception:
            return "Error: Invalid end_date format. Use 'YYYY-MM-DD'."

    if df_copy.empty:
        return "No product data found for this period."


    # 3. Determine Grouping Column
    if group_by == 'variant':
        group_col = 'product_variant'
        label = "Product Variant"
    elif group_by == 'category':
        group_col = 'productCategoryName'
        label = "Category"
    elif group_by == 'manufacturer':
        group_col = 'manufacturerName'
        label = "Manufacturer"
    else:
        return "Invalid 'group_by'. Use 'variant', 'category', or 'manufacturer'."

    # Safety check if column exists
    if group_col not in df_copy.columns:
        return f"Error: Column for {group_by} not found in data."

    # 4. Aggregation
    # Fill N/A in group column to avoid losing data
    df_copy[group_col] = df_copy[group_col].fillna('Unknown')

    product_agg = df_copy.groupby(group_col).agg(
        totalRevenue=('totalAmount', 'sum'),
        totalQuantity=('quantity', 'sum'),
        orderCount=('orderId', 'nunique'),
        customerCount=('customer_name', 'nunique')
    ).reset_index()

    # Calculate Metrics
    product_agg['avgRevenuePerOrder'] = product_agg.apply(
        lambda row: row['totalRevenue'] / row['orderCount'] if row['orderCount'] > 0 else 0,
        axis=1
    )

    # 5. Sort Logic
    if by_type == 'revenue':
        sort_column = 'totalRevenue'
    elif by_type == 'totalQuantity':
        sort_column = 'totalQuantity'
    elif by_type == 'orderCount':
        sort_column = 'orderCount'
    else:
        return "Invalid 'by_type'."

    is_ascending = (sort_order == 'asc')
    top_n_df = product_agg.sort_values(by=sort_column, ascending=is_ascending).head(n)

    # 6. Formatting
    period_info = f" (Since {start_date})" if start_date else " (All Time)"
    direction_label = "Bottom" if sort_order == 'asc' else "Top"
    
    output_strings = [
        f"--- {direction_label} {n} {label}s by {by_type.capitalize()}{period_info} ---",
        f"{label} | Revenue | Qty | Orders | Customers | Avg Rev/Order",
        "-" * 100
    ]

    for _, row in top_n_df.iterrows():
        formatted_row = (
            f"{row[group_col]} | "
            f"${row['totalRevenue']:,.2f} | "
            f"{int(row['totalQuantity'])} | "
            f"{row['orderCount']} | "
            f"{row['customerCount']} | "
            f"${row['avgRevenuePerOrder']:.2f}"
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
        logger1.info(f"Tool 'get_order_details' called called for: {user_id}")
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
    output_strings.append("\n--- Financial ---")
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
        logger1.info(f"Tool 'get_product_catalog' called called for: {user_id}")
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

    
    output_strings.append(f"Total Revenue:     ${total_revenue:,.2f}")
    output_strings.append(f"Total Units Sold:  {total_quantity:,}")
    output_strings.append(f"Total Orders:      {total_orders:,}")


    if total_quantity > 0:
        avg_price_per_unit = total_revenue / total_quantity
        output_strings.append(f"Avg. Price / Unit: ${avg_price_per_unit:.2f}")
    
    if total_orders > 0:
        avg_revenue_per_order = total_revenue / total_orders
        output_strings.append(f"Avg. Revenue / Order: ${avg_revenue_per_order:,.2f}")


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
    logger1.info(f"Tool 'get_product_details' called called for: {user_id}")
    df_products = pd.read_csv(os.path.join("data", user_id, "work_prod.csv"))
    if df_products is None:
        return "Error: The products DataFrame is None. Please check file loading."
    df_products['product_variant'] = df_products['name'].astype(str) + ' - ' + df_products['sku'].astype(str)    
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
    logger1.info(f"Tool 'get_orders_by_customer' called called for: {user_id} and {customer_id}")
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


import os
import numpy as np
import faiss
from openai import OpenAI
from agents import function_tool  # The specific decorator from the SDK

# Initialize standard OpenAI client for embeddings
client = OpenAI()
class SimpleVectorStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.is_initialized = False

    def _cosine_similarity(self, vec_a, matrix_b):
        """Calculates cosine similarity between vector A and all vectors in Matrix B."""
        # Normalize vector A
        norm_a = np.linalg.norm(vec_a)
        if norm_a == 0: return np.zeros(len(matrix_b))
        vec_a_norm = vec_a / norm_a

        # Normalize Matrix B (all chunks)
        norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
        matrix_b_norm = np.divide(matrix_b, norm_b, where=norm_b!=0)

        # Dot product
        return np.dot(matrix_b_norm, vec_a_norm)

    def load_and_index(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print("Loading FAQ file...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split by double newline (paragraphs)
        self.chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        if not self.chunks:
            print("Warning: No chunks found in file.")
            return

        print(f"Embedding {len(self.chunks)} entries...")
        
        # Get embeddings in one batch for speed
        response = client.embeddings.create(
            input=self.chunks,
            model="text-embedding-3-small"
        )
        
        # Store as numpy array
        self.embeddings = np.array([d.embedding for d in response.data]).astype('float32')
        self.is_initialized = True
        print("Indexing complete.")

    def search(self, query: str, top_k: int = 2):
        if not self.is_initialized:
            self.load_and_index()
            
        # Embed query
        query_embedding = client.embeddings.create(
            input=[query], 
            model="text-embedding-3-small"
        ).data[0].embedding
        
        query_vec = np.array(query_embedding).astype('float32')
        
        # Calculate similarities
        scores = self._cosine_similarity(query_vec, self.embeddings)
        
        # Get top_k indices (sorted high to low)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Optional: Filter low relevance (e.g., score < 0.3)
            if scores[idx] > 0.3:
                results.append(self.chunks[idx])
                
        return "\n---\n".join(results) if results else "No relevant info found."

FAQ_FILE_PATH = Path("AI/group_customer_analyze/Agents_rules/QA_SD2.txt")
faq_engine = SimpleVectorStore(FAQ_FILE_PATH)

@function_tool
def look_up_faq(question: str) -> str:
    """
    Searches the FAQ (Frequently Asked Questions) text file 
    to find answers to user questions about policies or features.

    Args:
        question: The specific question or topic the user is asking about.
    """
    logger1.info(f"Tool 'look_up_faq' called for: {question}")
    try:
        return faq_engine.search(question)
            
    except Exception as e:
        return f"Error retrieving FAQ: {str(e)}"

async def create_Ask_ai_single_c_agent(USER_ID:str) -> Tuple[Agent, AdvancedSQLiteSession]:
    """Initializes a new Inventory agent and session."""

    try:
        from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_Ask_ai_solo

        session_db = AdvancedSQLiteSession(
            session_id=USER_ID,
            create_tables=True,
            db_path=f"data/{USER_ID}/conversations.db",
            logger=logger1
        )
    except Exception as e:
        logger1.error(f"error creating session: {e}")

    try:
        instructions = await prompt_agent_Ask_ai_solo(USER_ID)
        agent = Agent(
            name="Warehouse_Inventory_Assistant",
            instructions=instructions,
            model=llm_model,
            tools=[
            General_statistics_tool,
            General_notes_statistics_tool,
            General_tasks_statistics_tool,
            General_activities_statistics_tool,
            get_top_n_orders,
            get_top_n_products,
            get_order_details,
            get_product_catalog,
            get_product_details,
            get_orders_by_customer_id,
            look_up_faq]

        )
        session = session_db

    except Exception as e:
        logger1.error(f"error creating agent: {e}")
        agent, session = None
    return agent, session