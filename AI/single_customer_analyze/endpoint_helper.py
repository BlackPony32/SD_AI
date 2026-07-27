
import asyncio
import json
import logging
import os
from pathlib import Path

import aiofiles
import requests


from AI.single_customer_analyze.Activities_AI import process_ai_activities_request
from AI.single_customer_analyze.Activities_analytics import analyze_activities
from AI.single_customer_analyze.Notes_analytics import notes_report
from AI.single_customer_analyze.Order_analytics import generate_sales_report
from AI.single_customer_analyze.Tasks_analytics import tasks_report

logger1 = logging.getLogger("reports")

DATA_ROOT = Path("data")

ALLOWED_ENTITIES = ["orders", "order_products", "customer", "notes", "tasks", "activities"]


class ExportError(Exception):
    """A required export from the source API could not be fetched.
    Always fatal for the caller — never swallow this and continue."""


def _get_error_detail(resp: requests.Response) -> str:
    try:
        err_json = resp.json()
        for key in ("error", "message", "detail"):
            if key in err_json:
                return f"{key}: {err_json[key]}"
        return str(err_json)
    except ValueError:
        return resp.text or "<no response body>"


def _get_exported_data_sync(customer_id: str, entity: str) -> bytes:
    """Blocking implementation (uses `requests`). Do not call directly from
    async code — use get_exported_data() below, which runs this in a worker
    thread so it never blocks the event loop."""
    if entity not in ALLOWED_ENTITIES:
        raise ValueError(f"Invalid entity: {entity}. Must be one of {ALLOWED_ENTITIES}")

    sd_api_url = os.getenv("SD_API_URL")
    if not sd_api_url:
        raise ExportError("SD_API_URL environment variable is not set")

    x_api_key = os.getenv("X_API_KEY")
    if not x_api_key:
        raise ExportError("X_API_KEY environment variable is not set")

    params = {"customer_id": customer_id, "entity": entity}
    headers = {"x-api-key": x_api_key}

    try:
        response = requests.get(sd_api_url, params=params, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        raise ExportError(f"Export request failed for entity '{entity}': {e}") from e

    if response.status_code == 401:
        raise ExportError(f"Authentication error (401) exporting '{entity}': {_get_error_detail(response)}")
    if response.status_code != 200:
        raise ExportError(
            f"Error exporting '{entity}': HTTP {response.status_code} — {_get_error_detail(response)}"
        )

    try:
        data = response.json()
    except ValueError:
        raise ExportError(f"Response for '{entity}' is not valid JSON — {_get_error_detail(response)}")

    exported_url = data.get("fileUrl")
    if not exported_url:
        raise ExportError(f"No 'fileUrl' found in export response for '{entity}'")

    try:
        file_response = requests.get(exported_url, timeout=10)
    except requests.exceptions.RequestException as e:
        raise ExportError(f"File download failed for '{entity}': {e}") from e

    if file_response.status_code != 200:
        raise ExportError(
            f"Failed to download file for '{entity}': "
            f"HTTP {file_response.status_code} — {_get_error_detail(file_response)}"
        )

    return file_response.content


async def get_exported_data(customer_id: str, entity: str) -> bytes:
    """Async-safe wrapper around the blocking export call."""
    return await asyncio.to_thread(_get_exported_data_sync, customer_id, entity)


async def _save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(content)


async def _save_text(path: Path, content: str) -> None:
    await _save_bytes(path, content.encode("utf-8"))


async def _read_text(path: Path) -> str:
    async with aiofiles.open(path, "r") as f:
        return await f.read()



def _orders_paths(customer_id: str) -> dict[str, Path]:
    order_dir = DATA_ROOT / customer_id / "orders"
    root = DATA_ROOT / customer_id
    return {
        "orders_csv": order_dir / "orders.csv",
        "products_csv": order_dir / "order_products.csv",
        "report_md": root / "report.md",
        "sections_json": root / "report_sections.json",
    }

def _activities_paths(customer_id: str) -> dict[str, Path]:
    act_dir = DATA_ROOT / customer_id / "activities"
    root = DATA_ROOT / customer_id
    return {
        "notes_csv": act_dir / "notes.csv",
        "tasks_csv": act_dir / "tasks.csv",
        "activities_csv": act_dir / "activities.csv",
        "report_activities_md": root / "report_activities.md",
        "report_task_md": root / "report_task.md",
        "report_notes_md": root / "report_notes.md",
        "report_final_md": root / "report_activities_answer.md",
        "sections_json": root / "report_activities_sections.json",
    }



async def ensure_sales_report(customer_id: str, force: bool = False) -> dict:
    """
    Guarantees data/{customer_id}/report.md (+ source CSVs) exist, building
    them only if missing or `force=True`. Returns the report text, its
    sections, and the CSV paths either way.
    """
    paths = _orders_paths(customer_id)
    is_cached = all(
        paths[k].exists() for k in ("orders_csv", "products_csv", "report_md")
    )

    if not force and is_cached:
        logger1.info(f"[orders] cache hit for customer '{customer_id}'")
        report_text = await _read_text(paths["report_md"])
        sections = None
        if paths["sections_json"].exists():
            sections = json.loads(await _read_text(paths["sections_json"]))
        return {
            "full_report": report_text,
            "sections": sections,
            "orders_path": str(paths["orders_csv"]),
            "products_path": str(paths["products_csv"]),
            "from_cache": True,
        }

    logger1.info(f"[orders] building report for customer '{customer_id}' (force={force})")

    orders_content, products_content = await asyncio.gather(
        get_exported_data(customer_id, "orders"),
        get_exported_data(customer_id, "order_products"),
    )
    await asyncio.gather(
        _save_bytes(paths["orders_csv"], orders_content),
        _save_bytes(paths["products_csv"], products_content),
    )

    result = await generate_sales_report(
        str(paths["orders_csv"]), str(paths["products_csv"]), customer_id
    )
    report_text = result["full_report"]
    sections = result["sections"]

    await asyncio.gather(
        _save_text(paths["report_md"], report_text),
        _save_text(paths["sections_json"], json.dumps(sections)),
    )

    return {
        "full_report": report_text,
        "sections": sections,
        "orders_path": str(paths["orders_csv"]),
        "products_path": str(paths["products_csv"]),
        "from_cache": False,
    }


async def ensure_activities_report(customer_id: str, force: bool = False) -> dict:
    """
    Guarantees the activities/tasks/notes source CSVs and all derived
    reports (including the final AI answer) exist on disk, building them
    only if missing or `force=True`.

    Note: in the original code, the result of process_ai_activities_request
    was recomputed on every /st_Ask_ai call and then never actually used by
    the SSE generator afterward — pure wasted work. Here it's computed once,
    persisted to report_final_md, and reused from disk on cache hits.
    """
    paths = _activities_paths(customer_id)
    is_cached = all(
        paths[k].exists()
        for k in (
            "notes_csv", "tasks_csv", "activities_csv",
            "report_activities_md", "report_task_md", "report_notes_md",
            "report_final_md",
        )
    )

    if not force and is_cached:
        logger1.info(f"[activities] cache hit for customer '{customer_id}'")
        full_report = await _read_text(paths["report_final_md"])
        sections = None
        if paths["sections_json"].exists():
            sections = json.loads(await _read_text(paths["sections_json"]))
        return {
            "full_report": full_report,
            "sections": sections,
            "activities_path": str(paths["activities_csv"]),
            "tasks_path": str(paths["tasks_csv"]),
            "notes_path": str(paths["notes_csv"]),
            "from_cache": True,
        }

    logger1.info(f"[activities] building report for customer '{customer_id}' (force={force})")

    notes_content, tasks_content, activities_content = await asyncio.gather(
        get_exported_data(customer_id, "notes"),
        get_exported_data(customer_id, "tasks"),
        get_exported_data(customer_id, "activities"),
    )
    await asyncio.gather(
        _save_bytes(paths["notes_csv"], notes_content),
        _save_bytes(paths["tasks_csv"], tasks_content),
        _save_bytes(paths["activities_csv"], activities_content),
    )

    report_activities = await analyze_activities(
        str(paths["notes_csv"]), str(paths["tasks_csv"]), str(paths["activities_csv"])
    )
    report_task = tasks_report(str(paths["tasks_csv"]))
    report_notes = notes_report(str(paths["notes_csv"]))

    await asyncio.gather(
        _save_text(paths["report_activities_md"], report_activities),
        _save_text(paths["report_task_md"], report_task),
        _save_text(paths["report_notes_md"], str(report_notes)),
    )

    report_text, section_report = await process_ai_activities_request(customer_id)
    full_report = report_text.get("model_answer") or "Could not analyze the activity of your customer"

    await asyncio.gather(
        _save_text(paths["report_final_md"], full_report),
        _save_text(paths["sections_json"], json.dumps(section_report)),
    )

    return {
        "full_report": full_report,
        "sections": section_report,
        "activities_path": str(paths["activities_csv"]),
        "tasks_path": str(paths["tasks_csv"]),
        "notes_path": str(paths["notes_csv"]),
        "from_cache": False,
    }