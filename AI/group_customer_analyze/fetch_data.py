import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type
import os
from enum import Enum
import aiofiles
import aiohttp
import httpx
import time
import json
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from AI.utils import is_data_ready
class AnalysisIdType(str, Enum):
    CUSTOMER = "customer"
    ORDER = "order"
    CATALOG = "catalog"
    

RAW_FILENAME_BY_ENTITY: Dict[str, str] = {
    "orders": "one_file_orders.csv",
    "order_products": "one_file_products.csv",
    "customer": "one_file_customers.csv",
    "catalog": "one_file_catalog.csv",
}
 
 
def raw_filename_for(entity: str) -> str:
    return RAW_FILENAME_BY_ENTITY.get(entity, f"one_file_{entity}.csv")


class BaseDataFetchStrategy(ABC):
    """
    One subclass per AnalysisIdType.

    entities:
        The canonical entity names this strategy fetches, e.g.
        ["orders", "order_products", "customer"]. This is the single
        thing the endpoint reads to know "how many files, and which
        ones" - nothing else needs updating when this list changes.

    cleanup_entities:
        Which two of those entities (orders-like, products-like) get fed
        into the existing prepared_big_data() cleanup step. Defaults to
        ("orders", "order_products") since that's what both current
        strategies use. Set to None for a future strategy whose data
        doesn't go through that cleanup at all.
    """

    id_type: AnalysisIdType
    entities: List[str]
    cleanup_entities: Optional[Tuple[str, str]] = ("orders", "order_products")

    @abstractmethod
    async def fetch_raw_data(self, ids: List[str]) -> Dict[str, Any]:
        """Returns {entity_name: raw_payload} for every name in self.entities."""
        ...

    async def fetch_and_write(self, ids: List[str], user_folder: str, distributor_id: Optional[str]) -> Dict[str, str]:
        from AI.utils import _process_and_save_file_data
        
        raw_data = await self.fetch_raw_data(ids)
        file_paths = {
            entity: os.path.join(user_folder, raw_filename_for(entity))
            for entity in self.entities
        }
        await asyncio.gather(*(
            _process_and_save_file_data(raw_data[entity], file_paths[entity])
            for entity in self.entities
        ))
        return file_paths

    async def validate_ids(self, ids: List[str]) -> List[str]:
        """
        Optional per-strategy hook for cheap, type-specific validation
        (format checks, de-duping, max-batch-size, etc.) before hitting the
        3rd-party API. Default: just guard against an empty list.
        """
        if not ids:
            raise ValueError(f"No {self.id_type.value} IDs provided.")
        return ids

    async def _gather_entities(
        self,
        ids: List[str],
        fetch_fn: Callable[[List[str], List[str]], Awaitable[Any]],
    ) -> Dict[str, Any]:
        """
        Shared fan-out helper: calls fetch_fn(ids, [entity]) once per
        entity in self.entities, concurrently, and zips the results back
        onto their entity names. This is what makes the entity count
        dynamic - subclasses just hand it the right underlying fetch
        function.
        """
        results = await asyncio.gather(*(fetch_fn(ids, [entity]) for entity in self.entities))
        return dict(zip(self.entities, results))


class CustomerIdFetchStrategy(BaseDataFetchStrategy):
    """Current/default behavior - lifted as-is from the existing endpoint."""

    id_type = AnalysisIdType.CUSTOMER
    entities = ["orders", "order_products", "customer"]

    async def fetch_raw_data(self, ids: List[str]) -> Dict[str, Any]:
        from AI.group_customer_analyze.many_customer import post_get_exported_data_one_file

        return await self._gather_entities(ids, post_get_exported_data_one_file)


class SalesIdFetchStrategy(BaseDataFetchStrategy):
    """
    Order/sales analysis.
    """
    id_type = AnalysisIdType.ORDER
    entities = ["orders", "order_products"]
    cleanup_entities = ("orders", "order_products")

    async def fetch_raw_data(self, ids: List[str]) -> Dict[str, Any]:
        from AI.group_customer_analyze.many_customer import post_group_orders

        return await post_group_orders(ids, ["sales"])

    async def fetch_and_write(self, ids: List[str], user_folder: str, distributor_id: Optional[str]) -> Dict[str, str]:
        from AI.utils import write_bytes_to_file_async

        raw = await self.fetch_raw_data(ids)
        files = raw.get("files", {})

        missing = [e for e in self.entities if e not in files]
        if missing:
            raise Exception(
                f"Sales export response missing expected file(s) {missing}; "
                f"got back: {list(files.keys())}"
            )

        file_paths = {
            entity: os.path.join(user_folder, raw_filename_for(entity))
            for entity in self.entities
        }


        await asyncio.gather(*(
            write_bytes_to_file_async(file_paths[entity], files[entity])
            for entity in self.entities
        ))
        return file_paths


class CatalogIdFetchStrategy(BaseDataFetchStrategy):
    """
    Catalog analysis.

    """
    id_type = AnalysisIdType.CATALOG
    entities = ["catalog"]
    cleanup_entities = None  # catalog doesn't go through the cleanup step

    async def fetch_raw_data(self, ids: List[str]) -> Dict[str, Any]:
        pass

    async def fetch_and_write(self, ids: List[str], user_folder: str, distributor_id: str) -> Dict[str, str]:
        from AI.group_customer_analyze.many_customer import post_get_exported_data_one_file, post_group_catalog
        from AI.MCP_tools.get_SD_data import get_distributor_data, handle_distributor_data
        
        catalog_data = await post_group_catalog(ids, ["catalog"])
        should_download_files = is_data_ready(distributor_id, "catalog")

        try:
            if not should_download_files:
                # --- STEP 1: FETCH DATA ---
                timeout_config = httpx.Timeout(5.0, read=120.0)
                async with httpx.AsyncClient(timeout=timeout_config) as shared_client:
                    fetch_tasks = [
                        get_distributor_data(distributor_id=distributor_id, entities=["orders"], client=shared_client),
                        get_distributor_data(distributor_id=distributor_id, entities=["order_products"], client=shared_client),
                        get_distributor_data(distributor_id=distributor_id, entities=["catalog"], client=shared_client),
                    ]
                    all_orders_data, all_products_data, all_catalog_data = await asyncio.gather(*fetch_tasks)

                # --- STEP 2: DOWNLOAD FILES ---
                #print(catalog_data)
                async with aiohttp.ClientSession() as download_session:
                    handle_tasks = [
                        handle_distributor_data(all_orders_data, "orders", distributor_id, download_session),
                        handle_distributor_data(all_products_data, "order_products", distributor_id, download_session),
                        handle_distributor_data(all_catalog_data, "catalog", distributor_id, download_session),
                        handle_distributor_data(catalog_data, "selected_catalog", distributor_id, download_session),
                    ]
                    await asyncio.gather(*handle_tasks)
            else:
                #case when full data is OK but need new catalog analysis:
                async with aiohttp.ClientSession() as download_session:
                    handle_tasks = [
                        handle_distributor_data(catalog_data, "selected_catalog", distributor_id, download_session)
                    ]
                    await asyncio.gather(*handle_tasks)

        except Exception as e:
            error_msg = str(e)
            if "HTTP Error" in error_msg:
                try:
                    # Extract the JSON payload from the exception string
                    json_part = error_msg.split("HTTP Error 404: ")[1]
                    upstream_detail = json.loads(json_part)
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={
                            "error": "Upstream Resource Missing",
                            "distributor_id": distributor_id,
                            "upstream_message": upstream_detail.get("message", "").strip()
                        }
                    )
                except (IndexError, json.JSONDecodeError):
                    pass # Fall through to generic handler if parsing fails
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data sync failed: {error_msg}"
            )

        return True


_STRATEGY_REGISTRY: Dict[AnalysisIdType, Type[BaseDataFetchStrategy]] = {
    AnalysisIdType.CUSTOMER: CustomerIdFetchStrategy,
    AnalysisIdType.ORDER: SalesIdFetchStrategy,
    AnalysisIdType.CATALOG: CatalogIdFetchStrategy,
}


def get_fetch_strategy(id_type: AnalysisIdType, distributor_id: Optional[str] = None) -> BaseDataFetchStrategy:
    """
    Single lookup point used by the endpoint. Adding a new AnalysisIdType
    means adding one entry here (plus a strategy class above with its own
    `entities` list) - the endpoint itself never branches on id_type or on
    entity count.
    """
    try:
        strategy_cls = _STRATEGY_REGISTRY[id_type]
    except KeyError:
        raise ValueError(f"No fetch strategy registered for id_type={id_type!r}")
    return strategy_cls()