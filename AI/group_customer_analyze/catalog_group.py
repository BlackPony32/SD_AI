import os
from itertools import combinations

import numpy as np
import pandas as pd

import aiofiles
from AI.utils import get_logger
from AI.group_customer_analyze.statistics_group_c import format_status, usd, top_new_contact, top_reorder_contact, peak_visit_time, \
  customer_insights, format_percentage

from AI.utils import get_logger, combine_sections, calculate_cost
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, AsyncOpenAI, OpenAIConversationsSession
from AI.group_customer_analyze.Agents_rules.prompts import prompt_agent_suggestions, prompt_mcp_topics_customer_agent, prompt_mcp_suggestions, prompt_catalog_grouped


logger2 = get_logger("logger2", "project_log_many.log", False)

from dotenv import load_dotenv
load_dotenv()

model = 'gpt-5.4-mini' #'gpt-5.4-mini'
llm_model = OpenAIResponsesModel(model=model, openai_client=AsyncOpenAI()) 

@function_tool
def get_prepared_statistics(user_id:str) -> str:
    """Each time, first call this tool to retrieve the user data that needs to be analyzed."""

    logger2.info(f"Tool 'mcp_get_prepared_statistics' called ")
    data_path = f"data/{user_id}/full_report.md"
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

ORDER_COLUMN_ALIASES = {
    "customerId": "customer_id",
}

_COST_COLUMN_CANDIDATES = [
    "wholesaleprice", "unit_cost", "cost", "costprice", "cost_price", "cogs",
    "landedcost", "landed_cost", "acquisitioncost", "acquisition_cost",
]

# Thresholds used to decide when a metric is worth flagging as an action item.
_OUTSTANDING_REVENUE_FLAG_PCT = 15      # % of net billed revenue still uncollected
_DISCOUNT_DEPTH_FLAG_PCT = 15           # % off list price on average
_CUSTOMER_CONCENTRATION_FLAG_PCT = 40   # % of a product's revenue from one customer
_LOW_STOCK_MONTHS_FLAG = 1              # months of stock remaining considered urgent
_STALE_PRODUCT_DAYS_FLAG = 60           # days since last order considered "gone quiet"
_TOP_N_ROWS = 10                        # cap for open-ended tables (customers, candidates)
_RECENT_TREND_MONTHS = 6                # months shown in the compact recent-trend table
_MOMENTUM_WINDOW_DAYS = 90              # trailing window used for the momentum comparison
_MIN_COOCCURRENCE_ORDERS = 2            # min co-occurring orders before a cross-sell candidate counts
_STRONG_LIFT = 2.0                      # lift above which a pairing is called out as a standout
_MAX_PAIRWISE_PRODUCTS = 6              # above this many selected products, skip the full pairwise matrix
_TARGET_ATTACH_LIFT_PP = 10             # percentage-point attach-rate lift used for the revenue projection
_SMALL_SAMPLE_ORDERS = 15               # focal-order count below which results get a small-sample caveat
_FULFILLMENT_SLA_DAYS = 7               # target days-to-ship used for the "% within SLA" stat
_BACKLOG_STALE_DAYS_FLAG = 14           # an unfulfilled order open this long gets flagged specifically


def _parse_created_at(series) -> pd.Series:
    raw = series.fillna('').astype(str)
    parsed = pd.to_datetime(raw, errors='coerce', utc=True, format='ISO8601')
    still_missing = parsed.isna()
    if still_missing.any():
        stripped = raw[still_missing].str.split(' (', n=1, regex=False).str[0]
        parsed.loc[still_missing] = pd.to_datetime(
            stripped, format='%a %b %d %Y %H:%M:%S GMT%z', errors='coerce', utc=True
        )
    return parsed



_ACTIVE_STATUS_CANDIDATES = ['status', 'isactive', 'is_active', 'active', 'enabled']
_ACTIVE_STATUS_TRUE_VALUES = {'active', 'enabled', 'true', '1', 'yes'}


def _resolve_active_ids(catalog_df) -> set:
    """Returns the set of product ids considered active/sellable in a FULL catalog
    export (as opposed to the selection-only catalog).

    If the file has a recognizable status-like column, only rows with an active-like
    value count. Otherwise every id in the file is treated as active — this matches
    the 'cleaned catalog' convention, where discontinued/inactive products are
    expected to already be excluded from the export rather than flagged in a column."""
    lower_map = {str(col).lower(): col for col in catalog_df.columns}
    status_col = None
    for candidate in _ACTIVE_STATUS_CANDIDATES:
        if candidate in lower_map:
            status_col = lower_map[candidate]
            break
    if status_col is None:
        return set(catalog_df['id'].dropna().unique())

    mask = catalog_df[status_col].apply(
        lambda v: str(v).strip().lower() in _ACTIVE_STATUS_TRUE_VALUES if pd.notna(v) else False
    )
    return set(catalog_df.loc[mask, 'id'].dropna().unique())


def _detect_cost_column(catalog_df):
    """Returns the actual column name holding a true unit cost, if the catalog has
    one, else None. Case-insensitive match against `_COST_COLUMN_CANDIDATES`."""
    lower_map = {str(col).lower(): col for col in catalog_df.columns}
    for candidate in _COST_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _canonical_names(df, cat_dict) -> dict:
    """Returns {productId: display_name}, one canonical name per productId.

    Prefers the catalog's name (+size/color) since that's deterministic. Falls back
    to the most common raw order-line 'name' when the catalog has none. This matters
    because the raw order-line 'name' is inconsistently spelled across lines for the
    same productId in real data (e.g. 'Coca Cola glass bottle' vs '...bottles') -
    grouping by that raw name instead of productId would silently split one physical
    product's revenue across multiple report rows."""
    names = {}
    for pid, group in df.groupby('productId'):
        cat_item = cat_dict.get(pid, {})
        cat_name = cat_item.get('name')
        if pd.notna(cat_name):
            base = cat_name
        else:
            mode = group['name'].mode(dropna=True)
            base = mode.iloc[0] if len(mode) else "Unknown Product"
            if pd.isna(base):
                base = "Unknown Product"

        parts = [str(base)]
        size = cat_item.get('size')
        color = cat_item.get('color')
        if pd.notna(size):
            parts.append(str(size))
        if pd.notna(color):
            parts.append(str(color))
        display = " ".join(parts)

        # Variants can share the same size/color (e.g. different flavors or designs of
        # otherwise-identical packaging) and would otherwise render as identical, in-
        # distinguishable row labels. SKU is usually the true variant differentiator in
        # that case, so append it when it isn't already reflected in the name.
        sku = cat_item.get('sku')
        if pd.notna(sku) and str(sku).strip() and str(sku).strip().lower() not in display.lower():
            display = f"{display} ({sku})"

        names[pid] = _truncate(display)
    return names


def _derive_month_column(orders_df):
    """Returns a Series of 'MM/YYYY' month labels for the orders.

    Uses the existing 'month' column when present (the 'cleaned' file schema already
    has it). Otherwise derives it from 'createdAt' via `_parse_created_at`."""
    if 'month' in orders_df.columns:
        return orders_df['month']
    if 'createdAt' not in orders_df.columns:
        return pd.Series(['Unknown'] * len(orders_df), index=orders_df.index)
    return _parse_created_at(orders_df['createdAt']).dt.strftime('%m/%Y').fillna('Unknown')


def _resolve_via_parent(cat_dict, pid, column, default=None):
    """Falls back to the parent row's value for a catalog attribute that's
    frequently only populated at the parent level rather than on the sellable
    variant row (observed for sellingOutOfStock, alongside name/category/
    manufacturer elsewhere) — only works if the parent row happens to be present
    in the same catalog file, which is a best-effort improvement, not a guarantee."""
    val = cat_dict.get(pid, {}).get(column)
    if pd.notna(val):
        return val
    parent_id = cat_dict.get(pid, {}).get('parentProductId')
    if pd.notna(parent_id):
        parent_val = cat_dict.get(parent_id, {}).get(column)
        if pd.notna(parent_val):
            return parent_val
    return default

def _resolve_with_fallback(preferred_value, fallback_series, default):
    """Returns `preferred_value` if it's populated; otherwise the most common
    non-null value in `fallback_series`; otherwise `default`.

    Needed because catalog exports often only populate name/category/manufacturer
    on the parent product row, not on the sellable variant row that order lines
    actually reference — so the catalog lookup for a variant can be genuinely
    empty even though the business clearly has this data (it's just sitting on
    the order line items instead)."""
    if pd.notna(preferred_value):
        return preferred_value
    if fallback_series is not None and len(fallback_series):
        modes = fallback_series.dropna().mode()
        if len(modes):
            return modes.iloc[0]
    return default

def _truncate(text, max_len=60) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "…"

def _fmt_money(value) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "N/A"

def _fmt_pct(value) -> str:
    return f"{value:.1f}%" if pd.notna(value) else "N/A"

def _log_empty(report_name, message, **context):
    """Logs the reason a report is returning early with no data — this is NOT an
    exception path, it's a normal 'nothing to report' outcome (bad selection, no
    order history, missing columns, etc). Without this, an intentionally-empty
    report and a silently-broken one look identical from the outside — same short
    output, nothing in the logs. Every early return in both report functions goes
    through this so a WARNING line always explains why."""
    ctx = " | ".join(f"{k}={v}" for k, v in context.items())
    logger2.warning(f"[{report_name}] Empty report — {message}" + (f" | {ctx}" if ctx else ""))
    return [message]


# --------------------------------------------------------------------------------- #
# 1. Revenue & Profitability
# --------------------------------------------------------------------------------- #
def _generate_revenue_profitability_report(catalog_path, orders_path, products_path) -> list:
    """
    Revenue & profitability breakdown for the product(s) in the given catalog selection.

    For each product in `catalog_path`, shows the full revenue waterfall (gross
    ordered -> discounts -> net billed -> collected -> refunded -> still outstanding),
    plus either true gross margin (if the catalog has a real cost column) or price
    realization vs. list price as a profitability proxy. When the selection contains
    more than one product, a portfolio total row and per-product action flags are added.

    Args:
        products_path: path to the order line items csv (cleaned_products.csv schema).
        orders_path: path to the orders csv (cleaned_orders.csv schema).
        catalog_path: path to the catalog csv, PRE-FILTERED to the user's selected
            product(s). Every product id present in this file is treated as selected;
            there is no separate id-list parameter.

    Returns:
        list[str]: markdown lines forming the report, or a single-item list with a
        user-facing error/status message if the report could not be generated.
    """
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(orders_path) and os.path.exists(catalog_path)):
                return _log_empty(
                    "Revenue & Profitability Report", "Error: One or more required data files (products, orders, catalog) are missing.",
                    products_path=products_path, orders_path=orders_path, catalog_path=catalog_path,
                    products_exists=os.path.exists(products_path), orders_exists=os.path.exists(orders_path),
                    catalog_exists=os.path.exists(catalog_path),
                )

            products_df = pd.read_csv(products_path)
            orders_df = pd.read_csv(orders_path)
            catalog_df = pd.read_csv(catalog_path)

            if catalog_df.empty:
                return _log_empty(
                    "Revenue & Profitability Report", "Please select at least one product to generate a Revenue & Profitability report.",
                    catalog_path=catalog_path,
                )
            if products_df.empty or orders_df.empty:
                return _log_empty(
                    "Revenue & Profitability Report", "Can not generate report: there is not enough data in the provided files.",
                    products_rows=len(products_df), orders_rows=len(orders_df),
                )

        except Exception as e:
            logger2.error(f"Data Loading Error in Revenue & Profitability Report: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Catalog Mapping & Selection Filtering ---
        try:
            if 'id' not in catalog_df.columns or 'productId' not in products_df.columns:
                return _log_empty(
                    "Revenue & Profitability Report", "Can not generate report: Missing required columns in catalog or products data.",
                    catalog_columns=list(catalog_df.columns), products_columns=list(products_df.columns),
                )

            cost_column = _detect_cost_column(catalog_df)
            catalog_cols = ['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name', 'sku']
            if cost_column:
                catalog_cols.append(cost_column)
            catalog_cols = [c for c in catalog_cols if c in catalog_df.columns]

            cat_dict = catalog_df.set_index('id')[catalog_cols].to_dict('index')
            selected_ids = set(catalog_df['id'].dropna().unique())

            df_selected = products_df[products_df['productId'].isin(selected_ids)].copy()
            if df_selected.empty:
                return _log_empty(
                    "Revenue & Profitability Report", "Can not generate report: the selected product(s) have no order history yet.",
                    selected_ids=sorted(selected_ids), products_rows_total=len(products_df),
                )

            name_map = _canonical_names(df_selected, cat_dict)
            df_selected['detailed_name'] = df_selected['productId'].map(name_map)

            no_history_ids = selected_ids - set(df_selected['productId'].unique())

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Revenue & Profitability Report: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Merge with Orders & Calculate Revenue Metrics ---
        try:
            required_order_cols = {'id', 'customer_name', 'paymentStatus'}
            if not required_order_cols.issubset(orders_df.columns):
                return _log_empty(
                    "Revenue & Profitability Report", "Can not generate report: Missing required columns in orders data.",
                    required=sorted(required_order_cols), missing=sorted(required_order_cols - set(orders_df.columns)),
                    orders_columns=list(orders_df.columns),
                )

            orders_scope = orders_df[orders_df['archived'] == False] if 'archived' in orders_df.columns else orders_df

            merged = pd.merge(
                df_selected,
                orders_scope[['id', 'customer_name', 'paymentStatus']],
                left_on='orderId', right_on='id', how='inner', suffixes=('', '_order')
            )
            if merged.empty:
                return _log_empty(
                    "Revenue & Profitability Report", "Can not generate report: No matching order history for the selected product(s).",
                    df_selected_rows=len(df_selected), orders_scope_rows=len(orders_scope),
                    sample_order_ids=df_selected['orderId'].unique()[:5].tolist(),
                )

            for col in ['quantity', 'paidQuantity', 'price', 'itemDiscountValue', 'amount', 'totalRawAmount', 'totalAmount']:
                if col not in merged.columns:
                    return _log_empty(
                        "Revenue & Profitability Report", f"Can not generate report: Missing required column '{col}' in products data.",
                        products_columns=list(products_df.columns),
                    )

            merged['item_discount_total'] = merged['itemDiscountValue'].fillna(0) * merged['quantity']
            merged['net_billed'] = merged['amount'].fillna(0) * merged['quantity']
            merged['refunded_amount'] = np.where(merged['paymentStatus'] == 'REFUNDED', merged['totalAmount'], 0)

            metrics = merged.groupby(['productId', 'detailed_name'], as_index=False).agg(
                gross_ordered_revenue=('totalRawAmount', 'sum'),
                item_discount_given=('item_discount_total', 'sum'),
                net_billed_revenue=('net_billed', 'sum'),
                collected_revenue_gross=('totalAmount', 'sum'),
                refunded_revenue=('refunded_amount', 'sum'),
                total_units_ordered=('quantity', 'sum'),
                order_count=('orderId', 'nunique'),
                unique_customers=('customer_name', 'nunique'),
                avg_selling_price=('price', 'mean'),
            )
            metrics['net_collected_revenue'] = metrics['collected_revenue_gross'] - metrics['refunded_revenue']
            metrics['outstanding_revenue'] = (metrics['net_billed_revenue'] - metrics['collected_revenue_gross']).clip(lower=0)

            metrics['list_price'] = metrics['productId'].map(lambda pid: cat_dict.get(pid, {}).get('wholesalePrice', np.nan))
            metrics['price_realization_pct'] = np.where(
                metrics['list_price'] > 0, (metrics['avg_selling_price'] / metrics['list_price']) * 100, np.nan
            )
            metrics['stock'] = metrics['productId'].map(
                lambda pid: int(cat_dict.get(pid, {}).get('inventory_onHand', 0)) if pd.notna(cat_dict.get(pid, {}).get('inventory_onHand', 0)) else 0
            )
            metrics['discount_rate_pct'] = np.where(
                metrics['gross_ordered_revenue'] > 0, (metrics['item_discount_given'] / metrics['gross_ordered_revenue']) * 100, 0
            )
            metrics['collection_rate_pct'] = np.where(
                metrics['net_billed_revenue'] > 0, (metrics['net_collected_revenue'] / metrics['net_billed_revenue']) * 100, np.nan
            )

            if cost_column:
                metrics['unit_cost'] = metrics['productId'].map(lambda pid: cat_dict.get(pid, {}).get(cost_column, np.nan))
                metrics['gross_profit'] = metrics['net_billed_revenue'] - (metrics['unit_cost'] * metrics['total_units_ordered'])
                metrics['margin_pct'] = np.where(
                    metrics['net_billed_revenue'] > 0, (metrics['gross_profit'] / metrics['net_billed_revenue']) * 100, np.nan
                )

        except Exception as e:
            logger2.error(f"Metrics Calculation Error in Revenue & Profitability Report: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Build the Markdown Report ---
        try:
            md = ["# Revenue & Profitability Analysis\n"]

            if no_history_ids:
                md.append(f"*Note: {len(no_history_ids)} selected item(s) have no order history yet and are excluded from this report.*\n")


            n_products = len(metrics)

            md.append("## Revenue Summary")
            md.append("| Product | Gross Ordered | Discounts Given | Net Billed | Collected (Net of Refunds) | Outstanding | Collection Rate |")
            md.append("|---|---|---|---|---|---|---|")
            for _, row in metrics.iterrows():
                md.append(
                    f"| **{row['detailed_name']}** | {_fmt_money(row['gross_ordered_revenue'])} | "
                    f"{_fmt_money(row['item_discount_given'])} | {_fmt_money(row['net_billed_revenue'])} | "
                    f"{_fmt_money(row['net_collected_revenue'])} | {_fmt_money(row['outstanding_revenue'])} | "
                    f"{_fmt_pct(row['collection_rate_pct'])} |"
                )
            if n_products > 1:
                gross_ordered_t = metrics['gross_ordered_revenue'].sum()
                discounts_t = metrics['item_discount_given'].sum()
                net_billed_t = metrics['net_billed_revenue'].sum()
                collected_t = metrics['net_collected_revenue'].sum()
                outstanding_t = metrics['outstanding_revenue'].sum()
                collection_rate_t = (collected_t / net_billed_t * 100) if net_billed_t else np.nan
                md.append(
                    f"| **PORTFOLIO TOTAL** | {_fmt_money(gross_ordered_t)} | {_fmt_money(discounts_t)} | "
                    f"{_fmt_money(net_billed_t)} | {_fmt_money(collected_t)} | {_fmt_money(outstanding_t)} | "
                    f"{_fmt_pct(collection_rate_t)} |"
                )
            md.append("\n---\n")


            md.append("## Flags & Action Items")
            flags = []
            for _, row in metrics.iterrows():
                if row['net_billed_revenue'] > 0 and (row['outstanding_revenue'] / row['net_billed_revenue']) * 100 > _OUTSTANDING_REVENUE_FLAG_PCT:
                    pct = row['outstanding_revenue'] / row['net_billed_revenue'] * 100
                    flags.append(f"- **{row['detailed_name']}** has {_fmt_money(row['outstanding_revenue'])} uncollected ({pct:.0f}% of net billed revenue) — consider a collections follow-up.")
                if row['refunded_revenue'] > 0:
                    flags.append(f"- **{row['detailed_name']}** had {_fmt_money(row['refunded_revenue'])} refunded — worth checking why (quality, mis-order, etc.).")
                if pd.notna(row['discount_rate_pct']) and row['discount_rate_pct'] > _DISCOUNT_DEPTH_FLAG_PCT:
                    flags.append(f"- **{row['detailed_name']}** is discounted {row['discount_rate_pct']:.0f}% off list on average — confirm this matches approved pricing policy.")
            if not flags:
                flags.append("- No major revenue-risk flags detected for the selected product(s).")
            md.extend(flags)

            logger2.info(
                f"[Revenue & Profitability Report] Generated OK — {n_products} product(s), "
                f"{len(merged)} order line(s), net billed={metrics['net_billed_revenue'].sum():.2f}"
            )
            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Revenue & Profitability Report: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    except Exception as e:
        logger2.error(f"Critical Error in Revenue & Profitability Report: {str(e)}")
        return ["This report is currently unavailable due to a temporary issue. Please check back later or contact support if you need assistance."]

# --------------------------------------------------------------------------------- #
# 2. Top Performer Deep Dive
# --------------------------------------------------------------------------------- #
def _generate_top_performer_deep_dive_report(catalog_path, orders_path, products_path) -> list:
    """
    Deep dive on the strongest product among the user's selection.

    If the selected catalog contains only one product, it is analyzed directly. If
    it contains several, they're first compared on net collected revenue (with
    unique customers as a tiebreaker) to pick a single winner, and that comparison
    is shown before the deep dive so the choice is transparent rather than a black box.

    The deep dive covers: revenue waterfall, monthly trend, customer concentration,
    repeat purchase rate, pricing consistency vs. list price, inventory/sell-through,
    and a directional fulfillment note.

    Args:
        products_path: path to the order line items csv (cleaned_products.csv schema).
        orders_path: path to the orders csv (cleaned_orders.csv schema).
        catalog_path: path to the catalog csv, PRE-FILTERED to the user's selected
            product(s). Every product id present in this file is treated as selected;
            there is no separate id-list parameter.

    Returns:
        list[str]: markdown lines forming the report, or a single-item list with a
        user-facing error/status message if the report could not be generated.
    """
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(orders_path) and os.path.exists(catalog_path)):
                return _log_empty(
                    "Top Performer Deep Dive", "Error: One or more required data files (products, orders, catalog) are missing.",
                    products_path=products_path, orders_path=orders_path, catalog_path=catalog_path,
                    products_exists=os.path.exists(products_path), orders_exists=os.path.exists(orders_path),
                    catalog_exists=os.path.exists(catalog_path),
                )

            products_df = pd.read_csv(products_path)
            orders_df = pd.read_csv(orders_path)
            catalog_df = pd.read_csv(catalog_path)

            if catalog_df.empty:
                return _log_empty(
                    "Top Performer Deep Dive", "Please select at least one product to generate a Top Performer Deep Dive.",
                    catalog_path=catalog_path,
                )
            if products_df.empty or orders_df.empty:
                return _log_empty(
                    "Top Performer Deep Dive", "Can not generate report: there is not enough data in the provided files.",
                    products_rows=len(products_df), orders_rows=len(orders_df),
                )

        except Exception as e:
            logger2.error(f"Data Loading Error in Top Performer Deep Dive: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Catalog Mapping & Selection Filtering ---
        try:
            if 'id' not in catalog_df.columns or 'productId' not in products_df.columns:
                return _log_empty(
                    "Top Performer Deep Dive", "Can not generate report: Missing required columns in catalog or products data.",
                    catalog_columns=list(catalog_df.columns), products_columns=list(products_df.columns),
                )

            catalog_cols = ['size', 'color', 'inventory_onHand', 'wholesalePrice', 'name',
                             'manufacturer_name', 'productCategory_name', 'sku']
            catalog_cols = [c for c in catalog_cols if c in catalog_df.columns]
            cat_dict = catalog_df.set_index('id')[catalog_cols].to_dict('index')
            selected_ids = set(catalog_df['id'].dropna().unique())

            df_selected = products_df[products_df['productId'].isin(selected_ids)].copy()
            if df_selected.empty:
                return _log_empty(
                    "Top Performer Deep Dive", "Can not generate report: the selected product(s) have no order history yet.",
                    selected_ids=sorted(selected_ids), products_rows_total=len(products_df),
                )

            name_map = _canonical_names(df_selected, cat_dict)
            df_selected['detailed_name'] = df_selected['productId'].map(name_map)

            no_history_ids = selected_ids - set(df_selected['productId'].unique())

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Top Performer Deep Dive: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Merge with Orders & Score Candidates ---
        try:
            required_order_cols = {'id', 'customer_name', 'paymentStatus'}
            if not required_order_cols.issubset(orders_df.columns):
                return _log_empty(
                    "Top Performer Deep Dive", "Can not generate report: Missing required columns in orders data.",
                    required=sorted(required_order_cols), missing=sorted(required_order_cols - set(orders_df.columns)),
                    orders_columns=list(orders_df.columns),
                )

            orders_scope = orders_df[orders_df['archived'] == False] if 'archived' in orders_df.columns else orders_df
            orders_scope = orders_scope.copy()
            orders_scope['month'] = _derive_month_column(orders_scope)
            orders_scope['_created_dt'] = _parse_created_at(orders_scope['createdAt']) if 'createdAt' in orders_scope.columns else pd.NaT

            merged = pd.merge(
                df_selected,
                orders_scope[['id', 'customer_name', 'paymentStatus', 'month', '_created_dt']],
                left_on='orderId', right_on='id', how='inner', suffixes=('', '_order')
            )
            if merged.empty:
                return _log_empty(
                    "Top Performer Deep Dive", "Can not generate report: No matching order history for the selected product(s).",
                    df_selected_rows=len(df_selected), orders_scope_rows=len(orders_scope),
                    sample_order_ids=df_selected['orderId'].unique()[:5].tolist(),
                )

            merged['net_billed'] = merged['amount'].fillna(0) * merged['quantity']
            merged['refunded_amount'] = np.where(merged['paymentStatus'] == 'REFUNDED', merged['totalAmount'], 0)

            candidate_scores = merged.groupby(['productId', 'detailed_name'], as_index=False).agg(
                collected_revenue_gross=('totalAmount', 'sum'),
                refunded_revenue=('refunded_amount', 'sum'),
                order_count=('orderId', 'nunique'),
                unique_customers=('customer_name', 'nunique'),
                total_units_ordered=('quantity', 'sum'),
            )
            candidate_scores['net_collected_revenue'] = candidate_scores['collected_revenue_gross'] - candidate_scores['refunded_revenue']
            candidate_scores = candidate_scores.sort_values(
                ['net_collected_revenue', 'unique_customers'], ascending=[False, False]
            ).reset_index(drop=True)

            winner_id = candidate_scores.iloc[0]['productId']
            winner_name = candidate_scores.iloc[0]['detailed_name']
            is_comparison = len(candidate_scores) > 1
            as_of_date = orders_scope['_created_dt'].max()

        except Exception as e:
            logger2.error(f"Candidate Scoring Error in Top Performer Deep Dive: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Deep-Dive Metrics for the Winning Product ---
        try:
            winner_lines = merged[merged['productId'] == winner_id].copy()

            net_billed = winner_lines['net_billed'].sum()
            collected_gross = winner_lines['totalAmount'].sum()
            refunded = winner_lines['refunded_amount'].sum()
            net_collected = collected_gross - refunded
            outstanding = max(net_billed - collected_gross, 0)
            total_units = winner_lines['quantity'].sum()
            order_count = winner_lines['orderId'].nunique()
            unique_customers = winner_lines['customer_name'].nunique()
            avg_units_per_order = (total_units / order_count) if order_count else 0

            cat_info = cat_dict.get(winner_id, {})
            list_price = cat_info.get('wholesalePrice', np.nan)
            stock = cat_info.get('inventory_onHand', 0)
            stock = int(stock) if pd.notna(stock) else 0
            sku = cat_info.get('sku', 'N/A')
            # Category/manufacturer often only live on the catalog's PARENT row, not
            # the sellable variant row order lines reference — fall back to the order
            # line data itself (which carries its own copy of these) before giving up.
            category = _resolve_with_fallback(
                cat_info.get('productCategory_name'), winner_lines.get('productCategoryName'), 'Uncategorized'
            )
            manufacturer = _resolve_with_fallback(
                cat_info.get('manufacturer_name'), winner_lines.get('manufacturerName'), 'Unknown'
            )

            # --- Pricing: median-based, with anomalies called out rather than blended in ---
            prices = winner_lines['price']
            price_anomaly_mask = prices <= 0
            anomaly_count = int(price_anomaly_mask.sum())
            clean_prices = prices[~price_anomaly_mask]
            median_price = clean_prices.median() if len(clean_prices) else np.nan
            reference_price = list_price if pd.notna(list_price) and list_price > 0 else median_price
            if len(clean_prices) and pd.notna(reference_price):
                at_list_mask = clean_prices >= reference_price * 0.999
                pct_at_list = at_list_mask.mean() * 100
                discounted = clean_prices[~at_list_mask]
                pct_discounted = 100 - pct_at_list
                avg_discount_pct = ((reference_price - discounted) / reference_price * 100).mean() if len(discounted) else 0
            else:
                pct_at_list = pct_discounted = avg_discount_pct = np.nan

            # Monthly trend (full history, chronological) — used for seasonality stats
            # and, capped to the most recent months, for the compact trend table.
            trend = winner_lines.groupby('month').agg(
                revenue=('totalAmount', 'sum'), units=('quantity', 'sum')
            ).reset_index()
            trend['_sort_key'] = pd.to_datetime(trend['month'], format='%m/%Y', errors='coerce')
            trend = trend.sort_values('_sort_key')
            dated_trend = trend[trend['_sort_key'].notna()]
            months_with_sales = len(dated_trend)

            seasonality_cv = None
            peak_month = trough_month = None
            if months_with_sales > 1 and dated_trend['revenue'].mean() > 0:
                seasonality_cv = dated_trend['revenue'].std() / dated_trend['revenue'].mean() * 100
                peak_month = dated_trend.loc[dated_trend['revenue'].idxmax()]
                trough_month = dated_trend.loc[dated_trend['revenue'].idxmin()]

            # --- Customer concentration & repeat behavior ---
            by_customer = winner_lines.groupby('customer_name').agg(
                revenue=('totalAmount', 'sum'), orders=('orderId', 'nunique')
            ).reset_index().sort_values('revenue', ascending=False)
            top_customers = by_customer.head(_TOP_N_ROWS)
            top_customer_share = (top_customers.iloc[0]['revenue'] / collected_gross * 100) if collected_gross and len(top_customers) else 0
            repeat_customers = (by_customer['orders'] > 1).sum()
            repeat_rate = (repeat_customers / unique_customers * 100) if unique_customers else 0

            # Average reorder interval for customers who ordered this product more than once.
            avg_reorder_days = None
            if pd.notna(as_of_date):
                intervals = []
                repeat_names = set(by_customer.loc[by_customer['orders'] > 1, 'customer_name'])
                for _, g in winner_lines[winner_lines['customer_name'].isin(repeat_names)].groupby('customer_name'):
                    dates = g['_created_dt'].dropna().drop_duplicates().sort_values()
                    if len(dates) > 1:
                        intervals.extend(dates.diff().dropna().dt.days.tolist())
                if intervals:
                    avg_reorder_days = sum(intervals) / len(intervals)

            # --- Sell-through & reorder runway ---
            sell_through_pct = (total_units / (total_units + stock) * 100) if (total_units + stock) > 0 else np.nan
            monthly_velocity = (total_units / months_with_sales) if months_with_sales else 0
            months_of_stock = (stock / monthly_velocity) if monthly_velocity > 0 else None

            # --- Fulfillment (directional only — 'delivered' tracking is inconsistently populated) ---
            fulfillment_pct = (winner_lines['delivered'].sum() / total_units * 100) if total_units and 'delivered' in winner_lines.columns else np.nan

            # --- Momentum: trailing window vs. the window before it ---
            momentum_pct = recency_days = None
            recent_revenue = prior_revenue = None
            new_vs_returning_note = None
            if pd.notna(as_of_date):
                last_order_date = winner_lines['_created_dt'].max()
                if pd.notna(last_order_date):
                    recency_days = (as_of_date - last_order_date).days

                recent_start = as_of_date - pd.Timedelta(days=_MOMENTUM_WINDOW_DAYS)
                prior_start = as_of_date - pd.Timedelta(days=2 * _MOMENTUM_WINDOW_DAYS)
                recent_mask = winner_lines['_created_dt'] > recent_start
                prior_mask = (winner_lines['_created_dt'] > prior_start) & (winner_lines['_created_dt'] <= recent_start)
                recent_revenue = winner_lines.loc[recent_mask, 'totalAmount'].sum()
                prior_revenue = winner_lines.loc[prior_mask, 'totalAmount'].sum()
                if prior_revenue > 0:
                    momentum_pct = (recent_revenue - prior_revenue) / prior_revenue * 100

                # New vs. returning customer mix within the trailing window.
                recent_lines = winner_lines[recent_mask]
                if len(recent_lines) and recent_lines['totalAmount'].sum() > 0:
                    customers_before_window = set(winner_lines.loc[winner_lines['_created_dt'] <= recent_start, 'customer_name'])
                    returning_rev = recent_lines[recent_lines['customer_name'].isin(customers_before_window)]['totalAmount'].sum()
                    window_total = recent_lines['totalAmount'].sum()
                    new_vs_returning_note = (returning_rev / window_total * 100, window_total)

        except Exception as e:
            logger2.error(f"Deep Dive Metrics Error in Top Performer Deep Dive: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 5. Build the Markdown Report ---
        try:
            md = ["# Top Performer Deep Dive\n"]

            if pd.notna(as_of_date):
                md.append(f"*Data runs through {as_of_date.strftime('%b %d, %Y')} (most recent order in your dataset). All \"days since/momentum\" stats below are measured against that date, not today's real-world date.*\n")

            if no_history_ids:
                md.append(f"*Note: {len(no_history_ids)} selected item(s) have no order history yet and are excluded from this report.*\n")

            if is_comparison:
                md.append("## Comparing Your Selections")
                shown = candidate_scores.head(_TOP_N_ROWS)
                md.append("| Product | Net Collected Revenue | Orders | Unique Customers |")
                md.append("|---|---|---|---|")
                for _, row in shown.iterrows():
                    marker = " 🏆" if row['productId'] == winner_id else ""
                    md.append(
                        f"| **{row['detailed_name']}**{marker} | {_fmt_money(row['net_collected_revenue'])} | "
                        f"{int(row['order_count'])} | {int(row['unique_customers'])} |"
                    )
                if len(candidate_scores) > _TOP_N_ROWS:
                    md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(candidate_scores)} selected products (all were considered when picking the winner).*")
                md.append(f"\n*'{winner_name}' was selected for the deep dive below — it has the highest net collected revenue of the {len(candidate_scores)} products you selected.*\n")
                md.append("\n---\n")

            md.append(f"## Deep Dive: {winner_name}")
            md.append(f"**Category:** {category}  |  **Manufacturer:** {manufacturer}  |  **SKU:** {sku if pd.notna(sku) else 'N/A'}\n")

            # --- Snapshot: the whole product at a glance in ~5 lines ---
            md.append("### Snapshot")
            md.append(
                f"- **Revenue:** {_fmt_money(net_collected)} collected (net of refunds) across {int(order_count)} orders"
                + (f", {_fmt_money(outstanding)} still outstanding" if outstanding > 0 else "")
            )
            md.append(f"- **Volume:** {int(total_units):,} units to {int(unique_customers)} customers ({repeat_rate:.0f}% repeat), avg {avg_units_per_order:.1f} units/order")
            stock_bit = "out of stock" if stock <= 0 else f"{stock:,} units on hand"
            runway_bit = (
                "" if stock <= 0 else
                f", ~{months_of_stock:.1f} mo. runway" if months_of_stock is not None else ""
            )
            md.append(f"- **Inventory:** {stock_bit}{runway_bit}  |  sell-through {_fmt_pct(sell_through_pct)}")
            md.append(f"- **Fulfillment:** {_fmt_pct(fulfillment_pct)} delivered vs. ordered *(directional — tracking has gaps)*")

            # --- Sales Patterns: the actual "big data" analysis over full history ---
            md.append("\n### Sales Patterns")
            if momentum_pct is not None:
                arrow = "📈" if momentum_pct > 5 else "📉" if momentum_pct < -5 else "➡️"
                caveat = " *(small prior-period base — interpret cautiously)*" if 0 < prior_revenue < (recent_revenue * 0.1) else ""
                md.append(f"- **Momentum:** {arrow} {momentum_pct:+.0f}% revenue in the last {_MOMENTUM_WINDOW_DAYS} days vs. the {_MOMENTUM_WINDOW_DAYS} days before that ({_fmt_money(recent_revenue)} vs. {_fmt_money(prior_revenue)}){caveat}")
            elif recent_revenue is not None:
                md.append(f"- **Momentum:** {_fmt_money(recent_revenue)} in the last {_MOMENTUM_WINDOW_DAYS} days — no prior-period activity to compare against")
            if recency_days is not None:
                stale_flag = " gone quiet" if recency_days > _STALE_PRODUCT_DAYS_FLAG else ""
                md.append(f"- **Recency:** {recency_days} day(s) since this product's last order (data runs through {as_of_date.strftime('%b %d, %Y')} — that's the most recent order anywhere in your dataset, not necessarily today){stale_flag}")
            if peak_month is not None and trough_month is not None:
                consistency = "highly seasonal/variable" if seasonality_cv is not None and seasonality_cv > 50 else "fairly steady"
                md.append(
                    f"- **Seasonality:** demand is {consistency} month to month (CV {seasonality_cv:.0f}%) — "
                    f"peak was {peak_month['_sort_key'].strftime('%b %Y')} ({_fmt_money(peak_month['revenue'])}), "
                    f"quietest was {trough_month['_sort_key'].strftime('%b %Y')} ({_fmt_money(trough_month['revenue'])})"
                )
            if avg_reorder_days is not None:
                md.append(f"- **Reorder Cadence:** repeat customers reorder roughly every {avg_reorder_days:.0f} days on average")
            if new_vs_returning_note is not None:
                returning_share, window_total = new_vs_returning_note
                md.append(f"- **Customer Mix (last {_MOMENTUM_WINDOW_DAYS} days):** {returning_share:.0f}% of {_fmt_money(window_total)} revenue came from customers who'd bought this product before, {100 - returning_share:.0f}% from customers new to it")

            # --- Recent Trend: compact, not the full history ---
            md.append(f"\n### Recent Trend (last {_RECENT_TREND_MONTHS} months)")
            recent_trend = dated_trend.tail(_RECENT_TREND_MONTHS)
            if len(recent_trend) > 0:
                md.append("| Month | Revenue | Units |")
                md.append("|---|---|---|")
                for _, row in recent_trend.iterrows():
                    md.append(f"| {row['_sort_key'].strftime('%b %Y')} | {_fmt_money(row['revenue'])} | {int(row['units'])} |")
                if months_with_sales > _RECENT_TREND_MONTHS:
                    md.append(f"\n*{months_with_sales - _RECENT_TREND_MONTHS} earlier month(s) omitted — see Sales Patterns above for full-history trend stats.*")
            else:
                md.append("*No dated sales history available.*")

            # --- Pricing: median + anomaly flag instead of a min/max range an outlier can distort ---
            md.append("\n### Pricing")
            md.append(f"- **Typical Price:** {_fmt_money(median_price)} (median)" + (f", list price {_fmt_money(list_price)}" if pd.notna(list_price) else ""))
            if pd.notna(pct_at_list):
                md.append(f"- **At List Price:** {pct_at_list:.0f}% of line items" + (f"  |  **Discounted:** {pct_discounted:.0f}% of line items, averaging {avg_discount_pct:.0f}% off" if pct_discounted > 0 else ""))
            if anomaly_count > 0:
                md.append(f"- **{anomaly_count} line item(s)** recorded at $0 or less — likely a data entry issue or comp order, excluded from the price stats above and worth a manual check")

            # --- Top Customers, capped ---
            md.append(f"\n### Top Customers{' (top ' + str(_TOP_N_ROWS) + ')' if len(by_customer) > _TOP_N_ROWS else ''}")
            md.append("| Customer | Revenue | Orders | Share of Product Revenue |")
            md.append("|---|---|---|---|")
            for _, row in top_customers.iterrows():
                share = (row['revenue'] / collected_gross * 100) if collected_gross else 0
                md.append(f"| {row['customer_name']} | {_fmt_money(row['revenue'])} | {int(row['orders'])} | {share:.1f}% |")
            if len(by_customer) > _TOP_N_ROWS:
                md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(by_customer)} customers.*")

            md.append("\n### Key Takeaways")
            takeaways = []
            if top_customer_share > _CUSTOMER_CONCENTRATION_FLAG_PCT:
                takeaways.append(f"- **Customer concentration risk:** {top_customers.iloc[0]['customer_name']} accounts for {top_customer_share:.0f}% of this product's revenue.")
            if stock <= 0 and total_units > 0:
                takeaways.append("- **Out of stock:** demand exists but there's no inventory left — reorder if this product should stay active.")
            elif months_of_stock is not None and months_of_stock < _LOW_STOCK_MONTHS_FLAG:
                takeaways.append(f"- **Reorder soon:** roughly {months_of_stock:.1f} months of stock left at current velocity.")
            if outstanding > 0 and net_billed > 0 and (outstanding / net_billed) * 100 > _OUTSTANDING_REVENUE_FLAG_PCT:
                takeaways.append(f"- **Collections:** {_fmt_money(outstanding)} is still uncollected for this product.")
            if refunded > 0:
                takeaways.append(f"- **Refunds:** {_fmt_money(refunded)} was refunded — worth a root-cause check.")
            if recency_days is not None and recency_days > _STALE_PRODUCT_DAYS_FLAG:
                takeaways.append(f"- **Gone quiet:** no orders in {recency_days} days despite {int(order_count)} orders historically — check if this product is being phased out or just needs a nudge.")
            if momentum_pct is not None and momentum_pct < -25:
                takeaways.append(f"- **Declining fast:** revenue is down {abs(momentum_pct):.0f}% over the last {_MOMENTUM_WINDOW_DAYS} days vs. the prior period.")
            if not takeaways:
                takeaways.append("- No major risk flags detected — this product is performing healthily on the available signals.")
            md.extend(takeaways)

            logger2.info(
                f"[Top Performer Deep Dive] Generated OK — winner='{winner_name}' out of {len(candidate_scores)} candidate(s), "
                f"net collected={net_collected:.2f}"
            )
            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Top Performer Deep Dive: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    except Exception as e:
        logger2.error(f"Critical Error in Top Performer Deep Dive: {str(e)}")
        return ["This report is currently unavailable due to a temporary issue. Please check back later or contact support if you need assistance."]

# --------------------------------------------------------------------------------- #
# 3. Cross-Sell & Bundle Actionability
# --------------------------------------------------------------------------------- #
def _generate_cross_sell_bundle_report(catalog_path, orders_path, products_path, full_catalog_path) -> list:
    """
    Market-basket analysis for the selected product(s): what else shows up in the
    same orders, how much more likely that is than chance, and whether the
    selection itself already behaves like a natural bundle.

    Methodology:
    - Looks at every order that contains at least one selected product ("focal
      orders"), regardless of payment/refund status — this is a behavioral
      co-purchase signal, not a revenue recognition metric, so it intentionally
      does NOT exclude refunded orders the way the revenue reports do.
    - For each other product found in those orders, computes:
        attach rate = % of focal orders that also contain it
        lift        = attach rate / that product's normal share of ALL orders
      Lift > 1 means it shows up with the selection more than its baseline
      popularity would predict — that's the actual cross-sell signal, not raw
      co-occurrence count (which just favors generically popular items).
    - Candidates are restricted to products that are ACTIVE in `full_catalog_path`
      — recommending a cross-sell of something no longer sellable isn't actionable.
      Their display names are also resolved against that full catalog (proper
      name/size/color/sku) rather than the raw order-line text.
    - If more than one product is selected, also reports how often the selection
      is *already* being bought as a set — i.e. validates whether a bundle makes
      sense before recommending one.

    Args:
        products_path: path to the order line items csv (cleaned_products.csv schema).
        orders_path: path to the orders csv (cleaned_orders.csv schema).
        catalog_path: path to the catalog csv, PRE-FILTERED to the user's selected
            product(s). Every product id present in this file is treated as selected.
        full_catalog_path: path to the FULL catalog csv (cleaned_catalog.csv schema —
            all products, not just the selection). Used to (a) resolve proper names
            for cross-sell candidates and (b) restrict candidates to active/sellable
            products. If the file has no recognizable status column, every product in
            it is treated as active (matches the 'cleaned catalog' convention of
            already excluding discontinued items from the export).

    Returns:
        list[str]: markdown lines forming the report, or a single-item list with a
        user-facing error/status message if the report could not be generated.
    """
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(orders_path)
                    and os.path.exists(catalog_path) and os.path.exists(full_catalog_path)):
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Error: One or more required data files (products, orders, catalog, full catalog) are missing.",
                    products_path=products_path, orders_path=orders_path, catalog_path=catalog_path, full_catalog_path=full_catalog_path,
                    products_exists=os.path.exists(products_path), orders_exists=os.path.exists(orders_path),
                    catalog_exists=os.path.exists(catalog_path), full_catalog_exists=os.path.exists(full_catalog_path),
                )

            products_df = pd.read_csv(products_path)
            orders_df = pd.read_csv(orders_path)
            catalog_df = pd.read_csv(catalog_path)
            full_catalog_df = pd.read_csv(full_catalog_path)

            if catalog_df.empty:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Please select at least one product to generate a Cross-Sell & Bundle report.",
                    catalog_path=catalog_path,
                )
            if products_df.empty or orders_df.empty:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: there is not enough data in the provided files.",
                    products_rows=len(products_df), orders_rows=len(orders_df),
                )

        except Exception as e:
            logger2.error(f"Data Loading Error in Cross-Sell & Bundle Actionability: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Catalog Mapping & Selection Filtering ---
        try:
            if 'id' not in catalog_df.columns or 'productId' not in products_df.columns:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: Missing required columns in catalog or products data.",
                    catalog_columns=list(catalog_df.columns), products_columns=list(products_df.columns),
                )
            for col in ['orderId', 'name', 'totalAmount']:
                if col not in products_df.columns:
                    return _log_empty(
                        "Cross-Sell & Bundle Actionability", f"Can not generate report: Missing required column '{col}' in products data.",
                        products_columns=list(products_df.columns),
                    )

            catalog_cols = [c for c in ['size', 'color', 'name', 'sku'] if c in catalog_df.columns]
            cat_dict = catalog_df.set_index('id')[catalog_cols].to_dict('index')
            selected_ids = set(catalog_df['id'].dropna().unique())

            df_selected = products_df[products_df['productId'].isin(selected_ids)].copy()
            if df_selected.empty:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: the selected product(s) have no order history yet.",
                    selected_ids=sorted(selected_ids), products_rows_total=len(products_df),
                )

            name_map = _canonical_names(df_selected, cat_dict)
            no_history_ids = selected_ids - set(df_selected['productId'].unique())
            selected_ids_with_history = selected_ids - no_history_ids

            if 'id' not in full_catalog_df.columns:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: Missing required columns in full catalog data.",
                    full_catalog_columns=list(full_catalog_df.columns),
                )
            full_catalog_cols = [c for c in ['size', 'color', 'name', 'sku'] if c in full_catalog_df.columns]
            full_cat_dict = full_catalog_df.set_index('id')[full_catalog_cols].to_dict('index')
            active_ids = _resolve_active_ids(full_catalog_df)

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Cross-Sell & Bundle Actionability: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Build the Basket Universe ---
        try:
            if 'id' not in orders_df.columns:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: Missing required columns in orders data.",
                    orders_columns=list(orders_df.columns),
                )

            orders_scope = orders_df[orders_df['archived'] == False] if 'archived' in orders_df.columns else orders_df
            order_universe = set(orders_scope['id'].dropna().unique())
            total_orders_universe = len(order_universe)

            products_scope = products_df[products_df['orderId'].isin(order_universe)].copy()

            focal_order_ids = set(products_scope.loc[products_scope['productId'].isin(selected_ids_with_history), 'orderId'].unique())
            total_focal_orders = len(focal_order_ids)

            if total_focal_orders == 0:
                return _log_empty(
                    "Cross-Sell & Bundle Actionability", "Can not generate report: No matching order history for the selected product(s).",
                    df_selected_rows=len(df_selected), orders_scope_rows=len(orders_scope),
                )

            # Baseline popularity of every product across the WHOLE order universe (not
            # just focal orders) — this is what "lift" is measured against.
            baseline_order_counts = products_scope.groupby('productId')['orderId'].nunique()

        except Exception as e:
            logger2.error(f"Basket Construction Error in Cross-Sell & Bundle Actionability: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Existing-Bundle Behavior (selection vs. itself) & Cross-Sell Candidates ---
        try:
            n_selected = len(selected_ids_with_history)
            pct_2plus = pct_all = None
            pairwise_rows = []
            pairwise_skipped = False

            if n_selected > 1:
                coverage = products_scope[
                    products_scope['orderId'].isin(focal_order_ids) & products_scope['productId'].isin(selected_ids_with_history)
                ]
                coverage_counts = coverage.groupby('orderId')['productId'].nunique()
                pct_2plus = (coverage_counts >= 2).mean() * 100
                pct_all = (coverage_counts == n_selected).mean() * 100

                if n_selected <= _MAX_PAIRWISE_PRODUCTS:
                    order_to_selected = coverage.groupby('orderId')['productId'].apply(set)
                    for a, b in combinations(sorted(selected_ids_with_history), 2):
                        a_orders = [s for s in order_to_selected if a in s]
                        b_orders = [s for s in order_to_selected if b in s]
                        both = sum(1 for s in order_to_selected if a in s and b in s)
                        if len(a_orders) and len(b_orders):
                            pairwise_rows.append({
                                'a_name': name_map.get(a, a), 'b_name': name_map.get(b, b),
                                'both': both,
                                'pct_of_a': both / len(a_orders) * 100,
                                'pct_of_b': both / len(b_orders) * 100,
                            })
                    pairwise_rows.sort(key=lambda r: r['both'], reverse=True)
                else:
                    pairwise_skipped = True

            other_products = products_scope[
                products_scope['orderId'].isin(focal_order_ids) & ~products_scope['productId'].isin(selected_ids)
            ].copy()

            inactive_excluded_count = other_products.loc[~other_products['productId'].isin(active_ids), 'productId'].nunique()
            other_products = other_products[other_products['productId'].isin(active_ids)]

            candidates = pd.DataFrame()
            if not other_products.empty:
                other_name_map = _canonical_names(other_products, full_cat_dict)
                candidates = other_products.groupby('productId', as_index=False).agg(
                    co_occurring_orders=('orderId', 'nunique'),
                    avg_bundle_revenue=('totalAmount', 'mean'),
                )
                candidates['detailed_name'] = candidates['productId'].map(other_name_map)
                dupe_names = candidates['detailed_name'].value_counts()
                dupe_names = set(dupe_names[dupe_names > 1].index)
                if dupe_names:
                    candidates['detailed_name'] = candidates.apply(
                        lambda r: f"{r['detailed_name']} (#{str(r['productId'])[:8]})" if r['detailed_name'] in dupe_names else r['detailed_name'],
                        axis=1
                    )
                candidates = candidates[candidates['co_occurring_orders'] >= _MIN_COOCCURRENCE_ORDERS]

                if not candidates.empty:
                    candidates['attach_rate_pct'] = candidates['co_occurring_orders'] / total_focal_orders * 100
                    candidates['baseline_rate_pct'] = candidates['productId'].map(
                        lambda pid: baseline_order_counts.get(pid, 0) / total_orders_universe * 100 if total_orders_universe else np.nan
                    )
                    candidates['lift'] = np.where(
                        candidates['baseline_rate_pct'] > 0,
                        candidates['attach_rate_pct'] / candidates['baseline_rate_pct'],
                        np.nan
                    )
                    candidates = candidates.sort_values(['lift', 'attach_rate_pct'], ascending=[False, False], na_position='last')

        except Exception as e:
            logger2.error(f"Candidate Scoring Error in Cross-Sell & Bundle Actionability: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 5. Build the Markdown Report ---
        try:
            selection_label = " + ".join(name_map.values()) if n_selected <= 3 else f"your {n_selected} selected products"
            md = ["# Cross-Sell & Bundle Actionability\n"]

            if no_history_ids:
                md.append(f"*Note: {len(no_history_ids)} selected item(s) have no order history yet and are excluded from this analysis.*\n")

            md.append(f"*Based on {total_focal_orders:,} order(s) containing {selection_label}, out of {total_orders_universe:,} total orders analyzed.*")
            if total_focal_orders < _SMALL_SAMPLE_ORDERS:
                md.append(f"*Small sample — with only {total_focal_orders} relevant order(s), treat the patterns below as directional, not conclusive.*")
            md.append("")

            if n_selected > 1:
                md.append("## Are These Already Bought Together?")
                md.append(f"- **{pct_2plus:.0f}%** of orders containing any of your selected products contain **2 or more** of them")
                md.append(f"- **{pct_all:.0f}%** contain **all {n_selected}** of them")
                if pairwise_rows:
                    md.append("\n| Pair | Orders Together | % of A's Orders | % of B's Orders |")
                    md.append("|---|---|---|---|")
                    for r in pairwise_rows[:_TOP_N_ROWS]:
                        md.append(f"| {r['a_name']} ↔ {r['b_name']} | {r['both']} | {r['pct_of_a']:.0f}% | {r['pct_of_b']:.0f}% |")
                elif pairwise_skipped:
                    md.append(f"\n*Pairwise detail omitted for {n_selected} products — showing aggregate coverage above instead.*")
                md.append(
                    "\n*High 'bought together' rates mean a bundle would mostly formalize existing behavior (low risk, modest upside). "
                    "Low rates mean a bundle would be trying to CREATE new behavior (higher upside if it works, less proven).*"
                )
                md.append("\n---\n")

            md.append("## Best Cross-Sell Candidates")
            md.append("*Other products that show up in orders with your selection more than their normal popularity would predict — limited to products still active in your catalog.*")
            if inactive_excluded_count > 0:
                md.append(f"*{inactive_excluded_count} co-occurring product(s) were excluded because they're no longer active in the catalog.*")
            if candidates.empty:
                md.append(f"\n*No cross-sell candidate cleared the minimum bar ({_MIN_COOCCURRENCE_ORDERS}+ co-occurring orders) yet — either basket size is small or this selection tends to be bought alone.*")
            else:
                shown = candidates.head(_TOP_N_ROWS)
                md.append("\n| Product | Attach Rate | Lift | Co-occurring Orders | Avg Revenue When Included |")
                md.append("|---|---|---|---|---|")
                for _, row in shown.iterrows():
                    lift_str = f"{row['lift']:.1f}×" if pd.notna(row['lift']) else "N/A"
                    flag = "" if pd.notna(row['lift']) and row['lift'] >= _STRONG_LIFT else ""
                    md.append(
                        f"| {row['detailed_name']}{flag} | {row['attach_rate_pct']:.0f}% | {lift_str} | "
                        f"{int(row['co_occurring_orders'])} | {_fmt_money(row['avg_bundle_revenue'])} |"
                    )
                if len(candidates) > _TOP_N_ROWS:
                    md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(candidates)} candidates that met the minimum co-occurrence bar.*")
                md.append(f"\n*lift ≥ {_STRONG_LIFT:.0f}× — appears with your selection at least {_STRONG_LIFT:.0f} times more often than its normal order share.*")
            md.append("\n---\n")

            md.append("## Recommendation")
            if not candidates.empty:
                top = candidates.iloc[0]
                if pd.notna(top['lift']) and top['lift'] >= 1.5 and top['attach_rate_pct'] >= 10:
                    projected_orders = total_focal_orders * (_TARGET_ATTACH_LIFT_PP / 100)
                    projected_revenue = projected_orders * top['avg_bundle_revenue']
                    md.append(
                        f"- **Prompt '{top['detailed_name']}'** alongside {selection_label} at checkout or in a bundle — "
                        f"it already appears in {top['attach_rate_pct']:.0f}% of relevant orders, "
                        f"{top['lift']:.1f}× more often than its baseline rate."
                    )
                    md.append(
                        f"- **Potential upside:** lifting its attach rate by {_TARGET_ATTACH_LIFT_PP} points across your "
                        f"{total_focal_orders:,} relevant orders is worth roughly {_fmt_money(projected_revenue)} in incremental revenue "
                        f"(rough estimate: {_TARGET_ATTACH_LIFT_PP}pp × current orders × its avg revenue per order — not adjusted for cost)."
                    )
                else:
                    md.append(
                        f"- The strongest candidate so far is **{top['detailed_name']}** (lift {top['lift']:.1f}×, "
                        f"{top['attach_rate_pct']:.0f}% attach rate) but it's not yet a strong enough signal to commit to a formal bundle — "
                        f"worth watching as more orders come in."
                    )
            else:
                md.append("- Not enough co-purchase data yet to recommend a specific cross-sell pairing.")

            if n_selected > 1 and pct_all is not None:
                if pct_all >= 50:
                    md.append(f"- Your selected products are already bought together as a full set in {pct_all:.0f}% of relevant orders — a formal bundle would mostly package up existing behavior, which is low-risk.")
                elif pct_2plus is not None and pct_2plus < 20:
                    md.append(f"- Only {pct_2plus:.0f}% of relevant orders contain 2+ of your selected products — bundling them together would be trying to create new purchase behavior, not reflect existing demand.")

            logger2.info(
                f"[Cross-Sell & Bundle Actionability] Generated OK — {n_selected} selected product(s), "
                f"{total_focal_orders} focal orders, {len(candidates)} candidate(s) found, "
                f"{len(active_ids)} active product(s) in full catalog, {inactive_excluded_count} excluded as inactive"
            )
            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Cross-Sell & Bundle Actionability: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    except Exception as e:
        logger2.error(f"Critical Error in Cross-Sell & Bundle Actionability: {str(e)}")
        return ["This report is currently unavailable due to a temporary issue. Please check back later or contact support if you need assistance."]

# --------------------------------------------------------------------------------- #
# 4. Buyer Health
# --------------------------------------------------------------------------------- #
def _generate_buyer_health_report(catalog_path, orders_path, products_path) -> list:
    """
    Customer-health analysis for the buyers of the selected product(s): who's core,
    who's new, who's drifting away, and whether they're paying cleanly.

    Segments every customer who has bought the selection into one of four buckets
    using a simple, explainable 2x2 grid (not a black-box score):
        Core     - ordered recently (<= _STALE_PRODUCT_DAYS_FLAG days ago) AND 2+ orders
        New      - ordered recently AND exactly 1 order so far
        At Risk  - hasn't ordered in a while, but used to be a repeat buyer (2+ orders)
        Lapsed   - hasn't ordered in a while and only ever ordered once
    "At Risk" is the actionable win-back list: customers who proved they'll reorder,
    then stopped — the highest-value list to chase, since Lapsed one-timers never
    demonstrated repeat intent in the first place.

    Also reports period-over-period retention (trailing _MOMENTUM_WINDOW_DAYS vs. the
    _MOMENTUM_WINDOW_DAYS before that) and payment health (how much of this customer
    base pays cleanly vs. carries an outstanding balance), so "health" covers both
    relationship strength and collections risk, not just repeat-buying.

    Args:
        products_path: path to the order line items csv (cleaned_products.csv schema).
        orders_path: path to the orders csv (cleaned_orders.csv schema).
        catalog_path: path to the catalog csv, PRE-FILTERED to the user's selected
            product(s). Every product id present in this file is treated as selected.

    Returns:
        list[str]: markdown lines forming the report, or a single-item list with a
        user-facing error/status message if the report could not be generated.
    """
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(orders_path) and os.path.exists(catalog_path)):
                return _log_empty(
                    "Buyer Health", "Error: One or more required data files (products, orders, catalog) are missing.",
                    products_path=products_path, orders_path=orders_path, catalog_path=catalog_path,
                    products_exists=os.path.exists(products_path), orders_exists=os.path.exists(orders_path),
                    catalog_exists=os.path.exists(catalog_path),
                )

            products_df = pd.read_csv(products_path)
            orders_df = pd.read_csv(orders_path)
            catalog_df = pd.read_csv(catalog_path)

            if catalog_df.empty:
                return _log_empty(
                    "Buyer Health", "Please select at least one product to generate a Buyer Health report.",
                    catalog_path=catalog_path,
                )
            if products_df.empty or orders_df.empty:
                return _log_empty(
                    "Buyer Health", "Can not generate report: there is not enough data in the provided files.",
                    products_rows=len(products_df), orders_rows=len(orders_df),
                )

        except Exception as e:
            logger2.error(f"Data Loading Error in Buyer Health: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Catalog Mapping & Selection Filtering ---
        try:
            if 'id' not in catalog_df.columns or 'productId' not in products_df.columns:
                return _log_empty(
                    "Buyer Health", "Can not generate report: Missing required columns in catalog or products data.",
                    catalog_columns=list(catalog_df.columns), products_columns=list(products_df.columns),
                )
            for col in ['orderId', 'name', 'quantity', 'amount', 'totalAmount']:
                if col not in products_df.columns:
                    return _log_empty(
                        "Buyer Health", f"Can not generate report: Missing required column '{col}' in products data.",
                        products_columns=list(products_df.columns),
                    )

            catalog_cols = [c for c in ['size', 'color', 'name', 'sku'] if c in catalog_df.columns]
            cat_dict = catalog_df.set_index('id')[catalog_cols].to_dict('index')
            selected_ids = set(catalog_df['id'].dropna().unique())

            df_selected = products_df[products_df['productId'].isin(selected_ids)].copy()
            if df_selected.empty:
                return _log_empty(
                    "Buyer Health", "Can not generate report: the selected product(s) have no order history yet.",
                    selected_ids=sorted(selected_ids), products_rows_total=len(products_df),
                )

            name_map = _canonical_names(df_selected, cat_dict)
            no_history_ids = selected_ids - set(df_selected['productId'].unique())
            n_selected = len(selected_ids) - len(no_history_ids)
            selection_label = " + ".join(name_map.values()) if n_selected <= 3 else f"your {n_selected} selected products"

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Buyer Health: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Merge with Orders & Build Customer-Level Aggregates ---
        try:
            required_order_cols = {'id', 'customer_name', 'paymentStatus'}
            if not required_order_cols.issubset(orders_df.columns):
                return _log_empty(
                    "Buyer Health", "Can not generate report: Missing required columns in orders data.",
                    required=sorted(required_order_cols), missing=sorted(required_order_cols - set(orders_df.columns)),
                    orders_columns=list(orders_df.columns),
                )

            orders_scope = orders_df[orders_df['archived'] == False] if 'archived' in orders_df.columns else orders_df
            orders_scope = orders_scope.copy()
            orders_scope['_created_dt'] = _parse_created_at(orders_scope['createdAt']) if 'createdAt' in orders_scope.columns else pd.NaT

            merged = pd.merge(
                df_selected,
                orders_scope[['id', 'customer_name', 'paymentStatus', '_created_dt']],
                left_on='orderId', right_on='id', how='inner', suffixes=('', '_order')
            )
            if merged.empty:
                return _log_empty(
                    "Buyer Health", "Can not generate report: No matching order history for the selected product(s).",
                    df_selected_rows=len(df_selected), orders_scope_rows=len(orders_scope),
                )

            merged['net_billed'] = merged['amount'].fillna(0) * merged['quantity']
            merged['refunded_amount'] = np.where(merged['paymentStatus'] == 'REFUNDED', merged['totalAmount'], 0)
            as_of_date = orders_scope['_created_dt'].max()

            # Payment-status counts must be per unique ORDER, not per line item — an
            # order with 4 selected-product lines would otherwise get counted 4x
            # against a single payment status, badly inflating the percentages.
            order_level = merged.drop_duplicates(subset='orderId')[['customer_name', 'orderId', 'paymentStatus']]
            payment_counts = order_level.groupby('customer_name', as_index=False).agg(
                paid_orders=('paymentStatus', lambda s: (s == 'PAID').sum()),
                pending_orders=('paymentStatus', lambda s: (s == 'PENDING').sum()),
                partial_orders=('paymentStatus', lambda s: (s == 'PARTIALLY_PAID').sum()),
                refunded_orders=('paymentStatus', lambda s: (s == 'REFUNDED').sum()),
            )

            customers = merged.groupby('customer_name', as_index=False).agg(
                orders=('orderId', 'nunique'),
                net_billed=('net_billed', 'sum'),
                collected_gross=('totalAmount', 'sum'),
                refunded=('refunded_amount', 'sum'),
                last_order=('_created_dt', 'max'),
                first_order=('_created_dt', 'min'),
            )
            customers = customers.merge(payment_counts, on='customer_name', how='left')
            customers['net_collected'] = customers['collected_gross'] - customers['refunded']
            customers['outstanding'] = (customers['net_billed'] - customers['collected_gross']).clip(lower=0)
            customers['recency_days'] = (as_of_date - customers['last_order']).dt.days if pd.notna(as_of_date) else np.nan
            total_customers = len(customers)

        except Exception as e:
            logger2.error(f"Metrics Calculation Error in Buyer Health: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Segmentation, Retention & Payment Health ---
        try:
            def _segment(row):
                recent = pd.notna(row['recency_days']) and row['recency_days'] <= _STALE_PRODUCT_DAYS_FLAG
                repeat = row['orders'] >= 2
                if recent and repeat:
                    return 'Core'
                if recent and not repeat:
                    return 'New'
                if not recent and repeat:
                    return 'At Risk'
                return 'Lapsed'

            customers['segment'] = customers.apply(_segment, axis=1)
            segment_summary = customers.groupby('segment', as_index=False).agg(
                customers=('customer_name', 'nunique'), revenue=('net_collected', 'sum')
            )
            total_revenue = customers['net_collected'].sum()

            # Retention: customers active in the prior window vs. the most recent window.
            retention_rate = new_customers_recent = churned_from_prior = None
            if pd.notna(as_of_date):
                recent_start = as_of_date - pd.Timedelta(days=_MOMENTUM_WINDOW_DAYS)
                prior_start = as_of_date - pd.Timedelta(days=2 * _MOMENTUM_WINDOW_DAYS)
                recent_mask = merged['_created_dt'] > recent_start
                prior_mask = (merged['_created_dt'] > prior_start) & (merged['_created_dt'] <= recent_start)
                recent_customers = set(merged.loc[recent_mask, 'customer_name'])
                prior_customers = set(merged.loc[prior_mask, 'customer_name'])
                if prior_customers:
                    retained = recent_customers & prior_customers
                    retention_rate = len(retained) / len(prior_customers) * 100
                    churned_from_prior = len(prior_customers - recent_customers)
                new_customers_recent = len(recent_customers - prior_customers)

            total_orders_all = customers['orders'].sum()
            pct_paid = customers['paid_orders'].sum() / total_orders_all * 100 if total_orders_all else 0
            pct_pending = customers['pending_orders'].sum() / total_orders_all * 100 if total_orders_all else 0
            pct_partial = customers['partial_orders'].sum() / total_orders_all * 100 if total_orders_all else 0
            pct_refunded_orders = customers['refunded_orders'].sum() / total_orders_all * 100 if total_orders_all else 0
            total_outstanding = customers['outstanding'].sum()
            customers_with_balance = int((customers['outstanding'] > 0).sum())

            at_risk = customers[customers['segment'] == 'At Risk'].sort_values('net_collected', ascending=False)
            collections_priority = customers[customers['outstanding'] > 0].sort_values('outstanding', ascending=False)

        except Exception as e:
            logger2.error(f"Segmentation Error in Buyer Health: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 5. Build the Markdown Report ---
        try:
            md = ["# Buyer Health\n"]

            if pd.notna(as_of_date):
                md.append(f"*Data runs through {as_of_date.strftime('%b %d, %Y')} (most recent order in your dataset). All \"days since\" stats are measured against that date.*\n")

            if no_history_ids:
                md.append(f"*Note: {len(no_history_ids)} selected item(s) have no order history yet and are excluded from this analysis.*\n")

            md.append(f"*Based on {total_customers} customer(s) who have bought {selection_label}.*")
            if total_customers < _SMALL_SAMPLE_ORDERS:
                md.append(f"*Small sample — with only {total_customers} customer(s), treat the segmentation below as directional.*")
            md.append("")

            md.append("## Customer Segments")
            md.append("*Core = recent + repeat. New = recent, first order. At Risk = used to reorder, gone quiet. Lapsed = one-time, gone quiet.*")
            md.append("| Segment | Customers | % of Base | Revenue | % of Revenue |")
            md.append("|---|---|---|---|---|")
            for seg in ['Core', 'New', 'At Risk', 'Lapsed']:
                seg_row = segment_summary[segment_summary['segment'] == seg]
                if len(seg_row):
                    r = seg_row.iloc[0]
                    rev_share = f"{r['revenue']/total_revenue*100:.0f}%" if total_revenue else "N/A"
                    md.append(f"| {seg} | {int(r['customers'])} | {r['customers']/total_customers*100:.0f}% | {_fmt_money(r['revenue'])} | {rev_share} |")
            md.append("\n---\n")

            md.append("## Retention & Churn")
            if retention_rate is not None:
                md.append(f"- **Retention:** {retention_rate:.0f}% of customers active {_MOMENTUM_WINDOW_DAYS}-{2*_MOMENTUM_WINDOW_DAYS} days before the data's end date also ordered in the most recent {_MOMENTUM_WINDOW_DAYS} days")
                md.append(f"- **New/Reactivated:** {new_customers_recent} customer(s) ordered in the most recent {_MOMENTUM_WINDOW_DAYS} days who hadn't before that")
                md.append(f"- **Went Quiet:** {churned_from_prior} customer(s) active {_MOMENTUM_WINDOW_DAYS}-{2*_MOMENTUM_WINDOW_DAYS} days before the data's end date haven't ordered since")
            else:
                md.append("*Not enough dated order history to compute period-over-period retention.*")
            md.append("\n---\n")

            md.append("## Payment Health")
            md.append(f"- **Order Mix:** {pct_paid:.0f}% paid, {pct_pending:.0f}% pending, {pct_partial:.0f}% partially paid, {pct_refunded_orders:.0f}% refunded")
            md.append(f"- **Accounts Receivable Balance:** {_fmt_money(total_outstanding)} across {customers_with_balance} customer(s)")
            if not collections_priority.empty:
                shown = collections_priority.head(_TOP_N_ROWS)
                md.append("\n| Customer | Accounts Receivable | Orders | Segment |")
                md.append("|---|---|---|---|")
                for _, row in shown.iterrows():
                    md.append(f"| {_truncate(row['customer_name'])} | {_fmt_money(row['outstanding'])} | {int(row['orders'])} | {row['segment']} |")
                if len(collections_priority) > _TOP_N_ROWS:
                    md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(collections_priority)} customers with an outstanding balance.*")
            md.append("\n---\n")

            md.append("## At-Risk Customers (Win-Back Priority)")
            md.append("*Proved they'll reorder, then went quiet — the highest-value win-back list.*")
            if at_risk.empty:
                md.append("\n*No customers currently fall into the At Risk segment.*")
            else:
                shown = at_risk.head(_TOP_N_ROWS)
                md.append("\n| Customer | Lifetime Revenue | Orders | Days Since Last Order |")
                md.append("|---|---|---|---|")
                for _, row in shown.iterrows():
                    days_str = str(int(row['recency_days'])) if pd.notna(row['recency_days']) else 'N/A'
                    md.append(f"| {_truncate(row['customer_name'])} | {_fmt_money(row['net_collected'])} | {int(row['orders'])} | {days_str} |")
                if len(at_risk) > _TOP_N_ROWS:
                    md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(at_risk)} at-risk customers by lifetime revenue.*")

            md.append("\n### Key Takeaways")
            takeaways = []
            at_risk_revenue = segment_summary.loc[segment_summary['segment'] == 'At Risk', 'revenue'].sum() if (segment_summary['segment'] == 'At Risk').any() else 0
            if at_risk_revenue > 0:
                takeaways.append(f"- **{_fmt_money(at_risk_revenue)}** in historical revenue sits with At Risk customers — prioritize win-back outreach to the list above.")
            if retention_rate is not None and retention_rate < 40:
                takeaways.append(f"- **Low retention:** only {retention_rate:.0f}% of previously-active customers are still ordering — worth investigating why.")
            if total_outstanding > 0 and total_revenue > 0 and (total_outstanding / total_revenue * 100) > _OUTSTANDING_REVENUE_FLAG_PCT:
                takeaways.append(f"- **Collections:** {_fmt_money(total_outstanding)} outstanding across {customers_with_balance} customers — meaningful relative to total revenue.")
            lapsed_count = segment_summary.loc[segment_summary['segment'] == 'Lapsed', 'customers'].sum() if (segment_summary['segment'] == 'Lapsed').any() else 0
            if total_customers and lapsed_count / total_customers > 0.4:
                takeaways.append(f"- **High one-and-done rate:** {int(lapsed_count)} of {total_customers} customers ({lapsed_count/total_customers*100:.0f}%) tried this once and never came back.")
            if not takeaways:
                takeaways.append("- No major risk flags detected — this buyer base looks healthy on the available signals.")
            md.extend(takeaways)

            at_risk_n = int(segment_summary.loc[segment_summary['segment'] == 'At Risk', 'customers'].sum()) if (segment_summary['segment'] == 'At Risk').any() else 0
            logger2.info(
                f"[Buyer Health] Generated OK — {total_customers} customer(s), {at_risk_n} at risk, "
                f"outstanding={total_outstanding:.2f}"
            )
            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Buyer Health: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    except Exception as e:
        logger2.error(f"Critical Error in Buyer Health: {str(e)}")
        return ["This report is currently unavailable due to a temporary issue. Please check back later or contact support if you need assistance."]

# --------------------------------------------------------------------------------- #
# 5. Inventory Health & Fulfillment Efficiency
# --------------------------------------------------------------------------------- #
def _generate_inventory_fulfillment_report(catalog_path, orders_path, products_path) -> list:
    """
    Two questions for the selected product(s): is there enough stock positioned
    correctly, and are orders for it actually getting shipped in good time?

    Inventory Health (per product, portfolio total if multiple selected):
    - Available to sell = inventory_onHand - inventory_allocated (standard
      available-to-promise; onHand alone overstates what's actually sellable if
      units are already reserved against open orders).
    - Monthly velocity & runway, same methodology as the deep dive's sell-through
      section, so the two reports agree with each other.
    - Distinguishes real stockouts from intentional backorder selling: if the
      catalog's `sellingOutOfStock` is True, a zero/negative available balance is
      treated as "backorder mode," not a blocking stockout.
    - Inventory value uses `wholesalePrice` (list price) as a proxy, same caveat as
      the Revenue & Profitability report: there's no true unit-cost field in this
      catalog, so this is "capital exposure at list price," not true COGS tied up.

    Fulfillment Efficiency (across orders containing the selection):
    - Uses `deliveryStatus` as the primary signal (FULFILLED/PARTIALLY_FULFILLED/
      UNFULFILLED) rather than the line-level `delivered` quantity — cross-checked
      against real data, deliveryStatus is highly consistent (100% of FULFILLED
      order lines show delivered >= quantity) while delivered-vs-quantity alone is
      noisier to interpret in isolation.
    - Time-to-ship is reported as a median (with the mean shown alongside) because
      lead times in this kind of data are typically right-skewed by a long tail of
      delayed orders — a straight average would overstate the typical experience.
    - Surfaces the actual backlog: currently unfulfilled orders containing the
      selection, oldest first, capped at _TOP_N_ROWS — the concrete "needs
      attention now" list, not just an aggregate rate.

    Args:
        products_path: path to the order line items csv (cleaned_products.csv schema).
        orders_path: path to the orders csv (cleaned_orders.csv schema).
        catalog_path: path to the catalog csv, PRE-FILTERED to the user's selected
            product(s). Every product id present in this file is treated as selected.

    Returns:
        list[str]: markdown lines forming the report, or a single-item list with a
        user-facing error/status message if the report could not be generated.
    """
    try:
        # --- 1. Load Data & Validate ---
        try:
            if not (os.path.exists(products_path) and os.path.exists(orders_path) and os.path.exists(catalog_path)):
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Error: One or more required data files (products, orders, catalog) are missing.",
                    products_path=products_path, orders_path=orders_path, catalog_path=catalog_path,
                    products_exists=os.path.exists(products_path), orders_exists=os.path.exists(orders_path),
                    catalog_exists=os.path.exists(catalog_path),
                )

            products_df = pd.read_csv(products_path)
            orders_df = pd.read_csv(orders_path)
            catalog_df = pd.read_csv(catalog_path)

            if catalog_df.empty:
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Please select at least one product to generate this report.",
                    catalog_path=catalog_path,
                )
            if products_df.empty or orders_df.empty:
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Can not generate report: there is not enough data in the provided files.",
                    products_rows=len(products_df), orders_rows=len(orders_df),
                )

        except Exception as e:
            logger2.error(f"Data Loading Error in Inventory Health & Fulfillment Efficiency: {str(e)}")
            return ["This report is currently unavailable due to a data loading error. Please check back later or contact support."]

        # --- 2. Catalog Mapping & Selection Filtering ---
        try:
            if 'id' not in catalog_df.columns or 'productId' not in products_df.columns:
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Can not generate report: Missing required columns in catalog or products data.",
                    catalog_columns=list(catalog_df.columns), products_columns=list(products_df.columns),
                )
            for col in ['orderId', 'name', 'quantity', 'totalAmount']:
                if col not in products_df.columns:
                    return _log_empty(
                        "Inventory Health & Fulfillment Efficiency", f"Can not generate report: Missing required column '{col}' in products data.",
                        products_columns=list(products_df.columns),
                    )

            inv_cols = ['size', 'color', 'name', 'sku', 'inventory_onHand', 'inventory_expected',
                        'inventory_allocated', 'wholesalePrice', 'sellingOutOfStock', 'parentProductId']
            inv_cols = [c for c in inv_cols if c in catalog_df.columns]
            cat_dict = catalog_df.set_index('id')[inv_cols].to_dict('index')
            selected_ids = set(catalog_df['id'].dropna().unique())

            df_selected = products_df[products_df['productId'].isin(selected_ids)].copy()
            if df_selected.empty:
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Can not generate report: the selected product(s) have no order history yet.",
                    selected_ids=sorted(selected_ids), products_rows_total=len(products_df),
                )

            name_map = _canonical_names(df_selected, cat_dict)
            no_history_ids = selected_ids - set(df_selected['productId'].unique())

        except Exception as e:
            logger2.error(f"Catalog Mapping Error in Inventory Health & Fulfillment Efficiency: {str(e)}")
            return ["This report is currently unavailable. Please check back later or contact support."]

        # --- 3. Merge with Orders & Calculate Inventory + Fulfillment Metrics ---
        try:
            required_order_cols = {'id', 'customer_name'}
            if not required_order_cols.issubset(orders_df.columns):
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Can not generate report: Missing required columns in orders data.",
                    required=sorted(required_order_cols), missing=sorted(required_order_cols - set(orders_df.columns)),
                    orders_columns=list(orders_df.columns),
                )

            orders_scope = orders_df[orders_df['archived'] == False] if 'archived' in orders_df.columns else orders_df
            orders_scope = orders_scope.copy()
            orders_scope['month'] = _derive_month_column(orders_scope)
            has_dates = 'createdAt' in orders_scope.columns
            if has_dates:
                orders_scope['_created_dt'] = _parse_created_at(orders_scope['createdAt'])
            has_ship_dates = has_dates and 'shippedAt' in orders_scope.columns
            if has_ship_dates:
                orders_scope['_shipped_dt'] = _parse_created_at(orders_scope['shippedAt'])
            has_delivery_status = 'deliveryStatus' in orders_scope.columns

            order_cols = ['id', 'customer_name', 'month']
            if has_dates:
                order_cols.append('_created_dt')
            if has_ship_dates:
                order_cols.append('_shipped_dt')
            if has_delivery_status:
                order_cols.append('deliveryStatus')

            merged = pd.merge(df_selected, orders_scope[order_cols], left_on='orderId', right_on='id', how='inner', suffixes=('', '_order'))
            if merged.empty:
                return _log_empty(
                    "Inventory Health & Fulfillment Efficiency", "Can not generate report: No matching order history for the selected product(s).",
                    df_selected_rows=len(df_selected), orders_scope_rows=len(orders_scope),
                )

            as_of_date = orders_scope['_created_dt'].max() if has_dates else None

            # --- Inventory: monthly velocity & runway per product ---
            inv_rows = []
            for pid in [i for i in selected_ids if i not in no_history_ids]:
                p_lines = merged[merged['productId'] == pid]
                cat_item = cat_dict.get(pid, {})
                on_hand = cat_item.get('inventory_onHand', 0)
                on_hand = float(on_hand) if pd.notna(on_hand) else 0.0
                allocated = cat_item.get('inventory_allocated', 0)
                allocated = float(allocated) if pd.notna(allocated) else 0.0
                expected = cat_item.get('inventory_expected', 0)
                expected = float(expected) if pd.notna(expected) else 0.0
                available = on_hand - allocated
                sells_oos_raw = _resolve_via_parent(cat_dict, pid, 'sellingOutOfStock', False)
                sells_oos = str(sells_oos_raw).strip().lower() == 'true' if not isinstance(sells_oos_raw, bool) else sells_oos_raw
                list_price = cat_item.get('wholesalePrice', np.nan)

                trend = p_lines.groupby('month')['quantity'].sum()
                months_active = trend[trend.index != 'Unknown'].shape[0] if len(trend) else 0
                total_units = p_lines['quantity'].sum()
                monthly_velocity = (total_units / months_active) if months_active else 0
                runway_months = (available / monthly_velocity) if monthly_velocity > 0 and available > 0 else None

                inv_rows.append({
                    'productId': pid, 'name': name_map.get(pid, pid),
                    'on_hand': on_hand, 'allocated': allocated, 'available': available, 'expected': expected,
                    'inventory_value': on_hand * list_price if pd.notna(list_price) else np.nan,
                    'monthly_velocity': monthly_velocity, 'runway_months': runway_months,
                    'sells_oos': sells_oos,
                })
            inventory = pd.DataFrame(inv_rows)

            # --- Fulfillment: funnel, lead time, and backlog ---
            order_level = merged.drop_duplicates(subset='orderId').copy()
            total_relevant_orders = len(order_level)
            funnel = order_level['deliveryStatus'].value_counts(normalize=True) * 100 if has_delivery_status else None

            lead_days = None
            if has_ship_dates:
                lead_days = (order_level['_shipped_dt'] - order_level['_created_dt']).dt.days
                lead_days = lead_days[lead_days >= 0]  # drop the rare negative-lead data artifact rather than let it skew stats

            backlog = pd.DataFrame()
            if has_delivery_status and has_dates:
                open_mask = order_level['deliveryStatus'].isin(['UNFULFILLED', 'PARTIALLY_FULFILLED'])
                backlog = order_level[open_mask].copy()
                if not backlog.empty:
                    backlog['days_open'] = (as_of_date - backlog['_created_dt']).dt.days
                    order_value = merged.groupby('orderId')['totalAmount'].sum()
                    backlog['order_value'] = backlog['id_order'].map(order_value)
                    backlog = backlog.sort_values('days_open', ascending=False)

        except Exception as e:
            logger2.error(f"Metrics Calculation Error in Inventory Health & Fulfillment Efficiency: {str(e)}")
            return ["This report is currently unavailable due to a calculation error. Please contact support."]

        # --- 4. Build the Markdown Report ---
        try:
            n_products = len(inventory)
            md = ["# Inventory Health & Fulfillment Efficiency\n"]

            if pd.notna(as_of_date) if as_of_date is not None else False:
                md.append(f"*Data runs through {as_of_date.strftime('%b %d, %Y')} (most recent order in your dataset).*\n")

            if no_history_ids:
                md.append(f"*Note: {len(no_history_ids)} selected item(s) have no order history yet and are excluded from this analysis.*\n")

            md.append("## Inventory Position")
            md.append("*Available = on hand minus units already allocated to open orders. Inventory value uses list price as a proxy — there's no true unit-cost field in this catalog, so treat it as capital exposure, not COGS.*")
            md.append("| Product | On Hand | Allocated | Available | Incoming | Value | Monthly Velocity | Runway |")
            md.append("|---|---|---|---|---|---|---|---|")
            for _, row in inventory.iterrows():
                if row['available'] <= 0:
                    runway_str = "backorder mode" if row['sells_oos'] else "OUT OF STOCK"
                elif row['runway_months'] is not None:
                    runway_str = f"~{row['runway_months']:.1f} mo."
                else:
                    runway_str = "N/A"
                md.append(
                    f"| **{row['name']}** | {row['on_hand']:,.0f} | {row['allocated']:,.0f} | {row['available']:,.0f} | "
                    f"{row['expected']:,.0f} | {_fmt_money(row['inventory_value'])} | {row['monthly_velocity']:.1f} units/mo | {runway_str} |"
                )
            if n_products > 1:
                md.append(
                    f"| **PORTFOLIO TOTAL** | {inventory['on_hand'].sum():,.0f} | {inventory['allocated'].sum():,.0f} | "
                    f"{inventory['available'].sum():,.0f} | {inventory['expected'].sum():,.0f} | "
                    f"{_fmt_money(inventory['inventory_value'].sum())} | — | — |"
                )
            md.append("\n---\n")

            md.append("## Fulfillment Performance")
            md.append(f"*Across {total_relevant_orders:,} order(s) containing the selection.*")
            if funnel is not None:
                funnel_str = ", ".join(f"{v:.0f}% {k.replace('_',' ').title()}" for k, v in funnel.items())
                md.append(f"- **Funnel:** {funnel_str}")
            else:
                md.append("- *No delivery status data available.*")
            if lead_days is not None and len(lead_days):
                within_sla = (lead_days <= _FULFILLMENT_SLA_DAYS).mean() * 100
                md.append(f"- **Time to Ship:** median {lead_days.median():.0f} day(s) (mean {lead_days.mean():.1f} — a long tail of delayed orders pulls the average up, median is more representative)")
                md.append(f"- **Within {_FULFILLMENT_SLA_DAYS}-Day Target:** {within_sla:.0f}% of shipped orders")
            else:
                md.append("- *No shipping date data available to measure lead time.*")
            md.append("\n---\n")

            md.append("## Fulfillment Backlog (Needs Attention)")
            if backlog.empty:
                md.append("*No open (unfulfilled/partially fulfilled) orders for the selection right now.*")
            else:
                shown = backlog.head(_TOP_N_ROWS)
                md.append("| Customer | Status | Days Open | Order Value |")
                md.append("|---|---|---|---|")
                for _, row in shown.iterrows():
                    flag = " ⚠️" if row['days_open'] >= _BACKLOG_STALE_DAYS_FLAG else ""
                    md.append(f"| {_truncate(row['customer_name'])} | {row['deliveryStatus'].replace('_',' ').title()} | {int(row['days_open'])}{flag} | {_fmt_money(row['order_value'])} |")
                if len(backlog) > _TOP_N_ROWS:
                    md.append(f"\n*Showing top {_TOP_N_ROWS} of {len(backlog)} open orders, oldest first.*")
                md.append(f"\n*⚠️ = open {_BACKLOG_STALE_DAYS_FLAG}+ days.*")

            md.append("\n### Key Takeaways")
            takeaways = []
            for _, row in inventory.iterrows():
                if row['available'] <= 0 and not row['sells_oos']:
                    takeaways.append(f"- **{row['name']}** is out of stock and not set up for backorders — active demand may be going unmet.")
                elif row['runway_months'] is not None and row['runway_months'] < _LOW_STOCK_MONTHS_FLAG:
                    takeaways.append(f"- **{row['name']}** has ~{row['runway_months']:.1f} months of runway left — reorder soon.")
            if not backlog.empty:
                stale_count = int((backlog['days_open'] >= _BACKLOG_STALE_DAYS_FLAG).sum())
                if stale_count:
                    takeaways.append(f"- **{stale_count} order(s)** containing this selection have been unfulfilled for {_BACKLOG_STALE_DAYS_FLAG}+ days — worth a fulfillment-team check-in.")
            if lead_days is not None and len(lead_days) and lead_days.median() > _FULFILLMENT_SLA_DAYS:
                takeaways.append(f"- **Typical fulfillment is slower than the {_FULFILLMENT_SLA_DAYS}-day target** (median {lead_days.median():.0f} days) — worth investigating the fulfillment pipeline for this selection.")
            if not takeaways:
                takeaways.append("- No major inventory or fulfillment risk flags detected for the selected product(s).")
            md.extend(takeaways)

            logger2.info(
                f"[Inventory Health & Fulfillment Efficiency] Generated OK — {n_products} product(s), "
                f"{total_relevant_orders} relevant orders, {len(backlog)} open backlog order(s)"
            )
            return md

        except Exception as e:
            logger2.error(f"Markdown Generation Error in Inventory Health & Fulfillment Efficiency: {str(e)}")
            return ["An error occurred while compiling the final report formatting. Please try again later."]

    except Exception as e:
        logger2.error(f"Critical Error in Inventory Health & Fulfillment Efficiency: {str(e)}")
        return ["This report is currently unavailable due to a temporary issue. Please check back later or contact support if you need assistance."]
    

AGENT_CONFIG = {
    "catalog_agent": {
        "revenue_profitability": _generate_revenue_profitability_report,
        "top_performers": _generate_top_performer_deep_dive_report,
        "cross_sell_bundling": _generate_cross_sell_bundle_report,
        "buyer_health": _generate_buyer_health_report,
        "inventory_fulfillment": _generate_inventory_fulfillment_report
    }
}
import asyncio
import inspect
import time
# Main Async Report Generator
async def group_catalog_statistics(
    catalog_path, 
    orders_path,
    products_path,
    full_catalog_path,
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
        if 'catalog_path' in sig.parameters: kwargs['catalog_path'] = catalog_path
        if 'orders_path' in sig.parameters: kwargs['orders_path'] = orders_path
        if 'products_path' in sig.parameters: kwargs['products_path'] = products_path
        if 'full_catalog_path' in sig.parameters: kwargs['full_catalog_path'] = full_catalog_path

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



async def create_agent_sectioned(USER_ID, topic, statistics, agent) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        PROMPT_DEFAULT = "You are a helpful assistant. Answer the user's query based on the data provided."
        topic_instruction_map = {
            "catalog_agent": prompt_catalog_grouped,
        }

        # 3. Select the correct instruction
        selected_instruction = topic_instruction_map.get(agent, PROMPT_DEFAULT)
        statistics_topic = statistics.get(topic, "No statistics available for this topic.")

        instructions = await selected_instruction(USER_ID, topic, statistics_topic)

        agent = Agent(
            name="Catalog_Assistant",
            instructions=instructions,
            model=llm_model
        )
        print(" New create_agent_sectioned are ready.")
    except Exception as e:
        print("create_agent_sectioned error: ", e)
    return agent

async def create_agent_suggestions(USER_ID) -> Agent:
    """Initializes a new Orders agent and session."""

    try:
        instructions = await prompt_mcp_suggestions(USER_ID)

        agent = Agent(
            name="Catalog_Assistant",
            instructions=instructions,
            model=llm_model,
            tools=[get_prepared_statistics]
        )
        print(" New create_agent_suggestions are ready.")
    except Exception as e:
        print("create_agent_suggestions error: ", e)
    return agent

async def process_standard_topic(topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent):
    """Logic for standard analysis topics."""
    try:
        start = time.perf_counter()
        
        # 1. Generate Statistics
        statistics_dict = await group_catalog_statistics(
            catalog_path=catalog_path, orders_path=orders_path, products_path=products_path, full_catalog_path=full_catalog_path, agent_type=agent, report_type=topic
        )
        
        # Safely extract the section
        statistics = statistics_dict.get('sections', {})
        if isinstance(statistics, str):
            statistics_topic = statistics
        else:
            statistics_topic = statistics.get(topic, "No statistics available for this topic.")
        
        # 2. CHECK FOR BROKEN/EMPTY STATISTICS
        error_phrases = [
            "currently unavailable",
            "not enough data",
            "can not generate report",
            "no statistics generated",
            "no statistics available"
        ]
        
        is_broken = False
        if isinstance(statistics_topic, str):
            stat_lower = statistics_topic.lower()
            if any(phrase in stat_lower for phrase in error_phrases):
                is_broken = True

        # 3. Conditional AI Analysis & Early Exit
        if is_broken:
            print(f"Topic '{topic}': Skipping AI analysis. Returning error message directly to user.")
            # Return the predefined error message directly so the front-end displays it
            return statistics_topic

        # --- Everything below this line ONLY runs if statistics are valid ---

        # 4. Generate AI Analysis
        agent = await create_agent_sectioned(uuid, topic, statistics, agent)
        runner = await Runner.run(
            agent, 
            input="Analyze my data and write useful tips for business"
        )
        answer = runner.final_output
        calculate_cost(runner, model=model)

        # 5. Combine Sections
        sectioned_answer = await combine_sections(topic, statistics_topic, answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        
        if isinstance(sectioned_answer, dict):
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        
        return sectioned_answer

    except Exception as e:
        logger2.error(f"Error in standard topic '{topic}': {e}")
        return None

async def process_suggestions_topic(topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent):
    """Logic for suggestions analysis topics."""
    try:
        start = time.perf_counter()

        # 1. Generate Statistics
        statistics_of_topic = await group_catalog_statistics(
            catalog_path=catalog_path, orders_path=orders_path, products_path=products_path, full_catalog_path=full_catalog_path, agent_type=agent, report_type='full_report'
        )

        async with aiofiles.open(f"data/{uuid}/full_report.md", "w", encoding="utf-8") as f:
                        await f.write(statistics_of_topic.get('full_report', '') )

        agent = await create_agent_suggestions(uuid)
        runner = await Runner.run(
            agent, 
            input="Analyze my data and write useful tips for business"
        )

        answer = runner.final_output
        answer = f"<div id=\"suggestions-block\">\n\n{answer}\n</div>"
        #print(answer)
        print(f"Topic {topic}", time.perf_counter() - start)
        calculate_cost(runner, model=model)

        sectioned_answer = {'suggestions_div' : answer}
        if isinstance(sectioned_answer, dict):
            # Try to get the value using the topic as key, otherwise take the first value
            return sectioned_answer.get(topic, list(sectioned_answer.values())[0])
        return sectioned_answer

    except Exception as e:
        logger2.error(f"Error in suggestions topic '{topic}': {e}")
        return None


async def worker(semaphore, topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent):
    """
    Router function: Decides which logic to run based on the topic name,
    constrained by the semaphore.
    """
    async with semaphore:
        #print(f"Processing: {topic}")
        
        if topic == "suggestions_div":
            return await process_suggestions_topic(topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent)
        else:
            return await process_standard_topic(topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent)


async def main_batch_catalog_process(
    catalog_path,
    orders_path, 
    products_path, 
    full_catalog_path,
    uuid, 
    agent, 
    specific_topic=None
):
    try:
        # Changed print to logger for consistency

        TOPIC_CONFIG = {
            "catalog_agent": [
                "revenue_profitability",
                "top_performers",
                "cross_sell_bundling",
                "buyer_health",
                "inventory_fulfillment",
                "suggestions_div"
            ] 
        }

        
        # 1. Override the topics list if a specific topic is requested
        all_agent_topics = TOPIC_CONFIG.get(agent, [])
        
        if specific_topic:
            if specific_topic in all_agent_topics:
                topics = [specific_topic]
            else:
                logger2.warning(f"Topic '{specific_topic}' is not valid for agent '{agent}'.")
                friendly_msg = f"The requested report topic '{specific_topic}' is currently unavailable for this agent."
                return friendly_msg, {"error": friendly_msg}
        else:
            topics = all_agent_topics

        if not topics:
            logger2.warning(f"No topics found for agent '{agent}'.")
            return "No reports available for this configuration.", {}

        # Limit concurrency to 10
        sem = asyncio.Semaphore(10)
        tasks = []
        for topic in topics:
            task = asyncio.create_task(
                worker(sem, topic, catalog_path, orders_path, products_path, full_catalog_path, uuid, agent)
            )
            tasks.append(task)

        logger2.info(f"Starting {len(topics)} topics...")
        
        # return_exceptions=True prevents gather from crashing if a single worker fails
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sectioned_report = {}
        for topic, result in zip(topics, results):
            if isinstance(result, Exception):
                # Log the actual technical error with full traceback
                logger2.error(f"Worker failed for topic '{topic}' (Agent: {agent}): {str(result)}", exc_info=result)
                
                # Assign a user-friendly message for this specific section
                clean_topic_name = topic.replace('_', ' ').title()
                sectioned_report[topic] = f"\n> **Notice:** We encountered an issue while generating the {clean_topic_name}. Please try again later.\n"
            else:
                #print(f"Worker succeeded for topic '{topic}' (Agent: {agent}): {str(result)}")
                sectioned_report[topic] = result
        
        # Compile the Report
        report_parts = []
        for key in topics:
            content = sectioned_report.get(key)
            if content:
                report_parts.append(str(content))
        
        def clean_markdown(text: str) -> str:
            if not text:
                return ""
            return text.replace('\n---\n', '\n\n').replace('\n---', '')

        raw_full_report = "\n".join(report_parts).strip()
        final_clean_report = await asyncio.to_thread(clean_markdown, raw_full_report)
        
        # Clean sections, ensuring we convert to string just in case
        clean_sections = await asyncio.to_thread(
            lambda: {k: clean_markdown(str(v)) for k, v in sectioned_report.items()}
        )
        #print(final_clean_report)
        return final_clean_report, clean_sections

    except Exception as e:
        # Global fallback for unexpected errors (e.g., missing files, memory issues, dict errors)
        logger2.error(f"Critical error in main_batch_catalog_process for UUID {uuid}: {str(e)}", exc_info=True)
        
        user_friendly_error = (
            "We encountered an unexpected error while generating your complete report. "
            "Our team has been notified. Please try again shortly."
        )
        return user_friendly_error, {"error": user_friendly_error}








async def main():
    #report = await group_catalog_statistics(
    #    products_path = "data/FULL_DIST_TEST/cleaned_products.csv",
    #    catalog_path = "data/f70070d6-6869-4544-99d7-539f40d7c70b/work_data_folder/raw_file_catalog.csv",
    #    orders_path = "data/FULL_DIST_TEST/cleaned_orders.csv",
    #    full_catalog_path = "data/FULL_DIST_TEST/cleaned_catalog.csv",
    #    agent_type="catalog_agent",
    #    report_type="inventory_fulfillment"
    #)
    #print(report.get("sections", "No full report generated.").get("inventory_fulfillment", "Report section not found."))
    report, sections = await main_batch_catalog_process(
        catalog_path="data\\f70070d6-6869-4544-99d7-539f40d7c70b\\work_data_folder\\raw_file_catalog.csv",
        orders_path="data\\FULL_DIST_TEST\\cleaned_orders.csv",
        products_path="data\\FULL_DIST_TEST\\cleaned_products.csv",
        full_catalog_path="data\\FULL_DIST_TEST\\cleaned_catalog.csv",
        uuid="FULL_DIST_TEST",
        agent="catalog_agent",
        specific_topic=None) #ange to None for full report
    print(report)

if __name__ == "__main__":
    asyncio.run(main())