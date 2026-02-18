from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utility helpers
# ============================================================

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names (strip spaces)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate column present in df (exact match)."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# -----------------------------
# Sheet name helpers (handles underscores / variants)
# -----------------------------
def _norm_sheet_name(name: str) -> str:
    x = str(name or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    x = " ".join(x.split())
    return x

def resolve_sheet_name(sheet_map: Dict[str, pd.DataFrame], canonical: str, aliases: Optional[Iterable[str]] = None) -> Optional[str]:
    """Return the actual sheet name in sheet_map that matches canonical / aliases (case/underscore tolerant)."""
    if not sheet_map:
        return None

    # 1) exact match
    if canonical in sheet_map:
        return canonical

    # 2) exact alias match
    if aliases:
        for a in aliases:
            if a in sheet_map:
                return a

    # 3) normalized exact match
    target_norms = [_norm_sheet_name(canonical)]
    if aliases:
        target_norms += [_norm_sheet_name(a) for a in aliases]

    key_by_norm = {_norm_sheet_name(k): k for k in sheet_map.keys()}
    for t in target_norms:
        if t in key_by_norm:
            return key_by_norm[t]

    # 4) fallback: contains match (useful for "Inventory Snapshot Index (v2)" etc.)
    for k in sheet_map.keys():
        kn = _norm_sheet_name(k)
        for t in target_norms:
            if t and t in kn:
                return k

    return None


def normalize_part_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s.lower() in ["nan", "none"]:
        return ""
    return s


def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def month_period(dt: pd.Series) -> pd.PeriodIndex:
    """Coerce dates to monthly PeriodIndex."""
    d = pd.to_datetime(dt, errors="coerce")
    return d.dt.to_period("M")


# ============================================================
# Required workbook sheets
# ============================================================

# -----------------------------
# Required workbook sheets
# -----------------------------
# NOTE:
# Some workbooks use underscores or slightly different sheet names.
# The app will try to auto-detect the correct sheet using aliases.
REQUIRED_BASE_SHEETS = [
    "Usage Issues Log",
    "AOG Stockout Log",
    "Purchase Orders",
    "Receipts Log",
    "Work Orders",
]

# Inventory can be provided in either of two formats:
# (A) Snapshot index that lists each monthly snapshot sheet
# (B) A single long-format sheet with a Month column (recommended for sharing)
INVENTORY_INDEX_SHEET = "Inventory Snapshot Index"
INVENTORY_LONG_SHEET = "Inventory_Monthly_Snapshot"

SHEET_ALIASES = {
    # inventory
    INVENTORY_INDEX_SHEET: [
        "Inventory_Snapshot_Index",
        "Inventory SnapshotIndex",
        "Inventory Index",
        "Snapshot Index",
    ],
    INVENTORY_LONG_SHEET: [
        "Inventory Monthly Snapshot",
        "Monthly Inventory Snapshot",
        "Inventory Snapshot (Long)",
        "Inventory_Snapshot_Long",
    ],

    # logs
    "Usage Issues Log": ["Usage_Issues_Log", "Usage Log", "Usage_Log"],
    "AOG Stockout Log": ["Stockout_AOG_Log", "AOG_Stockout_Log", "Stockout Log", "Stockout_Log"],
    "Purchase Orders": ["Purchase_Orders", "PO", "PurchaseOrders"],
    "Receipts Log": ["Receipts_Log", "Receipts", "Receipt Log", "Receipt_Log"],
    "Work Orders": ["Work_Orders", "WO", "WorkOrders"],
}


# ============================================================
# Workbook parsing
# ============================================================

@st.cache_data(show_spinner=False)
def read_excel_sheets(uploaded_file) -> Dict[str, pd.DataFrame]:
    """
    Reads all sheets in the uploaded Excel file and returns a dict of {sheet_name: dataframe}.
    Works with Streamlit's UploadedFile or a BytesIO-like object.
    """
    if uploaded_file is None:
        return {}

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
        bio = io.BytesIO(data)
    elif isinstance(uploaded_file, (bytes, bytearray)):
        bio = io.BytesIO(uploaded_file)
    else:
        # assume it's a file-like object
        bio = uploaded_file

    xls = pd.ExcelFile(bio, engine="openpyxl")
    out: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        out[name] = xls.parse(name)
    return out


def parse_workbook(sheet_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # -----------------------------
    # Resolve required sheet names (tolerant to underscores / variants)
    # -----------------------------
    inv_idx_name = resolve_sheet_name(sheet_map, INVENTORY_INDEX_SHEET, SHEET_ALIASES.get(INVENTORY_INDEX_SHEET))
    inv_long_name = resolve_sheet_name(sheet_map, INVENTORY_LONG_SHEET, SHEET_ALIASES.get(INVENTORY_LONG_SHEET))

    usage_name = resolve_sheet_name(sheet_map, "Usage Issues Log", SHEET_ALIASES.get("Usage Issues Log"))
    stockout_name = resolve_sheet_name(sheet_map, "AOG Stockout Log", SHEET_ALIASES.get("AOG Stockout Log"))
    po_name = resolve_sheet_name(sheet_map, "Purchase Orders", SHEET_ALIASES.get("Purchase Orders"))
    receipts_name = resolve_sheet_name(sheet_map, "Receipts Log", SHEET_ALIASES.get("Receipts Log"))
    wo_name = resolve_sheet_name(sheet_map, "Work Orders", SHEET_ALIASES.get("Work Orders"))

    missing: list[str] = []
    if usage_name is None:
        missing.append("Usage Issues Log")
    if stockout_name is None:
        missing.append("AOG Stockout Log")
    if po_name is None:
        missing.append("Purchase Orders")
    if receipts_name is None:
        missing.append("Receipts Log")
    if wo_name is None:
        missing.append("Work Orders")
    if inv_idx_name is None and inv_long_name is None:
        missing.append(f"{INVENTORY_INDEX_SHEET} or {INVENTORY_LONG_SHEET}")

    if missing:
        raise ValueError("Workbook is missing required sheet(s): " + ", ".join(missing))

    # -----------------------------
    # Inventory snapshots -> long format
    # Supports:
    #   (A) Inventory Snapshot Index + separate monthly sheets
    #   (B) Inventory_Monthly_Snapshot single sheet (Month, part_id, on_hand_qty)
    # -----------------------------
    inventory_frames: list[pd.DataFrame] = []

    if inv_idx_name is not None:
        idx = clean_cols(sheet_map[inv_idx_name])
        month_col = pick_col(idx, ["Month", "Snapshot Month", "snapshot_month", "month"])
        sheet_col = pick_col(idx, ["Sheet Name", "Sheet", "Worksheet", "sheet_name", "snapshot_sheet", "snapshot_sheet_name"])
        if month_col is None or sheet_col is None:
            raise ValueError("Inventory Snapshot Index must have 'Month' and 'Sheet Name' columns (or their equivalents).")

        for _, row in idx.iterrows():
            sheet_name = str(row[sheet_col]).strip()
            if not sheet_name or sheet_name.lower() in ["nan", "none"]:
                continue
            if sheet_name not in sheet_map:
                continue

            snap_month = pd.to_datetime(row[month_col], errors="coerce")
            if pd.isna(snap_month):
                continue
            snap_period = pd.Period(snap_month, freq="M")

            inv_raw = clean_cols(sheet_map[sheet_name])
            part_col = pick_col(inv_raw, ["Part Number", "Part No", "PN", "Part", "part_id"])
            qty_col = pick_col(inv_raw, ["Quantity", "Qty", "On Hand", "Stock", "on_hand_qty"])
            desc_col = pick_col(inv_raw, ["Item Nomenclature or Description", "Item Description", "Description", "Nomenclature", "description"])
            type_col = pick_col(inv_raw, ["Type", "Item Type", "part_type"])
            loc_col = pick_col(inv_raw, ["Location", "Bin", "Rack", "Shelf", "location"])

            if part_col is None or qty_col is None:
                continue

            part_series = inv_raw[part_col].map(normalize_part_id)
            qty_series = to_num(inv_raw[qty_col], 0.0).round().astype(int)

            df = pd.DataFrame(
                {
                    "month": snap_period,
                    "part_id": part_series,
                    "description": inv_raw[desc_col].astype(str).str.strip() if desc_col else "",
                    "part_type": inv_raw[type_col].astype(str).str.strip() if type_col else "",
                    "on_hand_qty": qty_series,
                    "location": inv_raw[loc_col].astype(str).str.strip() if loc_col else "",
                }
            )
            df = df[(df["part_id"] != "") & (~df["part_id"].str.lower().isin(["nan", "none"]))].copy()
            inventory_frames.append(df)

    else:
        # Long-format inventory sheet
        inv_raw = clean_cols(sheet_map[inv_long_name])  # type: ignore[index]
        month_col = pick_col(inv_raw, ["month", "Month", "snapshot_month", "Snapshot Month", "period"])
        date_col = pick_col(inv_raw, ["month_end_date", "Month End Date", "snapshot_date", "Snapshot Date", "Date", "date"])
        part_col = pick_col(inv_raw, ["part_id", "Part ID", "Part Number", "Part No", "PN", "Part"])
        qty_col = pick_col(inv_raw, ["on_hand_qty", "On Hand Qty", "On Hand", "Quantity", "Qty", "Stock"])
        desc_col = pick_col(inv_raw, ["description", "Description", "Item Description", "Nomenclature"])
        type_col = pick_col(inv_raw, ["part_type", "Type", "Item Type"])
        loc_col = pick_col(inv_raw, ["location", "Location", "Bin", "Rack", "Shelf"])

        if part_col is None or qty_col is None:
            raise ValueError("Inventory_Monthly_Snapshot must contain Part/part_id and Quantity/on_hand_qty columns.")

        # Month is preferred. If not present, derive from a date column.
        if month_col is not None:
            m_raw = inv_raw[month_col].astype(str).str.strip()
            # Accept formats like "2023-01" or full dates
            try:
                month_per = pd.PeriodIndex(m_raw, freq="M")
            except Exception:
                month_per = pd.to_datetime(m_raw, errors="coerce").dt.to_period("M")
        elif date_col is not None:
            month_per = month_period(inv_raw[date_col])
        else:
            raise ValueError("Inventory_Monthly_Snapshot must contain a Month column (e.g., 2023-01) or a Date column.")

        df = pd.DataFrame(
            {
                "month": month_per,
                "part_id": inv_raw[part_col].map(normalize_part_id),
                "description": inv_raw[desc_col].astype(str).str.strip() if desc_col else "",
                "part_type": inv_raw[type_col].astype(str).str.strip() if type_col else "",
                "on_hand_qty": to_num(inv_raw[qty_col], 0.0).round().astype(int),
                "location": inv_raw[loc_col].astype(str).str.strip() if loc_col else "",
            }
        )
        df = df[(df["part_id"] != "") & (~df["part_id"].str.lower().isin(["nan", "none"]))].copy()
        inventory_frames.append(df)

    if not inventory_frames:
        raise ValueError("No inventory snapshot data was read. Check your inventory sheet structure.")

    inv_long = pd.concat(inventory_frames, ignore_index=True)
    inv_long["month"] = inv_long["month"].astype("period[M]")

    # -----------------------------
    # Usage Issues Log
    # -----------------------------
    usage_raw = clean_cols(sheet_map[usage_name])  # type: ignore[index]
    u_date = pick_col(usage_raw, ["Date", "date", "Issue Date", "issue_date", "issued_date"])
    u_part = pick_col(usage_raw, ["Part Number", "Part No", "PN", "part_id", "Part ID", "part number"])
    u_qty = pick_col(usage_raw, ["Quantity Issued", "Qty Issued", "Quantity", "Qty", "qty_issued", "qty_used", "quantity_used"])
    u_check = pick_col(usage_raw, ["Check Type", "Maintenance Check Type", "check_type", "maintenance_check_type"])
    u_aircraft = pick_col(usage_raw, ["Aircraft Type", "Aircraft", "Fleet", "aircraft_type"])
    u_wo = pick_col(usage_raw, ["Work Order", "Work Order No.", "Work Order No", "WO", "work_order"])
    u_reason = pick_col(usage_raw, ["Issue Reason", "Reason", "issue_reason"])

    if u_date is None or u_part is None or u_qty is None:
        raise ValueError("Usage Issues Log must contain: Date, Part/part_id, Quantity Issued/qty_used.")

    usage_tbl = pd.DataFrame(
        {
            "date": pd.to_datetime(usage_raw[u_date], errors="coerce"),
            "part_id": usage_raw[u_part].map(normalize_part_id),
            "qty_issued": to_num(usage_raw[u_qty], 0.0).round().astype(int),
            "check_type": usage_raw[u_check].astype(str).str.strip() if u_check else "",
            "aircraft_type": usage_raw[u_aircraft].astype(str).str.strip() if u_aircraft else "",
            "work_order": usage_raw[u_wo].astype(str).str.strip() if u_wo else "",
            "issue_reason": usage_raw[u_reason].astype(str).str.strip() if u_reason else "",
        }
    ).dropna(subset=["date"]).copy()
    usage_tbl["month"] = month_period(usage_tbl["date"])
    usage_tbl = usage_tbl[(usage_tbl["part_id"] != "")].copy()

    # -----------------------------
    # AOG Stockout Log
    # -----------------------------
    so_raw = clean_cols(sheet_map[stockout_name])  # type: ignore[index]
    s_date = pick_col(so_raw, ["Date", "date", "Incident Date", "incident_date"])
    s_part = pick_col(so_raw, ["Part Number", "Part No", "PN", "part_id", "Part ID"])
    s_short = pick_col(so_raw, ["Short Qty", "Short Quantity", "Short", "short_qty"])
    s_down = pick_col(so_raw, ["Downtime Hours", "Downtime (Hours)", "Downtime", "downtime_hours"])
    s_check = pick_col(so_raw, ["Check Type", "Maintenance Check Type", "check_type", "maintenance_check_type"])
    s_aog = pick_col(so_raw, ["AOG?", "AOG", "Is AOG", "aog_flag"])
    s_req = pick_col(so_raw, ["Requested Qty", "Requested Quantity", "requested_qty"])
    s_iss = pick_col(so_raw, ["Issued Qty", "Issued Quantity", "issued_qty"])

    if s_date is None or s_part is None:
        raise ValueError("AOG Stockout Log must contain: Date/incident_date and Part/part_id.")

    stockout_tbl = pd.DataFrame(
        {
            "date": pd.to_datetime(so_raw[s_date], errors="coerce"),
            "part_id": so_raw[s_part].map(normalize_part_id),
            "short_qty": to_num(so_raw[s_short], 0.0) if s_short else 0.0,
            "downtime_hours": to_num(so_raw[s_down], 0.0) if s_down else 0.0,
            "check_type": so_raw[s_check].astype(str).str.strip() if s_check else "",
            "aog_flag": so_raw[s_aog].astype(str).str.strip() if s_aog else "",
            "requested_qty": to_num(so_raw[s_req], np.nan) if s_req else np.nan,
            "issued_qty": to_num(so_raw[s_iss], np.nan) if s_iss else np.nan,
        }
    ).dropna(subset=["date"]).copy()
    stockout_tbl["short_qty"] = to_num(stockout_tbl["short_qty"], 0.0).round().astype(int)
    stockout_tbl["downtime_hours"] = to_num(stockout_tbl["downtime_hours"], 0.0)
    stockout_tbl["month"] = month_period(stockout_tbl["date"])
    stockout_tbl = stockout_tbl[(stockout_tbl["part_id"] != "")].copy()

    # -----------------------------
    # Purchase Orders
    # -----------------------------
    po_raw = clean_cols(sheet_map[po_name])  # type: ignore[index]
    p_part = pick_col(po_raw, ["Part Number", "Part No", "PN", "part_id", "Part ID"])
    p_qty = pick_col(po_raw, ["Quantity Ordered", "Qty Ordered", "Quantity", "Qty", "qty_ordered"])
    p_cost = pick_col(po_raw, ["Unit Cost (PHP)", "Unit Cost", "Cost", "Unit Price", "unit_cost_php"])
    p_lt = pick_col(po_raw, ["Lead Time (Days)", "Lead Time", "LT (Days)", "lead_time_days"])
    p_date = pick_col(po_raw, ["Date Ordered", "Order Date", "Date", "date_ordered"])

    po_tbl = pd.DataFrame(
        {
            "date_ordered": pd.to_datetime(po_raw[p_date], errors="coerce") if p_date else pd.NaT,
            "part_id": po_raw[p_part].map(normalize_part_id) if p_part else "",
            "qty_ordered": to_num(po_raw[p_qty], 0.0).round().astype(int) if p_qty else 0,
            "unit_cost_php": to_num(po_raw[p_cost], np.nan) if p_cost else np.nan,
            "lead_time_days": to_num(po_raw[p_lt], np.nan) if p_lt else np.nan,
        }
    )
    po_tbl = po_tbl[(po_tbl["part_id"] != "")].copy()

    # -----------------------------
    # Receipts Log
    # -----------------------------
    r_raw = clean_cols(sheet_map[receipts_name])  # type: ignore[index]
    r_part = pick_col(r_raw, ["Part Number", "Part No", "PN", "part_id", "Part ID"])
    r_qty = pick_col(r_raw, ["Quantity Received", "Qty Received", "Quantity", "Qty", "qty_received"])
    r_cost = pick_col(r_raw, ["Unit Cost (PHP)", "Unit Cost", "Cost", "Unit Price", "unit_cost_php"])
    r_date = pick_col(r_raw, ["Date Received", "Received Date", "Date", "date_received"])

    receipts_tbl = pd.DataFrame(
        {
            "date_received": pd.to_datetime(r_raw[r_date], errors="coerce") if r_date else pd.NaT,
            "part_id": r_raw[r_part].map(normalize_part_id) if r_part else "",
            "qty_received": to_num(r_raw[r_qty], 0.0).round().astype(int) if r_qty else 0,
            "unit_cost_php": to_num(r_raw[r_cost], np.nan) if r_cost else np.nan,
        }
    )
    receipts_tbl = receipts_tbl[(receipts_tbl["part_id"] != "")].copy()
    receipts_tbl["month"] = month_period(receipts_tbl["date_received"])

    # -----------------------------
    # Work Orders
    # -----------------------------
    w_raw = clean_cols(sheet_map[wo_name])  # type: ignore[index]
    w_date = pick_col(w_raw, ["Date", "date", "Work Date", "work_date"])
    w_check = pick_col(w_raw, ["Check Type", "Maintenance Check Type", "check_type", "maintenance_check_type"])
    w_wo = pick_col(w_raw, ["Work Order No.", "Work Order No", "Work Order", "WO", "work_order"])
    w_aircraft = pick_col(w_raw, ["Aircraft Type", "Aircraft", "aircraft_type", "Fleet"])

    work_orders_tbl = pd.DataFrame(
        {
            "date": pd.to_datetime(w_raw[w_date], errors="coerce") if w_date else pd.NaT,
            "check_type": w_raw[w_check].astype(str).str.strip() if w_check else "",
            "work_order": w_raw[w_wo].astype(str).str.strip() if w_wo else "",
            "aircraft_type": w_raw[w_aircraft].astype(str).str.strip() if w_aircraft else "",
        }
    ).dropna(subset=["date"]).copy()
    work_orders_tbl["month"] = month_period(work_orders_tbl["date"])

    return {
        "inventory_long": inv_long,
        "usage": usage_tbl,
        "stockout": stockout_tbl,
        "purchase_orders": po_tbl,
        "receipts": receipts_tbl,
        "work_orders": work_orders_tbl,
    }


# ============================================================
# Metrics / analytics
# ============================================================

def build_parts_master(inventory_long: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a simple parts master from inventory_long.
    """
    cols = ["part_id", "description", "part_type"]
    pm = inventory_long[cols].copy()
    pm["description"] = pm["description"].fillna("").astype(str).str.strip()
    pm["part_type"] = pm["part_type"].fillna("").astype(str).str.strip()
    pm = pm.drop_duplicates(subset=["part_id"], keep="first")
    pm = pm.sort_values("part_id")
    return pm


def monthly_stockout_counts(stockout: pd.DataFrame) -> pd.DataFrame:
    """
    Count stockout incidents per month.
    """
    if stockout.empty:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "stockouts": []})

    grp = stockout.groupby("month", as_index=False).size()
    grp = grp.rename(columns={"size": "stockouts"})
    return grp.sort_values("month")


def monthly_usage_counts(usage: pd.DataFrame) -> pd.DataFrame:
    """
    Sum qty_issued per month.
    """
    if usage.empty:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "usage_qty": []})

    grp = usage.groupby("month", as_index=False)["qty_issued"].sum()
    grp = grp.rename(columns={"qty_issued": "usage_qty"})
    return grp.sort_values("month")


def monthly_inventory_onhand(inventory_long: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate on-hand quantities per month (total).
    """
    if inventory_long.empty:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "on_hand_total": []})

    grp = inventory_long.groupby("month", as_index=False)["on_hand_qty"].sum()
    grp = grp.rename(columns={"on_hand_qty": "on_hand_total"})
    return grp.sort_values("month")


def compute_fill_rate(stockout: pd.DataFrame) -> pd.DataFrame:
    """
    Simple fill rate estimate if requested_qty and issued_qty exist.
    fill_rate = issued_qty / requested_qty (per record), then average per month.
    """
    if stockout.empty or "requested_qty" not in stockout.columns or "issued_qty" not in stockout.columns:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "fill_rate": []})

    df = stockout.copy()
    df["requested_qty"] = to_num(df["requested_qty"], np.nan)
    df["issued_qty"] = to_num(df["issued_qty"], np.nan)
    df["fill_rate"] = df["issued_qty"] / df["requested_qty"]
    df.loc[df["requested_qty"] <= 0, "fill_rate"] = np.nan

    grp = df.groupby("month", as_index=False)["fill_rate"].mean()
    return grp.sort_values("month")


def compute_lead_time_stats(po_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean lead time per month (if date_ordered is present).
    """
    if po_tbl.empty or "lead_time_days" not in po_tbl.columns or "date_ordered" not in po_tbl.columns:
        return pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "lead_time_days_mean": []})

    df = po_tbl.copy()
    df["date_ordered"] = pd.to_datetime(df["date_ordered"], errors="coerce")
    df = df.dropna(subset=["date_ordered"])
    df["month"] = df["date_ordered"].dt.to_period("M")
    df["lead_time_days"] = to_num(df["lead_time_days"], np.nan)

    grp = df.groupby("month", as_index=False)["lead_time_days"].mean()
    grp = grp.rename(columns={"lead_time_days": "lead_time_days_mean"})
    return grp.sort_values("month")


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="MRO Inventory Dashboard", layout="wide")

st.title("MRO Inventory Dashboard")
st.caption(
    "Upload your workbook and explore inventory, usage, and stockout indicators. "
    "This version auto-detects sheet names with underscores and supports Inventory_Monthly_Snapshot."
)

uploaded = st.file_uploader("Upload Excel workbook", type=["xlsx"])

if uploaded is None:
    st.info("Please upload the 36-month workbook (e.g., 36mo dataset_sample_inventory.xlsx).")
    st.stop()

try:
    sheet_map = read_excel_sheets(uploaded)
    parsed = parse_workbook(sheet_map)
except Exception as e:
    st.error(str(e))
    st.stop()

inventory_long = parsed["inventory_long"]
usage = parsed["usage"]
stockout = parsed["stockout"]
purchase_orders = parsed["purchase_orders"]
receipts = parsed["receipts"]
work_orders = parsed["work_orders"]

parts_master = build_parts_master(inventory_long)

# Sidebar filters
st.sidebar.header("Filters")

min_month = inventory_long["month"].min()
max_month = inventory_long["month"].max()
month_range = st.sidebar.slider(
    "Month range",
    min_value=min_month.to_timestamp(),
    max_value=max_month.to_timestamp(),
    value=(min_month.to_timestamp(), max_month.to_timestamp()),
)
month_start = pd.Period(month_range[0], freq="M")
month_end = pd.Period(month_range[1], freq="M")

def filter_by_month(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    if df.empty:
        return df
    if month_col not in df.columns:
        return df
    return df[(df[month_col] >= month_start) & (df[month_col] <= month_end)].copy()

inventory_f = filter_by_month(inventory_long)
usage_f = filter_by_month(usage)
stockout_f = filter_by_month(stockout)
po_f = purchase_orders.copy()
receipts_f = filter_by_month(receipts)
wo_f = filter_by_month(work_orders)

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Months in range", int((month_end - month_start) + 1))

with col2:
    st.metric("Unique parts (inventory)", inventory_f["part_id"].nunique())

with col3:
    st.metric("Total stockout events", int(len(stockout_f)))

with col4:
    st.metric("Total usage qty", int(usage_f["qty_issued"].sum()) if not usage_f.empty else 0)

st.divider()

# Time series
st.subheader("Time Series")

m_stockouts = monthly_stockout_counts(stockout_f)
m_usage = monthly_usage_counts(usage_f)
m_onhand = monthly_inventory_onhand(inventory_f)
m_fill = compute_fill_rate(stockout_f)

ts = m_stockouts.merge(m_usage, on="month", how="outer").merge(m_onhand, on="month", how="outer").merge(m_fill, on="month", how="outer")
ts = ts.sort_values("month")
ts_display = ts.copy()
ts_display["month"] = ts_display["month"].astype(str)

st.dataframe(ts_display, use_container_width=True)

st.divider()

# Raw tables
st.subheader("Raw Tables (Filtered)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Inventory", "Usage", "Stockout", "PO", "Receipts", "Work Orders"])

with tab1:
    d = inventory_f.copy()
    d["month"] = d["month"].astype(str)
    st.dataframe(d, use_container_width=True)

with tab2:
    d = usage_f.copy()
    d["month"] = d["month"].astype(str)
    st.dataframe(d, use_container_width=True)

with tab3:
    d = stockout_f.copy()
    d["month"] = d["month"].astype(str)
    st.dataframe(d, use_container_width=True)

with tab4:
    st.dataframe(po_f, use_container_width=True)

with tab5:
    d = receipts_f.copy()
    d["month"] = d["month"].astype(str)
    st.dataframe(d, use_container_width=True)

with tab6:
    d = wo_f.copy()
    d["month"] = d["month"].astype(str)
    st.dataframe(d, use_container_width=True)
