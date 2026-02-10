# Thesis_dashboard_mro_simplified_fixed4.py
# Streamlit dashboard for MRO inventory performance comparison:
#   (1) Traditional / manual inventory practice (computed from workbook records)
#   (2) Predictive analytics (lightweight demand forecast + reorder planning)
#
# Changes vs fixed3 (per your request):
# - Carrying Cost is shown as:
#     (a) Avg monthly carrying cost
#     (b) Annual carrying cost (implied)
#     (c) 36-month total (period total)
# - Better numeric formatting (₱, commas, consistent decimals)
# - Value-based computations EXCLUDE parts with no written cost
#     ("written cost" = cost present in Receipts Log and/or Purchase Orders).
#   Specifically excluded from: inventory value, usage value, carrying cost, turnover.
# - Data Quality tab updated to reflect "Missing cost" instead of imputation.
#
# Run:
#   streamlit run Thesis_dashboard_mro_simplified_fixed4.py

from __future__ import annotations

import math
import time
from io import BytesIO
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Required workbook sheets
# -----------------------------
REQUIRED_SHEETS = [
    "Inventory Snapshot Index",
    "Usage Issues Log",
    "AOG Stockout Log",
    "Purchase Orders",
    "Receipts Log",
    "Work Orders",
]

APP_TITLE = "MRO Inventory Control Dashboard"
APP_SUBTITLE = "Manual (Traditional) vs Predictive Analytics (Lightweight Forecast + Reorder Plan)"

# -----------------------------
# Internal defaults (kept out of the UI to stay simple)
# -----------------------------
ANNUAL_CARRY_RATE = 0.25          # holding cost rate/year (25%)
SERVICE_LEVEL_V = 0.98            # Vital
SERVICE_LEVEL_E = 0.95            # Essential
SERVICE_LEVEL_D = 0.90            # Desirable
ORDER_COVER_MONTHS = 1.00         # recommended order quantity coverage
BLEND_WITH_CURRENT = 0.65         # blend recommended avg stock with current avg on-hand

# -----------------------------
# Styling
# -----------------------------
APP_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.card {
  border: 1px solid rgba(49, 51, 63, 0.20);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.80);
}
.card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
.muted { color: rgba(49, 51, 63, 0.7); font-size: 0.9rem; }

.loading-wrap {
  border: 1px solid rgba(49, 51, 63, 0.20);
  border-radius: 16px;
  padding: 18px 18px;
  background: linear-gradient(180deg, rgba(240,242,246,1) 0%, rgba(255,255,255,1) 100%);
}
.loading-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 6px; }
.loading-sub { color: rgba(49,51,63,0.75); margin-bottom: 10px; }

[data-testid="stMetricValue"] { font-size: 1.55rem; }

.dots { display: inline-block; margin-left: 6px; }
.dots span {
  display: inline-block;
  width: 7px; height: 7px; margin: 0 2px;
  border-radius: 50%;
  background: rgba(49,51,63,0.75);
  animation: dot-bounce 1s infinite ease-in-out;
}
.dots span:nth-child(2) { animation-delay: 0.15s; opacity: 0.85; }
.dots span:nth-child(3) { animation-delay: 0.30s; opacity: 0.70; }

@keyframes dot-bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-7px); }
}
</style>
"""

# -----------------------------
# Formatting helpers
# -----------------------------
def fmt_php(x: float, decimals: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "₱0.00"
    if np.isnan(v):
        return ""
    return f"₱{v:,.{decimals}f}"

def fmt_num(x: float, decimals: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "0"
    if np.isnan(v):
        return ""
    return f"{v:,.{decimals}f}"

def fmt_int(x: float) -> str:
    try:
        v = int(round(float(x)))
    except Exception:
        return "0"
    return f"{v:,}"

# -----------------------------
# Normal distribution helpers (no SciPy)
# -----------------------------
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))

def norm_pdf(z: float) -> float:
    return (1.0 / SQRT2PI) * math.exp(-0.5 * z * z)

def normal_loss(z: float) -> float:
    # L(z) = φ(z) - z(1-Φ(z))
    return norm_pdf(z) - z * (1.0 - norm_cdf(z))

def service_level_to_z(p: float) -> float:
    # Approx inverse CDF by binary search
    p = float(np.clip(p, 0.50, 0.999))
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

# -----------------------------
# Excel I/O
# -----------------------------
@st.cache_data(show_spinner=False)
def read_excel_sheets(source_bytes: bytes) -> Dict[str, pd.DataFrame]:
    bio = BytesIO(source_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    return {name: xls.parse(name) for name in xls.sheet_names}

# -----------------------------
# Column helpers
# -----------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    keep_cols = [c for c in out.columns if not str(c).strip().lower().startswith("unnamed")]
    return out.loc[:, keep_cols]

def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    colmap = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in colmap:
            return colmap[key]
    return None

def normalize_part_id(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    x = str(s).strip()
    x = " ".join(x.split())
    return x.upper()

def to_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    """
    Robust numeric conversion:
    - handles "₱", commas, spaces
    - handles parentheses for negatives "(1,234)"
    - strips non-numeric noise safely
    """
    if series is None:
        return pd.Series([], dtype=float)

    s = series.astype(str).str.strip()

    # parentheses negative
    neg = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    # remove currency symbols and commas
    s = s.str.replace("₱", "", regex=False)
    s = s.str.replace(",", "", regex=False)

    # keep digits, dot, minus
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)

    out = pd.to_numeric(s, errors="coerce")
    out = out.where(~neg, -out)
    return out.fillna(default)

def month_period(series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(series, errors="coerce")
    return dt_series.dt.to_period("M")

# -----------------------------
# Parse workbook -> normalized tables
# -----------------------------
def parse_workbook(sheet_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    missing = [s for s in REQUIRED_SHEETS if s not in sheet_map]
    if missing:
        raise ValueError(f"Workbook is missing required sheet(s): {', '.join(missing)}")

    # Snapshot index
    idx = clean_cols(sheet_map["Inventory Snapshot Index"])
    month_col = pick_col(idx, ["Month", "Snapshot Month"])
    sheet_col = pick_col(idx, ["Sheet Name", "Sheet", "Worksheet"])
    if month_col is None or sheet_col is None:
        raise ValueError("Inventory Snapshot Index must have 'Month' and 'Sheet Name' columns.")

    inventory_frames: list[pd.DataFrame] = []
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
        part_col = pick_col(inv_raw, ["Part Number", "Part No", "PN", "Part"])
        qty_col = pick_col(inv_raw, ["Quantity", "Qty", "On Hand", "Stock"])
        desc_col = pick_col(inv_raw, ["Item Nomenclature or Description", "Item Description", "Description", "Nomenclature"])
        type_col = pick_col(inv_raw, ["Type", "Item Type"])
        loc_col = pick_col(inv_raw, ["Location", "Bin", "Rack", "Shelf"])

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

    if not inventory_frames:
        raise ValueError("No inventory snapshot sheets were read. Check Inventory Snapshot Index sheet names.")

    inv_long = pd.concat(inventory_frames, ignore_index=True)
    inv_long["month"] = inv_long["month"].astype("period[M]")

    # Usage Issues Log
    usage_raw = clean_cols(sheet_map["Usage Issues Log"])
    u_date = pick_col(usage_raw, ["Date", "Issue Date"])
    u_part = pick_col(usage_raw, ["Part Number", "Part No", "PN"])
    u_qty = pick_col(usage_raw, ["Quantity Issued", "Qty Issued", "Quantity", "Qty"])
    u_check = pick_col(usage_raw, ["Check Type", "Maintenance Check Type"])
    u_aircraft = pick_col(usage_raw, ["Aircraft Type", "Aircraft", "Fleet"])
    u_wo = pick_col(usage_raw, ["Work Order", "Work Order No.", "Work Order No", "WO"])
    u_reason = pick_col(usage_raw, ["Issue Reason", "Reason"])

    if u_date is None or u_part is None or u_qty is None:
        raise ValueError("Usage Issues Log must contain: Date, Part Number, Quantity Issued.")

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

    # AOG Stockout Log
    so_raw = clean_cols(sheet_map["AOG Stockout Log"])
    s_date = pick_col(so_raw, ["Date", "Incident Date"])
    s_part = pick_col(so_raw, ["Part Number", "Part No", "PN"])
    s_short = pick_col(so_raw, ["Short Qty", "Short Quantity", "Short"])
    s_down = pick_col(so_raw, ["Downtime Hours", "Downtime (Hours)", "Downtime"])
    s_check = pick_col(so_raw, ["Check Type", "Maintenance Check Type"])
    s_aog = pick_col(so_raw, ["AOG?", "AOG", "Is AOG"])
    s_req = pick_col(so_raw, ["Requested Qty", "Requested Quantity"])
    s_iss = pick_col(so_raw, ["Issued Qty", "Issued Quantity"])

    if s_date is None or s_part is None:
        raise ValueError("AOG Stockout Log must contain: Date, Part Number.")

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

    # Purchase Orders
    po_raw = clean_cols(sheet_map["Purchase Orders"])
    p_part = pick_col(po_raw, ["Part Number", "Part No", "PN"])
    p_qty = pick_col(po_raw, ["Quantity Ordered", "Qty Ordered", "Quantity", "Qty"])
    p_cost = pick_col(po_raw, ["Unit Cost (PHP)", "Unit Cost", "Cost", "Unit Price"])
    p_lt = pick_col(po_raw, ["Lead Time (Days)", "Lead Time", "LT (Days)"])
    p_date = pick_col(po_raw, ["Date Ordered", "Order Date", "Date"])

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

    # Receipts
    r_raw = clean_cols(sheet_map["Receipts Log"])
    r_part = pick_col(r_raw, ["Part Number", "Part No", "PN"])
    r_qty = pick_col(r_raw, ["Quantity Received", "Qty Received", "Quantity", "Qty"])
    r_cost = pick_col(r_raw, ["Unit Cost (PHP)", "Unit Cost", "Cost", "Unit Price"])
    r_date = pick_col(r_raw, ["Date Received", "Received Date", "Date"])

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

    # Work Orders
    w_raw = clean_cols(sheet_map["Work Orders"])
    w_date = pick_col(w_raw, ["Date", "Work Date"])
    w_check = pick_col(w_raw, ["Check Type", "Maintenance Check Type"])
    w_wo = pick_col(w_raw, ["Work Order No.", "Work Order No", "Work Order", "WO"])
    w_aircraft = pick_col(w_raw, ["Aircraft Type", "Aircraft"])

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

# -----------------------------
# Parts master derivation
# -----------------------------
def infer_criticality_ved(desc: str) -> str:
    d = str(desc).lower()
    if any(k in d for k in ["generator", "flight control", "engine", "hydraulic", "avionics"]):
        return "V"
    if any(k in d for k in ["communication", "antenna", "pump", "sensor"]):
        return "E"
    return "D"

def infer_abc_class(cost_php: float) -> str:
    try:
        c = float(cost_php)
    except (TypeError, ValueError):
        c = 0.0
    if np.isnan(c):
        return "C"
    if c >= 50_000:
        return "A"
    if c >= 12_000:
        return "B"
    return "C"

def build_unit_cost_map(receipts_tbl: pd.DataFrame, po_tbl: pd.DataFrame) -> pd.Series:
    """
    Written cost source ONLY:
      - Weighted avg from receipts
      - fallback to weighted avg from PO
    No imputation. If no written cost exists for a part_id -> NaN.
    """
    r = receipts_tbl.copy()
    r["unit_cost_php"] = to_num(r["unit_cost_php"], np.nan)
    r["qty_received"] = to_num(r["qty_received"], 0.0)
    r = r[(r["qty_received"] > 0) & (~r["unit_cost_php"].isna())].copy()

    if len(r) > 0:
        num = (r["unit_cost_php"] * r["qty_received"]).groupby(r["part_id"]).sum()
        den = r["qty_received"].groupby(r["part_id"]).sum().replace(0.0, np.nan)
        cost_r = (num / den).rename("unit_cost_php")
    else:
        cost_r = pd.Series(dtype=float, name="unit_cost_php")

    p = po_tbl.copy()
    p["unit_cost_php"] = to_num(p["unit_cost_php"], np.nan)
    p["qty_ordered"] = to_num(p["qty_ordered"], 0.0)
    p = p[(p["qty_ordered"] > 0) & (~p["unit_cost_php"].isna())].copy()

    if len(p) > 0:
        num = (p["unit_cost_php"] * p["qty_ordered"]).groupby(p["part_id"]).sum()
        den = p["qty_ordered"].groupby(p["part_id"]).sum().replace(0.0, np.nan)
        cost_p = (num / den).rename("unit_cost_php")
    else:
        cost_p = pd.Series(dtype=float, name="unit_cost_php")

    # Prefer receipts-derived costs, fallback to PO-derived
    unit_cost = cost_r.combine_first(cost_p)
    return unit_cost

def build_lead_time_stats(po_tbl: pd.DataFrame) -> pd.DataFrame:
    p = po_tbl.copy()
    p["lead_time_days"] = to_num(p["lead_time_days"], np.nan)
    p = p[~p["lead_time_days"].isna()].copy()
    if len(p) == 0:
        return pd.DataFrame(columns=["part_id", "lead_time_days_mean", "lead_time_days_sd"])

    stats = (
        p.groupby("part_id")["lead_time_days"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "lead_time_days_mean", "std": "lead_time_days_sd"})
    )
    stats["lead_time_days_sd"] = to_num(stats["lead_time_days_sd"], 0.0)
    return stats

def build_parts_master(
    inv_long: pd.DataFrame,
    unit_cost: pd.Series,
    lt_stats: pd.DataFrame,
) -> pd.DataFrame:
    base = (
        inv_long.groupby("part_id", as_index=False)
        .agg(
            description=("description", "first"),
            part_type=("part_type", "first"),
            avg_on_hand_qty=("on_hand_qty", "mean"),
        )
        .copy()
    )
    base["avg_on_hand_qty"] = to_num(base["avg_on_hand_qty"], 0.0)

    # Written costs only (can be NaN if missing)
    base["unit_cost_php"] = base["part_id"].map(unit_cost)

    base = base.merge(lt_stats, on="part_id", how="left")
    base["lead_time_days_mean"] = to_num(base["lead_time_days_mean"], 28.0)
    base["lead_time_days_sd"] = to_num(base["lead_time_days_sd"], 9.0)

    base["criticality_VED"] = base["description"].apply(infer_criticality_ved)
    base["abc_class"] = base["unit_cost_php"].apply(infer_abc_class)

    base["unit_cost_source"] = np.where(base["unit_cost_php"].notna(), "Receipts/PO", "Missing")

    return base[
        [
            "part_id",
            "description",
            "part_type",
            "abc_class",
            "criticality_VED",
            "unit_cost_php",
            "unit_cost_source",
            "lead_time_days_mean",
            "lead_time_days_sd",
            "avg_on_hand_qty",
        ]
    ].copy()

# -----------------------------
# Manual monthly KPIs (from records)
# -----------------------------
def compute_manual_monthly_kpis(
    parts_master: pd.DataFrame,
    inv_long: pd.DataFrame,
    usage_tbl: pd.DataFrame,
    stockout_tbl: pd.DataFrame,
    check_type: str,
    annual_carry_rate: float,
) -> pd.DataFrame:
    parts = parts_master.copy()
    cost_map = parts.set_index("part_id")["unit_cost_php"]

    # Inventory value per month (EXCLUDE missing-cost parts)
    inv = inv_long.copy()
    inv["unit_cost_php"] = inv["part_id"].map(cost_map)
    inv = inv[inv["unit_cost_php"].notna()].copy()
    inv["line_value"] = inv["on_hand_qty"].astype(float) * inv["unit_cost_php"].astype(float)
    inv_value = inv.groupby("month")["line_value"].sum().sort_index()

    # Usage value (all) EXCLUDE missing-cost parts
    u_all = usage_tbl.copy()
    u_all["unit_cost_php"] = u_all["part_id"].map(cost_map)
    u_all = u_all[u_all["unit_cost_php"].notna()].copy()
    u_all["usage_value_php"] = u_all["qty_issued"].astype(float) * u_all["unit_cost_php"].astype(float)
    usage_all_month = u_all.groupby("month")["usage_value_php"].sum().sort_index()

    # Usage (filtered)
    u = u_all if check_type == "All" else u_all[u_all["check_type"] == check_type].copy()
    usage_month = u.groupby("month")["usage_value_php"].sum().sort_index()

    # Allocation share of inventory to that check type (for per-check reporting)
    if check_type == "All":
        alloc_share = pd.Series(1.0, index=inv_value.index)  # align to inventory months
    else:
        denom = usage_all_month.replace(0.0, np.nan)
        alloc_share = (usage_month / denom).fillna(0.0).clip(0.0, 1.0)

    # Align months
    all_months = sorted(set(inv_value.index) | set(usage_all_month.index))
    inv_value = inv_value.reindex(all_months).ffill().fillna(0.0)
    usage_month = usage_month.reindex(all_months, fill_value=0.0)
    alloc_share = alloc_share.reindex(all_months, fill_value=0.0)

    # Average inventory value using (prev+current)/2 when prev exists, else current
    inv_prev = inv_value.shift(1)
    avg_inv_value = inv_value.copy()
    mask = inv_prev.notna()
    avg_inv_value.loc[mask] = (inv_prev.loc[mask] + inv_value.loc[mask]) / 2.0

    alloc_avg_inv = (avg_inv_value * alloc_share).fillna(0.0)

    # Stockouts (not cost-based; keep all)
    s = stockout_tbl if check_type == "All" else stockout_tbl[stockout_tbl["check_type"] == check_type].copy()
    s_month = s.groupby("month").agg(
        stockout_events=("part_id", "count"),
        total_short_qty=("short_qty", "sum"),
        downtime_hours=("downtime_hours", "sum"),
    ).sort_index()
    s_month = s_month.reindex(all_months, fill_value=0)

    # Carrying cost (monthly)
    carry_month = alloc_avg_inv * (annual_carry_rate / 12.0)

    # Availability proxy (1 - downtime/hours_in_month)
    availability = []
    for per in all_months:
        m_start = per.to_timestamp()
        days_in_month = int((m_start + pd.offsets.MonthEnd(0)).day)
        hours = float(days_in_month * 24)
        down = float(s_month.loc[per, "downtime_hours"])
        availability.append(max(0.0, 1.0 - (down / hours if hours > 0 else 0.0)))
    availability = pd.Series(availability, index=all_months)

    # Annualized turnover (value-based; uses costed usage + costed avg inventory)
    num_months = len(all_months)
    num_years = max(num_months / 12.0, 1e-9)
    avg_monthly_inv = float(alloc_avg_inv.mean()) if num_months else 0.0
    annual_usage = float(usage_month.sum()) / num_years
    turnover_annual = (annual_usage / avg_monthly_inv) if avg_monthly_inv > 0 else 0.0

    out = pd.DataFrame(
        {
            "Month": [str(p) for p in all_months],
            "Usage Value (PHP)": usage_month.values,
            "Average Inventory Value (PHP)": alloc_avg_inv.values,
            "Inventory Turnover (Annualized)": turnover_annual,
            "Carrying Cost (PHP)": carry_month.values,
            "Stockout Events": s_month["stockout_events"].values,
            "Downtime Hours": s_month["downtime_hours"].values,
            "Availability Index": availability.values,
        }
    )
    out["Check Type"] = check_type
    out["System"] = "Manual (Traditional)"
    return out

# -----------------------------
# Lightweight forecast (no sklearn)
# -----------------------------
def ridge_fit_predict(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Multi-target ridge regression with intercept.
    x: (n, p), y: (n, k)
    returns: yhat (n, k)
    """
    x = x.astype(float)
    y = y.astype(float)

    ones = np.ones((x.shape[0], 1), dtype=float)
    x1 = np.hstack([ones, x])

    p = x1.shape[1]
    xtx = x1.T @ x1
    reg = alpha * np.eye(p, dtype=float)
    reg[0, 0] = 0.0  # don't penalize intercept
    b = np.linalg.solve(xtx + reg, x1.T @ y)
    return x1 @ b

def fit_monthly_demand_forecast(
    usage_tbl: pd.DataFrame,
    work_orders_tbl: pd.DataFrame,
    parts_master: pd.DataFrame,
    check_type: str,
) -> pd.DataFrame:
    if usage_tbl["month"].isna().all():
        return pd.DataFrame()

    months = pd.period_range(usage_tbl["month"].min(), usage_tbl["month"].max(), freq="M")

    # Work-order features per month
    wo = work_orders_tbl.copy()
    wo = wo[wo["month"].isin(months)].copy()
    wo_counts = wo.pivot_table(index="month", columns="check_type", values="work_order", aggfunc="count")
    wo_counts = wo_counts.fillna(0.0).reindex(months, fill_value=0.0)
    wo_counts.index.name = "month"
    for c in ["A", "B", "C", "D"]:
        if c not in wo_counts.columns:
            wo_counts[c] = 0.0
    wo_counts = wo_counts[["A", "B", "C", "D"]].copy()

    feat = wo_counts.reset_index()
    feat["t"] = np.arange(len(feat))

    # seasonality
    month_num = feat["month"].astype(str).str[-2:].astype(int)
    feat["sin"] = np.sin(2.0 * math.pi * month_num / 12.0)
    feat["cos"] = np.cos(2.0 * math.pi * month_num / 12.0)

    x = feat[["t", "sin", "cos", "A", "B", "C", "D"]].to_numpy(dtype=float)

    u = usage_tbl.copy()
    if check_type != "All":
        u = u[u["check_type"] == check_type].copy()

    demand = u.pivot_table(index="month", columns="part_id", values="qty_issued", aggfunc="sum").fillna(0.0)
    demand = demand.reindex(months, fill_value=0.0)

    part_list = parts_master["part_id"].astype(str).tolist()
    for pid in part_list:
        if pid not in demand.columns:
            demand[pid] = 0.0
    demand = demand[part_list]

    y = demand.to_numpy(dtype=float)
    yhat = ridge_fit_predict(x, y, alpha=1.0)
    yhat = np.clip(yhat, 0.0, None)

    return pd.DataFrame(yhat, index=months, columns=part_list)

# -----------------------------
# Planned monthly KPIs + reorder recommendations (formula-based)
# -----------------------------
def compute_planned_monthly_kpis(
    parts_master: pd.DataFrame,
    usage_tbl: pd.DataFrame,
    stockout_tbl: pd.DataFrame,
    work_orders_tbl: pd.DataFrame,
    check_type: str,
    annual_carry_rate: float,
    service_level_v: float,
    service_level_e: float,
    service_level_d: float,
    order_cover_months: float,
    blend_with_current: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = parts_master.copy()
    parts["part_id"] = parts["part_id"].astype(str)

    pred_qty = fit_monthly_demand_forecast(usage_tbl, work_orders_tbl, parts, check_type)
    if pred_qty.empty:
        empty_kpi = pd.DataFrame(columns=[
            "Month", "Usage Value (PHP)", "Average Inventory Value (PHP)", "Inventory Turnover (Annualized)",
            "Carrying Cost (PHP)", "Stockout Events", "Downtime Hours", "Availability Index", "Check Type", "System"
        ])
        return empty_kpi, pd.DataFrame()

    cost_map = parts.set_index("part_id")["unit_cost_php"]

    # Observed usage value per month (still from records) EXCLUDE missing-cost parts
    u = usage_tbl.copy()
    if check_type != "All":
        u = u[u["check_type"] == check_type].copy()
    u["unit_cost_php"] = u["part_id"].map(cost_map)
    u = u[u["unit_cost_php"].notna()].copy()
    u["usage_value_php"] = u["qty_issued"].astype(float) * u["unit_cost_php"].astype(float)
    usage_value_month = u.groupby("month")["usage_value_php"].sum().sort_index()

    months = pred_qty.index
    usage_value_month = usage_value_month.reindex(months, fill_value=0.0)

    # Learn downtime per short qty from observed stockout records
    s_obs = stockout_tbl.copy()
    if check_type != "All":
        s_obs = s_obs[s_obs["check_type"] == check_type].copy()
    total_short = float(to_num(s_obs["short_qty"], 0.0).sum()) if len(s_obs) else 0.0
    total_down = float(to_num(s_obs["downtime_hours"], 0.0).sum()) if len(s_obs) else 0.0
    downtime_per_short = (total_down / total_short) if total_short > 0 else 0.0

    # Service level by VED
    sl_map = {"V": service_level_v, "E": service_level_e, "D": service_level_d}
    parts["service_level"] = parts["criticality_VED"].map(sl_map).fillna(service_level_d)
    parts["z"] = parts["service_level"].map(service_level_to_z)

    # Historical monthly demand std (for safety stock)
    u_hist = usage_tbl.copy()
    if check_type != "All":
        u_hist = u_hist[u_hist["check_type"] == check_type].copy()
    dem_hist = u_hist.pivot_table(index="month", columns="part_id", values="qty_issued", aggfunc="sum").fillna(0.0)
    dem_hist = dem_hist.reindex(months, fill_value=0.0)

    pid_list = parts["part_id"].tolist()
    for pid in pid_list:
        if pid not in dem_hist.columns:
            dem_hist[pid] = 0.0
    dem_hist = dem_hist[pid_list]

    dem_std = dem_hist.std(axis=0, ddof=0).to_numpy(dtype=float)

    # Lead time in months
    lt_months = (to_num(parts["lead_time_days_mean"], 28.0) / 30.0).to_numpy(dtype=float)
    sigma_lt = dem_std * np.sqrt(np.clip(lt_months, 0.05, None))
    z = parts["z"].to_numpy(dtype=float)

    # Predicted monthly demand (months x parts)
    d = pred_qty[pid_list].to_numpy(dtype=float)

    order_cover_months = float(max(order_cover_months, 0.25))
    q = np.maximum(1.0, np.round(d * order_cover_months))

    mu_lt = d * lt_months  # broadcast
    safety_stock = z * sigma_lt  # (parts,) -> broadcast

    current_avg_qty = to_num(parts["avg_on_hand_qty"], 0.0).to_numpy(dtype=float)
    calc_avg = (q / 2.0) + mu_lt + safety_stock
    blend = float(np.clip(blend_with_current, 0.0, 1.0))
    avg_on_hand_reco = (blend * current_avg_qty) + ((1.0 - blend) * calc_avg)

    # Expected shortage & stockout events (standard formulas)
    loss = np.array([normal_loss(float(zz)) for zz in z], dtype=float)
    shortage_per_cycle = sigma_lt * loss
    cycles = np.divide(d, q, out=np.zeros_like(d), where=(q > 0))
    exp_short_qty = cycles * shortage_per_cycle

    prob_stockout = np.array([1.0 - norm_cdf(float(zz)) for zz in z], dtype=float)
    exp_stockout_events = cycles * prob_stockout

    # Inventory value & carrying cost: EXCLUDE missing-cost parts
    cost_mask = parts["unit_cost_php"].notna().to_numpy(dtype=bool)
    if cost_mask.any():
        unit_cost_costed = parts.loc[cost_mask, "unit_cost_php"].astype(float).to_numpy()
        avg_on_hand_costed = avg_on_hand_reco[:, cost_mask]
        inv_value_month = (avg_on_hand_costed * unit_cost_costed).sum(axis=1)
    else:
        inv_value_month = np.zeros(len(months), dtype=float)

    carrying_cost_month = inv_value_month * (annual_carry_rate / 12.0)

    exp_short_qty_month = exp_short_qty.sum(axis=1)
    exp_stockout_events_month = exp_stockout_events.sum(axis=1)
    downtime_month = exp_short_qty_month * downtime_per_short

    availability = []
    for per, down in zip(months, downtime_month):
        m_start = per.to_timestamp()
        days_in_month = int((m_start + pd.offsets.MonthEnd(0)).day)
        hours = float(days_in_month * 24)
        availability.append(max(0.0, 1.0 - (float(down) / hours if hours > 0 else 0.0)))
    availability = np.array(availability, dtype=float)

    # Annualized turnover (value-based; uses costed usage + costed avg inventory)
    num_months = len(months)
    num_years = max(num_months / 12.0, 1e-9)
    avg_monthly_inv = float(np.mean(inv_value_month)) if num_months else 0.0
    annual_usage = float(usage_value_month.sum()) / num_years
    turnover_annual = (annual_usage / avg_monthly_inv) if avg_monthly_inv > 0 else 0.0

    planned_kpis = pd.DataFrame(
        {
            "Month": [str(p) for p in months],
            "Usage Value (PHP)": usage_value_month.values,
            "Average Inventory Value (PHP)": inv_value_month,
            "Inventory Turnover (Annualized)": turnover_annual,
            "Carrying Cost (PHP)": carrying_cost_month,
            "Stockout Events": exp_stockout_events_month,
            "Downtime Hours": downtime_month,
            "Availability Index": availability,
        }
    )
    planned_kpis["Check Type"] = check_type
    planned_kpis["System"] = "Predictive Analytics"

    # Reorder recommendation table (policy summary)
    avg_d = d.mean(axis=0)
    avg_q = q.mean(axis=0)
    avg_mu_lt = mu_lt.mean(axis=0)
    avg_ss = np.broadcast_to(safety_stock, d.shape).mean(axis=0)

    policy_df = parts[[
        "part_id", "description", "part_type", "abc_class", "criticality_VED",
        "unit_cost_php", "unit_cost_source", "lead_time_days_mean"
    ]].copy()

    policy_df = policy_df.rename(
        columns={
            "part_id": "Part Number",
            "description": "Description",
            "part_type": "Type",
            "abc_class": "ABC",
            "criticality_VED": "VED",
            "unit_cost_php": "Unit Cost (PHP)",
            "unit_cost_source": "Unit Cost Source",
            "lead_time_days_mean": "Lead Time (Days)",
        }
    )

    policy_df["Forecast Avg Monthly Qty"] = np.round(avg_d, 2)
    policy_df["Recommended Order Qty"] = np.maximum(1, np.round(avg_q)).astype(int)
    policy_df["Reorder Point Qty"] = np.maximum(1, np.round(avg_mu_lt + avg_ss)).astype(int)
    policy_df["Safety Stock Qty"] = np.maximum(0, np.round(avg_ss)).astype(int)
    policy_df["Service Level Target"] = policy_df["VED"].map({"V": service_level_v, "E": service_level_e, "D": service_level_d}).fillna(service_level_d)
    policy_df["Check Type"] = check_type

    return planned_kpis, policy_df

# -----------------------------
# UI helpers
# -----------------------------
def show_loading(placeholder: st.delta_generator.DeltaGenerator, message: str, progress_value: float) -> None:
    with placeholder.container():
        st.markdown(
            f"""<div class="loading-wrap">
            <div class="loading-title">Loading<span class="dots"><span></span><span></span><span></span></span></div>
            <div class="loading-sub">{message}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(progress_value)

def bar_compare(a: float, b: float, title: str, labels: Tuple[str, str] = ("Manual", "Predictive")) -> plt.Figure:
    fig = plt.figure()
    plt.bar([labels[0], labels[1]], [a, b])
    plt.title(title)
    plt.tight_layout()
    return fig

def summarize_overall(monthly_df: pd.DataFrame) -> dict:
    if monthly_df is None or len(monthly_df) == 0:
        return {
            "months_count": 0,
            "usage_total": 0.0,
            "avg_inventory_value": 0.0,
            "carrying_cost_total": 0.0,
            "carrying_cost_avg_month": 0.0,
            "carrying_cost_annual_implied": 0.0,
            "stockouts_total": 0.0,
            "downtime_total": 0.0,
            "availability_avg": 0.0,
            "turnover_annual": 0.0,
        }

    months_count = int(len(monthly_df))
    usage_total = float(to_num(monthly_df["Usage Value (PHP)"], 0.0).sum())
    avg_inv = float(to_num(monthly_df["Average Inventory Value (PHP)"], 0.0).mean())

    carry_series = to_num(monthly_df["Carrying Cost (PHP)"], 0.0)
    carry_total = float(carry_series.sum())
    carry_avg_month = float(carry_series.mean())
    carry_annual_implied = float(carry_avg_month * 12.0)

    stockouts_total = float(to_num(monthly_df["Stockout Events"], 0.0).sum())
    downtime_total = float(to_num(monthly_df["Downtime Hours"], 0.0).sum())
    avail_avg = float(to_num(monthly_df["Availability Index"], 0.0).mean())
    turnover = float(to_num(monthly_df["Inventory Turnover (Annualized)"], 0.0).iloc[0]) if months_count else 0.0

    return {
        "months_count": months_count,
        "usage_total": usage_total,
        "avg_inventory_value": avg_inv,
        "carrying_cost_total": carry_total,
        "carrying_cost_avg_month": carry_avg_month,
        "carrying_cost_annual_implied": carry_annual_implied,
        "stockouts_total": stockouts_total,
        "downtime_total": downtime_total,
        "availability_avg": avail_avg,
        "turnover_annual": turnover,
    }

def pct_change(manual_val: float, planned_val: float) -> float:
    manual_val = float(manual_val)
    planned_val = float(planned_val)
    if manual_val == 0:
        return 0.0
    return (planned_val - manual_val) / manual_val * 100.0

def build_summary_by_check(manual_all: pd.DataFrame, planned_all: pd.DataFrame, check_list: list[str]) -> pd.DataFrame:
    rows = []
    for check in check_list:
        m_df = manual_all[manual_all["Check Type"] == check].copy()
        p_df = planned_all[planned_all["Check Type"] == check].copy()
        km = summarize_overall(m_df)
        kp = summarize_overall(p_df)
        rows.append(
            {
                "Check Type": check,
                "Manual Stockouts (Total)": km["stockouts_total"],
                "Planned Stockouts (Total)": kp["stockouts_total"],
                "Manual Downtime (Hours)": km["downtime_total"],
                "Planned Downtime (Hours)": kp["downtime_total"],
                "Manual Turnover (Annual)": km["turnover_annual"],
                "Planned Turnover (Annual)": kp["turnover_annual"],
                "Manual Carry Cost Avg/Month (PHP)": km["carrying_cost_avg_month"],
                "Planned Carry Cost Avg/Month (PHP)": kp["carrying_cost_avg_month"],
                "Manual Carry Cost Annual (PHP)": km["carrying_cost_annual_implied"],
                "Planned Carry Cost Annual (PHP)": kp["carrying_cost_annual_implied"],
                "Manual Carry Cost Total (PHP)": km["carrying_cost_total"],
                "Planned Carry Cost Total (PHP)": kp["carrying_cost_total"],
                "Manual Availability (Avg)": km["availability_avg"],
                "Planned Availability (Avg)": kp["availability_avg"],
            }
        )
    return pd.DataFrame(rows)

def style_kpi_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Safe formatting for KPI tables
    currency_cols = [
        "Usage Value (PHP)",
        "Average Inventory Value (PHP)",
        "Carrying Cost (PHP)",
    ]
    float_cols = [
        "Inventory Turnover (Annualized)",
        "Downtime Hours",
        "Availability Index",
        "Stockout Events",
    ]
    fmt = {}
    for c in currency_cols:
        if c in df.columns:
            fmt[c] = lambda v: fmt_php(v, 2)
    for c in float_cols:
        if c in df.columns:
            # stockouts might be float for planned; show 2 decimals
            fmt[c] = lambda v: fmt_num(v, 4) if c == "Inventory Turnover (Annualized)" else fmt_num(v, 2)
    if "Stockout Events" in df.columns:
        # override to 2 decimals for planned; keep consistent
        fmt["Stockout Events"] = lambda v: fmt_num(v, 2)
    return df.style.format(fmt)

def style_summary_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    fmt = {}
    for col in df.columns:
        if "PHP" in col or "Cost" in col:
            fmt[col] = lambda v: fmt_php(v, 2)
        elif "Hours" in col:
            fmt[col] = lambda v: fmt_num(v, 2)
        elif "Turnover" in col:
            fmt[col] = lambda v: fmt_num(v, 4)
        elif "Availability" in col:
            fmt[col] = lambda v: fmt_num(v, 4)
        elif "Stockouts" in col:
            fmt[col] = lambda v: fmt_num(v, 2)
    return df.style.format(fmt)

def build_data_quality_report(parts_master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pm = parts_master.copy()

    total_parts = int(len(pm))
    with_cost = int(pm["unit_cost_php"].notna().sum())
    missing_cost = int(pm["unit_cost_php"].isna().sum())

    summary = pd.DataFrame({
        "Metric": [
            "Parts count (master)",
            "Parts with written cost (Receipts/PO)",
            "Parts missing cost",
            "Parts with written cost (%)",
            "Parts missing cost (%)",
            "Unit cost (median of written costs)",
        ],
        "Value": [
            total_parts,
            with_cost,
            missing_cost,
            (with_cost / max(total_parts, 1)) * 100.0,
            (missing_cost / max(total_parts, 1)) * 100.0,
            float(pm.loc[pm["unit_cost_php"].notna(), "unit_cost_php"].median()) if with_cost else 0.0,
        ]
    })

    # Top missing-cost parts by avg on-hand qty (since we can't compute value without cost)
    top_missing = pm[pm["unit_cost_php"].isna()].copy()
    top_missing = top_missing.sort_values("avg_on_hand_qty", ascending=False).head(50)
    top_missing = top_missing.rename(columns={
        "part_id": "Part Number",
        "description": "Description",
        "avg_on_hand_qty": "Avg On-hand Qty",
        "unit_cost_source": "Cost Source",
    })[["Part Number", "Description", "Avg On-hand Qty", "Cost Source"]]

    return summary, top_missing

# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

st.sidebar.markdown(f"### {APP_TITLE}")
st.sidebar.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload the 36-month workbook (.xlsx)",
    type=["xlsx"],
)

selected_tab = st.sidebar.radio(
    "View",
    ["Overview", "KPI Tables", "Reorder Plan", "Data Preview", "Data Quality", "Exports"],
    index=0,
)

st.markdown(f"## {APP_TITLE}")
st.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown(
        """<div class="card">
        <h3>How to use</h3>
        <div class="muted">
        1) Upload the 36-month workbook (manual logs + monthly inventory snapshots).<br/>
        2) Manual KPIs are computed from the recorded history.<br/>
        3) Predictive view forecasts demand + reorder parameters, then estimates KPI impact using standard inventory formulas.<br/><br/>
        <b>Note:</b> Value-based KPIs (inventory value, carrying cost, turnover) exclude parts with no written cost in Receipts/PO.
        </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# Single loading screen
loading_placeholder = st.empty()
start_time = time.time()
file_bytes = uploaded_file.getvalue()

show_loading(loading_placeholder, "Reading workbook…", 0.20)
sheet_map = read_excel_sheets(file_bytes)

show_loading(loading_placeholder, "Parsing snapshots and logs…", 0.55)
parsed = parse_workbook(sheet_map)

show_loading(loading_placeholder, "Building cost and lead-time tables…", 0.80)
unit_cost_map = build_unit_cost_map(parsed["receipts"], parsed["purchase_orders"])
lt_stats = build_lead_time_stats(parsed["purchase_orders"])
parts_master = build_parts_master(parsed["inventory_long"], unit_cost_map, lt_stats)

# Keep loader visible for at least 2 seconds (smooth UX)
elapsed = time.time() - start_time
remaining = 2.0 - elapsed
if remaining > 0:
    time.sleep(remaining)

loading_placeholder.empty()

# Check-type selection
usage_tbl = parsed["usage"]
check_choices = ["All"]
if "check_type" in usage_tbl.columns:
    uniq_checks = sorted({str(x).strip() for x in usage_tbl["check_type"].dropna().unique() if str(x).strip()})
    check_choices += uniq_checks
selected_check = st.sidebar.selectbox("Maintenance Check Type", check_choices, index=0)

check_list = ["All"] + [c for c in check_choices if c != "All"]

def build_all_tables(
    check_types: list[str],
    parts_master_df: pd.DataFrame,
    parsed_tables: Dict[str, pd.DataFrame],
    annual_carry_rate: float,
    service_level_v: float,
    service_level_e: float,
    service_level_d: float,
    order_cover_months: float,
    blend_with_current: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual_rows: list[pd.DataFrame] = []
    planned_rows: list[pd.DataFrame] = []
    policy_rows: list[pd.DataFrame] = []

    for check in check_types:
        manual_ct = compute_manual_monthly_kpis(
            parts_master=parts_master_df,
            inv_long=parsed_tables["inventory_long"],
            usage_tbl=parsed_tables["usage"],
            stockout_tbl=parsed_tables["stockout"],
            check_type=check,
            annual_carry_rate=annual_carry_rate,
        )
        planned_ct, policy_ct = compute_planned_monthly_kpis(
            parts_master=parts_master_df,
            usage_tbl=parsed_tables["usage"],
            stockout_tbl=parsed_tables["stockout"],
            work_orders_tbl=parsed_tables["work_orders"],
            check_type=check,
            annual_carry_rate=annual_carry_rate,
            service_level_v=service_level_v,
            service_level_e=service_level_e,
            service_level_d=service_level_d,
            order_cover_months=order_cover_months,
            blend_with_current=blend_with_current,
        )
        manual_rows.append(manual_ct)
        planned_rows.append(planned_ct)
        if len(policy_ct):
            policy_rows.append(policy_ct)

    manual_all_df = pd.concat(manual_rows, ignore_index=True) if manual_rows else pd.DataFrame()
    planned_all_df = pd.concat(planned_rows, ignore_index=True) if planned_rows else pd.DataFrame()
    policy_all_df = pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()
    return manual_all_df, planned_all_df, policy_all_df

manual_all, planned_all, policy_all = build_all_tables(
    check_types=check_list,
    parts_master_df=parts_master,
    parsed_tables=parsed,
    annual_carry_rate=ANNUAL_CARRY_RATE,
    service_level_v=SERVICE_LEVEL_V,
    service_level_e=SERVICE_LEVEL_E,
    service_level_d=SERVICE_LEVEL_D,
    order_cover_months=ORDER_COVER_MONTHS,
    blend_with_current=BLEND_WITH_CURRENT,
)

manual_sel = manual_all[manual_all["Check Type"] == selected_check].copy() if len(manual_all) else pd.DataFrame()
planned_sel = planned_all[planned_all["Check Type"] == selected_check].copy() if len(planned_all) else pd.DataFrame()
policy_sel = policy_all[policy_all["Check Type"] == selected_check].copy() if len(policy_all) else pd.DataFrame()

k_manual = summarize_overall(manual_sel)
k_planned = summarize_overall(planned_sel)

months_label = f"{k_manual['months_count']} mo" if k_manual["months_count"] else "period"

summary_selected = pd.DataFrame(
    [
        {"Metric": "Stockout Events (total)", "Manual": k_manual["stockouts_total"], "Predictive": k_planned["stockouts_total"]},
        {"Metric": "Downtime Hours (total)", "Manual": k_manual["downtime_total"], "Predictive": k_planned["downtime_total"]},
        {"Metric": "Inventory Turnover (annualized)", "Manual": k_manual["turnover_annual"], "Predictive": k_planned["turnover_annual"]},

        {"Metric": "Carrying Cost (avg / month)", "Manual": k_manual["carrying_cost_avg_month"], "Predictive": k_planned["carrying_cost_avg_month"]},
        {"Metric": "Carrying Cost (annual implied)", "Manual": k_manual["carrying_cost_annual_implied"], "Predictive": k_planned["carrying_cost_annual_implied"]},
        {"Metric": f"Carrying Cost (total, {months_label})", "Manual": k_manual["carrying_cost_total"], "Predictive": k_planned["carrying_cost_total"]},

        {"Metric": "Availability Index (avg)", "Manual": k_manual["availability_avg"], "Predictive": k_planned["availability_avg"]},
    ]
)
summary_selected["% Change (Predictive vs Manual)"] = summary_selected.apply(
    lambda r: round(pct_change(r["Manual"], r["Predictive"]), 2), axis=1
)

summary_by_check = build_summary_by_check(manual_all, planned_all, check_list) if (len(manual_all) and len(planned_all)) else pd.DataFrame()

# -----------------------------
# Views
# -----------------------------
if selected_tab == "Overview":
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Manual (Traditional)")
        st.metric("Stockout Events (total)", fmt_int(k_manual["stockouts_total"]))
        st.metric("Downtime Hours (total)", fmt_num(k_manual["downtime_total"], 2))
        st.metric("Inventory Turnover (annualized)", fmt_num(k_manual["turnover_annual"], 4))
        st.metric("Avg Inventory Value (PHP / month)", fmt_php(k_manual["avg_inventory_value"], 2))

        st.metric("Carrying Cost (avg / month)", fmt_php(k_manual["carrying_cost_avg_month"], 2))
        st.metric("Carrying Cost (annual implied)", fmt_php(k_manual["carrying_cost_annual_implied"], 2))
        st.metric(f"Carrying Cost (total, {months_label})", fmt_php(k_manual["carrying_cost_total"], 2))

        st.metric("Availability Index (avg)", fmt_num(k_manual["availability_avg"], 4))

    with col2:
        st.subheader("Predictive Analytics (Forecast + Reorder Plan)")
        st.metric("Stockout Events (expected total)", fmt_num(k_planned["stockouts_total"], 2))
        st.metric("Downtime Hours (expected total)", fmt_num(k_planned["downtime_total"], 2))
        st.metric("Inventory Turnover (annualized)", fmt_num(k_planned["turnover_annual"], 4))
        st.metric("Avg Inventory Value (PHP / month)", fmt_php(k_planned["avg_inventory_value"], 2))

        st.metric("Carrying Cost (avg / month)", fmt_php(k_planned["carrying_cost_avg_month"], 2))
        st.metric("Carrying Cost (annual implied)", fmt_php(k_planned["carrying_cost_annual_implied"], 2))
        st.metric(f"Carrying Cost (total, {months_label})", fmt_php(k_planned["carrying_cost_total"], 2))

        st.metric("Availability Index (avg)", fmt_num(k_planned["availability_avg"], 4))

    st.divider()
    st.subheader(f"Results Summary (Check Type: {selected_check})")
    # format the summary table
    summary_display = summary_selected.copy()
    summary_display["Manual"] = summary_display["Metric"].apply(
        lambda m: m
    )
    # Use styler for proper formatting:
    def _format_metric_row(row):
        metric = row["Metric"]
        if "PHP" in metric or "Carrying Cost" in metric:
            return fmt_php(row["Manual"]), fmt_php(row["Predictive"])
        if "Downtime" in metric:
            return fmt_num(row["Manual"], 2), fmt_num(row["Predictive"], 2)
        if "Turnover" in metric:
            return fmt_num(row["Manual"], 4), fmt_num(row["Predictive"], 4)
        if "Availability" in metric:
            return fmt_num(row["Manual"], 4), fmt_num(row["Predictive"], 4)
        if "Stockout" in metric:
            return fmt_num(row["Manual"], 2), fmt_num(row["Predictive"], 2)
        return fmt_num(row["Manual"], 2), fmt_num(row["Predictive"], 2)

    tmp = summary_selected.copy()
    formatted = tmp.apply(lambda r: pd.Series(_format_metric_row(r), index=["Manual_fmt", "Predictive_fmt"]), axis=1)
    summary_show = tmp.copy()
    summary_show["Manual"] = formatted["Manual_fmt"]
    summary_show["Predictive"] = formatted["Predictive_fmt"]
    st.dataframe(summary_show, use_container_width=True)

    st.divider()
    st.subheader("Results Summary (All Check Types)")
    if len(summary_by_check):
        st.dataframe(style_summary_df(summary_by_check), use_container_width=True)
    else:
        st.info("No summary available.")

    st.divider()
    st.subheader("Comparison Charts (Selected Check Type)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.pyplot(bar_compare(k_manual["stockouts_total"], k_planned["stockouts_total"], "Stockout Events (lower is better)"))
    with c2:
        st.pyplot(bar_compare(k_manual["downtime_total"], k_planned["downtime_total"], "Downtime Hours (lower is better)"))
    with c3:
        st.pyplot(bar_compare(k_manual["turnover_annual"], k_planned["turnover_annual"], "Inventory Turnover (higher is better)"))
    with c4:
        st.pyplot(bar_compare(k_manual["carrying_cost_total"], k_planned["carrying_cost_total"], f"Carrying Cost Total ({months_label})"))

    st.divider()
    st.subheader("Monthly Trend (Selected Check Type)")
    if len(manual_sel) and len(planned_sel):
        trend_df = pd.DataFrame(
            {
                "Month": manual_sel["Month"],
                "Manual - Stockouts": manual_sel["Stockout Events"],
                "Predictive - Stockouts": planned_sel["Stockout Events"],
                "Manual - Downtime": manual_sel["Downtime Hours"],
                "Predictive - Downtime": planned_sel["Downtime Hours"],
                "Manual - Carrying Cost": manual_sel["Carrying Cost (PHP)"],
                "Predictive - Carrying Cost": planned_sel["Carrying Cost (PHP)"],
            }
        )
        st.line_chart(trend_df.set_index("Month"))
    else:
        st.info("Not enough data to show trends for this selection.")

elif selected_tab == "KPI Tables":
    st.subheader(f"Monthly KPI Tables (Check Type: {selected_check})")

    left, right = st.columns(2)
    with left:
        st.markdown("### Manual (Traditional)")
        if len(manual_sel):
            st.dataframe(style_kpi_df(manual_sel), use_container_width=True, height=420)
        else:
            st.info("No manual KPI data.")
    with right:
        st.markdown("### Predictive Analytics")
        if len(planned_sel):
            st.dataframe(style_kpi_df(planned_sel), use_container_width=True, height=420)
        else:
            st.info("No predictive KPI data.")

    st.divider()
    st.subheader("Summary by Check Type (Single Table)")
    if len(summary_by_check):
        st.dataframe(style_summary_df(summary_by_check), use_container_width=True)
    else:
        st.info("No summary available.")

elif selected_tab == "Reorder Plan":
    st.subheader(f"Reorder Plan (Check Type: {selected_check})")
    if len(policy_sel) == 0:
        st.info("No reorder recommendations available for this selection (insufficient data).")
    else:
        # light formatting for unit cost
        policy_show = policy_sel.copy()
        if "Unit Cost (PHP)" in policy_show.columns:
            policy_show["Unit Cost (PHP)"] = policy_show["Unit Cost (PHP)"].apply(lambda v: fmt_php(v, 2) if pd.notna(v) else "")
        st.dataframe(policy_show, use_container_width=True, height=520)

elif selected_tab == "Data Preview":
    st.subheader("Data Preview (what the app reads)")
    with st.expander("Parts Master (derived)", expanded=True):
        pm = parts_master.copy()
        pm["unit_cost_php"] = pm["unit_cost_php"].apply(lambda v: fmt_php(v, 2) if pd.notna(v) else "")
        st.dataframe(pm.head(200), use_container_width=True)

    with st.expander("Inventory Snapshots (long format)"):
        st.dataframe(parsed["inventory_long"].head(200), use_container_width=True)

    with st.expander("Usage Issues Log (normalized)"):
        st.dataframe(parsed["usage"].head(200), use_container_width=True)

    with st.expander("AOG Stockout Log (normalized)"):
        st.dataframe(parsed["stockout"].head(200), use_container_width=True)

    with st.expander("Purchase Orders (normalized)"):
        st.dataframe(parsed["purchase_orders"].head(200), use_container_width=True)

    with st.expander("Receipts Log (normalized)"):
        st.dataframe(parsed["receipts"].head(200), use_container_width=True)

    with st.expander("Work Orders (normalized)"):
        st.dataframe(parsed["work_orders"].head(200), use_container_width=True)

elif selected_tab == "Data Quality":
    st.subheader("Data Quality Checks (Written Cost Coverage)")
    summary_q, top_missing = build_data_quality_report(parts_master)

    # format the summary
    summary_show = summary_q.copy()
    def _fmt_summary(metric, value):
        if "(%)" in metric:
            return fmt_num(value, 2)
        if "median" in metric.lower():
            return fmt_php(value, 2)
        if "count" in metric.lower() or "parts" in metric.lower():
            return fmt_int(value)
        return fmt_num(value, 2)

    summary_show["Value"] = [
        _fmt_summary(m, v) for m, v in zip(summary_show["Metric"], summary_show["Value"])
    ]

    st.markdown("### Cost Coverage Summary")
    st.dataframe(summary_show, use_container_width=True)

    missing_pct = float(summary_q.loc[summary_q["Metric"] == "Parts missing cost (%)", "Value"].values[0])
    if missing_pct > 0:
        st.warning(
            f"{missing_pct:.1f}% of parts have no written cost in Receipts/PO. "
            "Those parts are excluded from all PHP-based KPIs (inventory value, carrying cost, turnover)."
        )

    st.markdown("### Top Missing-Cost Parts (by Avg On-hand Qty)")
    st.dataframe(top_missing, use_container_width=True, height=520)

else:
    st.subheader("Exports")

    monthly_export = pd.concat([manual_all, planned_all], ignore_index=True)

    st.download_button(
        "Download Monthly KPI Table (All check types) - CSV",
        data=monthly_export.to_csv(index=False).encode("utf-8"),
        file_name="monthly_kpis_all_check_types.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if len(summary_by_check):
        st.download_button(
            "Download Summary (Per check type) - CSV",
            data=summary_by_check.to_csv(index=False).encode("utf-8"),
            file_name="summary_by_check_type.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if len(policy_all) > 0:
        st.download_button(
            "Download Reorder Plan (All check types) - CSV",
            data=policy_all.to_csv(index=False).encode("utf-8"),
            file_name="reorder_plan_all_check_types.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        "Download Parts Master (with cost source) - CSV",
        data=parts_master.to_csv(index=False).encode("utf-8"),
        file_name="parts_master_with_cost_source.csv",
        mime="text/csv",
        use_container_width=True,
    )
