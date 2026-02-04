import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# Basic setup & logging
# =========================
def log_info(msg): print(msg)
def log_success(msg): print(msg)
def log_warning(msg): print(msg)
def log_error(msg): print(msg)

auth_url = "https://www.strava.com/oauth/token"
activities_url = "https://www.strava.com/api/v3/athlete/activities"

st.set_page_config(page_title="Strava Progress", layout="wide")
st.title("🎈 Strava Progress")
st.caption("Select a year, adjust your goal, sync from Strava (incremental for current year), and explore insights.")

# =========================
# Sidebar controls
# =========================
current_year = datetime.now().year
years_options = list(range(2015, current_year + 1))
with st.sidebar:
    st.header("Settings")
    YEAR_SELECTED = st.selectbox(
        "Year",
        options=years_options,
        index=len(years_options) - 1,
        help="Choose the year to analyze."
    )
    YEAR_GOAL_KM = st.slider(
        "Yearly goal (km)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50,
        help="Target total kilometers for the selected year.",
    )
    chart_view = st.radio(
        "Chart view",
        options=["Cumulative", "Daily"],
        index=0,
        help="Switch between cumulative or per-day view."
    )
    force_refresh = st.button("🔄 Force re-sync", help="Re-download data for the selected year (ignores incremental cache).")
    clear_cache_btn = st.button("🗑️ Clear disk cache", help="Remove cached data for the selected year.")

# =========================
# Cache paths (disk-backed, per year)
# =========================
def _cache_paths(year: int):
    base = Path(".strava_cache") / str(year)
    base.mkdir(parents=True, exist_ok=True)
    return base / "activities.json", base / "meta.json"

def load_cache_from_disk(year: int):
    activities_by_id, meta = {}, {}
    ACTIVITIES_FILE, META_FILE = _cache_paths(year)
    try:
        if ACTIVITIES_FILE.exists():
            with ACTIVITIES_FILE.open("r", encoding="utf-8") as f:
                activities_by_id = json.load(f)
        if META_FILE.exists():
            with META_FILE.open("r", encoding="utf-8") as f:
                meta = json.load(f)
    except Exception as e:
        log_warning(f"Cache load issue for {year}: {e}")
    return activities_by_id, meta

def save_cache_to_disk(year: int, activities_by_id: dict, meta: dict):
    ACTIVITIES_FILE, META_FILE = _cache_paths(year)
    try:
        with ACTIVITIES_FILE.open("w", encoding="utf-8") as f:
            json.dump(activities_by_id, f, ensure_ascii=False)
        with META_FILE.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception as e:
        log_warning(f"Cache save issue for {year}: {e}")

def clear_disk_cache(year: int):
    ACTIVITIES_FILE, META_FILE = _cache_paths(year)
    try:
        if ACTIVITIES_FILE.exists(): ACTIVITIES_FILE.unlink()
        if META_FILE.exists(): META_FILE.unlink()
    except Exception as e:
        log_warning(f"Cache clear issue for {year}: {e}")

# =========================
# Session cache (per year)
# =========================
if "cache" not in st.session_state:
    st.session_state["cache"] = {}  # year -> {id: activity}
if "meta" not in st.session_state:
    st.session_state["meta"] = {}   # year -> {last_after_epoch, last_synced_utc}

# =========================
# Strava auth & fetch
# =========================
def get_access_token():
    payload = {
        "client_id": st.secrets["strava_client_id"],
        "client_secret": st.secrets["strava_client_secret"],
        "refresh_token": st.secrets["strava_refresh_token"],
        "grant_type": "refresh_token",
    }
    res = requests.post(auth_url, data=payload)
    res.raise_for_status()
    return res.json()["access_token"]

def get_activities_page(token: str, page: int, per_page: int = 200,
                        after_epoch: int = None, before_epoch: int = None):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"per_page": per_page, "page": page}
    if after_epoch is not None:
        params["after"] = after_epoch
    if before_epoch is not None:
        params["before"] = before_epoch
    res = requests.get(activities_url, headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected data format: {type(data)} - {data}")
    return data

def minimalize_act(act: dict) -> dict:
    """Store only what we need to keep the cache compact."""
    return {
        "id": act.get("id"),
        "type": act.get("type"),
        "distance": float(act.get("distance", 0.0)),
        "start_date": act.get("start_date"),               # UTC ISO
        "start_date_local": act.get("start_date_local"),   # local ISO
    }

def to_epoch_utc(utc_iso: str) -> int:
    # Accept both Z and +00:00 forms
    return int(datetime.fromisoformat(utc_iso.replace("Z", "+00:00")).timestamp())

def fetch_activities_for_year(token: str, year: int, force: bool = False):
    """
    For the current year: incremental sync using 'after'.
    For past years: fetch once using [after Jan 1, before Jan 1 next year].
    Both use disk cache; 'force' bypasses and re-fetches.
    """
    # Load session cache for year; if empty or forced, load from disk
    if force or (year not in st.session_state["cache"]):
        disk_acts, disk_meta = load_cache_from_disk(year)
        st.session_state["cache"][year] = disk_acts or {}
        st.session_state["meta"][year] = disk_meta or {}

    # Clear cache if user clicked
    if clear_cache_btn:
        clear_disk_cache(year)
        st.session_state["cache"][year] = {}
        st.session_state["meta"][year] = {}
        st.toast(f"Disk cache for {year} cleared.", icon="🗑️")

    # Boundaries
    start_utc = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_utc_exclusive = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    start_epoch = int(start_utc.timestamp())
    end_epoch = int(end_utc_exclusive.timestamp())

    activities_by_id = st.session_state["cache"][year]
    meta = st.session_state["meta"][year]

    # Current year → incremental, else → full year fetch if not cached
    is_current = (year == current_year)
    need_fetch = force or (is_current) or (not activities_by_id)

    new_count = 0
    if need_fetch:
        # Determine after/before
        after_epoch = meta.get("last_after_epoch", start_epoch) if is_current else start_epoch
        before_epoch = None if is_current else end_epoch

        page = 1
        with st.spinner(f"🔍 Retrieving activities for {year}..."):
            while True:
                try:
                    page_data = get_activities_page(
                        token, page=page, after_epoch=after_epoch, before_epoch=before_epoch
                    )
                except requests.exceptions.HTTPError as e:
                    log_error(f"HTTP error: {e}")
                    st.error(f"HTTP error while fetching activities: {getattr(e, 'response', None) and e.response.text}")
                    break

                if not page_data:
                    break

                for act in page_data:
                    m = minimalize_act(act)
                    activities_by_id[str(m["id"])] = m
                new_count += len(page_data)
                page += 1

                if len(page_data) < 200:
                    break

        # For current year, slide the 'after' boundary to the newest activity
        if activities_by_id:
            latest_epoch = max(
                to_epoch_utc(a["start_date"]) for a in activities_by_id.values() if a.get("start_date")
            )
            if is_current:
                meta["last_after_epoch"] = latest_epoch
        meta["last_synced_utc"] = datetime.now(timezone.utc).isoformat()

        # Persist to disk
        save_cache_to_disk(year, activities_by_id, meta)

        if new_count > 0:
            st.toast(f"✨ Synced {new_count} activities for {year}.", icon="✅")
        else:
            st.toast(f"{year} is up to date. No new activities found.", icon="👌")

    # Reflect back into session
    st.session_state["cache"][year] = activities_by_id
    st.session_state["meta"][year] = meta

    # Return list of activities
    return list(activities_by_id.values())

# =========================
# Transformations (tz-naive dates)
# =========================
def process_activities(activities: list, year: int) -> pd.DataFrame:
    """
    Aggregate runs/virtual runs by local date within selected year, distance in km.
    Returns DataFrame with columns: date (Timestamp, tz-naive), distance.
    """
    rows = []
    for act in activities:
        if act.get("type") in ["Run", "VirtualRun"]:
            # Parse start_date_local and strip timezone info if present
            start_local_ts = pd.to_datetime(act.get("start_date_local"), errors="coerce")
            if pd.isna(start_local_ts):
                continue

            # If tz-aware, drop timezone to make tz-naive
            if getattr(start_local_ts, "tz", None) is not None:
                start_local_ts = start_local_ts.tz_localize(None)

            # Keep only rows for the selected year (local year)
            if start_local_ts.year != year:
                continue

            # Normalize time to midnight to keep date-only granularity
            start_local_ts = start_local_ts.normalize()

            dist_km = float(act.get("distance", 0.0)) / 1000.0
            rows.append({"date": start_local_ts, "distance": dist_km})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "distance"])

    # Enforce tz-naive dtype for the whole column (guardrail)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    return df.groupby("date", as_index=False).sum().sort_values("date")

def build_progress(df_actual: pd.DataFrame, year_goal_km: float, year: int) -> pd.DataFrame:
    df_actual = df_actual.copy()

    # Ensure pandas Timestamps and strip timezone info → tz-naive
    df_actual["date"] = pd.to_datetime(df_actual["date"], errors="coerce")
    if hasattr(df_actual["date"].dtype, "tz") and df_actual["date"].dtype.tz is not None:
        df_actual["date"] = df_actual["date"].dt.tz_localize(None)

    start_ts = pd.Timestamp(year=year, month=1, day=1)
    # For current year, end at today; for past years, end at Dec 31
    end_ts = pd.Timestamp(datetime.now().date()) if year == current_year else pd.Timestamp(year=year, month=12, day=31)

    # Goal date range (tz-naive)
    date_range = pd.date_range(start=start_ts, end=end_ts, freq="D")

    year_end_ts = pd.Timestamp(year=year, month=12, day=31)
    days_in_year = (year_end_ts - start_ts).days + 1

    df_goal = pd.DataFrame({"date": date_range})
    # (df_goal['date'] is tz-naive already)
    df_goal["day_number"] = (df_goal["date"] - start_ts).dt.days + 1
    df_goal["goal_cum"] = (df_goal["day_number"] / days_in_year) * year_goal_km
    df_goal["goal"] = df_goal["goal_cum"].diff().fillna(df_goal["goal_cum"])

    # Now both 'date' columns are tz-naive → safe to merge
    df_full = pd.merge(df_goal, df_actual, how="left", on="date")
    df_full["distance"] = pd.to_numeric(df_full["distance"], errors="coerce").fillna(0.0)

    df_full["actual_cum"] = df_full["distance"].cumsum()
    df_full["diff"] = df_full["actual_cum"] - df_full["goal_cum"]
    df_full["diff_day"] = df_full["distance"] - df_full["goal"]
    return df_full

# =========================
# Visualization
# =========================
def render_chart(df_progress: pd.DataFrame, year_goal_km: float, view: str, year: int):
    if df_progress.empty:
        st.info("No data to display yet.")
        return

    # Guardrail: ensure tz-naive in the plotted frame
    df_progress = df_progress.copy()
    df_progress["date"] = pd.to_datetime(df_progress["date"], errors="coerce")
    if hasattr(df_progress["date"].dtype, "tz") and df_progress["date"].dtype.tz is not None:
        df_progress["date"] = df_progress["date"].dt.tz_localize(None)

    if view == "Cumulative":
        st.subheader(f"Cumulative Goal vs. Actuals — {year} (Target: {year_goal_km} km)")
        bar_series = df_progress["diff"]
        line_goal = df_progress["goal_cum"]
        line_actual = df_progress["actual_cum"]
        y_title = "Kilometers (cumulative)"
        bar_name = "Difference (cum)"
        line_goal_name = "Target (cum)"
        line_actual_name = "Actual (cum)"
    else:
        st.subheader(f"Daily Goal vs. Actuals — {year} (Target: {year_goal_km} km)")
        bar_series = df_progress["diff_day"]
        line_goal = df_progress["goal"]
        line_actual = df_progress["distance"]
        y_title = "Kilometers per day"
        bar_name = "Difference (day)"
        line_goal_name = "Target (day)"
        line_actual_name = "Actual (day)"

    bar_colors = ["#2ca02c" if float(v) >= 0 else "#d62728" for v in bar_series]

    fig = go.Figure()

    # Bars for difference
    fig.add_trace(
        go.Bar(
            x=df_progress["date"],
            y=bar_series,
            name=bar_name,
            marker_color=bar_colors,
            opacity=0.5,
        )
    )
    # Goal line
    fig.add_trace(
        go.Scatter(
            x=df_progress["date"],
            y=line_goal,
            name=line_goal_name,
            mode="lines",
            line=dict(color="#6c757d", width=2, dash="dash"),
        )
    )
    # Actual line
    fig.add_trace(
        go.Scatter(
            x=df_progress["date"],
            y=line_actual,
            name=line_actual_name,
            mode="lines",
            line=dict(color="#1f77b4", width=3),
        )
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Date", type="date"),
        yaxis=dict(title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        shapes=[
            dict(
                type="line",
                xref="paper", x0=0, x1=1,
                yref="y", y0=0, y1=0,
                line=dict(color="#cccccc", width=1),
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Analysis section
# =========================
def compute_analysis(df_progress: pd.DataFrame, activities: list, year_goal_km: float, year: int):
    if df_progress.empty:
        st.info("No analysis yet — no runs recorded.")
        return

    latest = df_progress.iloc[-1]
    total_km = float(latest["actual_cum"])
    diff_km = float(latest["diff"])

    # Counts (filter activities to selected year using start_date_local)
    run_acts = []
    for a in activities:
        if a.get("type") not in ["Run", "VirtualRun"]:
            continue
        ts = pd.to_datetime(a.get("start_date_local"), errors="coerce")
        if pd.isna(ts):
            continue
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_localize(None)
        if ts.year == year:
            run_acts.append(a)
    run_count = len(run_acts)

    # Unique run days
    df_days = df_progress[df_progress["distance"] > 0.0]
    run_days = int((df_days["date"]).nunique())

    # Longest single run (by activity)
    longest_km = 0.0
    longest_date = None
    if run_acts:
        longest = max(run_acts, key=lambda x: float(x.get("distance", 0.0)))
        longest_km = float(longest.get("distance", 0.0)) / 1000.0
        longest_date = pd.to_datetime(longest.get("start_date_local")).date()

    # Weekly best (sum by ISO week)
    df_weekly = (
        df_progress.groupby(pd.to_datetime(df_progress["date"]).dt.isocalendar().week)["distance"]
        .sum()
        .reset_index(name="km")
    )
    best_week_km = float(df_weekly["km"].max()) if not df_weekly.empty else 0.0

    # Streaks (days with distance > 0)
    dates_with_run = sorted([pd.to_datetime(d).date() for d in df_days["date"].tolist()])
    best_streak = 0
    current_streak = 0
    if dates_with_run:
        streak = 1
        best_streak = 1
        for i in range(1, len(dates_with_run)):
            if dates_with_run[i] == dates_with_run[i - 1] + timedelta(days=1):
                streak += 1
            else:
                best_streak = max(best_streak, streak)
                streak = 1
        best_streak = max(best_streak, streak)

        # current streak for selected year (ends at last date in that year)
        last_date = dates_with_run[-1]
        if year == current_year and last_date == datetime.now().date():
            cs = 1
            for i in range(len(dates_with_run) - 2, -1, -1):
                if dates_with_run[i] == dates_with_run[i + 1] - timedelta(days=1):
                    cs += 1
                else:
                    break
            current_streak = cs
        else:
            current_streak = 0

    # Required pace to hit goal from "now" (for past years, 0 remaining)
    start_year = pd.Timestamp(year=year, month=1, day=1)
    end_year = pd.Timestamp(year=year, month=12, day=31)
    days_in_year = (end_year - start_year).days + 1

    if year == current_year:
        day_of_year = (pd.Timestamp(datetime.now().date()) - start_year).days + 1
        remaining_days = max(0, days_in_year - day_of_year)
    else:
        day_of_year = days_in_year
        remaining_days = 0

    remaining_km = max(0.0, year_goal_km - total_km)
    required_avg_per_day = (remaining_km / remaining_days) if remaining_days > 0 else 0.0
    goal_avg_per_day = year_goal_km / days_in_year
    days_ahead_equiv = (diff_km / goal_avg_per_day) if goal_avg_per_day > 0 else 0.0

    st.subheader("Analysis & Highlights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total distance (YTD)", f"{total_km:.1f} km")
    c2.metric("Runs", f"{run_count}")
    c3.metric("Active days", f"{run_days}")
    c4.metric("Ahead/Behind", f"{diff_km:+.1f} km ({days_ahead_equiv:+.1f} days)")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Longest run", f"{longest_km:.1f} km", f"{longest_date}" if longest_date else None)
    c6.metric("Best week total", f"{best_week_km:.1f} km")
    c7.metric("Current streak", f"{current_streak} days")
    c8.metric("Req. avg/day to hit goal", f"{required_avg_per_day:.2f} km")

    # Weekday distribution (sum of km by weekday)
    df_weekday = df_progress.copy()
    df_weekday["weekday"] = pd.to_datetime(df_weekday["date"]).dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_weekday = (
        df_weekday.groupby("weekday", as_index=False)["distance"]
        .sum()
        .set_index("weekday")
        .reindex(weekday_order, fill_value=0)
        .reset_index()
    )

    fig_wd = go.Figure(
        data=[
            go.Bar(
                x=df_weekday["weekday"],
                y=df_weekday["distance"],
                marker_color="#4e79a7",
                name="KM",
            )
        ]
    )
    fig_wd.update_layout(
        title="Distance by Weekday",
        xaxis_title="Weekday",
        yaxis_title="Kilometers",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    # Monthly totals
    df_month = df_progress.copy()
    df_month["month"] = pd.to_datetime(df_month["date"]).dt.to_period("M").astype(str)
    df_month = df_month.groupby("month", as_index=False)["distance"].sum()
    fig_mo = go.Figure(
        data=[
            go.Bar(
                x=df_month["month"],
                y=df_month["distance"],
                marker_color="#59a14f",
                name="KM",
            )
        ]
    )
    fig_mo.update_layout(
        title="Monthly Totals",
        xaxis_title="Month",
        yaxis_title="Kilometers",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    c9, c10 = st.columns(2)
    c9.plotly_chart(fig_wd, use_container_width=True)
    c10.plotly_chart(fig_mo, use_container_width=True)

    with st.expander("Details & raw data (processed)"):
        st.dataframe(df_progress)

# =========================
# Run pipeline
# =========================
def run():
    try:
        with st.spinner("🔐 Authenticating with Strava..."):
            token = get_access_token()
        log_success("Access token retrieved.")

        st.write(f"⏳ Checking for activities in {YEAR_SELECTED}…")
        activities = fetch_activities_for_year(token, YEAR_SELECTED, force=force_refresh)
        log_success(f"Cached activities for {YEAR_SELECTED}: {len(activities)}")

        st.write("🧮 Processing runs…")
        df_activities = process_activities(activities, YEAR_SELECTED)
        df_progress = build_progress(df_activities, YEAR_GOAL_KM, YEAR_SELECTED)

        # Last update info
        last_synced = st.session_state.get("meta", {}).get(YEAR_SELECTED, {}).get("last_synced_utc")
        if last_synced:
            ts = datetime.fromisoformat(last_synced).astimezone()
            st.caption(f"Last synced: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Chart
        render_chart(df_progress, YEAR_GOAL_KM, chart_view, YEAR_SELECTED)

        # Analysis
        compute_analysis(df_progress, activities, YEAR_GOAL_KM, YEAR_SELECTED)

    except Exception as e:
        log_error(f"Something went wrong: {e}")
        st.error(f"Something went wrong: {e}")

run()