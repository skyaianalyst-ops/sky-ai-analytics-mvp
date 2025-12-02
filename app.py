import os
import io
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

# ================================================================
# ENVIRONMENT SETUP
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_PATH = CONFIG_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI() if OPENAI_API_KEY else None

st.set_page_config(page_title="AI Analytics MVP", page_icon="📊", layout="wide")

st.title("📊 AI Analytics MVP")
st.caption("Excel / CSV → Auto Catalog → Auto Insights → NL Q&A (Agents 1–3)")


# ================================================================
# REPORT MANAGEMENT (POWER BI–STYLE)
# ================================================================

def get_reports() -> Dict[str, Dict[str, Any]]:
    """Return the dict of reports in session_state."""
    return st.session_state.setdefault("reports", {})


def get_saved_views() -> Dict[str, Dict[str, Any]]:
    """Return saved views (date + categorical filters)."""
    return st.session_state.setdefault("saved_views", {})


def ensure_active_report_id() -> Optional[str]:
    """Ensure there is a valid active_report_id if reports exist."""
    reports = get_reports()
    if not reports:
        return None
    rid = st.session_state.get("active_report_id")
    if rid is None or rid not in reports:
        rid = next(iter(reports.keys()))
        st.session_state["active_report_id"] = rid
    return rid


def get_active_report() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return (report_id, report_dict) for the active report."""
    reports = get_reports()
    rid = ensure_active_report_id()
    if rid is None or rid not in reports:
        return None, None
    return rid, reports[rid]


# ================================================================
# SIDEBAR – NAVIGATION + REPORT PICKER
# ================================================================

with st.sidebar:
    st.header("📁 Reports")

    reports = get_reports()
    active_id = ensure_active_report_id()

    if reports:
        report_ids = list(reports.keys())
        default_index = 0
        if active_id in report_ids:
            default_index = report_ids.index(active_id)

        selected_id = st.selectbox(
            "Active report",
            options=report_ids,
            index=default_index,
            format_func=lambda rid: reports[rid]["name"],
        )
        st.session_state["active_report_id"] = selected_id
        active_id = selected_id

        # Rename active report (like tab rename in Power BI)
        current_name = reports[active_id]["name"]
        name_key = f"report_name_{active_id}"
        new_name = st.text_input(
            "Report name",
            value=current_name,
            key=name_key,
        ).strip()
        if new_name:
            reports[active_id]["name"] = new_name

    else:
        st.info("No reports yet. Upload a file to create your first report.")

    st.header("🧭 Navigation")
    section = st.radio(
        "Go to:",
        [
            "1️⃣ Upload Data",
            "2️⃣ Catalog + Summary (Auto)",
            "3️⃣ Full Data Insights (Auto)",
            "4️⃣ Ask a Question (Agent 2)",
            "5️⃣ Logs & Debug",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("Data source: Excel / CSV")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY missing in config/.env – AI features limited.")


# ================================================================
# UTILITIES – CLEANING & PREPROCESSING
# ================================================================

def clean_column_name(col: str) -> str:
    """Convert column names to snake_case for LLM-friendly usage."""
    return (
        str(col)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes dataframe columns, converts potential date columns."""
    df = df.copy()

    # Rename columns to snake_case
    rename_map = {col: clean_column_name(col) for col in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Convert potential date/time columns
    for col in df.columns:
        if any(x in col for x in ["date", "time", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            except Exception:
                pass

    return df


def is_identifier_column(name: str) -> bool:
    """Heuristic to detect ID / serial columns that should not be main metrics."""
    n = name.lower()
    keywords = ["id", "code", "serial", "sno", "s_no", "srno", "sr_no", "no_"]
    return any(k in n for k in keywords)


def choose_main_metric_column(df: pd.DataFrame) -> Optional[str]:
    """
    Pick the best business metric column (price / amount / revenue / qty).
    Works generically across datasets.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    priority_keywords = [
        "net_amount",
        "gross_amount",
        "amount",
        "revenue",
        "sales",
        "gmv",
        "sellprice",
        "sale_price",
        "mrp",
        "price",
        "value",
        "qty",
        "quantity",
        "units",
    ]

    lower_map = {c: c.lower() for c in numeric_cols}
    for kw in priority_keywords:
        for col in numeric_cols:
            if kw in lower_map[col]:
                return col

    candidate_numeric = [c for c in numeric_cols if not is_identifier_column(c)]
    if candidate_numeric:
        return candidate_numeric[0]

    return numeric_cols[0]


def pretty_label(name: str) -> str:
    """Turn snake_case / lowercase into 'Nice Title Case'."""
    return name.replace("_", " ").title()


def sample_dataframe(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Return first max_rows rows to protect performance."""
    if len(df) > max_rows:
        return df.head(max_rows).copy()
    return df.copy()


# ================================================================
# CATALOG BUILDER + SEMANTIC INDEX
# ================================================================

def build_data_catalog(dataframes: Dict[str, pd.DataFrame]):
    """
    Builds a compact catalog + semantic index from uploaded dataframes.
    Returns:
        catalog: per-sheet summary
        semantic_index: per-column type info for LLM
    """
    catalog: Dict[str, Any] = {}
    semantic_index: Dict[str, Any] = {}

    for sheet, df in dataframes.items():
        processed_df = preprocess_dataframe(df)
        dataframes[sheet] = processed_df  # overwrite with processed version

        sheet_info = {
            "rows": len(processed_df),
            "columns": [],
        }

        for col in processed_df.columns:
            series = processed_df[col]
            col_info = {
                "name": col,
                "dtype": str(series.dtype),
                "sample_values": series.dropna().head(5).tolist(),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "is_date": bool(pd.api.types.is_datetime64_any_dtype(series)),
                "uniq_ratio": round(series.nunique() / max(len(series), 1), 4),
            }

            semantic_index[col] = {
                "sheet": sheet,
                "dtype": col_info["dtype"],
                "is_numeric": col_info["is_numeric"],
                "is_date": col_info["is_date"],
            }

            sheet_info["columns"].append(col_info)

        catalog[sheet] = sheet_info

    return catalog, semantic_index


def get_main_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    """Pick the 'main' sheet – the one with the most rows."""
    best_sheet = None
    best_df = None
    max_rows = -1

    for sheet, df in dataframes.items():
        n = len(df)
        if n > max_rows:
            max_rows = n
            best_sheet = sheet
            best_df = df

    return best_sheet, best_df


# ================================================================
# AGENT 1 – DATASET SUMMARY (AUTO) + VIZ BLUEPRINT
# ================================================================

def agent1_summary(catalog: Dict[str, Any]) -> str:
    """LLM-based dataset overview + KPI suggestions (text only)."""
    if not client:
        return "OPENAI_API_KEY missing – cannot generate AI summary."

    system_prompt = """
    You are a senior data analyst.
    You receive a JSON-like data catalog describing sheets, columns, types, and example values.
    Your job:
    1) Give a short dataset overview.
    2) Identify key entities (e.g., customers, orders, products, dates).
    3) Propose 8–12 practical business KPIs and metrics.
    4) Mention which columns/sheets each KPI would use.
    Focus on historical performance and descriptive analytics.
    """

    user_prompt = f"DATA CATALOG:\n{catalog}"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Agent 1 error: {e})"


def agent1_viz_blueprint(catalog: Dict[str, Any], main_sheet: str) -> Optional[List[Dict[str, Any]]]:
    """
    Agent 1 (extended) – analyse catalog + main sheet and design a viz_plan.
    viz_plan = list of chart specs:
    {
        "title": "string",
        "chart_type": "bar" | "column" | "line" | "area" |
                      "scatter" | "histogram" | "pie" |
                      "heatmap" | "boxplot" | "bubble",
        "x": "column_name",
        "y": "column_name or null",
        "agg": "sum" | "mean" | "count" | "none",
        "top_n": int or null,
        "size": "column_name or null",
        "color": "column_name or null"
    }
    """
    if not client:
        return None

    system_prompt = """
    You are AGENT 1: Data & Visualization Architect for an AI Analytics app.

    You receive:
    - A DATA CATALOG describing all sheets and columns.
    - The NAME of the MAIN SHEET that will be used for the primary dashboard.

    Your job:
    1) Analyse the main sheet and imagine a rich Power BI / Tableau style dashboard.
    2) Design a VIZ PLAN listing as many meaningful charts as possible (ideally 8–15),
       covering DIFFERENT visualization types, such as:
       - Line Charts (trend over time)
       - Bar / Column Charts (top N categories, rankings, comparisons)
       - Scatter Plots (relationship between two numeric metrics)
       - Histograms (distribution of one numeric field)
       - Pie Charts (share of total for a small number of categories)
       - Heat Maps (two dimensions with color intensity)
       - Box Plots (distribution & outliers by category)
       - Area Charts (cumulative trends)
       - Bubble Charts (scatter plot where bubble size shows magnitude)
    3) Use only columns that exist in the MAIN SHEET.
    4) You are primarily doing descriptive analytics on historical data;
       the app may overlay simple predictions on top of time trends.
    5) Allowed chart_type values:
       "bar", "column", "line", "area",
       "scatter", "histogram", "pie",
       "heatmap", "boxplot", "bubble".
    6) Aggregations: "sum", "mean", "count", or "none".

    OUTPUT FORMAT:
    - Return STRICT JSON ONLY, no markdown, no explanation.
    - Shape:
      {
        "viz_plan": [
          {
            "title": "...",
            "chart_type": "bar",
            "x": "category_col",
            "y": "net_amount",
            "agg": "sum",
            "top_n": 15,
            "size": null,
            "color": null
          }
        ]
      }
    """

    user_prompt = f"""
    DATA CATALOG:
    {catalog}

    MAIN SHEET NAME:
    {main_sheet}

    Remember:
    - Use ONLY columns from the main sheet.
    - Design as many meaningful charts as you can (ideally 8–15),
      mixing different chart types from the allowed list.
    - Return STRICT JSON with top-level key "viz_plan".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""

        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
                if raw.strip().lower().startswith("json"):
                    raw = raw.split("\n", 1)[1]
        raw = raw.strip()

        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

        data = json.loads(raw)
        viz_plan = data.get("viz_plan")
        if isinstance(viz_plan, list):
            return viz_plan
        return None
    except Exception as e:
        st.warning(f"Agent 1 viz blueprint failed: {e}")
        return None


# ================================================================
# AGENT 2 – NL → PANDAS ENGINE
# ================================================================

FORBIDDEN_TOKENS = [
    "import ",
    "open(",
    "os.",
    "subprocess",
    "eval(",
    "exec(",
    "__builtins__",
    "__import__",
    "system(",
    "popen(",
]


def is_safe_code(code: str) -> Tuple[bool, Optional[str]]:
    """Basic static check for obviously unsafe patterns."""
    low = code.lower()
    for token in FORBIDDEN_TOKENS:
        if token in low:
            return False, f"Unsafe token detected in generated code: {token}"
    return True, None


def agent2_generate_code(
    question: str,
    extra: str,
    catalog: Dict[str, Any],
    semantic_index: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    LLM generates Pandas code from NL question.
    Returns (code, error_message).
    """
    if not client:
        return None, "OPENAI_API_KEY missing – cannot generate code."

    system_prompt = """
    You are AGENT 2: Elite Pandas Code Generator.
    Your job: Convert user questions into pure Pandas code.

    ENVIRONMENT:
    - All dataframes are accessible via a dict called "dataframes".
    - Keys are sheet names like "CSV_File", "orders", etc.
    - Each value is a cleaned pandas DataFrame with snake_case column names.
    - Pandas is imported as 'pd', numpy as 'np'.

    ABSOLUTE RULES:
    1. DO NOT import anything (no 'import pandas', etc.).
    2. DO NOT read or write any files.
    3. DO NOT print anything.
    4. DO NOT use any OS, subprocess, system, or network operations.
    5. You MAY:
       - select, filter, groupby, aggregate
       - join dataframes using common key columns
       - create new calculated columns
    6. The FINAL result must be a pandas DataFrame assigned to a variable named: result_df
       Example last line:  result_df = ...
    7. Do NOT include explanations or comments. ONLY valid Python code.
    8. If question mentions things like "last 6 months", "last 30 days":
       - assume datetime columns are already converted using pd.to_datetime.
       - use pd.Timestamp.now() for current date and subtract appropriate offsets.
    9. Prefer readable, stepwise code over dense one-liners.
    10. You are working on historical data present in the tables; you may compute
        derived metrics, but do not simulate complex ML models.
    """

    memory: List[str] = st.session_state.get("query_memory", [])

    user_prompt = f"""
    USER QUESTION:
    {question}

    EXTRA INSTRUCTIONS:
    {extra}

    DATA CATALOG (JSON-like):
    {catalog}

    SEMANTIC INDEX (column → sheet + type info):
    {semantic_index}

    LAST QUERIES (to learn user style):
    {memory[-5:]}

    Now write ONLY the Python code that uses 'dataframes' and produces a pandas DataFrame
    assigned to 'result_df'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # changed from gpt-4.1
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        code = response.choices[0].message.content or ""

        if "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]
                if code.strip().lower().startswith("python"):
                    code = code.split("\n", 1)[1]

        code = code.strip()
        return code, None

    except Exception as e:
        return None, f"(Agent 2 code-generation error: {e})"


def agent2_self_heal(code: str, error: str, question: str) -> str:
    """Ask LLM to fix broken Pandas code."""
    if not client:
        return code

    prompt = f"""
    The following Pandas code failed with this error:

    ERROR:
    {error}

    ORIGINAL CODE:
    {code}

    USER QUESTION:
    {question}

    Fix the code. Follow these rules:
    - No imports.
    - Use only pandas (pd) and numpy (np).
    - Dataframes dict is named 'dataframes'.
    - The fixed code must assign the final DataFrame to 'result_df'.
    - Do NOT add explanations, only code.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You fix Pandas code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        new_code = response.choices[0].message.content or ""
        if "```" in new_code:
            parts = new_code.split("```")
            if len(parts) >= 3:
                new_code = parts[1]
                if new_code.strip().lower().startswith("python"):
                    new_code = new_code.split("\n", 1)[1]
        return new_code.strip()
    except Exception:
        return code


def run_safe_code(
    code: str, dataframes: Dict[str, pd.DataFrame]
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute code with a restricted global namespace.
    Expect 'result_df' to be produced.
    """
    safe_globals = {
        "pd": pd,
        "np": np,
        "dataframes": dataframes,
    }
    safe_locals: Dict[str, Any] = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        return None, f"Execution error: {e}"

    if "result_df" not in safe_locals:
        return None, "Generated code did not create 'result_df'."

    result = safe_locals["result_df"]
    if not isinstance(result, pd.DataFrame):
        return None, "result_df is not a pandas DataFrame."

    return result, None


# ================================================================
# AGENT 3 – RESULT-LEVEL CHART + INSIGHT + FOLLOW-UP QUESTIONS
# ================================================================

def detect_chart_type(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    """Choose the best chart type based on dataframe structure."""
    cols = list(df.columns)

    if df.shape == (1, 1) and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        return "kpi", cols[0], None

    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                return "line", col, numeric_cols[0]

    cat_cols = [c for c in cols if df[c].dtype == "object"]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols and num_cols:
        return "bar", cat_cols[0], num_cols[0]

    if len(num_cols) >= 2:
        return "scatter", num_cols[0], num_cols[1]

    return "table", None, None


def generate_chart(
    df: pd.DataFrame, chart_type: str, x_col: Optional[str], y_col: Optional[str]
):
    """Build an Altair chart or KPI text from the data."""
    if chart_type == "kpi":
        value = float(df.iloc[0, 0])
        col_name = df.columns[0]
        return f"### 💰 {pretty_label(col_name)}: **{value:,.2f}**"

    if chart_type == "line" and x_col and y_col:
        return (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{x_col}:T", title=pretty_label(x_col)),
                y=alt.Y(y_col, title=pretty_label(y_col)),
                tooltip=[x_col, y_col],
            )
            .properties(height=400)
        )

    if chart_type in ["bar", "column"] and x_col and y_col:
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(x_col, sort="-y", title=pretty_label(x_col)),
                y=alt.Y(y_col, title=pretty_label(y_col)),
                tooltip=[x_col, y_col],
            )
            .properties(height=400)
        )

    if chart_type == "scatter" and x_col and y_col:
        return (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x=alt.X(x_col, title=pretty_label(x_col)),
                y=alt.Y(y_col, title=pretty_label(y_col)),
                tooltip=df.columns.tolist(),
            )
            .properties(height=400)
        )

    if chart_type == "area" and x_col and y_col:
        return (
            alt.Chart(df)
            .mark_area(opacity=0.7)
            .encode(
                x=alt.X(f"{x_col}:T", title=pretty_label(x_col)),
                y=alt.Y(y_col, title=pretty_label(y_col)),
                tooltip=[x_col, y_col],
            )
            .properties(height=400)
        )

    return None


def agent3_insight(df: pd.DataFrame) -> str:
    """LLM-based short business insight from result_df."""
    if not client:
        return "LLM key missing – cannot generate automatic insight."

    prompt = f"""
    You are AGENT 3: Final Business Insight Layer.

    Analyze the following table and give ONE clear business insight in 1–2 sentences.

    DATAFRAME SAMPLE (dict form):
    {df.head(20).to_dict()}

    Rules:
    - No heavy technical/statistical jargon.
    - Explain like you are talking to a business manager.
    - Focus mainly on historical patterns; if there is a clear trend,
      you may briefly hint what it suggests for the future.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Insight unavailable: {e})"


def agent3_followup_questions(question: str, df: pd.DataFrame) -> List[str]:
    """
    Generate 3 follow-up analytics questions based on the user's question and result_df.
    """
    if not client:
        return []

    prompt = f"""
    You are AGENT 3 in an AI analytics system.

    The user asked this analytics question:
    {question}

    You are given a sampled result table from the previous analysis.

    DATA SAMPLE:
    {df.head(20).to_dict()}

    TASK:
    - Propose 3 highly relevant follow-up analytics questions that the user may ask next.
    - Questions must:
      * Be short natural-language questions (1 sentence each).
      * Focus on deeper analysis, breakdowns (by segment, region, product), trends, or comparisons.
      * Refer to business concepts (like revenue, orders, customers) inferred from the data.
      * NOT mention 'above table', 'dataframe', or 'data sample'.

    OUTPUT:
    Return STRICT JSON ONLY:

    {{
      "questions": [
        "question 1 ...",
        "question 2 ...",
        "question 3 ..."
      ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
                if raw.strip().lower().startswith("json"):
                    raw = raw.split("\n", 1)[1]
        raw = raw.strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

        data = json.loads(raw)
        questions = data.get("questions", [])
        if isinstance(questions, list):
            return [str(q) for q in questions if isinstance(q, str)]
        return []
    except Exception as e:
        st.warning(f"Could not generate follow-up questions: {e}")
        return []


def explain_chart_button(df_small: pd.DataFrame, title: str, context: str, key: str) -> None:
    """LLM explanation for a specific chart."""
    if not client:
        return

    if st.button("Explain this chart", key=key):
        prompt = f"""
        You are a senior business intelligence analyst.

        Chart title:
        {title}

        Chart context / spec:
        {context}

        Data (first rows as dict):
        {df_small.head(50).to_dict()}

        TASK:
        - Explain this chart in 2–3 short bullet points.
        - Use simple business language (no heavy statistics).
        - Highlight key trends, top/bottom categories, or outliers.
        - Speak as if you are presenting to a business manager.
        """

        with st.spinner("Explaining chart..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                text = response.choices[0].message.content
                st.markdown(text)
            except Exception as e:
                st.error(f"Chart explanation failed: {e}")


def run_agent_3(df: Optional[pd.DataFrame]) -> None:
    """
    Safe Agent 3 runner:
    - Validates df
    - Limits rows for plotting (5k)
    - Wraps chart + insight generation in try/except
    """
    try:
        if df is None or df.empty:
            st.warning("No data available for Agent 3. Please run Agent 2 first.")
            return

        max_rows = 5000
        if len(df) > max_rows:
            st.info(
                f"Result has {len(df)} rows. Using a sample of {max_rows} rows "
                f"for faster charting in Agent 3."
            )
            df = df.head(max_rows)

        st.markdown("### 📈 Auto Chart for This Result (Agent 3)")
        chart_type, x_col, y_col = detect_chart_type(df)
        chart = generate_chart(df, chart_type, x_col, y_col)

        if chart_type == "kpi" and isinstance(chart, str):
            st.markdown(chart)
        elif chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No suitable chart type detected for this result. Showing table instead.")
            st.dataframe(df.head(50))

        st.markdown("### 💡 Quick Insight (Agent 3)")
        insight = agent3_insight(df)
        st.success(insight)

    except Exception as e:
        st.error("Agent 3 encountered an error and stopped safely instead of crashing the app.")
        st.exception(e)


# ================================================================
# DATASET-LEVEL AUTO INSIGHTS (FULL DATA – ANALYST MODE)
# ================================================================

def dataset_auto_insights_text(df: pd.DataFrame, sheet_name: str) -> str:
    """LLM narrative over the (possibly filtered) main dataframe."""
    if not client:
        return "LLM key missing – cannot generate narrative insights."

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    stats = {}
    if numeric_cols:
        stats["numeric_describe"] = df[numeric_cols].describe().to_dict()

    cat_samples = {}
    for c in cat_cols[:5]:
        cat_samples[c] = df[c].value_counts().head(10).to_dict()

    prompt = f"""
    You are an expert BI analyst (Power BI/Tableau style).
    You received a main dataset from sheet '{sheet_name}'.

    META:
    - Rows: {len(df)}
    - Columns: {list(df.columns)}
    - Numeric columns: {numeric_cols}
    - Categorical columns: {cat_cols}
    - Date columns: {date_cols}

    NUMERIC STATS:
    {stats}

    CATEGORY DISTRIBUTIONS (top values):
    {cat_samples}

    TASK:
    - Write a rich but concise analysis (3–6 short paragraphs).
    - Focus on trends, segments, outliers, distribution, top performers, etc.
    - You may gently mention what the trend implies for upcoming periods,
      but do not do complex forecasting.
    - Talk in plain business language, no 'dataframe' or raw column names.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Dataset insight unavailable: {e})"


# ================================================================
# VIZ PLAN RENDERING (Agent 1 blueprint → charts + simple prediction)
# ================================================================

def render_viz_plan(vis_df: pd.DataFrame, viz_plan: List[Dict[str, Any]]) -> None:
    """
    Execute Agent 1's viz_plan on a sampled + filtered dataframe.
    Uses Altair charts. Skips invalid specs safely.
    Adds a simple one-step-ahead prediction for time-series bar/column/line/area charts.
    """
    if not viz_plan:
        st.info("No visualization plan from Agent 1 – using fallback visuals.")
        return

    max_charts = 20  # safety cap
    count = 0

    for spec in viz_plan:
        if count >= max_charts:
            break

        title = spec.get("title", "Untitled Chart")
        chart_type = str(spec.get("chart_type", "bar")).lower()
        x = spec.get("x")
        y = spec.get("y")
        agg = (spec.get("agg") or "sum").lower()
        top_n = spec.get("top_n")
        size_col = spec.get("size")
        color_col = spec.get("color")

        if not x:
            continue
        if x not in vis_df.columns:
            continue
        if y is not None and y not in vis_df.columns:
            continue
        if size_col is not None and size_col not in vis_df.columns:
            size_col = None
        if color_col is not None and color_col not in vis_df.columns:
            color_col = None

        df = vis_df.copy()

        if chart_type == "column":
            chart_type = "bar"
        if chart_type == "box":
            chart_type = "boxplot"

        try:
            # ===================== BAR / LINE / AREA (aggregated, with prediction overlay) =====================
            if chart_type in ["bar", "line", "area"]:
                if y is None:
                    chart_df = df.groupby(x).size().reset_index(name="value")
                    y_col = "value"
                else:
                    if agg == "mean":
                        chart_df = df.groupby(x)[y].mean().reset_index()
                    elif agg == "count":
                        chart_df = df.groupby(x)[y].count().reset_index()
                    elif agg == "none":
                        chart_df = df[[x, y]].copy()
                    else:  # default sum
                        chart_df = df.groupby(x)[y].sum().reset_index()
                    y_col = y

                is_time_x = pd.api.types.is_datetime64_any_dtype(chart_df[x])

                chart_df["is_prediction"] = False

                if is_time_x and len(chart_df) >= 2:
                    chart_df = chart_df.sort_values(x)

                    last_date = chart_df[x].iloc[-1]
                    prev_date = chart_df[x].iloc[-2]
                    delta = last_date - prev_date
                    next_date = last_date + delta

                    last_val = float(chart_df[y_col].iloc[-1])
                    prev_val = float(chart_df[y_col].iloc[-2])
                    next_val = last_val + (last_val - prev_val)

                    pred_row = {x: next_date, y_col: next_val, "is_prediction": True}
                    chart_df = pd.concat([chart_df, pd.DataFrame([pred_row])], ignore_index=True)

                chart_df["series"] = np.where(chart_df["is_prediction"], "Prediction", "Actual")

                if (not is_time_x) and top_n and isinstance(top_n, int) and top_n > 0:
                    chart_df = chart_df.sort_values(y_col, ascending=False).head(top_n)

                st.markdown(f"#### {title}")

                if chart_type == "bar":
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(x, sort="-y", title=pretty_label(x)),
                            y=alt.Y(y_col, title=pretty_label(y_col)),
                            tooltip=[x, y_col, "series"],
                            color=alt.Color(
                                "series:N",
                                title="Type",
                                scale=alt.Scale(
                                    domain=["Actual", "Prediction"],
                                    range=["#4c78a8", "#f97316"],
                                ),
                            ),
                        )
                        .properties(height=350)
                    )
                elif chart_type == "line":
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(x, title=pretty_label(x)),
                            y=alt.Y(y_col, title=pretty_label(y_col)),
                            tooltip=[x, y_col, "series"],
                            color=alt.Color(
                                "series:N",
                                title="Type",
                                scale=alt.Scale(
                                    domain=["Actual", "Prediction"],
                                    range=["#4c78a8", "#f97316"],
                                ),
                            ),
                            strokeDash=alt.StrokeDash("series:N", legend=None),
                        )
                        .properties(height=350)
                    )
                else:  # area
                    chart = (
                        alt.Chart(chart_df)
                        .mark_area(opacity=0.7)
                        .encode(
                            x=alt.X(x, title=pretty_label(x)),
                            y=alt.Y(y_col, title=pretty_label(y_col)),
                            tooltip=[x, y_col, "series"],
                            color=alt.Color(
                                "series:N",
                                title="Type",
                                scale=alt.Scale(
                                    domain=["Actual", "Prediction"],
                                    range=["#4c78a8", "#f97316"],
                                ),
                            ),
                        )
                        .properties(height=350)
                    )

                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type={chart_type}, x={x}, y={y_col}, agg={agg}, top_n={top_n}, prediction_overlay={is_time_x}"
                explain_chart_button(chart_df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            # ===================== SCATTER / BUBBLE =====================
            if chart_type in ["scatter", "bubble"]:
                if not y:
                    continue

                st.markdown(f"#### {title}")
                df["series"] = "Actual"

                base = alt.Chart(df).encode(
                    x=alt.X(x, title=pretty_label(x)),
                    y=alt.Y(y, title=pretty_label(y)),
                    tooltip=df.columns.tolist(),
                )

                if chart_type == "bubble":
                    size_field = size_col if size_col else y
                    chart = base.mark_circle(opacity=0.7).encode(
                        size=alt.Size(size_field, legend=None),
                        color=alt.Color(color_col) if color_col else alt.value("#4c78a8"),
                    )
                else:
                    chart = base.mark_circle(size=60, opacity=0.7).encode(
                        color=alt.Color(color_col) if color_col else alt.value("#4c78a8"),
                    )

                chart = chart.properties(height=350)
                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type={chart_type}, x={x}, y={y}, size={size_col}, color={color_col}"
                explain_chart_button(df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            # ===================== HISTOGRAM =====================
            if chart_type == "histogram":
                if not pd.api.types.is_numeric_dtype(df[x]):
                    continue

                st.markdown(f"#### {title}")
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X(x, bin=alt.Bin(maxbins=40), title=pretty_label(x)),
                        y=alt.Y("count()", title="Record Count"),
                        tooltip=[x, "count()"],
                        color=alt.Color(color_col) if color_col else alt.value("#4c78a8"),
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type=histogram, x={x}"
                explain_chart_button(df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            # ===================== PIE =====================
            if chart_type == "pie":
                st.markdown(f"#### {title}")
                if y is None:
                    pie_df = df[x].value_counts().reset_index()
                    pie_df.columns = [x, "value"]
                    y_col = "value"
                else:
                    if agg == "mean":
                        pie_df = df.groupby(x)[y].mean().reset_index()
                    elif agg == "count":
                        pie_df = df.groupby(x)[y].count().reset_index()
                    elif agg == "none":
                        pie_df = df[[x, y]].copy()
                    else:
                        pie_df = df.groupby(x)[y].sum().reset_index()
                    y_col = y

                if top_n and isinstance(top_n, int) and top_n > 0:
                    pie_df = pie_df.sort_values(y_col, ascending=False).head(top_n)

                chart = (
                    alt.Chart(pie_df)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(y_col, stack=True),
                        color=alt.Color(x, legend=True, title=pretty_label(x)),
                        tooltip=[x, y_col],
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type=pie, x={x}, y={y_col}, top_n={top_n}"
                explain_chart_button(pie_df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            # ===================== HEATMAP =====================
            if chart_type == "heatmap":
                if not y:
                    continue

                st.markdown(f"#### {title}")
                hm_df = df.groupby([x, y]).size().reset_index(name="value")

                chart = (
                    alt.Chart(hm_df)
                    .mark_rect()
                    .encode(
                        x=alt.X(x, title=pretty_label(x)),
                        y=alt.Y(y, title=pretty_label(y)),
                        color=alt.Color("value", title="Count"),
                        tooltip=[x, y, "value"],
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type=heatmap, x={x}, y={y}"
                explain_chart_button(hm_df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            # ===================== BOXPLOT =====================
            if chart_type == "boxplot":
                if not y:
                    continue
                if not pd.api.types.is_numeric_dtype(df[y]):
                    continue

                st.markdown(f"#### {title}")
                chart = (
                    alt.Chart(df)
                    .mark_boxplot()
                    .encode(
                        x=alt.X(x, title=pretty_label(x)),
                        y=alt.Y(y, title=pretty_label(y)),
                        color=alt.Color(color_col) if color_col else alt.value("#4c78a8"),
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)

                context = f"chart_type=boxplot, x={x}, y={y}"
                explain_chart_button(df, title, context, key=f"explain_viz_{count}")

                count += 1
                continue

            continue

        except Exception as e:
            st.warning(f"Skipping one chart in viz_plan due to error: {e}")
            continue

    if count == 0:
        st.info("No valid charts could be rendered from Agent 1's viz_plan.")


def fallback_visuals(vis_df: pd.DataFrame, main_metric: Optional[str], cat_cols: List[str], date_cols: List[str]) -> None:
    """
    Simple heuristic visuals in case viz_plan fails or is missing.
    Uses 5K-row sample already (no prediction here).
    """
    if not main_metric:
        st.info("No numeric metric detected – fallback visuals limited.")
        return

    # Bar chart: top categories
    if cat_cols:
        sel_cat = cat_cols[0]
        cat_agg = (
            vis_df.groupby(sel_cat)[main_metric]
            .sum()
            .sort_values(ascending=False)
            .head(25)
        ).reset_index()

        st.markdown(f"#### {pretty_label(main_metric)} by {pretty_label(sel_cat)} (Top 25)")
        chart = (
            alt.Chart(cat_agg)
            .mark_bar()
            .encode(
                x=alt.X(sel_cat, sort="-y", title=pretty_label(sel_cat)),
                y=alt.Y(main_metric, title=pretty_label(main_metric)),
                tooltip=[sel_cat, main_metric],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

        context = f"chart_type=bar, x={sel_cat}, y={main_metric}, top_n=25"
        explain_chart_button(cat_agg, f"{pretty_label(main_metric)} by {pretty_label(sel_cat)}", context, key="explain_fb_bar")

    # Time series if date exists
    if date_cols:
        date_col = date_cols[0]
        tmp = vis_df.dropna(subset=[date_col]).copy()
        if not tmp.empty:
            time_agg = (
                tmp.groupby(date_col)[main_metric]
                .sum()
                .sort_index()
                .reset_index()
            )
            st.markdown(f"#### {pretty_label(main_metric)} over Time ({pretty_label(date_col)})")
            chart = (
                alt.Chart(time_agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X(date_col, title=pretty_label(date_col)),
                    y=alt.Y(main_metric, title=pretty_label(main_metric)),
                    tooltip=[date_col, main_metric],
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)

            context = f"chart_type=line, x={date_col}, y={main_metric}"
            explain_chart_button(time_agg, f"{pretty_label(main_metric)} over Time", context, key="explain_fb_line")

    # Histogram
    if pd.api.types.is_numeric_dtype(vis_df[main_metric]):
        st.markdown(f"#### Distribution of {pretty_label(main_metric)}")
        hist_df = pd.DataFrame({main_metric: vis_df[main_metric]})
        chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                x=alt.X(main_metric, bin=alt.Bin(maxbins=30), title=pretty_label(main_metric)),
                y=alt.Y("count()", title="Record Count"),
                tooltip=[main_metric, "count()"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

        context = f"chart_type=histogram, x={main_metric}"
        explain_chart_button(hist_df, f"Distribution of {pretty_label(main_metric)}", context, key="explain_fb_hist")


# ================================================================
# SECTION ROUTING
# ================================================================

# 1️⃣ Upload Data
if section == "1️⃣ Upload Data":
    st.subheader("📂 Upload Excel or CSV File")

    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                file_bytes = uploaded.read()
                df_raw = None
                last_err = None

                for enc in ["utf-8", "latin1", "iso-8859-1"]:
                    try:
                        df_raw = pd.read_csv(
                            io.BytesIO(file_bytes),
                            encoding=enc,
                            on_bad_lines="skip",
                        )
                        break
                    except Exception as e:
                        last_err = e
                        df_raw = None

                if df_raw is None:
                    raise last_err or UnicodeDecodeError(
                        "utf-8", b"", 0, 1, "Unable to decode CSV"
                    )

                df = preprocess_dataframe(df_raw)
                dataframes: Dict[str, pd.DataFrame] = {"CSV_File": df}
            else:
                raw_dfs = pd.read_excel(uploaded, sheet_name=None)
                dataframes = {name: preprocess_dataframe(df) for name, df in raw_dfs.items()}

            reports = get_reports()
            new_id = f"report_{len(reports) + 1}"
            default_name = uploaded.name.rsplit(".", 1)[0]

            reports[new_id] = {
                "name": default_name,
                "dataframes": dataframes,
                "catalog": None,
                "semantic_index": None,
                "agent1_summary": None,
                "viz_plan": None,
                "dataset_narrative": None,
                "last_result": None,
                "last_code": None,
                "query_memory": [],
            }

            st.session_state["active_report_id"] = new_id

            st.success(
                f"✅ Data loaded into new report: **{default_name}**.\n"
                "Switch between reports and rename them from the left sidebar."
            )

        except Exception as e:
            st.error(f"Error loading file: {e}")


# 2️⃣ Catalog + Summary (Auto)
elif section == "2️⃣ Catalog + Summary (Auto)":
    st.subheader("📚 Data Catalog & Agent 1 Summary (Auto)")

    report_id, report = get_active_report()
    if report is None:
        st.info("Upload a file in 'Upload Data' to create a report.")
    else:
        st.markdown(f"**Active report:** `{report['name']}`")

        if report.get("catalog") is None or report.get("semantic_index") is None:
            dfs = report["dataframes"]
            catalog, semantic_index = build_data_catalog(dfs)
            report["catalog"] = catalog
            report["semantic_index"] = semantic_index
        else:
            catalog = report["catalog"]
            semantic_index = report["semantic_index"]

        # Agent 1 summary FIRST
        if OPENAI_API_KEY:
            if report.get("agent1_summary") is None:
                with st.spinner("Running Agent 1 – analyzing dataset..."):
                    summary = agent1_summary(catalog)
                report["agent1_summary"] = summary

            st.markdown("### 🤖 Agent 1 – Data Overview & KPIs")
            st.markdown(report["agent1_summary"])
        else:
            st.warning("Set OPENAI_API_KEY in config/.env to enable AI summary.")

        st.markdown("---")
        st.markdown("### 📘 Raw Catalog (for reference)")
        st.json(catalog)


# 3️⃣ Full Data Insights (Auto) – VISUALS + FILTERS (5K ROWS, with prediction overlay)
elif section == "3️⃣ Full Data Insights (Auto)":
    st.subheader("📊 Full Data Insights – Auto (5K-row visuals + prediction overlay)")

    report_id, report = get_active_report()
    if report is None:
        st.info("Upload a file and build the catalog first.")
    else:
        st.markdown(f"**Active report:** `{report['name']}`")

        if not report.get("dataframes"):
            st.info("No dataframes found in this report.")
        else:
            dataframes = report["dataframes"]
            sheet_name, main_df = get_main_dataframe(dataframes)

            if main_df is None or main_df.empty:
                st.warning("Main dataset is empty.")
            else:
                st.markdown(
                    f"**Using main sheet:** `{sheet_name}` – "
                    f"{len(main_df):,} rows × {len(main_df.columns)} columns"
                )

                # ---------------- GLOBAL DATE FILTER (POWER BI STYLE) ----------------
                date_cols_main = [
                    c for c in main_df.columns if pd.api.types.is_datetime64_any_dtype(main_df[c])
                ]
                working_main_df = main_df.copy()

                if date_cols_main:
                    st.markdown("### 🕒 Global Date Filter")

                    date_col_for_filter = st.selectbox(
                        "Date column for global filter",
                        options=date_cols_main,
                        key="fdi_date_col",
                    )

                    range_options = [
                        "All Time",
                        "Last 30 Days",
                        "Last 90 Days",
                        "This Year",
                        "Last Year",
                    ]
                    date_range_choice = st.selectbox(
                        "Date range",
                        options=range_options,
                        key="fdi_date_range",
                    )

                    ser = pd.to_datetime(
                        working_main_df[date_col_for_filter], errors="coerce"
                    )
                    mask = pd.Series(True, index=working_main_df.index)
                    today = pd.Timestamp.today().normalize()

                    if date_range_choice == "Last 30 Days":
                        start = today - pd.Timedelta(days=30)
                        mask = ser >= start
                    elif date_range_choice == "Last 90 Days":
                        start = today - pd.Timedelta(days=90)
                        mask = ser >= start
                    elif date_range_choice == "This Year":
                        start = pd.Timestamp(year=today.year, month=1, day=1)
                        end = pd.Timestamp(year=today.year + 1, month=1, day=1)
                        mask = (ser >= start) & (ser < end)
                    elif date_range_choice == "Last Year":
                        start = pd.Timestamp(year=today.year - 1, month=1, day=1)
                        end = pd.Timestamp(year=today.year, month=1, day=1)
                        mask = (ser >= start) & (ser < end)
                    else:
                        mask = pd.Series(True, index=working_main_df.index)

                    if date_range_choice != "All Time":
                        mask = mask & ser.notna()

                    working_main_df = working_main_df[mask]
                    st.info(
                        f"Global date filter applied on {pretty_label(date_col_for_filter)} – "
                        f"{len(working_main_df):,} rows remain."
                    )

                # ---------------- FULL ANALYST PIPELINE BUTTON ----------------
                if OPENAI_API_KEY:
                    if st.button("🚀 Run Full Analyst Pipeline", key="full_pipeline"):
                        # Step 1: catalog + semantic index
                        with st.spinner("Step 1/4: Building data catalog..."):
                            if report.get("catalog") is None or report.get("semantic_index") is None:
                                catalog, semantic_index = build_data_catalog(report["dataframes"])
                                report["catalog"] = catalog
                                report["semantic_index"] = semantic_index
                            else:
                                catalog = report["catalog"]
                                semantic_index = report["semantic_index"]

                        # Step 2: Agent 1 summary
                        with st.spinner("Step 2/4: Running Agent 1 summary..."):
                            summary = agent1_summary(report["catalog"])
                            report["agent1_summary"] = summary

                        # Step 3: Agent 1 viz_plan
                        with st.spinner("Step 3/4: Designing viz_plan (Agent 1)..."):
                            viz_plan = agent1_viz_blueprint(report["catalog"], sheet_name)
                            report["viz_plan"] = viz_plan

                        # Step 4: Narrative over filtered dataset
                        with st.spinner("Step 4/4: Generating narrative insights..."):
                            narrative = dataset_auto_insights_text(working_main_df, sheet_name)
                            report["dataset_narrative"] = narrative

                        st.success("✅ Full Analyst Pipeline completed for current dataset view.")

                # ---------------- KPI + VISUAL DATASET (5K ROWS) ----------------
                VIS_ROWS = 5000
                vis_df = sample_dataframe(working_main_df, VIS_ROWS)

                numeric_cols = [
                    c for c in working_main_df.columns
                    if pd.api.types.is_numeric_dtype(working_main_df[c])
                ]
                cat_cols = [
                    c for c in working_main_df.columns
                    if working_main_df[c].dtype == "object"
                ]
                date_cols = [
                    c for c in working_main_df.columns
                    if pd.api.types.is_datetime64_any_dtype(working_main_df[c])
                ]

                main_metric = choose_main_metric_column(working_main_df)

                # --- KPI SECTION ---
                st.markdown("### 📌 Key Metrics")
                kpi_cols = st.columns(3)
                with kpi_cols[0]:
                    st.metric("📦 Total Records", f"{len(working_main_df):,}")

                if main_metric:
                    total_val = float(working_main_df[main_metric].sum())
                    avg_val = float(working_main_df[main_metric].mean())
                    with kpi_cols[1]:
                        st.metric(f"💰 Total {pretty_label(main_metric)}", f"{total_val:,.2f}")
                    with kpi_cols[2]:
                        st.metric(f"📊 Avg {pretty_label(main_metric)}", f"{avg_val:,.2f}")
                else:
                    with kpi_cols[1]:
                        st.metric("🔢 Numeric Columns", str(len(numeric_cols)))
                    with kpi_cols[2]:
                        st.metric("🏷️ Categorical Columns", str(len(cat_cols)))

                kpi_html = f"""
                <div style='display:flex; gap:1rem; margin-top:0.75rem;'>
                  <div style='background:#1d4ed8; color:white; padding:0.75rem 1rem; border-radius:0.75rem; flex:1;'>
                    <div style='font-size:0.7rem; text-transform:uppercase; opacity:0.8;'>Main Metric</div>
                    <div style='font-size:1.1rem; font-weight:bold;'>{pretty_label(main_metric) if main_metric else "Not Detected"}</div>
                  </div>
                  <div style='background:#15803d; color:white; padding:0.75rem 1rem; border-radius:0.75rem; flex:1;'>
                    <div style='font-size:0.7rem; text-transform:uppercase; opacity:0.8;'>Rows Used for Visuals</div>
                    <div style='font-size:1.1rem; font-weight:bold;'>{len(vis_df):,}</div>
                  </div>
                </div>
                """
                st.markdown(kpi_html, unsafe_allow_html=True)

                st.markdown("---")

                # Filters + Saved Views
                saved_views = get_saved_views()
                filtered_df = vis_df.copy()
                cat_filter_state: Dict[str, List[Any]] = {}

                if cat_cols:
                    st.markdown("### 🎛️ Filters & Saved Views (Power BI Style)")
                    with st.expander("Open Filters & Saved Views", expanded=False):
                        max_filter_cols = min(3, len(cat_cols))
                        filter_cols = cat_cols[:max_filter_cols]

                        active_view_cfg = st.session_state.get("active_view_config", {})

                        for col in filter_cols:
                            top_vals = (
                                vis_df[col]
                                .dropna()
                                .value_counts()
                                .head(30)
                                .index.tolist()
                            )
                            default_selected = active_view_cfg.get("cat_filters", {}).get(col, [])
                            selected = st.multiselect(
                                f"Filter by {pretty_label(col)}",
                                options=top_vals,
                                default=default_selected,
                                key=f"filter_{col}",
                            )
                            cat_filter_state[col] = selected
                            if selected:
                                filtered_df = filtered_df[filtered_df[col].isin(selected)]

                        st.markdown("---")
                        st.markdown("#### 💾 Saved Views")

                        view_name = st.text_input("View name", key="save_view_name")
                        if st.button("Save current view", key="save_view_btn"):
                            if view_name:
                                current_date_col = st.session_state.get(
                                    "fdi_date_col",
                                    date_cols_main[0] if date_cols_main else None,
                                )
                                current_date_range = st.session_state.get(
                                    "fdi_date_range", "All Time"
                                )
                                saved_views[view_name] = {
                                    "date_col": current_date_col,
                                    "date_range": current_date_range,
                                    "cat_filters": cat_filter_state,
                                }
                                st.success(f"Saved view '{view_name}'")
                            else:
                                st.error("Please enter a view name before saving.")

                        if saved_views:
                            selected_view_name = st.selectbox(
                                "Load view",
                                options=list(saved_views.keys()),
                                key="load_view_name",
                            )
                            if st.button("Apply view", key="apply_view_btn"):
                                cfg = saved_views[selected_view_name]
                                if cfg.get("date_col") is not None:
                                    st.session_state["fdi_date_col"] = cfg.get("date_col")
                                st.session_state["fdi_date_range"] = cfg.get("date_range", "All Time")

                                cat_cfg = cfg.get("cat_filters", {})
                                for col, vals in cat_cfg.items():
                                    st.session_state[f"filter_{col}"] = vals

                                st.session_state["active_view_config"] = cfg
                                st.experimental_rerun()

                    st.info(f"Filtered visual dataset has {len(filtered_df):,} rows (max 5,000).")

                st.markdown("---")

                # Ensure catalog & semantic_index exist for viz_plan usage
                if report.get("catalog") is None or report.get("semantic_index") is None:
                    dfs = report["dataframes"]
                    catalog, semantic_index = build_data_catalog(dfs)
                    report["catalog"] = catalog
                    report["semantic_index"] = semantic_index
                else:
                    catalog = report["catalog"]
                    semantic_index = report["semantic_index"]

                # Agent 1 viz blueprint if possible
                if OPENAI_API_KEY:
                    if report.get("viz_plan") is None:
                        with st.spinner("Agent 1 is designing a visualization blueprint (viz_plan)..."):
                            viz_plan = agent1_viz_blueprint(catalog, sheet_name)
                        report["viz_plan"] = viz_plan

                    viz_plan = report.get("viz_plan")
                    if viz_plan:
                        st.markdown("### 📈 Agent 1–Driven Visuals (Multiple Types + Prediction Overlay)")
                        render_viz_plan(filtered_df, viz_plan)
                    else:
                        st.markdown("### 📈 Visuals (Fallback – no viz_plan)")
                        fallback_visuals(filtered_df, main_metric, cat_cols, date_cols)
                else:
                    st.markdown("### 📈 Visuals (No API key – heuristic only)")
                    fallback_visuals(filtered_df, main_metric, cat_cols, date_cols)

                st.markdown("---")
                # Narrative insights for filtered working_main_df
                if OPENAI_API_KEY:
                    st.markdown("### 💡 Narrative Insights (Analyst View)")
                    if report.get("dataset_narrative") is None:
                        with st.spinner("Generating narrative insights from current dataset view..."):
                            narrative = dataset_auto_insights_text(working_main_df, sheet_name)
                        report["dataset_narrative"] = narrative
                    st.markdown(report["dataset_narrative"])
                else:
                    st.info("Set OPENAI_API_KEY to enable narrative auto-insights.")


# 4️⃣ Ask a Question (Agent 2)
elif section == "4️⃣ Ask a Question (Agent 2)":
    st.subheader("🤖 Ask a Question – Agent 2 (NL → Pandas)")

    report_id, report = get_active_report()
    if report is None:
        st.info("Upload a file and build the catalog first.")
    elif report.get("catalog") is None or report.get("semantic_index") is None:
        st.info("Please open 'Catalog + Summary (Auto)' once to build the catalog.")
    elif not OPENAI_API_KEY:
        st.warning("Set OPENAI_API_KEY to use Agent 2 (code generation).")
    else:
        st.markdown(f"**Active report:** `{report['name']}`")

        # --- Init session_state for agent2_question so widgets & callbacks play nicely ---
        if "agent2_question" not in st.session_state:
            st.session_state["agent2_question"] = ""

        question = st.text_input(
            "Your analytics question:",
            placeholder="e.g., 'Show total net_amount by product_category for last 6 months'",
            key="agent2_question",
        )
        extra = st.text_area(
            "Additional instructions (optional):",
            placeholder="e.g., 'sort by revenue desc', 'only top 10 categories'",
            height=80,
        )

        if st.button("▶ Run Analysis"):
            if not question.strip():
                st.error("Please enter a question first.")
            else:
                catalog = report["catalog"]
                semantic_index = report["semantic_index"]
                dataframes = report["dataframes"]

                with st.spinner("Agent 2 generating Pandas code..."):
                    code, err = agent2_generate_code(
                        question.strip(), extra.strip(), catalog, semantic_index
                    )

                if err:
                    st.error(err)
                elif not code:
                    st.error("Agent 2 did not return any code.")
                else:
                    safe, msg = is_safe_code(code)
                    if not safe:
                        st.error(msg)
                    else:
                        report["last_code"] = code

                        st.markdown("### 🧾 Generated Pandas Code")
                        with st.expander("View code", expanded=False):
                            st.code(code, language="python")

                        with st.spinner("Executing generated code..."):
                            result_df, exec_err = run_safe_code(code, dataframes)

                        if exec_err:
                            st.warning(
                                f"Initial execution failed, triggering self-heal. Error: {exec_err}"
                            )
                            healed_code = agent2_self_heal(code, exec_err, question)
                            report["last_code"] = healed_code

                            with st.spinner("Executing self-healed code..."):
                                result_df, exec_err = run_safe_code(
                                    healed_code, dataframes
                                )

                        if exec_err:
                            st.error(exec_err)
                        else:
                            st.success(
                                f"✅ Analysis completed. Result has {len(result_df)} rows × {len(result_df.columns)} columns."
                            )
                            report["last_result"] = result_df

                            mem: List[str] = report.get("query_memory", [])
                            mem.append(question.strip())
                            report["query_memory"] = mem
                            st.session_state["query_memory"] = mem

                            st.markdown("### 📊 Result Preview (top 50 rows)")
                            st.dataframe(result_df.head(50))

        # 🧠 Agent 3 – Visuals & Insight (fixed to use stored last_result)
        st.markdown("---")
        st.markdown("### 🧠 Agent 3 – Visuals & Insight")
        last_result = report.get("last_result")
        if last_result is not None:
            if st.button("▶ Run Agent 3: Visuals + Insight"):
                run_agent_3(last_result)
        else:
            st.info("Run an analysis first so Agent 3 can visualize the latest result.")

        # --- Callback for follow-up question buttons ---
        def _set_agent2_question(q: str):
            st.session_state["agent2_question"] = q

        # Follow-up questions
        if OPENAI_API_KEY and report.get("last_result") is not None:
            st.markdown("---")
            st.markdown("### 🔁 Suggested Follow-up Questions")
            followups = agent3_followup_questions(
                question.strip() if question else "",
                report["last_result"],
            )
            if followups:
                for i, fq in enumerate(followups):
                    st.button(
                        f"💬 {fq}",
                        key=f"followup_{i}",
                        on_click=_set_agent2_question,
                        args=(fq,),
                    )
            else:
                st.info("No follow-up suggestions available for this result.")


# 5️⃣ Logs & Debug
elif section == "5️⃣ Logs & Debug":
    st.subheader("📝 Logs & Debug")

    report_id, report = get_active_report()
    if report is None:
        st.info("No reports yet.")
    else:
        st.markdown(f"**Active report:** `{report['name']}`")

        st.markdown("### Last Generated Code (Agent 2)")
        last_code = report.get("last_code")
        if last_code:
            st.code(last_code, language="python")
        else:
            st.info("No code generated yet for this report.")

        st.markdown("### Query Memory")
        st.json(report.get("query_memory", []))

        if report.get("last_result") is not None:
            df = report["last_result"]
            st.markdown("### Last Result Shape")
            st.write(f"{len(df)} rows × {len(df.columns)} columns")
        else:
            st.info("No result dataframe stored yet for this report.")

st.markdown("---")
st.caption("AI Analytics MVP – Agent 1–3, multi-visual dashboards with global date filters, saved views (date + filters) & follow-up Q&A.")
