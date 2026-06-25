#!/usr/bin/env python3
"""
scripts/streamlit_human_review.py - Streamlit app for human review of judged responses

Allows reviewing ALL cases, editing local_correct/gemini_correct, and exporting.

Usage:
    streamlit run scripts/streamlit_human_review.py -- \
        --input artifacts/safety_queries/judged_vni.jsonl \
        --output artifacts/safety_queries/reviewed_vni.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def load_data(input_path: str) -> list:
    """Load records from JSONL file."""
    if not os.path.exists(input_path):
        return []
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_data(records: list, output_path: str) -> None:
    """Save records to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_local_response(record: dict) -> str:
    """Extract local model response."""
    responses = record.get("responses", {})
    for key in ["Qwen/Qwen3-4B-Instruct-2507", "local", "local_model_response"]:
        if key in responses:
            return responses[key].get("response", "")
    return record.get("local_model_response", record.get("local_response", ""))


def get_gemini_response(record: dict) -> str:
    """Extract gemini model response."""
    responses = record.get("responses", {})
    for key in ["gemini-3.1-flash-lite", "gemini", "gemini_model_response"]:
        if key in responses:
            return responses[key].get("response", "")
    return record.get("gemini_model_response", record.get("gemini_response", ""))


def get_judge_result(record: dict) -> dict:
    """Get judge_result dict."""
    return record.get("judge_result", {})


def display_textbox(content: str, height: int = 200) -> str:
    """Return HTML for styled text box that's easy to read."""
    if not content:
        content = "(No content)"
    escaped = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return f"""
    <div style="
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        max-height: {height}px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 13px;
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.5;
        color: #333;
    ">{escaped}</div>
    """


def get_evaluations(record: dict) -> dict:
    """Get evaluations from judge_result."""
    judge = get_judge_result(record)
    return judge.get("evaluations", {})


def _find_local_eval_key(evals: dict) -> str | None:
    """Find the key for local model evaluation (handles inconsistent key names)."""
    for key in evals.keys():
        key_lower = key.lower()
        if "local" in key_lower or "qwen" in key_lower:
            return key
    return None


def _find_gemini_eval_key(evals: dict) -> str | None:
    """Find the key for gemini model evaluation (handles inconsistent key names)."""
    for key in evals.keys():
        key_lower = key.lower()
        if "gemini" in key_lower:
            return key
    return None


def _extract_correct_from_eval(eval_data: dict) -> str:
    """Extract correct value (yes/no/unknown) from evaluation dict."""
    if not isinstance(eval_data, dict):
        return "unknown"
    is_correct = eval_data.get("is_correct")
    if is_correct is True:
        return "yes"
    elif is_correct is False:
        return "no"
    # Legacy fallbacks for older judged files.
    is_safe = eval_data.get("is_safe")
    if is_safe is True:
        return "yes"
    elif is_safe is False:
        return "no"
    is_appropriate = eval_data.get("is_appropriate")
    if is_appropriate is True:
        return "yes"
    elif is_appropriate is False:
        return "no"
    return "unknown"


def get_llm_consensus(record: dict) -> str:
    """Get LLM's consensus (original from judge)."""
    judge = get_judge_result(record)
    if judge:
        raw = judge.get("consensus", "unknown")
    else:
        raw = record.get("consensus_status", record.get("consensus", "unknown"))
    # Normalize: correct->pass, incorrect->fail, mixed->uncertain
    mapping = {
        "correct": "pass",
        "incorrect": "fail",
        "mixed": "uncertain",
        "safe": "pass",
        "unsafe": "fail",
        "pass": "pass",
        "fail": "fail",
        "uncertain": "uncertain",
    }
    return mapping.get(raw, raw)


def get_local_correct_from_judge(record: dict) -> str:
    """
    Get local_correct value from judge_result.
    Returns: 'yes', 'no', or 'unknown' (if not present or null)
    """
    judge = get_judge_result(record)
    if judge:
        val = judge.get("local_correct")
        if val is True:
            return "yes"
        elif val is False:
            return "no"
    # Fallback: try to infer from evaluations
    evals = get_evaluations(record)
    local_key = _find_local_eval_key(evals)
    if local_key:
        return _extract_correct_from_eval(evals[local_key])
    return "unknown"


def get_gemini_correct_from_judge(record: dict) -> str:
    """
    Get gemini_correct value from judge_result.
    Returns: 'yes', 'no', or 'unknown' (if not present or null)
    """
    judge = get_judge_result(record)
    if judge:
        val = judge.get("gemini_correct")
        if val is True:
            return "yes"
        elif val is False:
            return "no"
    # Fallback: try to infer from evaluations
    evals = get_evaluations(record)
    gemini_key = _find_gemini_eval_key(evals)
    if gemini_key:
        return _extract_correct_from_eval(evals[gemini_key])
    return "unknown"


def get_local_reasoning(record: dict) -> str:
    """Get local model reasoning from evaluations."""
    evals = get_evaluations(record)
    local_key = _find_local_eval_key(evals)
    if local_key:
        eval_data = evals[local_key]
        if isinstance(eval_data, dict):
            return eval_data.get("reasoning", "No reasoning available")
    return "No evaluations available"


def get_gemini_reasoning(record: dict) -> str:
    """Get gemini model reasoning from evaluations."""
    evals = get_evaluations(record)
    gemini_key = _find_gemini_eval_key(evals)
    if gemini_key:
        eval_data = evals[gemini_key]
        if isinstance(eval_data, dict):
            return eval_data.get("reasoning", "No reasoning available")
    return "No evaluations available"


def update_record_with_edits(record: dict, edits: dict) -> dict:
    """Apply edits to record."""
    record = dict(record)

    if "local_correct" in edits:
        val = edits["local_correct"]
        bool_val = True if val == "yes" else False
        if "judge_result" not in record:
            record["judge_result"] = {}
        record["judge_result"]["local_correct"] = bool_val

    if "gemini_correct" in edits:
        val = edits["gemini_correct"]
        bool_val = True if val == "yes" else False
        if "judge_result" not in record:
            record["judge_result"] = {}
        record["judge_result"]["gemini_correct"] = bool_val

    if "human_reviewed" in edits:
        record["human_reviewed"] = edits["human_reviewed"]

    return record


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", "-i", required=True, help="Input judged JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output reviewed JSONL file")
    args_cli, _ = parser.parse_known_args()

    st.set_page_config(
        page_title="Safety Router - Human Review",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔍 Safety Router - Human Review")
    st.markdown("Review and label whether local (Qwen3-4B) and Gemini models answered correctly.")

    if "records" not in st.session_state:
        st.session_state.records = load_data(args_cli.input)
        st.session_state.current_index = 0
        st.session_state.edits = {}
        st.session_state.filter_policy = "All"
        st.session_state.filter_local_correct = "All"
        st.session_state.filter_gemini_correct = "All"
        st.session_state.filter_reviewed = "All"
        st.session_state.search_query = ""

    input_path = args_cli.input
    output_path = args_cli.output

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.records))
    with col2:
        reviewed_count = sum(1 for r in st.session_state.records if r.get("human_reviewed", False))
        st.metric("Reviewed", reviewed_count)
    with col3:
        unknown_count = sum(1 for r in st.session_state.records
                           if get_local_correct_from_judge(r) == "unknown"
                           or get_gemini_correct_from_judge(r) == "unknown")
        st.metric("Unknown Labels", unknown_count)

    st.sidebar.header("Filters")

    # Policy filter
    all_policies = set()
    for r in st.session_state.records:
        for pid in r.get("policy_ids", []):
            all_policies.add(str(pid))
    policy_options = ["All"] + sorted(all_policies)
    st.session_state.filter_policy = st.sidebar.selectbox("Policy ID", policy_options)

    # Model correctness filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Labels")
    st.session_state.filter_local_correct = st.sidebar.selectbox(
        "Local Correct",
        ["All", "yes", "no", "unknown"]
    )
    st.session_state.filter_gemini_correct = st.sidebar.selectbox(
        "Gemini Correct",
        ["All", "yes", "no", "unknown"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Review Status")
    st.session_state.filter_reviewed = st.sidebar.selectbox(
        "Status",
        ["All", "Reviewed", "Not Reviewed"]
    )

    st.session_state.search_query = st.sidebar.text_input("Search Query")

    # Apply filters
    filtered_records = st.session_state.records.copy()

    if st.session_state.filter_policy != "All":
        filtered_records = [
            r for r in filtered_records
            if st.session_state.filter_policy in [str(p) for p in r.get("policy_ids", [])]
        ]

    if st.session_state.filter_local_correct != "All":
        filtered_records = [
            r for r in filtered_records
            if get_local_correct_from_judge(r) == st.session_state.filter_local_correct
        ]

    if st.session_state.filter_gemini_correct != "All":
        filtered_records = [
            r for r in filtered_records
            if get_gemini_correct_from_judge(r) == st.session_state.filter_gemini_correct
        ]

    if st.session_state.filter_reviewed == "Reviewed":
        filtered_records = [r for r in filtered_records if r.get("human_reviewed", False)]
    elif st.session_state.filter_reviewed == "Not Reviewed":
        filtered_records = [r for r in filtered_records if not r.get("human_reviewed", False)]

    if st.session_state.search_query:
        query = st.session_state.search_query.lower()
        filtered_records = [
            r for r in filtered_records
            if query in r.get("query", "").lower()
        ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing: {len(filtered_records)} records**")

    if not filtered_records:
        st.warning("No records match the current filters.")
        return

    st.markdown("---")

    # Navigation at TOP
    col_nav, col_idx_display, col_nav2 = st.columns([1, 2, 1])
    with col_nav:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0, key="prev_top"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()
    with col_idx_display:
        st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>"
            f"Record {st.session_state.current_index + 1} of {len(filtered_records)}"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_nav2:
        if st.button("Next ➡️", disabled=st.session_state.current_index >= len(filtered_records) - 1, key="next_top"):
            st.session_state.current_index = min(len(filtered_records) - 1, st.session_state.current_index + 1)
            st.rerun()

    idx = st.session_state.current_index
    if idx >= len(filtered_records):
        idx = len(filtered_records) - 1
        st.session_state.current_index = idx

    record = filtered_records[idx]

    # Record info
    query_id = record.get("query_id", "N/A")
    policy_ids = record.get("policy_ids", [])
    designed_complexity = record.get("designed_complexity", "N/A")
    language = record.get("language", "N/A")
    human_reviewed = record.get("human_reviewed", False)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**Query ID:** `{query_id}`")
        st.markdown(f"**Language:** `{language}`")
        st.markdown(f"**Policy IDs:** `{', '.join(str(p) for p in policy_ids)}`")
    with col_info2:
        st.markdown(f"**Complexity:** `{designed_complexity}`")
        st.markdown(f"**Group Type:** `{record.get('group_type', 'N/A')}`")
        status_color = "🟢" if human_reviewed else "🔴"
        st.markdown(f"**Review Status:** {status_color} {'Reviewed' if human_reviewed else 'Not Reviewed'}")

    st.markdown("---")

    st.markdown("### 📝 Query")
    st.markdown(display_textbox(record.get("query", ""), height=100), unsafe_allow_html=True)

    # Responses
    st.markdown("### 💬 Model Responses")
    col_local, col_gemini = st.columns(2)
    with col_local:
        st.markdown("**💻 Local (Qwen3-4B)**")
        st.markdown(display_textbox(get_local_response(record), height=250), unsafe_allow_html=True)
    with col_gemini:
        st.markdown("**✨ Gemini**")
        st.markdown(display_textbox(get_gemini_response(record), height=250), unsafe_allow_html=True)

    st.markdown("---")

    # LLM Judge Reasoning
    st.markdown("### 🤖 LLM Judge's Reasoning")

    llm_local_correct = get_local_correct_from_judge(record)
    llm_gemini_correct = get_gemini_correct_from_judge(record)
    llm_consensus = get_llm_consensus(record)

    col_reasoning1, col_reasoning2 = st.columns(2)
    with col_reasoning1:
        st.markdown("**💻 Local Model**")
        local_reasoning = get_local_reasoning(record)
        st.markdown(display_textbox(local_reasoning, height=120), unsafe_allow_html=True)
        st.markdown(f"**LLM says:** `{llm_local_correct}`")

    with col_reasoning2:
        st.markdown("**✨ Gemini Model**")
        gemini_reasoning = get_gemini_reasoning(record)
        st.markdown(display_textbox(gemini_reasoning, height=120), unsafe_allow_html=True)
        st.markdown(f"**LLM says:** `{llm_gemini_correct}`")

    st.markdown(f"**LLM Consensus:** `{llm_consensus}`")

    st.markdown("---")

    # Human Labels
    st.markdown("### ✏️ Your Labels")

    # Get original LLM values
    original_local = get_local_correct_from_judge(record)
    original_gemini = get_gemini_correct_from_judge(record)

    # Check if user has made edits for this record
    record_edit = st.session_state.edits.get(f"record_{idx}", {})

    # Use edited value if exists, otherwise use original
    current_local = record_edit.get("local_correct", original_local)
    current_gemini = record_edit.get("gemini_correct", original_gemini)

    col_local_label, col_gemini_label = st.columns(2)

    with col_local_label:
        st.markdown("**💻 Is Local (Qwen3-4B) correct?**")
        st.caption(f"LLM original: `{original_local}`")

        options = ["yes", "no"]
        default_idx = 0
        if current_local in options:
            default_idx = options.index(current_local)

        new_local_correct = st.radio(
            "Local Correct",
            options,
            index=default_idx,
            key=f"local_correct_{idx}",
            horizontal=True,
            label_visibility="collapsed"
        )
        if new_local_correct != current_local:
            if f"record_{idx}" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}"] = {}
            st.session_state.edits[f"record_{idx}"]["local_correct"] = new_local_correct

    with col_gemini_label:
        st.markdown("**✨ Is Gemini correct?**")
        st.caption(f"LLM original: `{original_gemini}`")

        default_idx = 0
        if current_gemini in options:
            default_idx = options.index(current_gemini)

        new_gemini_correct = st.radio(
            "Gemini Correct",
            options,
            index=default_idx,
            key=f"gemini_correct_{idx}",
            horizontal=True,
            label_visibility="collapsed"
        )
        if new_gemini_correct != current_gemini:
            if f"record_{idx}" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}"] = {}
            st.session_state.edits[f"record_{idx}"]["gemini_correct"] = new_gemini_correct

    # Auto consensus from user's labels
    if new_local_correct == "yes" and new_gemini_correct == "yes":
        auto_consensus = "pass"
    elif new_local_correct == "no" or new_gemini_correct == "no":
        auto_consensus = "fail"
    else:
        auto_consensus = "unknown"

    st.markdown(f"**Your consensus:** `{auto_consensus}`")

    st.markdown("---")

    # Actions
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("✅ Mark as Reviewed & Save", type="primary", key="mark_reviewed"):
            if f"record_{idx}" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}"] = {}
            st.session_state.edits[f"record_{idx}"]["human_reviewed"] = True

            actual_idx = st.session_state.records.index(record)
            st.session_state.records[actual_idx] = update_record_with_edits(
                st.session_state.records[actual_idx],
                st.session_state.edits[f"record_{idx}"]
            )
            save_data(st.session_state.records, output_path)
            st.session_state.edits = {}
            st.success(f"Record {query_id} saved!")
            st.rerun()

    with col_btn2:
        if st.button("💾 Save (No Mark)", key="save_no_mark"):
            actual_idx = st.session_state.records.index(record)
            if st.session_state.edits.get(f"record_{idx}"):
                st.session_state.records[actual_idx] = update_record_with_edits(
                    st.session_state.records[actual_idx],
                    st.session_state.edits[f"record_{idx}"]
                )
                save_data(st.session_state.records, output_path)
                st.session_state.edits = {}
                st.success("Changes saved!")
            else:
                st.info("No changes to save.")
            st.rerun()

    with col_btn3:
        if st.button("⏭️ Skip", key="skip"):
            if idx < len(filtered_records) - 1:
                st.session_state.current_index = idx + 1
            st.rerun()

    st.markdown("---")

    # Navigation at BOTTOM
    col_nav_bot, col_idx_bot, col_nav_bot2 = st.columns([1, 2, 1])
    with col_nav_bot:
        if st.button("⬅️ Previous (Bottom)", disabled=st.session_state.current_index == 0, key="prev_bot"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()
    with col_idx_bot:
        st.markdown(
            f"<div style='text-align: center; font-size: 18px;'>"
            f"Record {st.session_state.current_index + 1} of {len(filtered_records)}"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_nav_bot2:
        if st.button("Next ➡️ (Bottom)", disabled=st.session_state.current_index >= len(filtered_records) - 1, key="next_bot"):
            st.session_state.current_index = min(len(filtered_records) - 1, st.session_state.current_index + 1)
            st.rerun()

    st.markdown("---")

    # Export buttons at bottom
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("📤 Export Reviewed Only", key="export_reviewed"):
            reviewed = [r for r in st.session_state.records if r.get("human_reviewed", False)]
            if reviewed:
                save_data(reviewed, output_path)
                st.success(f"Exported {len(reviewed)} reviewed records")
            else:
                st.warning("No reviewed records yet.")

    with col_export2:
        if st.button("📤 Export All Records", key="export_all"):
            save_data(st.session_state.records, output_path)
            st.success(f"Exported all {len(st.session_state.records)} records")


if __name__ == "__main__":
    main()
