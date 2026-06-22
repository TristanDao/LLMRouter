#!/usr/bin/env python3
"""
scripts/streamlit_human_review.py - Streamlit app for human review of judged responses

Allows reviewing ALL cases, editing responses/judgments/consensus, and exporting.

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


def get_judge_consensus(record: dict) -> str:
    """Get judge consensus."""
    judge_result = record.get("judge_result", {})
    if judge_result:
        return judge_result.get("consensus", "uncertain")
    return record.get("consensus_status", record.get("consensus", "uncertain"))


def get_is_pass(record: dict) -> bool | None:
    """Get is_pass value."""
    judge_result = record.get("judge_result", {})
    if judge_result:
        val = judge_result.get("is_pass")
        if val is not None:
            return val
    pass_val = record.get("pass")
    if pass_val is not None:
        return pass_val
    return None


def update_record_with_edits(record: dict, edits: dict) -> dict:
    """Apply edits to record."""
    record = dict(record)

    if "local_response" in edits:
        local_resp = edits["local_response"]
        if "responses" in record:
            for key in record["responses"]:
                if "Qwen" in key or "local" in key.lower():
                    record["responses"][key]["response"] = local_resp
                    break
        record["local_model_response"] = local_resp

    if "gemini_response" in edits:
        gemini_resp = edits["gemini_response"]
        if "responses" in record:
            for key in record["responses"]:
                if "gemini" in key.lower():
                    record["responses"][key]["response"] = gemini_resp
                    break
        record["gemini_model_response"] = gemini_resp

    if "consensus" in edits:
        consensus = edits["consensus"]
        if "judge_result" not in record:
            record["judge_result"] = {}
        record["judge_result"]["consensus"] = consensus
        record["consensus_status"] = consensus

        if consensus == "pass":
            record["judge_result"]["is_pass"] = True
            record["pass"] = True
        elif consensus == "fail":
            record["judge_result"]["is_pass"] = False
            record["pass"] = False
        else:
            record["judge_result"]["is_pass"] = None
            record["pass"] = None

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
        initial_sidebar_state="collapsed",
    )

    st.title("🔍 Safety Router - Human Review")
    st.markdown("Review ALL judged cases, edit responses/judgments if needed, then export.")

    if "records" not in st.session_state:
        st.session_state.records = load_data(args_cli.input)
        st.session_state.current_index = 0
        st.session_state.edits = {}
        st.session_state.filter_policy = "All"
        st.session_state.filter_consensus = "All"
        st.session_state.filter_reviewed = "All"
        st.session_state.search_query = ""

    input_path = args_cli.input
    output_path = args_cli.output

    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric("Total Records", len(st.session_state.records))
    with col2:
        reviewed_count = sum(1 for r in st.session_state.records if r.get("human_reviewed", False))
        st.metric("Reviewed", reviewed_count)

    st.sidebar.header("Filters")

    all_policies = set()
    for r in st.session_state.records:
        for pid in r.get("policy_ids", []):
            all_policies.add(str(pid))
    policy_options = ["All"] + sorted(all_policies)

    st.session_state.filter_policy = st.sidebar.selectbox(
        "Filter by Policy ID", policy_options
    )
    st.session_state.filter_consensus = st.sidebar.selectbox(
        "Filter by Consensus",
        ["All", "pass", "fail", "uncertain"]
    )
    st.session_state.filter_reviewed = st.sidebar.selectbox(
        "Filter by Review Status",
        ["All", "Reviewed", "Not Reviewed"]
    )
    st.session_state.search_query = st.sidebar.text_input(
        "Search Query (contains)", ""
    )

    filtered_records = st.session_state.records.copy()
    if st.session_state.filter_policy != "All":
        filtered_records = [
            r for r in filtered_records
            if st.session_state.filter_policy in [str(p) for p in r.get("policy_ids", [])]
        ]
    if st.session_state.filter_consensus != "All":
        filtered_records = [
            r for r in filtered_records
            if get_judge_consensus(r) == st.session_state.filter_consensus
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

    idx = st.session_state.current_index
    if idx >= len(filtered_records):
        idx = len(filtered_records) - 1
        st.session_state.current_index = idx

    record = filtered_records[idx]

    col_nav, col_idx_display, col_nav2 = st.columns([1, 2, 1])
    with col_nav:
        if st.button("⬅️ Previous", disabled=idx == 0):
            st.session_state.current_index = max(0, idx - 1)
            st.rerun()
    with col_idx_display:
        st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>"
            f"Record {idx + 1} of {len(filtered_records)}"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_nav2:
        if st.button("Next ➡️", disabled=idx == len(filtered_records) - 1):
            st.session_state.current_index = min(len(filtered_records) - 1, idx + 1)
            st.rerun()

    query = record.get("query", "")
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
    st.subheader("📝 Query")
    st.text_area("Query Content", query, height=100, disabled=True, key=f"query_{idx}")

    col_local, col_gemini = st.columns(2)

    local_response = get_local_response(record)
    gemini_response = get_gemini_response(record)

    with col_local:
        st.subheader("💻 Local (Qwen3-4B)")
        new_local = st.text_area(
            "Local Response",
            local_response,
            height=200,
            key=f"local_{idx}",
            label_visibility="collapsed"
        )
        if new_local != local_response:
            if f"record_{idx}_local" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}_local"] = {}
            st.session_state.edits[f"record_{idx}_local"]["local_response"] = new_local

    with col_gemini:
        st.subheader("✨ Gemini")
        new_gemini = st.text_area(
            "Gemini Response",
            gemini_response,
            height=200,
            key=f"gemini_{idx}",
            label_visibility="collapsed"
        )
        if new_gemini != gemini_response:
            if f"record_{idx}_gemini" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}_gemini"] = {}
            st.session_state.edits[f"record_{idx}_gemini"]["gemini_response"] = new_gemini

    st.markdown("---")
    st.subheader("⚖️ Judge Verdict")

    current_consensus = get_judge_consensus(record)
    current_is_pass = get_is_pass(record)

    col_cons, col_pass = st.columns(2)
    with col_cons:
        new_consensus = st.selectbox(
            "Consensus",
            ["pass", "fail", "uncertain"],
            index=["pass", "fail", "uncertain"].index(current_consensus) if current_consensus in ["pass", "fail", "uncertain"] else 2,
            key=f"consensus_{idx}"
        )
        if new_consensus != current_consensus:
            if f"record_{idx}" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}"] = {}
            st.session_state.edits[f"record_{idx}"]["consensus"] = new_consensus

    with col_pass:
        if current_is_pass is True:
            pass_display = "pass (correct)"
        elif current_is_pass is False:
            pass_display = "fail (incorrect)"
        else:
            pass_display = "uncertain"
        st.markdown(f"**Judgment:** `{pass_display}`")

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("✅ Mark as Reviewed", type="primary"):
            if f"record_{idx}" not in st.session_state.edits:
                st.session_state.edits[f"record_{idx}"] = {}
            st.session_state.edits[f"record_{idx}"]["human_reviewed"] = True

            actual_idx = st.session_state.records.index(record)
            st.session_state.records[actual_idx] = update_record_with_edits(
                st.session_state.records[actual_idx],
                {"human_reviewed": True}
            )
            save_data(st.session_state.records, output_path)
            st.success(f"Marked record {query_id} as reviewed and saved!")
            st.rerun()

    with col_btn2:
        if st.button("💾 Save Changes"):
            actual_idx = st.session_state.records.index(record)
            all_edits = {}
            for key in st.session_state.edits:
                if key.startswith(f"record_{idx}_"):
                    all_edits.update(st.session_state.edits[key])
            if f"record_{idx}" in st.session_state.edits:
                all_edits.update(st.session_state.edits[f"record_{idx}"])

            if all_edits:
                st.session_state.records[actual_idx] = update_record_with_edits(
                    st.session_state.records[actual_idx],
                    all_edits
                )
                save_data(st.session_state.records, output_path)
                st.session_state.edits = {}
                st.success("Changes saved!")
            else:
                st.info("No changes to save.")
            st.rerun()

    with col_btn3:
        if st.button("⏭️ Skip"):
            if idx < len(filtered_records) - 1:
                st.session_state.current_index = idx + 1
            st.rerun()

    st.markdown("---")

    if st.session_state.edits:
        st.info(f"Pending edits: {len(st.session_state.edits)} record(s) with unsaved changes")

    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("📤 Export All Reviewed Records"):
            reviewed = [r for r in st.session_state.records if r.get("human_reviewed", False)]
            if reviewed:
                export_path = output_path
                save_data(reviewed, export_path)
                st.success(f"Exported {len(reviewed)} reviewed records to {export_path}")
            else:
                st.warning("No reviewed records to export.")

    with col_export2:
        if st.button("📤 Export All Records (including unreviewed)"):
            save_data(st.session_state.records, output_path)
            st.success(f"Exported all {len(st.session_state.records)} records to {output_path}")


if __name__ == "__main__":
    main()
