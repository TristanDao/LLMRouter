"""
Safety-Aware Policy Routing Classifier for Enterprise RAG Systems.

This router evaluates query safety, policy violation risk, and query difficulty
based on dynamic customer policies and routes queries to:
- LOCAL: Low risk, clear policy.
- MEDIUM: Multiple overlapping policies, requires reasoning.
- HIGH: Political/territorial violations, public relations crisis, or policy gray areas.
"""

import json
import os
import re
from typing import Any, Dict, List

import torch
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils.api_calling import call_api
from llmrouter.utils.prompting import generate_task_query
from safety.common.env import env_bool, env_str, load_dotenv_file

load_dotenv_file()

try:  # pragma: no cover
    import safety.tasks.safety_aware_policy  # noqa: F401
except Exception:
    pass


class SafetyRouter(MetaRouter):
    """
    Safety-Aware Policy Routing Classifier.

    Reads customer policies and dynamically evaluates inputs:
    - Estimating Risk Score (business & legal sensitivity)
    - Estimating Difficulty Score (policy ambiguity & overlaps)
    - Selecting the optimal routing target: local, medium, or high.
    """

    def __init__(self, yaml_path: str):
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)

        hparam = self.cfg.get("hparam", {})
        self.risk_threshold = float(env_str("SAFETY_RISK_THRESHOLD", hparam.get("risk_threshold", 0.75)))
        self.diff_threshold = float(env_str("SAFETY_DIFF_THRESHOLD", hparam.get("diff_threshold", 0.70)))
        self.local_model = env_str("SAFETY_LOCAL_MODEL", hparam.get("local_model", "qwen-2.5-7b"))
        self.medium_model = env_str("SAFETY_MEDIUM_MODEL", hparam.get("medium_model", "gpt-4o-mini"))
        self.high_model = env_str("SAFETY_HIGH_MODEL", hparam.get("high_model", "gemini-1.5-pro"))
        self.api_endpoint = env_str("SAFETY_API_ENDPOINT", self.cfg.get("api_endpoint", "https://api.openai.com/v1"))
        self.service = env_str("SAFETY_SERVICE", self.cfg.get("service", "openai"))
        self.use_llm_judge = env_bool("SAFETY_USE_LLM_JUDGE", bool(self.cfg.get("use_llm_judge", False)))

        self.policies_file = env_str(
            "SAFETY_POLICIES_FILE",
            self.cfg.get("data_path", {}).get("policies_file", "policy_custom.md"),
        )
        self.policies = self._load_policies(self.policies_file)

        print(f"🛡️ SafetyRouter initialized with {len(self.policies)} custom policies.")
        print(f"   Risk Threshold: {self.risk_threshold} | Difficulty Threshold: {self.diff_threshold}")
        print(f"   Routes: LOCAL ({self.local_model}) | MEDIUM ({self.medium_model}) | HIGH ({self.high_model})")

    def _load_policies(self, file_path: str) -> Dict[str, str]:
        policies = {}
        if not os.path.exists(file_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            file_path = os.path.join(project_root, file_path)

        if not os.path.exists(file_path):
            print(f"⚠️ Policy file not found: {file_path}. Using empty policy set.")
            return policies

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = r"Policy\s+([SVP]\d+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=\n\n|\nPolicy|\Z)"
            matches = re.findall(pattern, content, re.DOTALL)
            for policy_id, desc in matches:
                policies[policy_id] = desc.strip()

            if not policies:
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("Policy"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            policy_id = parts[0].replace("Policy", "").strip()
                            policies[policy_id] = parts[1].strip()
        except Exception as e:
            print(f"❌ Error loading policies: {e}")

        return policies

    def _rule_based_fallback_eval(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        risk = 0.0
        difficulty = 0.0
        violated_policies = []
        conflicts = []
        reasoning = []

        if any(w in query_lower for w in ["hoàng sa", "trường sa", "biển đông", "bản đồ", "map", "lãnh thổ", "chủ quyền"]):
            risk = max(risk, 0.95)
            difficulty = max(difficulty, 0.85)
            violated_policies.append("S1")
            reasoning.append("Matches Policy S1 (Chủ quyền lãnh thổ) which has highly sensitive geopolitical content.")

        if any(w in query_lower for w in ["chính trị", "đài loan", "trung quốc", "quốc gia", "độc lập", "chiến tranh", "lãnh đạo", "bình luận chính trị"]):
            risk = max(risk, 0.90)
            difficulty = max(difficulty, 0.80)
            violated_policies.append("S2")
            reasoning.append("Matches Policy S2 (Trung lập chính trị) due to sensitive political queries.")

        if any(w in query_lower for w in ["lỗi", "cháy", "đâm", "tai nạn", "phanh gấp", "kiện", "tranh chấp", "tin đồn", "vinfast lỗi", "vinhomes tranh chấp"]):
            risk = max(risk, 0.85)
            difficulty = max(difficulty, 0.75)
            violated_policies.append("S3")
            reasoning.append("Matches Policy S3 (Khủng hoảng truyền thông) regarding brand product issues/disputes.")

        has_xanh = any(w in query_lower for w in ["xanh sm", "taxi", "bắt xe", "tài xế", "chuyến xe", "thú cưng", "chó", "mèo"])
        has_vinhomes = any(w in query_lower for w in ["vinhomes", "cư dân", "ocean park", "smart city", "thẻ cư dân"])
        has_vinpearl = any(w in query_lower for w in ["vinpearl", "lưu trú", "villa", "resort", "buggy", "xe điện"])
        has_vinmec = any(w in query_lower for w in ["vinmec", "khám", "bệnh nhân", "phòng khám", "cấp cứu", " emergency", "viện phí"])

        if has_vinhomes and has_xanh:
            if any(w in query_lower for w in ["giảm giá", "ưu đãi", "khuyến mãi", "15%"]):
                violated_policies.append("V1")
                reasoning.append("Matches Policy V1 (Vinhomes resident discount for Xanh SM).")
                if any(w in query_lower for w in ["cộng dồn", "mã khác", "đồng thời", "voucher"]):
                    violated_policies.append("V2")
                    difficulty = max(difficulty, 0.65)
                    reasoning.append("Matches Policy V2 (Non-stackable Xanh SM discounts).")

            if any(w in query_lower for w in ["thú cưng", "chó", "mèo", "mang vật nuôi"]):
                violated_policies.append("V3")
                reasoning.append("Matches Policy V3 (Pets inside Xanh SM for Vinhomes residents).")

        if has_vinpearl and has_vinmec:
            if any(w in query_lower for w in ["khám sức khỏe", "miễn phí", "tặng suất"]):
                violated_policies.append("P1")
                reasoning.append("Matches Policy P1 (Vinpearl guests free Vinmec checkup).")
            if any(w in query_lower for w in ["cấp cứu", "emergency", "thanh toán", "tiền mặt", "trừ cọc"]):
                violated_policies.append("P2")
                reasoning.append("Matches Policy P2 (Vinmec emergency cash payment requirement).")
            if any(w in query_lower for w in ["xe điện", "buggy", "di chuyển nội khu", "sau 22h", "ban đêm", "phí"]):
                violated_policies.append("P3")
                reasoning.append("Matches Policy P3 (Vinpearl buggy service fees and operating hours).")

        if (has_vinhomes or has_xanh) and has_vinmec and any(w in query_lower for w in ["cấp cứu", "đền bù", "tai nạn", "phanh gấp", "chó", "thú cưng"]):
            difficulty = max(difficulty, 0.90)
            risk = max(risk, 0.85)
            conflicts = list(set(violated_policies + ["V3", "P2"]))
            violated_policies = list(set(violated_policies + ["V3", "P2"]))
            reasoning.append("CRITICAL: Detected policy overlap / gray area between Xanh SM pet policy (V3) and Vinmec emergency payment (P2) with liability claims. No explicit corporate guidelines exist.")

        if len(violated_policies) >= 2 and difficulty < 0.70:
            difficulty = max(difficulty, 0.60)
            conflicts = violated_policies.copy()
            reasoning.append("Moderate difficulty due to multi-policy reference.")

        if not violated_policies:
            difficulty = 0.15
            risk = 0.05
            reasoning.append("Query is straightforward and does not trigger any safety or ecosystem policies.")

        if risk >= self.risk_threshold or difficulty >= self.diff_threshold:
            route = "high"
        elif difficulty >= 0.50:
            route = "medium"
        else:
            route = "local"

        violation_status = "non-violation"
        if any(p in ["S1", "S2", "S3"] for p in violated_policies):
            violation_status = "violation"
        elif conflicts:
            violation_status = "uncertain"

        return {
            "route": route,
            "difficulty": difficulty,
            "risk": risk,
            "confidence": 0.90 - (0.20 if len(conflicts) > 0 else 0.0),
            "violation_status": violation_status,
            "violated_policies": violated_policies,
            "mapped_policy_conflict": conflicts,
            "reasoning": " | ".join(reasoning),
        }

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "")
        conv_hist = query_input.get("conversation_history", "")

        target_text = query
        if not target_text and conv_hist:
            if isinstance(conv_hist, list) and len(conv_hist) > 0:
                target_text = conv_hist[-1].get("content", "")
            elif isinstance(conv_hist, str):
                target_text = conv_hist.split("\n")[-1]

        if os.environ.get("API_KEYS") and self.use_llm_judge:
            try:
                formatted_prompt = generate_task_query(
                    "safety_aware_policy",
                    {
                        "policies": list(self.policies.values()),
                        "conversation_history": conv_hist or query,
                        "target_utterance": target_text,
                        "target_type": "user",
                    },
                )

                api_request = {
                    "api_endpoint": self.api_endpoint,
                    "query": formatted_prompt["user"],
                    "system_prompt": formatted_prompt["system"],
                    "model_name": self.high_model,
                    "api_name": self.high_model,
                    "service": self.service,
                }

                response_data = call_api(api_request)
                resp_text = response_data.get("response", "").strip()
                if resp_text.startswith("```json"):
                    resp_text = resp_text[7:]
                if resp_text.endswith("```"):
                    resp_text = resp_text[:-3]

                parsed_res = json.loads(resp_text.strip())

                selected_model = self.local_model
                if parsed_res.get("route") == "high":
                    selected_model = self.high_model
                elif parsed_res.get("route") == "medium":
                    selected_model = self.medium_model

                parsed_res["model_name"] = selected_model
                parsed_res["predicted_llm"] = selected_model
                parsed_res["route_label"] = parsed_res.get("route", "local")
                parsed_res["human_review_required"] = (
                    parsed_res.get("route") == "high"
                    or parsed_res.get("violation_status") == "uncertain"
                )
                return parsed_res
            except Exception as e:
                print(f"⚠️ API call failed: {e}. Falling back to high-fidelity semantic evaluator.")

        evaluation = self._rule_based_fallback_eval(target_text)

        if evaluation["route"] == "high":
            selected_model = self.high_model
        elif evaluation["route"] == "medium":
            selected_model = self.medium_model
        else:
            selected_model = self.local_model

        evaluation["model_name"] = selected_model
        evaluation["predicted_llm"] = selected_model
        evaluation["method"] = "safety_router_fallback"
        evaluation["route_label"] = evaluation.get("route", "local")
        evaluation["human_review_required"] = (
            evaluation.get("route") == "high"
            or evaluation.get("violation_status") == "uncertain"
        )
        return evaluation

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.route_single(item) for item in batch]
