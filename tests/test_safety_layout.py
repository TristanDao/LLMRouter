from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmrouter.prompts import load_prompt_template
from safety.dataset.pipeline import SafetyGoldenDatasetBuilder, summarize_records


class SafetyLayoutSmokeTest(unittest.TestCase):
    def test_canonical_dataset_builder_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = SafetyGoldenDatasetBuilder(
                router_config_path="configs/safety/router.yaml",
                policy_path="policy.csv",
                seed=7,
                output_dir=tmpdir,
            )
            queries = builder.build_queries(
                single_per_policy=1,
                multi_groups=1,
                multi_per_group=1,
                no_policy_per_complexity=1,
            )
            self.assertGreater(len(queries), 0)
            records = builder.benchmark(queries[:3], call_models=False)
            summary = summarize_records(records)
            self.assertIn("total", summary)
            self.assertGreater(summary["total"], 0)

            manifest = builder.export(records)
            self.assertTrue(Path(tmpdir, "manifest.json").exists())
            self.assertEqual(manifest["total"], len(records))

    def test_prompt_loader_finds_canonical_safety_prompt(self) -> None:
        self.assertIsInstance(load_prompt_template("task_safety_aware_policy"), str)


if __name__ == "__main__":
    unittest.main()
