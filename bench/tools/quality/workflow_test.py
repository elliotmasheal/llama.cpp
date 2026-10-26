import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["qrt.exe", "model.gguf", "tokens.txt", "model.qwen36rt.idx",
                     "model.qwen36rt.bin", "base.bin"]:
            (self.root / name).write_bytes(b"fixture")
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        workflow.add_parser(sub)
        self.args = parser.parse_args([
            "run", "--qrt", str(self.root / "qrt.exe"), "--model", str(self.root / "model.gguf"),
            "--store", str(self.root / "model"), "--tokens", str(self.root / "tokens.txt"),
            "--output-dir", str(self.root / "run"),
        ])

    def manifest(self):
        return json.loads((self.args.output_dir / "workflow.json").read_text(encoding="utf-8"))

    def test_success_is_diagnostic_not_acceptance(self):
        with patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(workflow.run(self.args), 0)
        self.assertEqual(run.call_count, 5)
        report = self.manifest()
        self.assertEqual(report["status"], "format_verified")
        self.assertFalse(report["performance_acceptance"])
        self.assertFalse(report["model_quality_acceptance"])
        self.assertEqual([s["name"] for s in report["stages"]],
                         ["selftest", "config", "index", "model", "verify"])
        model_cmd = report["stages"][3]["command"]
        self.assertIn(str(self.args.tokens.resolve()), model_cmd)
        self.assertIn("--layer-store", model_cmd)
        self.assertIn("--kv-page-tokens", model_cmd)

    def test_stage_failure_stops_consumers(self):
        for failed_at in range(5):
            self.args.output_dir = self.root / ("failure-%d" % failed_at)
            codes = [subprocess.CompletedProcess([], 0)] * failed_at
            codes.append(subprocess.CompletedProcess([], 2))
            with self.subTest(stage=failed_at):
                with patch.object(workflow.subprocess, "run", side_effect=codes) as run:
                    self.assertEqual(workflow.run(self.args), 2)
                self.assertEqual(run.call_count, failed_at + 1)
                self.assertEqual(self.manifest()["status"], "failed")

    def test_quality_failure_is_preserved(self):
        self.args.baseline = self.root / "base.bin"
        codes = [subprocess.CompletedProcess([], 0)] * 5 + [subprocess.CompletedProcess([], 3)]
        with patch.object(workflow.subprocess, "run", side_effect=codes):
            self.assertEqual(workflow.run(self.args), 3)
        self.assertEqual(self.manifest()["stages"][-1]["name"], "compare")
        self.assertEqual(self.manifest()["stages"][-1]["exit_code"], 3)

    def test_existing_evidence_is_not_overwritten(self):
        self.args.output_dir.mkdir()
        marker = self.args.output_dir / "model.bin"
        marker.write_bytes(b"keep")
        with patch.object(workflow.subprocess, "run") as run:
            self.assertEqual(workflow.run(self.args), 2)
        run.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_timeout_is_reported(self):
        with patch.object(workflow.subprocess, "run", side_effect=subprocess.TimeoutExpired(["qrt"], 1)):
            self.assertEqual(workflow.run(self.args), 2)
        self.assertIn("error", self.manifest()["stages"][0])

    def test_missing_input_is_rejected_before_launch(self):
        self.args.tokens = self.root / "missing.txt"
        with patch.object(workflow.subprocess, "run") as run:
            self.assertEqual(workflow.run(self.args), 2)
        run.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
