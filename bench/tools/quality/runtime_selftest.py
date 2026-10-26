import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RuntimeCliTests(unittest.TestCase):
    binary = None

    def run_qrt(self, *args, env=None):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("QRT_")}
        if env:
            clean.update(env)
        return subprocess.run([str(self.binary), *args], env=clean, text=True,
                              encoding="utf-8", errors="replace", capture_output=True,
                              timeout=30)

    def test_defaults(self):
        result = self.run_qrt("--print-config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prefill_chunk=2048", result.stdout)
        self.assertIn("kv_k=q8_0", result.stdout)
        self.assertIn("kv_v=q4_0", result.stdout)

    def test_config_precedence(self):
        result = self.run_qrt("--print-config", "--spec-k=2", "--spec-k", "8",
                              env={"QRT_SPEC_K": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("spec_k=8", result.stdout)

    def test_cli_overrides_invalid_environment(self):
        result = self.run_qrt("--print-config", "--spec-k", "4", env={"QRT_SPEC_K": "bad"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("spec_k=4", result.stdout)
        result = self.run_qrt("--print-config", env={"QRT_SPEC_K": "bad"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_help_is_complete_and_environment_independent(self):
        result = self.run_qrt("--help", env={"QRT_SPEC_K": "bad"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ["--model-check", "--chunk-check", "--prefill-check", "--logits-dump"]:
            self.assertIn(command, result.stdout)

    def test_probe_requires_supported_submission_mode(self):
        result = self.run_qrt("--prefill-check", "missing", "--prefill-compute", "probe")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("perexpert", result.stderr)

    def test_boolean_cli_flag_sets_its_member_not_a_command(self):
        # --spec-mtp is the one flag entry that turns a member on instead of selecting a
        # command; the two failure modes to keep are a value on a flag and two commands
        result = self.run_qrt("--spec-check", "missing", "--spec-mtp=1")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("takes no value", result.stderr)
        result = self.run_qrt("--spec-check", "missing", "--generate", "missing")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("conflicting commands", result.stderr)
        result = self.run_qrt("--print-config", "--spec-mtp")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--spec-mtp needs", result.stderr)

    def test_invalid_numbers(self):
        for flag, value in [("--layer", "3junk"), ("--prefill-pipeline", "maybe"),
                            ("--prefill-pipeline", "2"), ("--prefill-group-kib", "abc"),
                            ("--logits-layers", "junk"), ("--dense-rows", "2147483648"),
                            ("--fullattn-resident", "999999999999999999999999")]:
            with self.subTest(flag=flag, value=value):
                result = self.run_qrt("--print-config", flag, value)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(flag, result.stderr)

    def test_unknown_arguments(self):
        for arg in ["typo", "-bad", "--unknown"]:
            with self.subTest(arg=arg):
                result = self.run_qrt("--print-config", arg)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_missing_values(self):
        for flag in ["--layer", "--logits-out", "--prefill-chunk"]:
            with self.subTest(flag=flag):
                result = self.run_qrt(flag)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_diagnostic_equals_syntax(self):
        result = self.run_qrt("--print-config", "--layer=3", "--prefill-pipeline=0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_conflicting_commands(self):
        for args in [("--print-config", "--gguf-index", "missing.gguf"),
                     ("--gguf-index", "missing.gguf", "--logits-dump", "missing.gguf")]:
            with self.subTest(args=args):
                result = self.run_qrt(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("command", result.stderr.lower())

    def test_no_command_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "telemetry.csv"
            result = self.run_qrt("--telemetry", "1", "--telemetry-path", str(output))
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(output.exists(), "no command must not create a telemetry file")

    def test_moe_batch_requires_store(self):
        result = self.run_qrt("--moe-batch-check", "missing.gguf")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--layer-store", result.stderr)
        self.assertIn("--moe-batch-check", self.run_qrt("--help").stdout)

    def test_cache_option_contract(self):
        cases = [
            ("--cache-check",),
            ("--cache-trace", "trace.bin"),
            ("--cache-check", "missing-store", "--cache-decode", "0"),
            ("--cache-check", "missing-store", "--cache-protected", "101"),
            ("--cache-check", "missing-store", "positional"),
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_qrt(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--cache-check", self.run_qrt("--help").stdout)

    def test_model_requires_explicit_expert_store(self):
        result = self.run_qrt("--logits-dump", "missing.gguf", "--logits-tokens", "ids.txt",
                              "--logits-out", "dump.bin")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--layer-store", result.stderr)

    def test_builtin_selftest(self):
        result = self.run_qrt("--selftest")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def main(argv):
    parser = argparse.ArgumentParser(description="CPU-only runtime CLI contract regression")
    parser.add_argument("--qrt", type=Path, required=True)
    args = parser.parse_args(argv)
    RuntimeCliTests.binary = args.qrt.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeCliTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
