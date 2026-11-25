import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vllm.vllm_bench_metrics import parse_summary_text
from vllm.vllm_fake_prefill_client import (
    RequestRecord,
    build_dataset,
    build_summary,
    collect_payload,
    page_align_prefix_ratio,
)

FAKE_IDS = list(range(1000, 3000))


def find_bash():
    """可用的 bash; Windows 下 WSL stub 不可用则返回 None."""
    bash = shutil.which("bash")
    if not bash:
        return None
    try:
        proc = subprocess.run([bash, "--version"], capture_output=True, timeout=15)
        return bash if proc.returncode == 0 else None
    except OSError:
        return None


class PageAlignTest(unittest.TestCase):
    def test_zero_hit(self):
        self.assertEqual(page_align_prefix_ratio(4096, 0, 64), 0)
        self.assertEqual(page_align_prefix_ratio(4096, -1, 64), 0)

    def test_full_hit_keeps_one_unique_token(self):
        self.assertEqual(page_align_prefix_ratio(4096, 100, 64), 4032)

    def test_floor_page_alignment(self):
        self.assertEqual(page_align_prefix_ratio(1000, 50, 64), 448)
        self.assertEqual(page_align_prefix_ratio(100, 90, 64), 64)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            page_align_prefix_ratio(1024, 150, 64)


class BuildDatasetTest(unittest.TestCase):
    def test_lengths_and_warmup(self):
        warmup, prompts = build_dataset(
            tokenizer_ids=FAKE_IDS, num_prompts=8, input_len=1024,
            prefix_hit_pct=100, page_size=64, seed=7,
        )
        cached = page_align_prefix_ratio(1024, 100, 64)  # 960
        self.assertEqual(cached, 960)
        self.assertEqual(len(prompts), 8)
        self.assertTrue(all(len(p) == 1024 for p in prompts))
        self.assertEqual(len(warmup), cached + 1)
        # 共享前缀一致, warmup 的 start token 不与主请求重复
        starts_main = {p[cached] for p in prompts}
        self.assertEqual(warmup[:cached], prompts[0][:cached])
        self.assertNotIn(warmup[cached], starts_main)
        self.assertEqual(len(starts_main), 8)

    def test_zero_hit_has_no_warmup(self):
        warmup, prompts = build_dataset(
            tokenizer_ids=FAKE_IDS, num_prompts=4, input_len=256,
            prefix_hit_pct=0, page_size=64, seed=1,
        )
        self.assertIsNone(warmup)
        self.assertTrue(all(len(p) == 256 for p in prompts))

    def test_deterministic_seed(self):
        _, a = build_dataset(tokenizer_ids=FAKE_IDS, num_prompts=4,
                             input_len=512, prefix_hit_pct=50, page_size=64, seed=9)
        _, b = build_dataset(tokenizer_ids=FAKE_IDS, num_prompts=4,
                             input_len=512, prefix_hit_pct=50, page_size=64, seed=9)
        _, c = build_dataset(tokenizer_ids=FAKE_IDS, num_prompts=4,
                             input_len=512, prefix_hit_pct=50, page_size=64, seed=10)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


def make_record(ttft_ms, latency_ms, output_tokens, prompt_tokens, cached_tokens):
    rec = RequestRecord()
    rec.ttft_s = ttft_ms / 1000.0
    rec.latency_s = latency_ms / 1000.0
    rec.output_tokens = output_tokens
    rec.prompt_tokens = prompt_tokens
    rec.cached_tokens = cached_tokens
    return rec


class SummaryFormatTest(unittest.TestCase):
    def payload(self):
        records = [
            make_record(4.5, 10.0, 1, 4096, 4032),
            make_record(5.5, 12.0, 1, 4096, 4032),
        ]
        return collect_payload(
            records, {},
            num_prompts=2, input_len=4096, output_len=1,
            prefix_hit_pct=100, cache_salt="s", warmup_used=True,
            duration_s=1.0,
        )

    def test_payload_basics(self):
        payload = self.payload()
        self.assertEqual(payload["successful_requests"], 2)
        self.assertEqual(payload["failed_requests"], 0)
        self.assertAlmostEqual(payload["measured_cache_hit_pct"], 98.44, places=2)
        self.assertEqual(payload["mean_cached_tokens"], 4032.0)
        self.assertIsNone(payload["mean_tpot_ms"])  # output_len=1 -> 无 TPOT
        # 失败请求计为 failed
        failed = collect_payload(
            [make_record(1, 0, 1, 0, 0), make_record(2, 2, 1, 10, 0)], {},
            num_prompts=2, input_len=10, output_len=1,
            prefix_hit_pct=0, cache_salt="", warmup_used=False,
            duration_s=1.0,
        )
        self.assertEqual(failed["failed_requests"], 1)
        self.assertEqual(failed["successful_requests"], 1)

    def test_summary_parseable_by_metrics_module(self):
        text = build_summary(self.payload())
        data = parse_summary_text(text)
        self.assertEqual(data["successful_requests"], 2)
        self.assertEqual(data["failed_requests"], 0)
        self.assertIn("benchmark_duration_s", data)
        self.assertIn("request_throughput", data)
        self.assertIn("input_throughput", data)
        self.assertIn("output_throughput", data)
        self.assertIn("total_throughput", data)
        self.assertIn("mean_ttft_ms", data)
        self.assertIn("p99_ttft_ms", data)
        self.assertIn("mean_e2e_latency_ms", data)  # 模块别名: mean_e2el_ms


class DriverShellTest(unittest.TestCase):
    script = REPO_ROOT / "vllm" / "vllm_fake_prefill_bench.sh"

    def run_bash(self, *args, env_extra=None):
        import os
        env = {**os.environ, "PYTHON_BIN": sys.executable.replace("\\", "/")}
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [find_bash(), str(self.script), *args],
            capture_output=True, text=True, timeout=60, env=env,
        )

    @unittest.skipUnless(find_bash(), "usable bash not available")
    def test_help(self):
        proc = self.run_bash("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("fake-prefill", proc.stdout)

    @unittest.skipUnless(find_bash(), "usable bash not available")
    def test_dry_run_creates_trials_header(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = self.run_bash("--new-run", "--case", "4k,1,0|100",
                                 env_extra={"OUTPUT_ROOT": temp, "NUM_PROMPTS": "8"})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("[dry-run]", proc.stdout)
            run_ids = list(Path(temp).glob("bench_vllm_fake_prefill_*"))
            self.assertEqual(len(run_ids), 1)
            trials = run_ids[0] / "vllm_fake_prefill_trials.csv"
            self.assertTrue(trials.is_file())
            header = trials.read_text(encoding="utf-8").strip().split(",")
            self.assertIn("measured_hit_pct", header)
            self.assertIn("ttft_ratio_vs_cold", header)
            self.assertIn("verdict", header)

    def test_metrics_hit_fallback(self):
        # usage 未上报 cached_tokens -> 回退 Prometheus 差值
        payload = collect_payload(
            [make_record(4.0, 9.0, 1, 4096, 0)], {},
            num_prompts=1, input_len=4096, output_len=1,
            prefix_hit_pct=100, cache_salt="s", warmup_used=True,
            duration_s=1.0, pc_queries_delta=128.0, pc_hits_delta=64.0,
        )
        self.assertEqual(payload["measured_hit_source"], "metrics")
        self.assertEqual(payload["measured_cache_hit_pct"], 50.0)
        self.assertEqual(payload["prefix_cache_query_tokens"], 128.0)
        # usage 可用时优先 usage
        payload2 = collect_payload(
            [make_record(1.0, 2.0, 1, 100, 50)], {},
            num_prompts=1, input_len=100, output_len=1,
            prefix_hit_pct=100, cache_salt="", warmup_used=False,
            duration_s=1.0, pc_queries_delta=100.0, pc_hits_delta=10.0,
        )
        self.assertEqual(payload2["measured_hit_source"], "usage")
        self.assertEqual(payload2["measured_cache_hit_pct"], 50.0)
        # 两者皆无
        payload3 = collect_payload(
            [make_record(1.0, 2.0, 1, 100, 0)], {},
            num_prompts=1, input_len=100, output_len=1,
            prefix_hit_pct=0, cache_salt="", warmup_used=False,
            duration_s=1.0,
        )
        self.assertEqual(payload3["measured_hit_source"], "none")
        self.assertEqual(payload3["measured_cache_hit_pct"], 0.0)
        self.assertIsNone(payload3["prefix_cache_query_tokens"])


if __name__ == "__main__":
    unittest.main()
