import json
import unittest
from pathlib import Path

from vllm.vllm_bench_metrics import (
    build_rate_sequence,
    normalize_row,
    parse_prefill_case,
    parse_summary_from_log,
    select_sla_rows,
)


FIXTURE = Path(__file__).parent / "fixtures" / "bench_serving.log"


class SharedMetricsTest(unittest.TestCase):
    def test_vllm_log_fields_are_normalized(self):
        row = normalize_row(parse_summary_from_log(FIXTURE), FIXTURE)
        self.assertEqual(row["rps"], 0.20)
        self.assertEqual(row["request_rate"], "inf")
        self.assertEqual(row["input_throughput_tok_s"], 819.20)
        self.assertEqual(row["generate_throughput_tok_s"], 0.20)
        self.assertEqual(row["total_throughput_tok_s"], 819.40)
        self.assertEqual(row["mean_ttft_ms"], 150.0)
        self.assertEqual(row["p95_ttft_ms"], 180.0)
        self.assertEqual(row["p99_ttft_ms"], 200.0)
        self.assertEqual(row["mean_e2e_latency_ms"], 160.0)

    def test_single_request_percentiles_fall_back_to_mean(self):
        row = normalize_row(
            {
                "successful_requests": 1,
                "mean_ttft_ms": 123.0,
                "median_ttft_ms": 123.0,
            }
        )
        self.assertEqual(row["p50_ttft_ms"], 123.0)
        self.assertEqual(row["p95_ttft_ms"], 123.0)
        self.assertEqual(row["p99_ttft_ms"], 123.0)


class PrefillCoreTest(unittest.TestCase):
    def test_case_supports_binary_and_decimal_k(self):
        case = parse_prefill_case("0.5k,p99_ttft_ms,5|6", 1024)
        self.assertEqual(case.input_tokens, 512)
        self.assertEqual(case.output_tokens, 1)
        self.assertEqual(case.targets_s, (5.0, 6.0))
        self.assertEqual(
            parse_prefill_case("0.5k,mean_ttft_ms,3", 1000).input_tokens,
            500,
        )

    def test_rate_sequence_is_descending_and_uses_point_one_step(self):
        self.assertEqual(
            build_rate_sequence(1.03, 0.1),
            [1.03, 0.93, 0.83, 0.73, 0.63, 0.53, 0.43, 0.33, 0.23, 0.13],
        )

    def test_selects_highest_actual_rps_for_each_target(self):
        case = parse_prefill_case("4k,mean_ttft_ms,3|4")
        rows = [
            {
                "exit_code": 0,
                "successful_requests": 10,
                "failed_requests": 0,
                "rps": 1.0,
                "mean_ttft_ms": 3500,
            },
            {
                "exit_code": 0,
                "successful_requests": 10,
                "failed_requests": 0,
                "rps": 0.9,
                "mean_ttft_ms": 2800,
            },
            {
                "exit_code": 0,
                "successful_requests": 10,
                "failed_requests": 0,
                "rps": 0.8,
                "mean_ttft_ms": 2200,
            },
        ]
        selected = select_sla_rows(rows, case)
        self.assertEqual(selected[3.0]["rps"], 0.9)
        self.assertEqual(selected[4.0]["rps"], 1.0)


if __name__ == "__main__":
    unittest.main()
