import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vllm.vllm_bench_metrics import (
    append_trial_row,
    parse_prefill_case,
    write_prefill_summary,
)
from vllm.vllm_prefill_plot import build_matrix


class PrefillOutputTest(unittest.TestCase):
    def make_row(self, case, request_rate, rps, input_tp, ttft):
        return {
            "case_id": case.case_id,
            "case_name": case.case_name,
            "input_len": case.input_tokens,
            "output_len": 1,
            "parallel_config": "TP8",
            "SLO": case.slo_json,
            "request_rate": request_rate,
            "duration_s": 60,
            "rps": rps,
            "input_throughput_tok_s": input_tp,
            "generate_throughput_tok_s": rps,
            "total_throughput_tok_s": input_tp + rps,
            "successful_requests": 60,
            "failed_requests": 0,
            "num_prompts": 60,
            "max_concurrency": "",
            "concurrency": 12,
            "peak_output_throughput": rps,
            "peak_concurrency": 12,
            "mean_ttft_ms": ttft,
            "p95_ttft_ms": ttft,
            "p99_ttft_ms": ttft,
            "exit_code": 0,
            "case_status": "PASS",
            "search_phase": "descending" if request_rate != "inf" else "inf_trial_1",
            "log_file": f"{case.case_id}_{request_rate}.log",
        }

    def test_summary_keeps_best_inf_and_descending_finite_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trials = root / "trials.csv"
            summary = root / "summary.tsv"
            case = parse_prefill_case("4k,mean_ttft_ms,3|4")
            append_trial_row(trials, self.make_row(case, "inf", 1.1, 1200, 3500))
            append_trial_row(trials, self.make_row(case, "inf", 1.0, 1000, 3600))
            append_trial_row(trials, self.make_row(case, "0.93", 0.93, 1100, 2800))
            append_trial_row(trials, self.make_row(case, "0.83", 0.83, 1000, 2200))
            write_prefill_summary(trials, summary, [case])

            with summary.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                rows = list(reader)
                header = reader.fieldnames

            self.assertEqual(
                header[:5],
                ["case_name", "input_len", "output_len", "parallel_config", "SLO"],
            )
            self.assertEqual(header[header.index("rps") + 1], "input_throughput_tok_s")
            self.assertEqual(rows[0]["request_rate"], "inf")
            self.assertEqual(
                [row["request_rate"] for row in rows[1:]],
                ["0.93", "0.83"],
            )
            self.assertEqual(rows[0]["SLO"], '{"ttft":{"mean":[3,4]}}')

    def test_matrix_chooses_highest_actual_rps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trials = root / "trials.csv"
            summary = root / "summary.tsv"
            case = parse_prefill_case("4k,mean_ttft_ms,3|4")
            append_trial_row(trials, self.make_row(case, "inf", 1.1, 1200, 3500))
            append_trial_row(trials, self.make_row(case, "0.93", 0.93, 1100, 2800))
            append_trial_row(trials, self.make_row(case, "0.83", 0.83, 1000, 2200))
            write_prefill_summary(trials, summary, [case])

            matrix = build_matrix(summary, "rps")
            self.assertEqual(matrix["columns"], [case.case_name])
            self.assertEqual(matrix["winners"][(case.case_name, ("mean", 3.0))], 0.93)
            self.assertEqual(matrix["winners"][(case.case_name, ("mean", 4.0))], 0.93)

    def test_cli_append_trial_and_summary(self):
        repo = Path(__file__).resolve().parents[2]
        metrics = repo / "vllm" / "vllm_bench_metrics.py"
        fixture = repo / "vllm" / "tests" / "fixtures" / "bench_serving.log"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trials = root / "trials.csv"
            summary = root / "summary.tsv"
            cases = root / "cases.json"

            parsed_case = subprocess.run(
                [sys.executable, str(metrics), "parse-case", "4k,mean_ttft_ms,3|4", "1024"],
                check=True,
                capture_output=True,
                text=True,
            )
            case = json.loads(parsed_case.stdout)
            cases.write_text(json.dumps([case]), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(metrics),
                    "append-trial",
                    str(trials),
                    str(fixture),
                    json.dumps(case),
                    "inf",
                    "2",
                    "",
                    "TP8",
                    "0",
                    "inf_trial_1",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, str(metrics), "summary", str(trials), str(summary), str(cases)],
                check=True,
            )

            with summary.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["request_rate"], "inf")
            self.assertEqual(rows[0]["input_len"], "4096")
            self.assertEqual(rows[0]["p95_ttft_ms"], "180.0")


if __name__ == "__main__":
    unittest.main()
