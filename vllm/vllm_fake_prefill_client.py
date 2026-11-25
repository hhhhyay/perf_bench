#!/usr/bin/env python3
"""vLLM fake-prefill / warmed shared-prefix benchmark client.

Self-contained async OpenAI client (aiohttp), no sglang dependency.

Protocol (mirrors sla_bench/prefix_hit.py):
  1. One warmup request bakes the shared random prefix into the server's
     Automatic Prefix Caching (APC) KV cache - the prefill is computed once.
  2. Main requests = warmed prefix + unique start token + random tail, so
     only the prefix can be served from cache ("fake prefill"); the rest of
     the prompt and all decode steps are fully live.

Result interpretation:
  PREFIX_HIT=0   -> cold random workload (all prompts independent).
  PREFIX_HIT=100 -> the timed run is effectively decode-only: the entire
                    input except one unique start token is served from KV
                    cache, i.e. client-side fake-prefill without any PD
                    disaggregation / server-side fake backend.

Verification signals (written to --result-json):
  measured_cache_hit_pct : token-level prefix cache hit rate. Preferred
                           source is usage.prompt_tokens_details.cached_tokens;
                           some server builds omit it in usage (e.g. vLLM 0.34
                           streaming), in that case fallback to Prometheus
                           deltas of vllm:prefix_cache_{queries,hits}_total
                           around the main timed window (source=metrics).

The printed summary block mimics `vllm bench serve` so
vllm_bench_metrics.py parse-log can consume the log unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
import urllib.request
from typing import Any
from urllib.parse import urlsplit

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


# --------------------------------------------------------------------------- #
# dataset generation (pure, offline testable)
# --------------------------------------------------------------------------- #

def load_tokenizer(vocab_source: str):
    """Return (available_token_ids, vocab_size) from a HF tokenizer path."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vocab_source, trust_remote_code=True)
    vocab_size = tok.vocab_size
    specials: set[int] = set()
    for name in (
        "all_special_ids",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        value = getattr(tok, name, None)
        if isinstance(value, int):
            specials.add(value)
        elif isinstance(value, (list, tuple, set)):
            specials.update(int(v) for v in value)
    ids = [i for i in range(vocab_size) if i not in specials]
    if len(ids) < 2:
        raise RuntimeError("tokenizer vocabulary too small after special-id filter")
    return ids, vocab_size


def page_align_prefix_ratio(
    input_len: int, prefix_hit_pct: float, page_size: int
) -> int:
    """Cached prefix length: page aligned, always keep >=1 unique token."""
    if prefix_hit_pct <= 0:
        return 0
    if prefix_hit_pct > 100:
        raise ValueError("prefix hit (%) must be 0..100")
    cached = int(math.floor(input_len * prefix_hit_pct / 100.0 / page_size)) * page_size
    cached = min(cached, (input_len - 1) // page_size * page_size)
    return max(cached, 0)


def build_dataset(
    *,
    tokenizer_ids: list[int],
    num_prompts: int,
    input_len: int,
    prefix_hit_pct: float,
    page_size: int,
    seed: int,
) -> tuple[list[list[int]] | None, list[list[int]]]:
    """Return (warmup_prompt, prompts).

    prompts[i] = shared_prefix + unique_start_i + random_tail_i
    total length == input_len for every prompt.
    warmup_prompt = shared_prefix + one extra unique start (not used by mains).
    """
    if input_len < 2:
        raise ValueError(f"input_len must be >= 2, got {input_len}")
    cached = page_align_prefix_ratio(input_len, prefix_hit_pct, page_size)
    rng = random.Random(seed)
    prefix = rng.choices(tokenizer_ids, k=cached) if cached else []
    # num_prompts unique starts + 1 extra for the warmup request
    starts = rng.sample(tokenizer_ids, num_prompts + 1)
    prompts = [
        prefix + [starts[i + 1]] + rng.choices(tokenizer_ids, k=input_len - cached - 1)
        for i in range(num_prompts)
    ]
    warmup_prompt = prefix + [starts[0]] if cached else None
    return warmup_prompt, prompts


# --------------------------------------------------------------------------- #
# metrics accumulation
# --------------------------------------------------------------------------- #

class RequestRecord:
    __slots__ = ("ttft_s", "latency_s", "output_tokens", "prompt_tokens", "cached_tokens")

    def __init__(self) -> None:
        self.ttft_s = 0.0
        self.latency_s = 0.0
        self.output_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def tpot_ms_from(rec: RequestRecord) -> float | None:
    if rec.output_tokens > 1:
        return (rec.latency_s - rec.ttft_s) * 1000.0 / (rec.output_tokens - 1)
    return None


def itl_ms_from(rec: RequestRecord, deltas: list[float]) -> float | None:
    if deltas:
        return sum(deltas) * 1000.0 / len(deltas)
    return None


def build_summary(payload: dict[str, Any]) -> str:
    """Render a `vllm bench serve` compatible summary block."""
    p = payload
    lines = [
        f"Successful requests: {p['successful_requests']}",
        f"Failed requests: {p['failed_requests']}",
        f"Number Of Prompts: {p['num_prompts']}",
        f"Random input len: {p['input_len']}",
        f"Random output len: {p['output_len']}",
        f"Prefix hit configured (%): {p['prefix_hit_pct']:.1f}",
        f"Cache salt: {p['cache_salt'] or '-'}",
        f"Warmup enabled: {1 if p['warmup_used'] else 0}",
        f"Benchmark duration (s): {p['duration_s']:.5f}",
        f"Request throughput (req/s): {p['request_throughput']:.5f}",
        f"Input token throughput (tok/s): {p['input_throughput']:.2f}",
        f"Output token throughput (tok/s): {p['output_throughput']:.2f}",
        f"Total token throughput (tok/s): {p['total_throughput']:.2f}",
        f"Mean TTFT (ms): {p['mean_ttft_ms']:.5f}",
        f"P50 TTFT (ms): {p['p50_ttft_ms']:.5f}",
        f"P95 TTFT (ms): {p['p95_ttft_ms']:.5f}",
        f"P99 TTFT (ms): {p['p99_ttft_ms']:.5f}",
        f"Max TTFT (ms): {p['max_ttft_ms']:.5f}",
        f"Mean prefix cache hit (%): {p['measured_cache_hit_pct']:.2f}",
        f"Hit source: {p.get('measured_hit_source', 'usage')}",
        f"Mean cached tokens per prompt: {p['mean_cached_tokens']:.2f}",
    ]
    if p["mean_tpot_ms"] is not None:
        lines += [
            f"Mean TPOT (ms): {p['mean_tpot_ms']:.5f}",
            f"P50 TPOT (ms): {p['p50_tpot_ms']:.5f}",
            f"P95 TPOT (ms): {p['p95_tpot_ms']:.5f}",
            f"P99 TPOT (ms): {p['p99_tpot_ms']:.5f}",
            f"Max TPOT (ms): {p['max_tpot_ms']:.5f}",
        ]
    if p["mean_itl_ms"] is not None:
        lines += [
            f"Mean ITL (ms): {p['mean_itl_ms']:.5f}",
            f"P50 ITL (ms): {p['p50_itl_ms']:.5f}",
            f"P95 ITL (ms): {p['p95_itl_ms']:.5f}",
            f"P99 ITL (ms): {p['p99_itl_ms']:.5f}",
            f"Max ITL (ms): {p['max_itl_ms']:.5f}",
        ]
    lines += [
        f"Mean E2EL (ms): {p['mean_e2el_ms']:.5f}",
        f"P50 E2EL (ms): {p['p50_e2el_ms']:.5f}",
        f"P95 E2EL (ms): {p['p95_e2el_ms']:.5f}",
        f"P99 E2EL (ms): {p['p99_e2el_ms']:.5f}",
        f"Max E2EL (ms): {p['max_e2el_ms']:.5f}",
    ]
    return "\n".join(lines)


def collect_payload(
    records: list[RequestRecord],
    itl_samples: dict[int, list[float]],
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    prefix_hit_pct: float,
    cache_salt: str,
    warmup_used: bool,
    duration_s: float,
    pc_queries_delta: float | None = None,
    pc_hits_delta: float | None = None,
) -> dict[str, Any]:
    ok = [r for r in records if r.prompt_tokens > 0]
    failed = len(records) - len(ok)
    ttfts = [r.ttft_s * 1000.0 for r in ok]
    e2els = [r.latency_s * 1000.0 for r in ok]
    tpots = [v for v in (tpot_ms_from(r) for r in ok) if v is not None]
    itls: list[float] = []
    for v in (itl_ms_from(r, itl_samples.get(i, [])) for i, r in enumerate(ok)):
        if v is not None:
            itls.append(v)
    total_prompt = sum(r.prompt_tokens for r in ok)
    total_cached = sum(r.cached_tokens for r in ok)
    total_output = sum(r.output_tokens for r in ok)

    def block(name: str, values: list[float], mean_default: float | None) -> dict[str, Any]:
        if not values:
            return {f"{s}_{name}_ms": mean_default for s in ("mean", "p50", "p95", "p99", "max")}
        return {
            f"mean_{name}_ms": sum(values) / len(values),
            f"p50_{name}_ms": percentile(values, 50),
            f"p95_{name}_ms": percentile(values, 95),
            f"p99_{name}_ms": percentile(values, 99),
            f"max_{name}_ms": max(values),
        }

    def flat(d: dict[str, Any]) -> dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}

    duration = max(duration_s, 1e-9)
    metrics_hit: float | None = None
    if pc_queries_delta and pc_queries_delta > 0 and pc_hits_delta is not None:
        metrics_hit = round(pc_hits_delta * 100.0 / pc_queries_delta, 2)
    if total_cached > 0:
        measured_hit, hit_source = round(total_cached * 100.0 / total_prompt, 2), "usage"
    elif metrics_hit is not None:
        measured_hit, hit_source = metrics_hit, "metrics"
    else:
        measured_hit, hit_source = 0.0, "none"
    payload: dict[str, Any] = {
        "successful_requests": len(ok),
        "failed_requests": failed,
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "prefix_hit_pct": prefix_hit_pct,
        "cache_salt": cache_salt,
        "warmup_used": warmup_used,
        "duration_s": round(duration_s, 5),
        "request_throughput": round(len(ok) / duration, 5),
        "input_throughput": round(total_prompt / duration, 2),
        "output_throughput": round(total_output / duration, 2),
        "total_throughput": round((total_prompt + total_output) / duration, 2),
        "measured_cache_hit_pct": measured_hit,
        "measured_hit_source": hit_source,
        "mean_cached_tokens": round(total_cached / len(ok), 2) if ok else 0.0,
        "prefix_cache_query_tokens": pc_queries_delta,
        "prefix_cache_hit_tokens": pc_hits_delta,
    }
    payload.update(flat(block("ttft", ttfts, None)))
    payload.update(flat(block("tpot", tpots, None)))
    payload.update(flat(block("itl", itls, None)))
    payload.update(flat(block("e2el", e2els, None)))
    return payload


# --------------------------------------------------------------------------- #
# networking
# --------------------------------------------------------------------------- #

def parse_sse(data: bytes) -> tuple[bool, int]:
    """Parse one SSE message; return (has_content, completion_tokens_so_far_delta)."""
    has_content = False
    n_new_tokens = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            has_content = True
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if choices:
            text = choices[0].get("text")
            if text:
                has_content = True
        usage = payload.get("usage")
        if usage:
            has_content = True
    # token counting handled by caller via usage detection
    return has_content, n_new_tokens


async def stream_one(
    session: "aiohttp.ClientSession",
    url: str,
    body: dict[str, Any],
    timeout_s: float,
    record: RequestRecord,
    itl_out: list[float],
) -> None:
    """POST /v1/completions with streaming; fill the RequestRecord."""
    payload = dict(body)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    first_token_at: float | None = None
    prev_token_at = t0
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=timeout_s, sock_read=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        async for raw, _ in resp.content.iter_chunks():
            if not raw.strip():
                continue
            now = time.perf_counter()
            used_usage = False
            usage_prompt = 0
            usage_cached = 0
            text_content = False
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    continue
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if choices and choices[0].get("text"):
                    text_content = True
                usage = event.get("usage")
                if usage:
                    details = usage.get("prompt_tokens_details") or {}
                    usage_prompt = int(usage.get("prompt_tokens") or 0)
                    usage_cached = int(details.get("cached_tokens") or 0)
            if text_content and first_token_at is None:
                first_token_at = now
                prev_token_at = now
            if usage_prompt:
                record.latency_s = now - t0
                record.prompt_tokens = usage_prompt
                record.cached_tokens = usage_cached
            elif text_content:
                record.latency_s = now - t0
            if text_content:
                if first_token_at is None:
                    first_token_at = now
                else:
                    itl_out.append(now - prev_token_at)
                prev_token_at = now
                record.output_tokens += 1
    if first_token_at is not None:
        record.ttft_s = first_token_at - t0
        record.output_tokens = max(record.output_tokens, 1)
    elif record.latency_s > 0:
        # non-streaming fallback shape
        record.output_tokens = max(record.output_tokens, 1)


def fetch_prefix_cache_counters(base_url: str) -> tuple[float, float] | None:
    """(query_tokens, hit_tokens) totals from /metrics, or None if unavailable."""
    parts = urlsplit(base_url)
    origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme else base_url
    try:
        with urllib.request.urlopen(origin + "/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    queries = 0.0
    hits = 0.0
    for line in text.splitlines():
        m = re.match(r"vllm:prefix_cache_queries_total\{[^}]*\}\s+([0-9.eE+]+)", line)
        if m:
            queries += float(m.group(1))
            continue
        m = re.match(r"vllm:prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+]+)", line)
        if m:
            hits += float(m.group(1))
    if queries <= 0 and hits <= 0:
        return None
    return queries, hits


async def warmup_prefix(base_url: str, prompt: list[int], body_base: dict[str, Any], timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/v1/completions"
    payload = dict(body_base)
    payload["prompt"] = prompt
    payload["max_tokens"] = 1
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=timeout_s, sock_read=timeout_s)
    async with aiohttp.ClientSession(trust_env=False) as session:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
    if not isinstance(data, dict) or "choices" not in data:
        raise RuntimeError(f"unexpected warmup response: {str(data)[:400]}")


def run_self_check(args: argparse.Namespace) -> dict[str, Any]:
    """Offline dataset construction check (no network)."""
    tokenizer_ids, vocab_size = load_tokenizer(args.tokenizer)
    cached_len = page_align_prefix_ratio(args.input_len, args.prefix_hit_pct, args.page_size)
    warmup_prompt, prompts = build_dataset(
        tokenizer_ids=tokenizer_ids,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        prefix_hit_pct=args.prefix_hit_pct,
        page_size=args.page_size,
        seed=args.seed,
    )
    lens = sorted({len(p) for p in prompts})
    print("[self-check] vocab_size:", vocab_size)
    print("[self-check] input_len:", args.input_len, "prompt lens:", lens)
    print("[self-check] prefix_hit:", args.prefix_hit_pct,
          "-> cached_prefix_len:", cached_len)
    print("[self-check] warmup prompt len:", len(warmup_prompt) if warmup_prompt else 0)
    assert not lens or lens == [args.input_len], "prompt lengths must equal input_len"
    assert len(warmup_prompt) == cached_len + 1 if warmup_prompt else True
    print("[self-check] OK (offline, no request sent)")
    return {"self_check": True, "cached_prefix_len": cached_len}


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        return run_self_check(args)
    tokenizer_ids, vocab_size = load_tokenizer(args.tokenizer)
    cached_len = page_align_prefix_ratio(args.input_len, args.prefix_hit_pct, args.page_size)
    warmup_prompt, prompts = build_dataset(
        tokenizer_ids=tokenizer_ids,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        prefix_hit_pct=args.prefix_hit_pct,
        page_size=args.page_size,
        seed=args.seed,
    )
    print(f"[INFO] vocab_size={vocab_size} input_len={args.input_len} "
          f"prefix_hit={args.prefix_hit_pct}% cached_prefix_len={cached_len} "
          f"(page={args.page_size}) prompts={args.num_prompts} output_len={args.output_len}")

    rng = random.Random(args.seed + 1)
    body_base = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": not args.eos,
        "max_tokens": args.output_len,
    }
    if args.cache_salt:
        body_base["cache_salt"] = args.cache_salt

    warmup_used = False
    if warmup_prompt is not None and args.warmup > 0:
        await warmup_prefix(args.base_url, warmup_prompt, body_base, args.timeout)
        warmup_used = True
        print(f"[INFO] warmup done: shared prefix of {cached_len} tokens baked into cache")
        await asyncio.sleep(args.warmup_sleep)

    url = args.base_url.rstrip("/") + "/v1/completions"
    pc0 = fetch_prefix_cache_counters(args.base_url)  # after warmup, before mains
    records: list[RequestRecord | None] = [None] * args.num_prompts
    itl_samples: dict[int, list[float]] = {}
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency > 0 else None

    async with aiohttp.ClientSession(
        trust_env=False,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
    ) as session:
        start = time.perf_counter()

        async def worker(index: int, when: float) -> None:
            delay = max(0.0, when - (time.perf_counter() - start))
            await asyncio.sleep(delay)
            record = RequestRecord()
            itl_out: list[float] = []
            if sem is not None:
                await sem.acquire()
            try:
                await stream_one(session, url, {**body_base, "prompt": prompts[index]},
                                 args.timeout, record, itl_out)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] request {index} failed: {exc}", file=sys.stderr)
            finally:
                if sem is not None:
                    sem.release()
                records[index] = record
                if itl_out:
                    itl_samples[index] = itl_out

        if args.request_rate == "inf":
            tasks = [asyncio.create_task(worker(i, 0.0)) for i in range(args.num_prompts)]
        else:
            rate = float(args.request_rate)
            cumulative: list[float] = []
            gap_total = 0.0
            for _ in range(args.num_prompts):
                gap_total += rng.expovariate(rate)  # Poisson arrivals: exp gaps, cumulative sum
                cumulative.append(gap_total)
            tasks = [asyncio.create_task(worker(i, cumulative[i])) for i in range(args.num_prompts)]
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start

    pc1 = fetch_prefix_cache_counters(args.base_url)
    pc_queries_delta = pc_hits_delta = None
    if pc0 and pc1:
        pc_queries_delta = max(0.0, pc1[0] - pc0[0])
        pc_hits_delta = max(0.0, pc1[1] - pc0[1])

    valid_records = [r or RequestRecord() for r in records]

    payload = collect_payload(
        valid_records,
        itl_samples,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        prefix_hit_pct=args.prefix_hit_pct,
        cache_salt=args.cache_salt,
        warmup_used=warmup_used,
        duration_s=duration,
        pc_queries_delta=pc_queries_delta,
        pc_hits_delta=pc_hits_delta,
    )
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(build_summary(payload))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:18901")
    parser.add_argument("--model", default="")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path for vocab sampling")
    parser.add_argument("--input-len", type=int, default=4096)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prefix-hit", type=float, default=100.0, dest="prefix_hit_pct",
                        help="0..100, warmed shared prefix ratio; 0 = cold")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--request-rate", default="inf",
                        help="'inf' or requests/second (Poisson arrivals)")
    parser.add_argument("--max-concurrency", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--warmup", type=int, default=1, help="1 = bake shared prefix before the run")
    parser.add_argument("--warmup-sleep", type=float, default=1.0)
    parser.add_argument("--cache-salt", default="", help="cache_salt for APC isolation between runs")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument("--eos", action="store_true", help="respect EOS (default ignore_eos)")
    parser.add_argument("--self-check", action="store_true",
                        help="offline: validate tokenizer + dataset construction, no request")
    parser.add_argument("--result-json", default="", help="path to write full metrics JSON")
    args = parser.parse_args(argv)
    if not args.self_check and not args.model:
        parser.error("--model is required unless --self-check")
    if args.request_rate != "inf":
        value = float(args.request_rate)
        if not math.isfinite(value) or value <= 0:
            parser.error("request-rate must be 'inf' or a positive number")
    return args


def main(argv: list[str] | None = None) -> int:
    if aiohttp is None:
        print("[ERROR] aiohttp is required", file=sys.stderr)
        return 2
    args = parse_args(argv)
    payload = asyncio.run(run_benchmark(args))
    if payload.get("self_check"):
        return 0
    return 0 if payload["failed_requests"] == 0 and payload["successful_requests"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
