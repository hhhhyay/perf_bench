"""Native SGLang benchmark: warmed shared-prefix protocol, fully parameterized.

Derivative of the on-prem prefix90.py (hit rate generalized, no longer fixed 90%):
  - PREFIX_HIT env (0..100, default 90) sets the warmed prefix ratio; 0 = cold;
  - arbitrary input/output lengths (no whitelist; the 4096-cold convention is
    dropped - pass hit=0 for a cold case);
  - one warmup request bakes `cached` shared tokens into the radix cache;
  - main requests = warmed prefix + unique start token + random tail, so
    nothing beyond the warmed prefix is shared.
"""

import json
import os
import random
import time
import urllib.request
from pathlib import Path

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow, get_available_tokens


def dataset(args, tokenizer, model_id=None):
    total = args.random_input_len
    if total < 2:
        raise ValueError(f"random_input_len must be >= 2, got {total}")
    hit = int(os.environ.get("PREFIX_HIT", "90"))
    if not 0 <= hit <= 100:
        raise ValueError(f"PREFIX_HIT must be 0..100, got {hit}")
    cached = (total * hit // 100 // 64) * 64
    if cached > total - 1:  # keep >=1 token for the unique start + tail
        cached = (total - 1) // 64 * 64
    rng = random.Random(int(os.environ["PREFIX_SEED"]))
    tokens = get_available_tokens(tokenizer)
    prefix = rng.choices(tokens, k=cached)
    starts = rng.sample(tokens, args.num_prompts + 1)

    if cached:
        body = dict(
            input_ids=prefix + [starts[0]],
            sampling_params=dict(
                max_new_tokens=1, temperature=0.6, top_p=0.95, ignore_eos=True
            ),
            stream=False,
        )
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3600) as response:
            warmup = json.load(response)
        if not isinstance(warmup, dict) or "meta_info" not in warmup:
            raise RuntimeError(f"Unexpected warmup response: {str(warmup)[:400]}")
        if warmup["meta_info"].get("finish_reason", {}).get("type") == "abort":
            raise RuntimeError(f"Warmup aborted: {warmup['meta_info']}")
        Path(args.output_file).with_suffix(".warmup.json").write_text(
            json.dumps(warmup, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        time.sleep(1)

    return [
        DatasetRow(
            prompt=prefix + [starts[i + 1]] + rng.choices(tokens, k=total - cached - 1),
            prompt_len=total,
            output_len=args.random_output_len,
        )
        for i in range(args.num_prompts)
    ]


if __name__ == "__main__":
    if os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true"):
        raise RuntimeError("Unset SGLANG_IS_IN_CI: it would flush the warmed prefix")
    serving.get_dataset = dataset
    serving.cli_main()
