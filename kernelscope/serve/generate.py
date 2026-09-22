"""Local raw-text continuation demo; single-request timing is not a serving benchmark."""
import json
import math
from pathlib import Path
import time

def generate(model, pool, tokenizer, prompt, max_new_tokens, policy, eos_token_ids=()):
    import torch
    with torch.inference_mode():
        return _generate(model, pool, tokenizer, prompt, max_new_tokens, policy, eos_token_ids)


def _generate(model, pool, tokenizer, prompt, max_new_tokens, policy, eos_token_ids):
    import torch
    from kernelscope.serve.engine import _CallTimer
    from kernelscope.serve.kvcache import PAGE
    from kernelscope.serve.model import AttentionTimer
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be a positive integer")
    prompt_tokens = tokenizer.encode(prompt).ids
    if not prompt_tokens:
        raise ValueError("the tokenizer produced an empty prompt")
    if math.ceil((len(prompt_tokens) + max_new_tokens - 1) / PAGE) > pool.free_pages:
        raise MemoryError("prompt plus generation exceeds the KV pool; increase --kv-gib")
    if policy.name in {"model", "table"} and (model.cfg.head_dim != 128 or model.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError("model/table policies require d=128 fp16/bf16")
    generated, stamps, steps = [], [], []
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    try:
        timer = _CallTimer(model.device)
        timer.start()
        logits = model.prefill("text_demo", prompt_tokens, pool)
        prefill_us = timer.stop()
        while len(generated) < max_new_tokens:
            token = int(logits.argmax().item())
            generated.append(token)
            stamps.append((time.perf_counter() - started) * 1e6)
            if token in eos_token_ids or len(generated) == max_new_tokens:
                break
            selection_start = time.perf_counter()
            splits = policy.choose([pool.length("text_demo") + 1], model.cfg.n_heads, model.cfg.n_kv_heads)
            selection_us = (time.perf_counter() - selection_start) * 1e6
            attention = AttentionTimer(device=model.device)
            timer = _CallTimer(model.device)
            timer.start()
            logits = model.decode(["text_demo"], [token], pool, num_splits=splits, timer=attention)[0]
            step_us = timer.stop()
            steps.append({"position": len(generated), "num_splits": splits, "policy_us": selection_us,
                          "step_us": step_us, "attn_us": attention.total_us()})
        elapsed = (time.perf_counter() - started) * 1e6
    finally:
        pool.release("text_demo")
    return {"schema_version": 1, "evidence_kind": "single_request_text_demo", "performance_claim": False,
            "prompt_mode": "raw_text_continuation_no_chat_template", "prompt": prompt,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
            "prompt_tokens": len(prompt_tokens), "generated_tokens": len(generated), "token_ids": generated,
            "token_timestamps_us": stamps, "stopped_on_eos": bool(generated and generated[-1] in eos_token_ids),
            "policy": policy.name, "prefill_us": prefill_us, "ttft_us": stamps[0], "total_wall_us": elapsed,
            "tpot_us": (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else None,
            "steps": steps, "limitations": ["One request, no warmup or repeated comparison",
                                               "Raw continuation without instruction/chat templating"]}


def command(args):
    from kernelscope.serve.cli import _json, _policies, preflight
    from kernelscope.serve.hf import load_tokenizer, snapshot_dir
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.model import DecoderModel
    try:
        if not args.prompt.strip() or args.max_new_tokens < 1:
            raise ValueError("provide nonempty --prompt and positive --max-new-tokens")
        if not math.isfinite(args.kv_gib) or args.kv_gib <= 0:
            raise ValueError("--kv-gib must be finite and positive")
        out = Path(args.out) if args.out else None
        if out is not None and out.exists():
            raise FileExistsError(f"output already exists: {out}")
        status = preflight(args.model)
        if not status["ready"]:
            raise RuntimeError("Generation preflight failed: " + "; ".join(status["errors"]))
        args.policy = [args.policy]
        policy = _policies(args)[0]
        path = snapshot_dir(args.model)
        tokenizer = load_tokenizer(path)
        config = json.loads((path / "config.json").read_text())
        eos = config.get("eos_token_id", [])
        eos = [eos] if isinstance(eos, int) else eos or []
        model = DecoderModel.from_pretrained(args.model)
        pool = PagePool.for_budget(model.cfg, int(args.kv_gib * 2**30), device=model.device, dtype=model.dtype)
        result = generate(model, pool, tokenizer, args.prompt, args.max_new_tokens, policy, () if args.ignore_eos else eos)
        result.update(model=args.model, model_path=str(path.resolve()), preflight=status)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            _json(out, result)
        print(result["generated_text"])
        print(f"\nRaw continuation · {result['generated_tokens']} tokens · TTFT {result['ttft_us'] / 1000:.2f} ms · single-request demo")
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        raise SystemExit(str(error)) from None


def register_parser(subparsers):
    parser = subparsers.add_parser("generate", help="continue local raw text with the trained model (single-request demo)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--policy", default="heuristic")
    parser.add_argument("--kv-gib", type=float, default=1)
    parser.add_argument("--machine")
    parser.add_argument("--params")
    parser.add_argument("--cache-state", choices=("cold", "warm"), default="cold")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--out")
    parser.set_defaults(func=command)
