"""Teacher-force Qwen over the exact annotated answers so labels and signals align."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.runtime'), str(ROOT)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/signals/qwen2.5-1.5b-aligned')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=min(12, os.cpu_count() or 2))
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    return parser.parse_args()


def rows():
    for split in ('train', 'val', 'test'):
        path = ROOT/'data'/'processed'/f'{split}.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                yield split, json.loads(line)


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from extraction.hooks import compute_logit_entropy

    model_dir = ROOT/'models-local/qwen-research'
    if not (model_dir/'model.safetensors').exists():
        raise FileNotFoundError('Run: python scripts/setup_verifier.py --research')
    output = ROOT/args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').eval()
    layer_ids = (0, 7, 14, 21, 27)
    captured = {}
    hooks = []
    for layer_id in layer_ids:
        def capture(_module, _inputs, result, index=layer_id):
            value = result[0] if isinstance(result, tuple) else result
            captured[index] = value.detach()
        hooks.append(model.model.layers[layer_id].register_forward_hook(capture))
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('shard-index must be in [0, shard-count)')
    all_rows = list(rows())
    dataset = [row for index, row in enumerate(all_rows)
               if index % args.shard_count == args.shard_index]
    if args.limit:
        dataset = dataset[:args.limit]
    complete = skipped = 0
    timings = []
    try:
        for offset, (split, record) in enumerate(dataset, start=1):
            target = output/f"{record['id']}.pt"
            if args.resume and target.exists():
                skipped += 1
                continue
            prompt = str(record.get('prompt') or record.get('question') or '')
            answer = str(record.get('response') or '')
            response_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not response_ids:
                raise ValueError(f"Empty annotated response: {record['id']}")
            full_prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_ids = full_prompt_ids
            available = args.max_length - len(response_ids)
            if available < 16:
                response_ids = response_ids[:args.max_length-16]
                available = 16
            prompt_ids = prompt_ids[-available:]
            input_ids = torch.tensor([prompt_ids + response_ids])
            captured.clear()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model(input_ids=input_ids, use_cache=False)
            # Position t predicts token t+1. Select predictions for every answer token.
            start = len(prompt_ids)-1
            logits = result.logits[0, start:start+len(response_ids)].float()
            probabilities = logits.softmax(-1)
            observed = torch.tensor(response_ids).unsqueeze(1)
            answer_probabilities = probabilities.gather(1, observed).squeeze(1).cpu()
            entropy = compute_logit_entropy(logits).cpu()
            hidden = {index:value[0, len(prompt_ids):len(prompt_ids)+len(response_ids)].float().cpu()
                      for index, value in captured.items()}
            elapsed = time.perf_counter()-started
            payload = {
                'features': {}, 'logit_entropy': entropy,
                'token_probs': answer_probabilities,
                'hidden_states': hidden, 'attention_entropy': {},
                'label': int(record.get('is_hallucinated', record.get('label', 0))),
                'generated_text': answer,
                'metadata': {
                    'example_id': str(record['id']), 'source_id': str(record.get('source_id','')),
                    'split': split, 'model': 'qwen2.5-1.5b',
                    'extraction': 'teacher-forced-exact-response-v1',
                    'response_tokens': len(response_ids), 'prompt_tokens_used': len(prompt_ids),
                    'prompt_tokens_original': len(full_prompt_ids),
                    'prompt_truncated': len(full_prompt_ids) > len(prompt_ids),
                    'label_target': 'exact annotated response',
                },
                'extraction_time_s': elapsed,
            }
            torch.save(payload, target)
            complete += 1
            timings.append(elapsed)
            if complete == 1 or complete % 10 == 0:
                mean = sum(timings)/len(timings)
                print(f"{complete+skipped}/{len(dataset)} aligned records · {mean:.2f}s/record", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    manifest = {
        'status': 'complete' if complete+skipped == len(dataset) else 'partial',
        'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'method': 'teacher-forced-exact-response-v1',
        'records': complete+skipped, 'new_records': complete, 'skipped': skipped,
        'layers': list(layer_ids), 'max_length': args.max_length, 'threads': args.threads,
        'label_alignment': 'generated_text equals the annotated response by construction',
    }
    manifest['shard'] = {'index': args.shard_index, 'count': args.shard_count}
    name = 'manifest.json' if args.shard_count == 1 else f'manifest-shard-{args.shard_index}.json'
    (output/name).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
