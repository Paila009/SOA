"""Install the approved NLI weights locally; no inference-time downloads."""
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--research', action='store_true')
    args = parser.parse_args()
    if args.research:
        target = Path(__file__).resolve().parents[1] / 'models-local' / 'qwen-research'
        snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=target,
                          allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'])
        print(target)
        raise SystemExit(0)
    target = Path(__file__).resolve().parents[1] / 'models-local' / 'nli-minilm'
    snapshot_download('cross-encoder/nli-MiniLM2-L6-H768', local_dir=target,
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer*',
                                      'special_tokens_map.json', 'vocab.json', 'merges.txt'])
    print(target)
