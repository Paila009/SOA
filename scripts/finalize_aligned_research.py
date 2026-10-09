"""Validate a complete aligned extraction, train probes, and publish a summary."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT)]
PYTHON=sys.executable
SIGNALS=ROOT/'outputs/signals/qwen2.5-1.5b-aligned'
PROBES=ROOT/'outputs/probes-aligned'


def expected_ids():
    result=[]
    for split in ('train','val','test'):
        path=ROOT/'data/processed'/f'{split}.jsonl'
        result.extend(str(json.loads(line)['id']) for line in path.read_text(encoding='utf-8').splitlines() if line)
    return result


def main():
    ids=expected_ids()
    missing=[item for item in ids if not (SIGNALS/f'{item}.pt').exists()]
    if missing:
        raise RuntimeError(f'Aligned extraction incomplete: {len(missing)} missing, first={missing[:3]}')
    PROBES.mkdir(parents=True,exist_ok=True)
    for kind in ('entropy-only','logistic','mlp'):
        command=[PYTHON,str(ROOT/'scripts/train_probe.py'),'--type',kind,
                 '--signal-dir',str(SIGNALS),'--output-dir',str(PROBES)]
        subprocess.run(command,cwd=ROOT,check=True)
    results={kind:json.loads((PROBES/f'results_{kind}.json').read_text(encoding='utf-8'))
             for kind in ('entropy-only','logistic','mlp')}
    truncation=0
    import torch
    for identifier in ids:
        row=torch.load(SIGNALS/f'{identifier}.pt',map_location='cpu',weights_only=False)
        truncation+=int(row.get('metadata',{}).get('prompt_truncated',False))
    summary={'status':'complete','records':len(ids),'label_alignment':'exact annotated responses',
             'prompt_truncated_records':truncation,'results':results,
             'warning':'Teacher-forced detection experiment; not generation-time causal intervention or cross-model validation.'}
    target=ROOT/'outputs/results/aligned_research_summary.json'
    target.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
