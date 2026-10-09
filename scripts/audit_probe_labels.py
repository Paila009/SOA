"""Audit label provenance without altering existing signals or reported results."""
import collections
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = {}
    source_ids = {}
    for split in ('train','val','test'):
        path = ROOT / 'data' / 'processed' / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        records.update({str(row['id']):row for row in rows})
        source_ids[split] = {str(row.get('source_id',row['id'])) for row in rows}
    counts = collections.Counter()
    for path in (ROOT / 'outputs/signals/qwen2.5-1.5b').glob('*.pt'):
        signal = torch.load(path, map_location='cpu', weights_only=False)
        row = records.get(str(signal.get('metadata',{}).get('example_id',path.stem)))
        if not row:
            counts['missing_record'] += 1
            continue
        counts['records_checked'] += 1
        generated = ' '.join(str(signal.get('generated_text','')).split())
        labeled = ' '.join(str(row.get('response','')).split())
        counts['different_response' if generated != labeled else 'matching_response'] += 1
        counts['label_matches_dataset'] += int(signal['label'] == int(row.get('is_hallucinated',row.get('label',0))))
    confusion = {}
    rows = json.loads((ROOT/'outputs/probes/predictions_logistic.json').read_text(encoding='utf-8'))
    for split in ('val','test'):
        subset = [row for row in rows if row['split']==split]
        c = collections.Counter(('tp' if row['label'] else 'fp') if row['probability']>=.5 else
                                ('fn' if row['label'] else 'tn') for row in subset)
        confusion[split] = dict(c)
    overlaps = {f'{a}/{b}':len(source_ids[a]&source_ids[b])
                for a,b in [('train','val'),('train','test'),('val','test')]}
    report = {'counts':dict(counts), 'source_id_overlap':overlaps, 'confusion_at_0.5':confusion,
              'interpretation':'Labels for dataset responses cannot be transferred automatically to newly generated answers. Re-extract on labeled responses or re-label generated answers before interpreting F1 as hallucination accuracy.'}
    target=ROOT/'outputs/results/label_provenance_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
