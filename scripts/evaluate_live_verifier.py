"""Small authored diagnostic set; not an independent research benchmark."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'dashboard'),str(ROOT/'.runtime')]
from claim_verifier import verify_claim
from nli_runtime import NLI


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lexical',action='store_true')
    parser.add_argument('--cases', type=Path, help='JSON array of independently labeled source/claim/label cases')
    args=parser.parse_args()
    cases=json.loads((args.cases or ROOT/'tests/fixtures/claim_cases.json').read_text(encoding='utf-8'))
    if not cases or any(row.get('label') not in {'supported','partial','contradicted','unverified'} for row in cases):
        raise ValueError('Provide nonempty cases with supported/partial/contradicted/unverified labels')
    details=[]
    for case in cases:
        result=verify_claim(case['claim'],[{'snippet':case['source']}] if case['source'] else [],use_nli=not args.lexical)
        details.append({**case,'predicted':result['status'],'score':result['score'],
                        'reason':result['reason'],'correct':result['status']==case['label']})
    supported=[row for row in details if row['label']=='supported']
    unsupported=[row for row in details if row['label']!='supported']
    tp=sum(row['predicted']!='supported' for row in unsupported)
    fp=sum(row['predicted']!='supported' for row in supported)
    fn=sum(row['predicted']=='supported' for row in unsupported)
    report={'scope':str(args.cases) if args.cases else '20 authored diagnostic cases, not independent human validation',
            'verifier':NLI.status(),'method':'lexical' if args.lexical else 'nli',
            'samples':len(details),'status_accuracy':sum(row['correct'] for row in details)/len(details),
            'unsupported_detection':{'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),
              'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)},'cases':details}
    path=ROOT/'outputs/results'/('live_verifier_lexical.json' if args.lexical else 'live_verifier_nli.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='cases'},indent=2))
    print('Failures:',[(row.get('category','custom'),row['label'],row['predicted']) for row in details if not row['correct']])


if __name__=='__main__':
    main()
