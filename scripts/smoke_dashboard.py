"""Opt-in real localhost inference checks; requires the dashboard and local models."""
import json
import time
from urllib.request import Request, urlopen

BASE = 'http://127.0.0.1:8766'


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request(BASE+path, data=data, headers={'Content-Type':'application/json'}), timeout=30) as response:
        return json.load(response)


def complete(payload):
    job = request('/api/jobs', payload)
    deadline = time.monotonic()+120
    while time.monotonic() < deadline:
        snapshot = request('/api/jobs/'+job['id'])
        if snapshot['done']:
            errors = [event for event in snapshot['events'] if event['type']=='error']
            assert not errors, errors
            return next(event['data'] for event in snapshot['events'] if event['type']=='result'), snapshot
        time.sleep(.2)
    request('/api/jobs/'+job['id']+'/cancel', {})
    raise TimeoutError('Inference exceeded 120 seconds')


def main():
    print('HEALTH', request('/api/health')['status'], flush=True)
    frozen = request('/api/evidence', {'question':'What is the capital of France?',
        'context':'Paris is the capital of France.', 'search_enabled':False})
    payload = {'question':frozen['question'], 'evidence_snapshot':frozen,
               'search_enabled':False,'guardrail':False,'max_tokens':40,'temperature':0}
    ids = []
    for model in ('qwen2.5-1.5b','phi-3-mini'):
        result, snapshot = complete({**payload,'model':model})
        assert 'Paris' in result['answer'], result
        assert result['snapshot_id'] == frozen['snapshot_id']
        assert any(e['type']=='claim' for e in snapshot['events'])
        ids.append(result['snapshot_id'])
        print('REAL INFERENCE', model, result['answer'], result['verifier'], flush=True)
    job = request('/api/jobs', {**payload,'model':'qwen2.5-1.5b','question':'Describe France in detail.',
                               'max_tokens':512,'evidence_snapshot':None,
                               'context':'France has many regions. Paris is its capital.'})
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        snapshot=request('/api/jobs/'+job['id'])
        if any(e['type']=='progress' for e in snapshot['events']):
            request('/api/jobs/'+job['id']+'/cancel',{})
            break
        time.sleep(.05)
    else:
        raise AssertionError('No inference progress before cancellation')
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        snapshot=request('/api/jobs/'+job['id'])
        if snapshot['done']:
            assert snapshot['cancelled'], snapshot
            assert not any(e['type']=='result' for e in snapshot['events'])
            print('INFERENCE CANCELLED', flush=True)
            break
        time.sleep(.1)
    else:
        raise AssertionError('Cancelled inference did not finish')
    print('FROZEN EVIDENCE MATCH', ids, flush=True)


if __name__=='__main__':
    main()
