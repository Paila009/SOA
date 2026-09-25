"""Quick script to inspect extracted signal files."""
import torch
import os
import sys

signal_dir = "outputs/signals/qwen2.5-1.5b"
files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.pt')])
print(f"Signal files found: {len(files)}")

for fname in files:
    path = os.path.join(signal_dir, fname)
    d = torch.load(path, map_location='cpu', weights_only=False)
    
    print(f"\n--- {fname} ---")
    print(f"  Keys: {list(d.keys())}")
    print(f"  Label: {d.get('label')}")
    
    ent = d.get('logit_entropy')
    if ent is not None:
        print(f"  Logit entropy: shape={ent.shape}, mean={ent.mean().item():.4f}, max={ent.max().item():.4f}")
    
    tp = d.get('token_probs')
    if tp is not None:
        print(f"  Token probs: shape={tp.shape}, mean={tp.mean().item():.4f}, min={tp.min().item():.4f}")
    
    hs = d.get('hidden_states', {})
    print(f"  Hidden state layers: {list(hs.keys())}")
    for layer_idx, h in hs.items():
        print(f"    Layer {layer_idx}: shape={h.shape}")
    
    ae = d.get('attention_entropy', {})
    print(f"  Attention entropy layers: {list(ae.keys())}")
    
    gen = d.get('generated_text', '')
    print(f"  Generated text ({len(gen)} chars): {gen[:120]}...")
    print(f"  Extraction time: {d.get('extraction_time_s', 0):.1f}s")
