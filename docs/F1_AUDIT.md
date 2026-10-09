# F1 audit and aligned re-extraction

## Legacy result

The saved logistic-probe artifacts report validation F1 0.6018 and test F1 0.6552
at threshold 0.5. Test precision is 38/59 = 64.4%; recall is 38/57 = 66.7%.
F1 = 2*38/(2*38 + 21 + 19) = 65.5%. The 115 test records comprise 38 true
positives, 21 false positives, 19 false negatives, and 37 true negatives.
F1 is not the percentage of all answers that were correct.

## Confirmed label-provenance defect

The 2026-09-26 audit of all 749 saved signal files found that 748 generated texts
are different from the original labeled RAGTruth responses. All 749 signal labels
equal the original dataset labels. `SignalExtractor.run` generated fresh Qwen
answers and reused labels for answers produced by other models. Those labels
cannot be assumed to describe the newly generated answers. Consequently the
reported F1 is a score against inherited labels, not a validated hallucination
metric for Qwen generation. This defect can introduce label noise; the audit
does not quantify how much of the score loss it explains.

Source IDs have zero overlap across train/validation/test in the local dataset.
Training has 522 examples for 331 features. Limited data, feature quality, and
threshold choice may also affect performance, but require controlled experiments
after fixing label alignment. Threshold tuning alone cannot repair this defect.

## Completed repair (2026-09-26)

The repair teacher-forced Qwen over every exact annotated response, recorded the
probability of each observed response token and hidden states from layers 0, 7,
14, 21, and 27, fitted PCA only on the 522 training records, and evaluated once
on the untouched 115-record test split. The 112 validation records remained
separate. All 749 records passed response-label alignment checks. Because the
extraction window was capped at 1,024 tokens, 172 prompts were left-truncated.

Aligned held-out results are: entropy AUROC 0.7232 / F1 0.6496; logistic AUROC
0.7979 / F1 0.7167; and MLP AUROC 0.8194 / F1 0.7611. These are teacher-forced,
same-model classification results. They are not calibrated probabilities, live
chat accuracy, causal evidence, cross-model transfer, or steering validation.

The training entry point refuses mismatched response labels. No legacy files
were overwritten. The real NLI verifier's
20 authored development examples passed after fixes; that small diagnostic result
is not a replacement for the probe's held-out F1 or an independent benchmark.

Reproduce: `python scripts/audit_probe_labels.py` with the research runtime.
Full audit: `outputs/results/label_provenance_audit.json` (local generated file).
Aligned summary: `outputs/results/aligned_research_summary.json`.
