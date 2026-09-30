# mlp (mini-training) workload

Deterministic 2-layer MLP forward + one SGD backward step (a minimal "training
step"), used to study **error propagation across an operator chain**.

```
X -> Z1=XW1+b1 -> A1=ReLU(Z1) -> Z2=A1W2+b2 -> P=softmax(Z2) -> loss
backward: dZ2 -> dW2 -> dZ1 -> dW1
```

All reductions are fixed-order, so the golden is bit-exact. The workload emits
**six named tensors** for the multi-output oracle:

| name | shape | meaning |
|---|---|---|
| `A1` | B×H | hidden activation |
| `logits` | B×O | pre-softmax |
| `probs` | B×O | softmax output |
| `loss` | B | per-sample cross-entropy |
| `dW1` | D×H | gradient wrt W1 |
| `dW2` | H×O | gradient wrt W2 |

A fault at a given site may be visible in some outputs and masked in others;
`observations.jsonl` records per-output metrics so propagation can be traced
(e.g. present in `probs` but absent in `dW1`).

Inputs `X`,`W1`,`W2` can be supplied via `--input/--weight1/--weight2`
(input_spec). Labels are deterministic (`y_b = (7b+3) mod O`).
