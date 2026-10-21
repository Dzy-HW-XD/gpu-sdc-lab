# High-bit fault injection → output corruption

Status: **executed 2026-10-20**. Results and figures are in
`docs/REPORT_HIGHBIT_OUTPUT.md`; this document is the protocol record.

## 1. Objective

Given a fixed prompt suite, inject a **single high-order bit flip** into the
model during prefill, generate an output, and quantify whether the output is

1. **literally garbled** (invalid/replacement characters, control characters,
   empty output, degenerate repetition), or
2. **semantically completely wrong** (the required answer disappears and the
   output is far from the clean reference).

The experiment must produce one publication-style summary figure.

## 2. Scope

- Models: all five planned base models:
  - `EleutherAI/pythia-160m` (12 layers, 768 hidden, MHA)
  - `EleutherAI/pythia-410m` (24 layers, 1024 hidden, MHA)
  - `EleutherAI/pythia-1b` (16 layers, 2048 hidden, MHA)
  - `Qwen/Qwen2.5-0.5B` (24 layers, 896 hidden, GQA 14Q/2KV)
  - `Qwen/Qwen2.5-1.5B` (28 layers, 1536 hidden, GQA 12Q/2KV)
- Staged rollout: run Pythia-160M first and measure wall time; if the per-model
  cost is acceptable, continue with the other four models in the listed order.
- Decode is required because the endpoint is the generated output. This is a
  separate experiment from the prefill-only propagation study; there is no
  KV-cache injection.
- Generation: greedy, deterministic, `max_new_tokens=48`, batch size 1.
- Injection: activation high-bit flip in the residual stream, after prompt
  encoding and before a chosen transformer block. One element is flipped at
  the last prompt token.
- Prompt suite: 24 fixed prompts, balanced across six categories.
- Model extension: all five models use the same protocol, prompt set and
  metrics; layer indices are model-specific E/M/L.

## 3. Prompt suite

24 prompts, four per category.  The executed suite uses short few-shot/cloze
continuations so that base models produce a checkable first answer:

| category | what it tests | example | reference |
|---|---|---|---|
| factual | short factual recall | "The capital of France is Paris. The capital of Japan is" | contains "Tokyo" |
| arithmetic | exact arithmetic | "1 + 1 = 2. 2 + 2 = 4. 2 + 3 =" | contains "5" |
| code | short code completion | "def add(a, b): ... def mul(a, b): return" | contains "a * b" |
| list / format | ordered list or format | "The first three letters are A, B, and" | contains "C" |
| definition | one-sentence explanation | "Hot is the opposite of cold. Up is the opposite of" | contains "down" |
| continuation | open-ended continuation | "The quick brown fox jumps over the lazy" | contains "dog" |

- Closed prompts have an expected substring or regex.
- Open prompts use the clean output as the reference, not as a correctness
  label.
- Prompts whose clean output is already wrong are flagged and excluded from
  the "newly wrong" denominator; they remain in the raw CSV.
- Executed clean-correct prompt counts: Pythia-160M 11/24, Pythia-410M 17/24,
  Pythia-1B 20/24, Qwen2.5-0.5B 23/24, Qwen2.5-1.5B 24/24.

## 4. Injection design

### Default arm: activation

- Layer: input to block `l`, for `l ∈ {0, 6, 11}`.
- Position: last prompt token.
- Element: the largest-magnitude activation element at that position.
- Bit: `{28, 29, 30, 31}` (FP32 sign bit and top exponent bits).
- Mechanism: view the FP32 hidden state as int32, XOR one bit in one element,
  then continue the forward pass. The hook is active during prefill and the
  corrupted state enters the normal decode path.
- Verification: assert that exactly one bit differs from the clean activation;
  record the original and faulted values.
- NaN/Inf: catch and count as a catastrophic outcome, not as a silent skip.

### Optional arm: weight

If requested, flip one bit in one weight element of a selected matrix
(`mlp.down_proj.weight` or `attention.dense.weight`) at layer 6. The same
corrupted model is used for all prompts. This is a model-level fault, whereas
the activation arm is prompt-level.

## 5. Metrics

### Literal garble

- `replacement_char_rate`: count of U+FFFD divided by character length.
- `control_char_rate`: Unicode control characters excluding newline/tab.
- `empty_output`: zero decoded characters.
- `degenerate_repetition`: a repeated 3-gram above a threshold calibrated on
  clean outputs, or very low distinct-2.
- `script_mix` (English prompts): unexpected CJK/other-script ratio above a
  calibrated threshold.
- `literal_garble = replacement/control/empty OR degenerate_repetition`.

### Semantic correctness / preservation

- Closed prompts: expected answer present (exact/contains/regex).
- `char_sim = 1 - Levenshtein(output, clean_output) / max(len)`.
- `token_f1` between output and clean output (secondary).
- `clean_nll`: teacher-forced negative log-likelihood of the clean output
  under the faulted model; report `PPL_ratio = exp(NLL_fault - NLL_clean)`.
  This measures semantic corruption even when the output is fluent.
- `semantic_wrong`:
  - closed: expected answer missing **and** `char_sim < 0.4`, or
    `PPL_ratio > 5`;
  - open: `char_sim < 0.3` **and** `token_f1 < 0.3`, or `PPL_ratio > 5`.
- `catastrophic = literal_garble OR semantic_wrong`.
- `partial`: not garbled, but answer missing or similarity in the middle band.
- `preserved`: answer correct and `char_sim ≥ 0.8`.

### Aggregation

- Per condition: `garbled_rate`, `wrong_rate`, `catastrophic_rate` with Wilson
  95% confidence intervals.
- Per prompt category and per layer/bit.
- Store 3–5 representative clean, partial, wrong, and garbled examples in the
  report for manual inspection.

## 6. Experimental matrix (default)

| factor | values |
|---|---|
| models | 5 (Pythia-160M/410M/1B, Qwen2.5-0.5B/1.5B) |
| prompts | 24 |
| layers | 3 per model: E/M/L = 0 / N/2 / N-1 |
| bits | 28, 29, 30, 31 |
| injection elements | 1 (largest magnitude) |
| clean baseline | 24 generations per model; Pythia-160M repeated twice |
| sham hook | 72 generations per model (3 layers × 24 prompts) |
| fault runs | 5 × 24 × 3 × 4 = 1,440 |
| total generations | 1,944 (including controls) |

Executed runtime on the shared A100: Pythia-160M 164 s, Pythia-410M 292 s,
Pythia-1B 197 s, Qwen2.5-0.5B 341 s, Qwen2.5-1.5B 373 s (about 23 minutes
total including model startup). No OOM.

## 7. Controls and sanity checks

- Greedy generation, fixed seed, deterministic algorithms.
- Clean baseline repeated to confirm identical outputs.
- Sham injection must reproduce the clean output exactly.
- Verify the injected tensor has exactly one changed bit.
- Record tokenizer output, decoded text, token ids, and special-token flags.
- If bit 28 produces no measurable effect in a 3-prompt pilot, escalate to the
  top-4 largest elements and record that change as a protocol deviation.

## 8. Outputs

```
results/llm_highbit_output/
  prompts.json
  runs.csv
  summary_by_condition.csv
  summary_by_category.csv
  examples.md
  figures/highbit_output_summary.pdf
  figures/highbit_output_summary.png
docs/REPORT_HIGHBIT_OUTPUT.md
```

One figure, three panels:

- (a) heatmap of `catastrophic_rate` over model × bit, annotated with
  percentages;
- (b) heatmap of `catastrophic_rate` over model × injection depth (E/M/L);
- (c) outcome composition (garbled / wrong / partial / preserved) for bits
  28–29, which are the most consistently damaging pair.

The executed figure is `results/llm_highbit_output/figures/highbit_output_summary.pdf`
and `docs/figures/highbit_output_summary.pdf`.

## 9. Risks

- A one-element activation flip may be too weak at lower high bits; the pilot
  gate handles this.
- Pythia-160M may fail some clean prompts; the clean-correct filter prevents
  attributing baseline errors to injection.
- Twenty-four prompts give wide confidence intervals; per-category claims are
  descriptive, not inferential.
- A single bit can produce NaN/Inf; these are valid catastrophic outcomes and
  must be reported explicitly.
- Garbled-text heuristics need calibration on clean outputs; examples are
  retained for manual review.

## 10. Confirmed decisions

1. Injection object: **activation** (residual-stream input, last prompt token).
2. Model scope: **all five models**, with a Pythia-160M timing gate before the
   remaining four.
3. High-bit range: **bits 28–31**.
4. Prompt count: **24 prompts**, six categories × four.

Execution begins with the Pythia-160M pilot; the other models follow if the
measured runtime is acceptable.

### Post-hoc baseline-validity correction

After execution, the clean-correct rates showed that Pythia-160M (11/24),
Pythia-410M (17/24), and Pythia-1B (20/24) do not pass a 90% clean-accuracy
gate. They are retained as low-baseline stress tests but are **not** used for
the primary model comparison. The primary result is restricted to
Qwen2.5-0.5B and Qwen2.5-1.5B (23/24 and 24/24); see
`docs/REPORT_HIGHBIT_OUTPUT.md` §0. A formal rerun should calibrate prompts
first, use ≥90% clean accuracy, and increase the prompt count to 60–100.
