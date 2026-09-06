# Benchmark — env=all

| arm | success | exec/GT | LLM calls/task | latency (s) | UNSAT | repairs | clauses |
|---|---|---|---|---|---|---|---|
| FOLBP (full, ours) | 96/100 | 1.03 | 1.5 | 20.4 | 1.12 | 0.7/1.1 | 0.3 |
| - verification | 85/100 | 1.03 | 2.9 | 30.7 | 0.00 | — | 0.0 |
| repair: LLM re-prompt | 93/99 | 1.03 | 4.0 | 59.5 | 1.52 | 0.7/1.5 | 0.4 |
| - CDCL | 94/100 | 1.03 | 2.4 | 25.8 | 1.04 | 0.7/1.0 | 0.0 |
| constraints: hard | — not run — | | | | | | |
| naive one-shot | 65/100 | 1.02 | 1.0 | 11.1 | [0.53] | — | 0.0 |
| PEFA (baseline) | 95/100 | 1.03 | 54.5 | 201.7 | — | — | — |

`[n]` = shadow-mode UNSAT: detected by Z3, deliberately not acted on.
