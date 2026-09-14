# Case studies (`positive_retry`)

Illustrative orchestrator logs from the miniF2F-rocq **test** evaluation with `PositiveRetryLemmaSelectionStrategy` and a **$0.04** per-problem budget. Full per-problem logs are not all kept in the repository; these two files sit next to the campaign `report.txt` and `proved_vs_cost.png`.

## `mathd_algebra_44.log`

**Role.** Typical easy success — the shape of most proved theorems in this campaign.

| | |
|--|--|
| Goal | Linear system over `R`: from `s = 9 - 2t` and `t = 3s + 1`, conclude `s = 1 ∧ t = 4` |
| Outcome | Proved in **1** attempt |
| Cost | ~**$0.0006** |
| Shape | Direct About repair → `intros` / `split` / `lra`; **no** helper lemmas |

## `mathd_numbertheory_765.log`

**Role.** Harder success that needs a **coherent multi-lemma** plan, not a one-shot tactic.

| | |
|--|--|
| Goal | For `x : Z`, if `x < 0` and `24 * x mod 1199 = 15`, then `x ≤ -449` |
| Outcome | Proved after **27** attempts |
| Cost | ~**$0.030** |
| Shape | Failed `lia`-style directs, several decompositions, then helpers such as `mod_repr`, `negative_quotient_bound`, and `target_from_quotient` assembled into one script |

Together, the pair shows the campaign’s range: many cheap shallow wins, and occasional deeper searches where decomposition and retry scheduling matter.
