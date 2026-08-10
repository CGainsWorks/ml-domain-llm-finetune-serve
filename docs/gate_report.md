# Promotion gate

**Decision: PROMOTE**

## Checks

| | Check | Metric | Threshold | Baseline | Observed |
|:--:|---|---|---:|---:|---:|
| ✅ | min_absolute | `general_guard.non_empty_rate` | +0.9800 | — | 1.0000 |
| ✅ | min_absolute | `sql_exec_holdout.schema_adherence` | +0.9500 | — | 0.9960 |
| ✅ | min_absolute | `sql_exec_holdout.valid_sql_rate` | +0.9800 | — | 0.9960 |
| ✅ | min_delta | `sql_exec_holdout.exec_acc_nonempty` | +0.0500 | 0.7865 | 0.8932 |
| ✅ | max_regression | `general_guard.answers_in_prose` | +0.1000 | 0.9950 | 0.9950 |
| ✅ | max_regression | `sql_exec_holdout.valid_sql_rate` | +0.0100 | 0.9980 | 0.9960 |

## Scorecard: `baseline` vs `finetuned`

- Baseline run: `2026-08-05T15:34:08+00:00` (commit `3a66dc9`)
- Candidate run: `2026-08-10T11:55:55+00:00` (commit `b9a348d`)
- Prompt versions: `{'sql': 'sql_v1', 'guard': 'guard_v1'}`

| Task | Metric | Base | Fine-tuned | Δ | |
|---|---|---:|---:|---:|:--:|
| `general_guard` | `answers_in_prose` | 0.9950 | 0.9950 | +0.0000 | = |
| `general_guard` | `keyword_overlap` | 0.2997 | 0.2946 | -0.0051 | ▼ |
| `general_guard` | `mean_chars` | 232.9400 | 214.0000 | -18.9400 | ▼ |
| `general_guard` | `non_empty_rate` | 1.0000 | 1.0000 | +0.0000 | = |
| `sql_exec_holdout` | `exact_match_norm` | 0.5960 | 0.7960 | +0.2000 | ▲ |
| `sql_exec_holdout` | `exec_acc` | 0.7860 | 0.8940 | +0.1080 | ▲ |
| `sql_exec_holdout` | `exec_acc_nonempty` | 0.7865 | 0.8932 | +0.1068 | ▲ |
| `sql_exec_holdout` | `exec_error_rate` | 0.0120 | 0.0060 | -0.0060 | ▲ |
| `sql_exec_holdout` | `schema_adherence` | 0.9880 | 0.9960 | +0.0080 | ▲ |
| `sql_exec_holdout` | `unparseable_rate` | 0.0020 | 0.0040 | +0.0020 | ▼ |
| `sql_exec_holdout` | `valid_sql_rate` | 0.9980 | 0.9960 | -0.0020 | ▼ |

### Item counts

| Task | n | scorable | gold unexecutable |
|---|---:|---:|---:|
| `general_guard` | 200 | — | — |
| `sql_exec_holdout` | 500 | 500 | 0 |
