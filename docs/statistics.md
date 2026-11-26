# Offline statistics reproduction

## Y2/Y3 three-model INDUCE

```bash
python scripts/calculate.py --workflow y2y3 \
  --package /path/to/final_judged_package.zip \
  --inventory /path/to/n21_source_inventory.json \
  --output outputs/y2y3/reproduced.json
```

The inputs are the paired final scored package and its frozen inventory from the team's November 21 delivery. Public download: **To be released**. No private server location or raw answer is distributed in this repository. The package member layout must match the final delivery; a generation-only handoff is insufficient.

This entry is adapted from the actual final `prepare_n21.py`, not a newly guessed paper metric implementation. It verifies the package SHA, each model's judgment SHA, raw answer/row hashes, full unique key coverage, exceptional record sets and the published per-condition CSV numerators/denominators. It outputs the original six model/prompt aggregates, per-condition details and exceptional records; preserve this output as internal result data until release approval.

It reads existing judgments only: no model inference, API calls, retries or scoring campaigns. It requires only Python's standard library. JSON output has been compared against the unchanged original script on the frozen actual delivery and is identical. A wrong package hash is rejected. Output filenames must differ from the input package and inventory.

Workbook rendering is separate. The original Node renderer depends on `@oai/artifact-tool`; that dependency is not assumed to be publicly installable or included. This entry reproduces the statistics payload and checks published condition counts; it does not reproduce all paper figures or claim independent correctness of upstream judgments.

## Remaining paper tables

The owner-provided November 25 reproduction index now identifies the scripts and frozen inputs for MAIN/TEXT, closed models, teacher608, mixed SFT, STAR and paired comparisons. Their portable implementations are still being integrated. Some historical scripts combine publishing with paid scoring; never execute an `--execute` recovery command to recompute a report.

`python scripts/calculate.py --workflow coverage ...` retains the structural generation audit. Coverage is not a semantic score.
