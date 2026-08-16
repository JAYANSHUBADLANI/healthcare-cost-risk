# Progress

All four phases are complete. Every number in the README and the case write-up comes from
one clean run of the whole pipeline (`make clean` then `make all`) against the real CMS
extract, finishing in about six minutes.

## Done

**Phase 1: data, cleaning, audit**

- Verified the CMS download structure against the live site rather than assuming it. Found
  that the page for subsample 1 links to the subsample **20** file for the 2010
  beneficiary summary, and that no 2010 summary for subsample 1 exists at the canonical
  path. Switched to subsample 2, which is complete. Documented in the README and in
  `scripts/download_data.sh`.
- All seven record counts match the counts CMS publishes for subsample 2, exactly:
  116,395 / 114,618 / 112,845 beneficiaries, 66,494 inpatient, 792,562 outpatient,
  4,745,914 carrier, 5,561,154 drug events. The check runs in the pipeline.
- 2008 chronic condition prevalence matches the published figures to within 0.25
  percentage points on all seven conditions the user guide tabulates.
- Cost rebuilt from the four claim files reconciles against the annual totals on the
  summary file at correlation 0.9984 or better in all three years, aggregate within 1.9
  percent.
- Cohort is built from the beneficiary summary, not from claims, so members who used no
  services survive as genuine zeros instead of vanishing.
- Carrier claims stream in chunks. The distinct physician count is built by deduplicating
  member, year and physician triples across every chunk rather than summing per chunk
  counts, which would double count members whose claims span a chunk boundary.

**Phase 2: features and the two part model**

- Logistic participation model plus Gamma severity model with a log link, chosen over log
  OLS so no smearing retransformation is needed. Gradient boosted Poisson challenger
  alongside.
- Strictly prospective: fit on 2008 to 2009, evaluated on 2009 to 2010, which is never
  seen by either part, by the scaler, or by the feature pruning.
- Held out results: participation AUC 0.9681, high cost AUC 0.7644, rank correlation
  0.7167, 23.97 percent of spend reached in the top decile against 19.92 percent for
  ranking on prior cost.

**Phase 3: high cost identification, tiering, business case**

- High cost defined as the top decile of realised 2010 cost, 11,300 members above $7,050,
  holding 49.96 percent of all spending.
- Three tiers at the 80th and 95th percentiles. High tier is 5.0 percent of members and
  13.77 percent of spend, a concentration of 2.75x.
- Business case states its three assumed inputs explicitly and reports the break even
  avoidable fraction of 16.41 percent, which is the number that survives changing them.

**Phase 4: explainability and figures**

- Permutation importance on the metric the score is actually used for, plus a feature
  block ablation, plus the marginal condition profile.
- Eight figures, all drawn from the same CSV tables the written numbers quote.

**Testing**

- 40 tests, no downloaded data required. They cover leakage (the target and anything
  derived from it must stay out of the design matrix), the rank guard, the two part
  identity, curve monotonicity, the oracle and random bounds, tier proportions, business
  case arithmetic, and the break even definition.

## Three things the build found that were worth finding

- **The Gamma fit silently diverged** on the first run, producing coefficients in the
  billions with identical p values. Cause was exact collinearity: the chronic condition
  count is the arithmetic sum of the eleven condition flags, so including both made the
  design matrix singular. Fixed by dropping the count from the design matrix, and the
  pipeline now raises rather than continuing if the matrix is ever rank deficient again.
- **Two of the planned features are duplicates in this data.** The distinct drug product
  count equals the fill count for 99.6 percent of members and the distinct physician count
  tracks carrier claim volume at 0.9989, because provider and product identifiers were
  randomised during synthesis. Both are built, both are reported, and the second of each
  pair is pruned before fitting with the decision recorded in a table.
- **The chronic conditions carry no incremental signal.** Each looks strongly predictive
  alone, between 1.65x and 2.57x on realised cost. Refitting on prior utilisation only
  gives AUC 0.7646 against 0.7644 for the full model. This is reported as a null rather
  than buried, with the argument for why it probably does not generalise to real claims.

## Pending, and things worth deciding

**Nothing blocking. The project runs end to end and is demoable as it stands.**

Things I would add next, in the order I would do them:

- **Repeat across subsamples.** There is one honest prospective split in three years of
  data, so the 4 percentage point advantage over ranking on prior cost has no interval
  around it. The other nineteen tranches are identically structured and would give a
  distribution for that gap. This is the weakest claim in the project as it stands.
- **A rolling twelve month feature window.** Scoring on a calendar year lag means the most
  recent information about a member can be twelve months stale.
- **Calibration to a target year level.** The model over predicts 2010 by 1.82x because
  mean spend falls 40.7 percent. Ranking is unaffected and every targeting number is rank
  based, but any absolute dollar output needs a trend factor the data cannot supply
  prospectively.

**Two things I still need to settle:**

- The three economic assumptions in `config/config.yaml` ($500 outreach cost per member,
  10 percent avoidable share, 35 percent engagement) are placeholders. They change every
  dollar figure in the business case and none of the model results. If I find figures
  I would rather defend, I change them there and rerun phase 3.
- The high cost threshold is the top decile and the tier cutoffs are the 80th and 95th
  percentiles. Both are conventional and both are in the config. A plan with a fixed
  outreach capacity would set the high tier to that capacity instead.

Not pushed yet. Local files only for now. The raw CMS files are excluded by
`.gitignore` and rebuild with `make data`.
