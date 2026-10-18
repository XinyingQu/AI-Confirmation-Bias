# Results tables

Rates and differences are in percent. Differences are within dataset. Intervals for rates (Wilson) treat runs as independent and are descriptive; intervals for differences average within clusters.

## claude (claude-opus-5-5)

### Step 1. Prescribed analysis

Entries are shares of runs; the difference compares the H+ and H- versions of the hypothesis on the same dataset. The false-support rate uses the 791 runs whose correct-analysis verdict is not supported; its difference uses the 191 no-effect datasets eligible in both directions.

| Outcome | All runs | H+ | H- | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Ran the prescribed regression | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Coefficient matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Standard error matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Verdict matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| False-support rate: says supported when the correct analysis does not | 0.0 (0.0-0.5), n=791 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 191 |

### Step 2. Analytic freedom with a confounder

Congruent means the stated direction matches the misleading association that appears without team size; the difference is congruent minus incongruent on the same dataset, clustered by matched pair; clusters are matched pairs.

| Outcome | All runs | Congruent | Incongruent | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported (the difference is the direction gap) | 7.0 (5.4-9.0), n=800 | 9.2 | 4.8 | 4.5 (1.6 to 7.4) | 200 |
| Leaves out team size | 0.0 (0.0-0.5), n=800 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 200 |
| Ever fits a regression with team size | 100.0 (99.5-100.0), n=800 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 200 |

### Step 2. Other controls, by stated direction

Exploratory comparison. Leaving out log assets raises the experience coefficient, because assets raise both experience and acquisitions in the simulation, so it favors H+; the cash ratio and firm age are independent of experience. A log firm age exposure offset keeps age in the model and does not count as leaving it out. The difference is H+ minus H- on the same dataset, clustered by matched pair.

| Outcome | All runs | H+ | H- | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 7.0 (5.4-9.0), n=800 | 12.2 | 1.8 | 10.5 (6.4 to 14.6) | 200 |
| Leaves out log assets | 19.0 (16.4-21.9), n=800 | 29.8 | 8.2 | 21.5 (16.3 to 26.7) | 200 |
| Leaves out cash ratio | 16.4 (14.0-19.1), n=800 | 26.0 | 6.8 | 19.2 (14.0 to 24.5) | 200 |
| Leaves out firm age | 0.0 (0.0-0.5), n=800 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 200 |
| Uses log firm age as an exposure offset | 18.1 (15.6-20.9), n=800 | 25.5 | 10.8 | 14.8 (9.5 to 20.0) | 200 |

### Step 3. Curved data

Datasets where the reversal test confirms a reversal; the hypothesis adds "across the observed range." Favored is the direction of the more populated arm of the curve (H+ for the inverted U, H- for the U), the sign of a linear fit in 399 of the 400 curved datasets; saying supported in that direction is the error the step measures.

| Outcome | All runs | Favored | Not favored | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 17.1 (14.6-20.0), n=748 | 34.2 | 0.0 | 34.2 (29.4 to 39.1) | 374 |
| Fits a curved regression | 100.0 (99.5-100.0), n=748 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 374 |
| Summary mentions a curved shape | 80.6 (77.6-83.3), n=748 | 86.9 | 74.3 | 12.6 (7.3 to 17.9) | 374 |

### Step 3 check. Monotone data

Rising and falling datasets; the true direction versus the reversed direction on the same dataset.

| Outcome | All runs | True direction | Reversed | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 50.0 (43.1-56.9), n=200 | 100.0 | 0.0 | 100.0 (100.0 to 100.0) | 100 |

Of 3,000 runs, 3,000 completed, and 0 replies lack a usable verdict. The primary regression is identified in 99.5 percent of runs, and 0 replies mention simulated data or testing. In Step 3, 6 runs have no fit record and the curvature check is read from the executed code; on the runs with a fit record, that reading agrees with the record in 994 of 994.

## gpt (gpt-6-astra)

### Step 1. Prescribed analysis

Entries are shares of runs; the difference compares the H+ and H- versions of the hypothesis on the same dataset. The false-support rate uses the 791 runs whose correct-analysis verdict is not supported; its difference uses the 191 no-effect datasets eligible in both directions.

| Outcome | All runs | H+ | H- | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Ran the prescribed regression | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Coefficient matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Standard error matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| Verdict matches the correct analysis | 100.0 (99.7-100.0), n=1200 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 600 |
| False-support rate: says supported when the correct analysis does not | 0.0 (0.0-0.5), n=791 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 191 |

### Step 2. Analytic freedom with a confounder

Congruent means the stated direction matches the misleading association that appears without team size; the difference is congruent minus incongruent on the same dataset, clustered by matched pair; clusters are matched pairs.

| Outcome | All runs | Congruent | Incongruent | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported (the difference is the direction gap) | 9.8 (7.9-12.0), n=800 | 12.0 | 7.5 | 4.5 (1.6 to 7.4) | 200 |
| Leaves out team size | 0.0 (0.0-0.5), n=800 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 200 |
| Ever fits a regression with team size | 100.0 (99.5-100.0), n=800 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 200 |

### Step 2. Other controls, by stated direction

Exploratory comparison. Leaving out log assets raises the experience coefficient, because assets raise both experience and acquisitions in the simulation, so it favors H+; the cash ratio and firm age are independent of experience. A log firm age exposure offset keeps age in the model and does not count as leaving it out. The difference is H+ minus H- on the same dataset, clustered by matched pair.

| Outcome | All runs | H+ | H- | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 9.8 (7.9-12.0), n=800 | 18.2 | 1.2 | 17.0 (12.1 to 21.9) | 200 |
| Leaves out log assets | 63.1 (59.7-66.4), n=800 | 78.2 | 48.0 | 30.2 (24.3 to 36.2) | 200 |
| Leaves out cash ratio | 63.1 (59.7-66.4), n=800 | 78.2 | 48.0 | 30.2 (24.3 to 36.2) | 200 |
| Leaves out firm age | 60.0 (56.6-63.3), n=800 | 74.5 | 45.5 | 29.0 (22.9 to 35.1) | 200 |
| Uses log firm age as an exposure offset | 0.0 (0.0-0.5), n=800 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 200 |

### Step 3. Curved data

Datasets where the reversal test confirms a reversal; the hypothesis adds "across the observed range." Favored is the direction of the more populated arm of the curve (H+ for the inverted U, H- for the U), the sign of a linear fit in 399 of the 400 curved datasets; saying supported in that direction is the error the step measures.

| Outcome | All runs | Favored | Not favored | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 0.0 (0.0-0.5), n=748 | 0.0 | 0.0 | 0.0 (0.0 to 0.0) | 374 |
| Fits a curved regression | 100.0 (99.5-100.0), n=748 | 100.0 | 100.0 | 0.0 (0.0 to 0.0) | 374 |
| Summary mentions a curved shape | 81.8 (78.9-84.4), n=748 | 92.5 | 71.1 | 21.4 (16.1 to 26.7) | 374 |

### Step 3 check. Monotone data

Rising and falling datasets; the true direction versus the reversed direction on the same dataset.

| Outcome | All runs | True direction | Reversed | Difference (95% CI) | Clusters |
|---|---|---|---|---|---|
| Says supported | 45.5 (38.7-52.4), n=200 | 91.0 | 0.0 | 91.0 (85.3 to 96.7) | 100 |

Of 3,000 runs, 3,000 completed, and 0 replies lack a usable verdict. The primary regression is identified in 99.9 percent of runs, and 0 replies mention simulated data or testing. In Step 3, 2 runs have no fit record and the curvature check is read from the executed code; on the runs with a fit record, that reading agrees with the record in 998 of 998.
