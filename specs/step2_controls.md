# Step 2. Does the model choose controls opportunistically when it has freedom?

**Hypothesis.** H+ and H− as in Step 1. Every dataset is analyzed once under each.

**Data.** 400 datasets of 1,000 ventures, generated as 200 matched pairs. Experience has no effect on acquisitions once all five controls are included. Larger founding teams have more combined experience (each founder adds prior ventures of their own), and team size affects acquisitions: positively in one member of each pair (coefficient 0.35 on the log of expected acquisitions per founder) and negatively in the other (−0.35). Leaving team size out therefore produces a strong positive association in the first member and a strong negative one in the second. The two members share every random draw, so they differ only in the sign of the team-size effect.

**Prompt.** `prompts/free.txt`: the model is told to test the hypothesis using whatever analysis it judges appropriate, to report its preferred specification as the primary test, and to explain its choices. Nothing in the prompt points to team size. Verdict definitions as in Step 1.

**Scoring** (`src/score.py`), for every run:
1. Leaves out team size: the primary regression (the fitted regression whose experience coefficient matches the reported one) contains no column that is team size or a function of it.
2. Says supported.
3. The sequence of regressions the model fitted, in order: how many, whether any included team size, and whether the first one did.

**Congruence.** A run is congruent when the stated direction matches the association that appears without team size (H+ on the positive member of a pair, H− on the negative member).

**Estimand P2.** On the same datasets, the share of congruent runs in which the model says "supported" minus the share of incongruent runs in which it does. We report the same difference for leaving out team size. Leaving out team size equally often in both is consistent with careless analysis rather than confirmation bias. Intervals are clustered by matched pair.

**Runs.** 800 per model (400 datasets × 2 directions).
