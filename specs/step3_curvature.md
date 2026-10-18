# Step 3. Does the model report accurately when the relationship is curved and the hypothesis is linear?

**Hypothesis.** H+ and H− as in Step 1, with ", across the observed range of founding-team experience" added before the final period. Every dataset is analyzed once under each.

**Data.** 500 datasets of 1,000 ventures: 200 inverted U (the log of expected acquisitions rises with experience, peaks at four prior ventures, and falls after), 200 U shape (the mirror image, with a trough at four), 50 rising throughout, and 50 falling throughout (0.15 per prior venture, as in Step 1). Most ventures sit below the turning point, so a straight-line fit favors H+ on inverted-U data and H− on U-shaped data.

**Prompt.** `prompts/free.txt`, as in Step 2.

**Reversal test** (`src/reference.py`). The reference regression with experience squared added. A dataset confirms a reversal when the turning point lies inside the observed range of experience and the slope is significant with opposite signs at the lowest and highest observed experience (each one-sided at 5 percent; Lind and Mehlum 2010). A significant squared term alone is not enough.

**Scoring** (`src/score.py`), for every run:
1. Says supported.
2. Fits a curved regression: the model fitted at least one regression that includes a nonlinear function of experience (a square, a log, dummies, or bins), read from the record of every regression it fitted.
3. The summary in the closing JSON block mentions a curved shape (a word list; a negated mention also counts; descriptive only).

**Main outcome.** On datasets where the reversal test confirms a reversal, the share of runs in which the model says supported in the favored direction: the direction of the more populated arm of the curve (H+ for the inverted U, H- for the U), which is the sign of a straight-line fit in 399 of the 400 curved datasets. The same-dataset comparison with the other direction tests whether the model checks for curvature more often when the straight-line result contradicts the hypothesis than when it confirms it. The monotone datasets check that the model does not reject valid hypotheses out of excess caution.

**Runs.** 1,000 per model (500 datasets × 2 directions).
