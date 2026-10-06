Threshold algorithms for prophet gain-from-trade, from https://arxiv.org/abs/2609.30742.

`cargo test`

This checks the paper's threshold rules, the closed-form ratios R(n), the balanced bound, and Lemma 4.3 by direct evaluation, plus a seeded Monte Carlo on Uniform[0,1]. It does not replay the appendix proofs, and the general-case prophet is not a multi-unit DP; the ratio check uses the single-seller optimum.
