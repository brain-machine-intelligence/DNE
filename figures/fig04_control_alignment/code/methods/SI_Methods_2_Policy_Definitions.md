# Methods (one line for the main text)

The reference parameter vector was fixed at \(\theta^\star = (1.3, -0.9, 0.4)\) for all runs; it was not resampled across seeds.

# Supplementary Methods 2. Stimulus-selection policies

All ten policies share the same Bayesian learner and the same 50-candidate Latin-hypercube pool within a run. They differ only in the selection rule `select()`. The reference parameter vector was fixed at \(\theta^\star = (w_x, w_y, b) = (1.3, -0.9, 0.4)\) for all runs and was not resampled across seeds. Unless noted, a policy does not observe the reference response probabilities \(p^\star\).

**DNE.** Selects the candidate that maximises the squared discrepancy \((p_{\mathrm{learn}}(s) - p^\star(s))^2\) between the learner’s current predictive probability and the reference probability at that stimulus. The controller therefore sees \(p^\star\); it does not optimise parameter \(\ell_2\) error directly.

**Expected information gain (EIG).** Selects the candidate that maximises predictive entropy minus expected Bernoulli entropy under the posterior,
\[
\mathrm{EIG}(s) = H[p(y\mid s)] - \mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}[H[p(y\mid s,\theta)]].
\]
EIG does not observe \(p^\star\).

**BALD.** In this Bernoulli observation model BALD coincides with the EIG score above. Separate BALD runs were stored independently of the EIG runs, so the empirical trajectories need not be identical.

**BOED-H2.** Two-step Bayesian optimal experimental design: maximises the expected reduction in posterior entropy over a two-step horizon, taking the second stimulus as the contingent optimum after the first observation.

**BOED-H3.** Three-step approximate dynamic programme that minimises expected posterior entropy; after the first step, later steps use greedy, branch-wise expected-entropy minimisation.

**Thompson sampling.** Draws one parameter vector from the discrete posterior grid and selects the candidate whose Bernoulli probability under that draw is closest to \(1/2\).

**Uncertainty sampling.** Selects the candidate whose current predictive probability \(p(y=1\mid s)\) is closest to \(1/2\). (This is not the entropy-bin analysis used elsewhere in the Supplement.)

**Random.** Selects uniformly at random among the 50 candidates.

**RNN-info.** An online recurrent baseline that does not observe \(p^\star\). At each trial a GRU cell (hidden size 32) with a one-hidden-layer MLP head is updated for a few AdamW steps to imitate exact EIG scores on the current candidates, then the candidate with the highest predicted score is chosen. Features per candidate are stimulus coordinates, the posterior-mean parameters, posterior entropy, and the predictive probability. The network is reset at the start of each episode; there is no offline pretraining.

**RNN-target.** The same online architecture and training schedule as RNN-info, but each candidate feature vector also includes the reference probability \(p^\star(s)\). The teacher signal is the Bernoulli Jensen–Shannon divergence between \(p_{\mathrm{learn}}(s)\) and \(p^\star(s)\), not the squared discrepancy used by DNE. Thus RNN-target is target-aware but does not clone the DNE objective.
