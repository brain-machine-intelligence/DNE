"""
Bayesian causal inference for one-shot learning (Lee et al., PLOS Biology 2015).
Latent class model with Dirichlet prior: causal strength E(θ|D), causal uncertainty Var(θ|D),
learning rate controlled by softmax over uncertainties; primacy/recency and additive/sole effect.
"""
import numpy as np

# Outcome encoding (from task design)
OUTCOME_NONNOVEL = 3
OUTCOME_NOVEL = 4
N_STIMULI = 3


def _dirichlet_causal_strength(alpha):
    """E(θ_i|D) = α_i / α_0 (posterior mean)."""
    a0 = alpha.sum()
    if a0 <= 0:
        return np.ones(N_STIMULI, dtype=np.float64) / N_STIMULI
    return alpha / a0


def _dirichlet_causal_uncertainty(alpha):
    """Var(θ_i|D) = α_i(α_0 - α_i) / (α_0^2 (α_0+1))."""
    a0 = alpha.sum()
    if a0 <= 0:
        return np.ones(N_STIMULI, dtype=np.float64) / N_STIMULI
    var = np.zeros(N_STIMULI, dtype=np.float64)
    for i in range(N_STIMULI):
        var[i] = alpha[i] * (a0 - alpha[i]) / (a0 * a0 * (a0 + 1.0))
    return var


def _learning_rates_from_uncertainty(var, inv_temp):
    """γ_i = softmax(τ * Var(θ_i))."""
    if inv_temp <= 0:
        return np.ones(N_STIMULI, dtype=np.float64) / N_STIMULI
    logits = inv_temp * var
    logits = np.clip(logits, -1e2, 1e2)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def _position_weights(n_positions, primacy, recency):
    """
    Positional weights for primacy and recency (Lee et al. 2015, Eq. 10 idea).

    Base weight is 1.0 for all positions; the first position gets extra
    credit (1 + primacy), and the last position gets (1 + recency).
    Middle positions remain at 1.0. We do NOT normalize the weights so that
    increased primacy/recency directly increases total credit.
    """
    w = np.ones(n_positions, dtype=np.float64)
    if n_positions == 0:
        return w
    if n_positions == 1:
        w[0] = 1.0 + primacy + recency
        return w
    w[0] = 1.0 + primacy
    w[-1] = 1.0 + recency
    return w


class os_sangwan(object):
    """
    Bayesian causal learning model (Lee et al., 2015).
    Dirichlet posterior over cause; learning rate = softmax(causal uncertainty);
    primacy/recency and additive (count) or sole (presence) salience.
    """

    def __init__(self, lr=0.1, primacy=0.36, recency=0.37, inv_temp=10.0, additive=False):
        """
        lr: baseline learning rate (scaling for updates; paper also fits this).
        primacy: weight for first item in trial (δ_p).
        recency: weight for last item in trial (δ_r).
        inv_temp: inverse temperature τ for softmax over uncertainties.
        additive: True = x_i = count of stimulus i; False = presence (0/1).
        """
        self.lr = float(lr)
        self.primacy = float(primacy)
        self.recency = float(recency)
        self.inv_temp = float(inv_temp)
        self.additive = bool(additive)

    def perform(self, exps):
        """
        exps: (5, 6) after argmax, or one-hot (5, 6, 5).
            [:, :5] = stimulus indices 0,1,2  (three cue types)
            [:, 5]  = outcome 3 (non-novel) or 4 (novel)
        Returns: (6, 3) array — causal strength E(θ) before trial 0 and after trials 0..4
        (Dirichlet posterior mean, following Lee et al. 2015 BayesU dynamics).
        """
        exps = np.asarray(exps)
        if exps.ndim == 3:
            exps = exps.argmax(-1)
        if exps.ndim == 1:
            exps = exps.reshape(-1, 6)
        if exps.ndim != 2 or exps.shape[1] < 6:
            exps = exps.argmax(-1)
            if exps.ndim == 1:
                exps = exps.reshape(-1, 6)

        n_trials = exps.shape[0]
        # Dirichlet parameters; uniform prior
        alpha = np.ones(N_STIMULI, dtype=np.float64)
        # Causal strength after each trial (including initial)
        tot_oslrs = [_dirichlet_causal_strength(alpha).astype(np.float32)]

        for t in range(n_trials):
            stims = exps[t, :5].flatten()
            outcome = exps[t, 5]

            # Only indices 0,1,2 are stimuli; ignore invalid
            stims = np.clip(stims.astype(np.int32), 0, N_STIMULI - 1)

            # Salience x_i with primacy/recency (Lee et al. 2015, Eq. 10):
            #   - additive: sum of positional weights over all occurrences
            #   - sole: presence/absence (0/1) within the block
            pos_w = _position_weights(len(stims), self.primacy, self.recency)
            x = np.zeros(N_STIMULI, dtype=np.float64)
            for idx, s in enumerate(stims):
                if 0 <= s < N_STIMULI:
                    if self.additive:
                        x[s] += pos_w[idx]
                    else:
                        x[s] = 1.0 if x[s] == 0.0 else x[s]

            # Learning rate γ from causal uncertainty (eq. 9): softmax over Var(θ)
            var = _dirichlet_causal_uncertainty(alpha)
            gamma = _learning_rates_from_uncertainty(var, self.inv_temp)
            gamma = self.lr * gamma

            # Update (Eq. 7 & surrounding text):
            #   Δα_i =  γ_i x_i  when paired with this outcome type
            #   Δα_i = -γ_i x_i  when paired with the other outcome type
            if outcome == OUTCOME_NOVEL:
                delta = gamma * x
            else:
                delta = -gamma * x

            alpha = alpha + delta
            alpha = np.maximum(alpha, 0.0)

            tot_oslrs.append(_dirichlet_causal_strength(alpha).astype(np.float32))

        return np.stack(tot_oslrs)
