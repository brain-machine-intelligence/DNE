from __future__ import annotations
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bernoulli_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def kl_bernoulli(p, q, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    q = np.clip(q, eps, 1-eps)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def js_bernoulli(p, q, eps=1e-12):
    m = 0.5*(p+q)
    return 0.5*kl_bernoulli(p, m, eps) + 0.5*kl_bernoulli(q, m, eps)
