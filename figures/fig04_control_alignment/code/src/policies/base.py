from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

@dataclass
class PolicyOutput:
    index: int
    score: float

class BasePolicy:
    def __init__(self, name: str):
        self.name = name

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        raise NotImplementedError
