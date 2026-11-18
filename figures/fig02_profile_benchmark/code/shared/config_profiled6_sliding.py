"""Paths for profiled6 sliding-window training & evaluation (separate from mdl13 baseline)."""

from __future__ import annotations

from pathlib import Path

from profile_benchmark.analysis.profiled6_paths import resolve_profiled6_root

_DATA_ROOT = resolve_profiled6_root()
PROFILED6_SLIDING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROFILED6_SLIDING_ROOT.parents[2]

DATA_CSV = _DATA_ROOT / "data" / "simulated_profiled6.csv"
SPLIT_MANIFEST = PROFILED6_SLIDING_ROOT / "splits" / "split_seed123_profiled6_sliding_episode_ids.json"

CKPT_DIR = PROFILED6_SLIDING_ROOT / "checkpoints_profiled6_sliding_mdl13"
LOG_DIR = PROFILED6_SLIDING_ROOT / "logs_profiled6_sliding_mdl13"

RESULTS_SLIDING = PROFILED6_SLIDING_ROOT / "results_profiled6_sliding"
GEOMETRY_SLIDING = PROFILED6_SLIDING_ROOT / "geometry_profiled6_sliding"
MECH_TRACES = PROFILED6_SLIDING_ROOT / "mechanistic_traces_profiled6_sliding"
ANALYSIS_SUMMARY = PROFILED6_SLIDING_ROOT / "analysis_summary_profiled6_sliding"
# Full forward diagnostics (logits, z, hidden, attention) + shared episode behaviour tables
DIAGNOSTICS_DIR = PROFILED6_SLIDING_ROOT / "diagnostics_profiled6_sliding"

PROFILED6_TASKS = ["causal", "bandit", "field2d", "switching_medium"]
PROFILED6_AGENTS = ["bayes", "softmax", "wsls", "noisy", "switching", "random"]

TRAIN_L_VALUES = (10, 20, 40, 80)
EVAL_L_VALUES = (5, 10, 20, 40, 80)
TARGET_LEN = 20
SLIDING_STRIDE = 20
MAX_PREFIX_L = 80
TRAIN_SEEDS = tuple(range(20))

MODELS_13 = [
    "logreg",
    "mlp_pool",
    "rnn",
    "tinyrnn",
    "transformer",
    "lstm_attn",
    "lstm",
    "single_layer",
    "mlp",
    "cnn",
    "bilstm",
    "pre_lstm",
    "pre_bilstm",
    "post_bilstm",
]

# Paper TinyRNN (Ji-An et al. 2025 GRU d=4); trained into checkpoints_tinyrnn_paper/
MODELS_TINYRNN_PAPER = ["tinyrnn_paper"]
CKPT_DIR_TINYRNN_PAPER = PROFILED6_SLIDING_ROOT / "checkpoints_tinyrnn_paper"
LOG_DIR_TINYRNN_PAPER = PROFILED6_SLIDING_ROOT / "logs_tinyrnn_paper"

CKPT_DIR_INPUTPAD_V2 = PROFILED6_SLIDING_ROOT / "checkpoints_inputpad_v2"
LOG_DIR_INPUTPAD_V2 = PROFILED6_SLIDING_ROOT / "logs_inputpad_v2"
CKPT_DIR_INPUTPAD_V2_EPOCHS = PROFILED6_SLIDING_ROOT / "checkpoints_inputpad_v2_epochs"
RESULTS_INPUTPAD_V2 = PROFILED6_SLIDING_ROOT / "results_inputpad_v2"
GEOMETRY_INPUTPAD_V2 = PROFILED6_SLIDING_ROOT / "geometry_inputpad_v2"
ANALYSIS_INPUTPAD_V2 = PROFILED6_SLIDING_ROOT / "analysis_summary_inputpad_v2"
FIG_DIR_INPUTPAD_V2 = (
    PROFILED6_SLIDING_ROOT
    / "handoff_analysis_workspace"
    / "profile_module_storyline_final"
    / "figures"
    / "unified_main_inputpad_v2"
)

MODELS_INPUTPAD_V2 = [
    "lstm",
    "bilstm",
    "gru",
    "lstm_attn",
    "post_bilstm",
    "pre_lstm",
    "pre_bilstm",
    "transformer",
    "tinyrnn_paper",
    "cnn",
    "mlp_pool",
    "mlp",
    "single_layer",
]
