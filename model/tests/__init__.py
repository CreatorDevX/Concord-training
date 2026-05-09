"""
Concord-α Test Suite.

Test categories:
    test_shapes.py              — Forward pass shape correctness (Gate 1)
    test_training.py            — End-to-end training (Gate 2)
    test_param_counts.py        — Pure math param counts (Gate 2)
    test_param_counts_materialized.py — Materialized param counts (Gate 3)
    test_param_groups.py        — Optimizer param group deduplication
    test_checkpointing.py       — Activation recomputation correctness
    test_optimizer_normalized.py — NormalizedSGD optimizer tests
    test_training_full.py       — Full training loop correctness
    test_load_balance.py        — Router load distribution (Gate 4)
    test_attnres_stability.py   — AttnRes stability (Gate 5)
    test_dataset_loading.py     — Dataset loading
    test_harness_masking.py     — Harness formatting and masking
    test_progress_tracker.py    — Progress tracker
"""
