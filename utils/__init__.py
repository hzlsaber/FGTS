"""
Utility modules for Fisher-Guided Token Selection (FGTS)
"""

from .data import (
    SimpleImageDataset,
    build_dataloader,
    get_all_test_datasets,
    resolve_dataset_paths,
    create_dataset_readme
)

from .models import (
    load_model,
    load_hf_model,
    get_token_layout,
    freeze_model
)

from .features import (
    extract_features,
    select_tokens,
    compute_fisher_scores,
    filter_special_tokens,
    select_top_k_tokens,
    pool_features,
    compute_register_subspace,
    decompose_tokens
)

from .metrics import (
    compute_metrics,
    evaluate_model,
    print_metrics,
    aggregate_results,
    format_result_table
)

from .io import (
    save_json,
    load_json,
    save_numpy,
    load_numpy,
    save_checkpoint,
    load_checkpoint,
    generate_report,
    save_fisher_scores,
    load_fisher_scores
)

__all__ = [
    # data
    'SimpleImageDataset',
    'build_dataloader',
    'get_all_test_datasets',
    'resolve_dataset_paths',
    'create_dataset_readme',
    # models
    'load_model',
    'load_hf_model',
    'get_token_layout',
    'freeze_model',
    # features
    'extract_features',
    'select_tokens',
    'compute_fisher_scores',
    'filter_special_tokens',
    'select_top_k_tokens',
    'pool_features',
    'compute_register_subspace',
    'decompose_tokens',
    # metrics
    'compute_metrics',
    'evaluate_model',
    'print_metrics',
    'aggregate_results',
    'format_result_table',
    # io
    'save_json',
    'load_json',
    'save_numpy',
    'load_numpy',
    'save_checkpoint',
    'load_checkpoint',
    'generate_report',
    'save_fisher_scores',
    'load_fisher_scores',
]
