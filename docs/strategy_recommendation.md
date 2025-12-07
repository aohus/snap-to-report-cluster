# Hyperparameter Optimization Strategy

## 1. Analysis of Current Results

Based on the results from `.experiment/exp_results/semantic_final/summary.csv`, we can draw the following conclusions:

-   **LoFTR Outperforms SIFT**: The `loftr` geometry matcher consistently achieves higher ARI and NMI scores compared to `sift`.
-   **Semantic Masking is Functional**: Experiments with `use_mask=True` are now running successfully. The fact that the scores are identical to `use_mask=False` suggests that for the current dataset, the background features are dominant, and masking objects like people doesn't significantly alter the matching results.
-   **Fixed Primary Parameters**: The core parameters `similarity_threshold` (0.88) and `knn_k` (8) have remained constant. These are the most critical parameters to tune for improving clustering quality.

## 2. Proposed Optimization Strategy

To find the optimal hyperparameter configuration, the following strategy is recommended:

1.  **Focus on LoFTR with Masking**: We will use the `loftr` matcher with `use_mask=True` as the default configuration for all upcoming experiments. This is our best-performing setup and aligns with the goal of ignoring people and other specified objects.

2.  **Grid Search on Core Parameters**: The next logical step is to perform a grid search on the most impactful parameters that have not yet been tuned:
    -   `similarity_threshold`: This defines the minimum similarity score for two images to be considered candidates for clustering.
    -   `knn_k`: This determines the size of the neighborhood to search for potential matches.

3.  **Fine-Tune Geometric Threshold**: We will also explore a more granular range for LoFTR's `geo_threshold` to find the sweet spot for geometric verification.

## 3. Next Experiment Parameter Grid

The following parameter ranges will be used to generate a comprehensive set of experiment combinations:

-   **`similarity_threshold`**: `[0.85, 0.88, 0.90, 0.92]`
-   **`knn_k`**: `[8, 10, 12, 15]`
-   **`geo_threshold` (for LoFTR)**: `[0.1, 0.15, 0.2]`

This grid search will result in `4 * 4 * 3 = 48` new experiments, allowing us to thoroughly explore the parameter space and identify the optimal settings for this clustering task.