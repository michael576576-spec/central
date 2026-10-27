# Hub Structure Experiments

Run the commands below from the project directory. Dependencies: `numpy`, `pandas`, `networkx`, and `matplotlib`.

## Synthetic network test

```bash
python simulate_graph_test.py
```

Entry point: `simulate_graph_test.evaluate()`. The script builds synthetic graphs with known positive and negative labels, tests for significant hub structure, and prints per-graph results plus precision, recall, and F1. To change the null model or sample count, edit the settings at the end of the file.

## Real network test

```bash
python real_network_test.py
```

Entry point: `real_network_test.evaluate()`. The script reads `.txt` edge lists from `network/`, tests each graph, and prints the results. The current loader labels every real network as positive, so its evaluation metrics depend on that assumption.

## Hub weight experiment

```bash
python multi_hub_pvalue_experiment.py
```

Entry point: `multi_hub_pvalue_experiment.run_hub_weight_sweep_experiment()`. With node and edge counts fixed, the script varies hub connection weights (`hub_weights`) and compares spectral-radius p-values. The current script settings use `N=50`, `M=100`, 1–15 hub nodes (up to `0.3N`), 50 random seeds per condition, and 100 null graphs per test. CSV files and plots are saved under `experiment_results/hub_weight_sweep_N50_M100_K1_15_gnm_50seed/`.

## Batch experiment

```bash
python batch_experiment.py
```

Entry point: `batch_experiment.run_model_sample_grid()`. The script combines real and synthetic data with four null models (`gnm`, `configuration-base`, `configuration-swap`, `hidden`) and three sample counts (100, 500, 1000), for 24 experiment groups. It writes `experiment_results/summary_grid.csv`, `detail_grid.csv`, and one log per model. Rerunning it overwrites files with those names.
