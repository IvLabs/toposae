"""Visualization utilities for research results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os


def setup_plotting_style():
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.2)
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 1.0})


def plot_training_curves(metrics, output_path, title="Training Curves"):
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(metrics['train_loss']) + 1)
    ax1.plot(epochs, metrics['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in metrics:
        ax1.plot(epochs, metrics['val_loss'], label='Val Loss', marker='s', markersize=3)
    if 'topo_loss' in metrics:
        ax1.plot(epochs, metrics['topo_loss'], label='TopoLoss', marker='^', markersize=3)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss Over Time'); ax1.legend(); ax1.grid(True, alpha=0.3)
    if 'val_acc' in metrics:
        ax2.plot(epochs, metrics['val_acc'], label='Val Accuracy', marker='s', color='green')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Validation Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14); plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path); plt.close(fig)


def plot_monosemanticity_distribution(scores_dict, output_path, title="Monosemanticity Score Distribution"):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, scores in scores_dict.items():
        scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
        sns.kdeplot(scores_np, ax=ax, label=run_id, linewidth=2, fill=True, alpha=0.3)
    ax.set_xlabel('Monosemanticity Score'); ax.set_ylabel('Density'); ax.set_title(title); ax.legend()
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path); plt.close(fig)


def plot_cortical_heatmap(selectivity_map, output_path, title="Cortical Selectivity Heatmap", class_name=""):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(selectivity_map, cmap='hot', aspect='equal')
    fig.colorbar(im, ax=ax, label='Selectivity Score')
    ax.set_title(f'{title} - {class_name}' if class_name else title)
    ax.set_xlabel('Cortical X'); ax.set_ylabel('Cortical Y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path); plt.close(fig)


def plot_comparison_bar_chart(metrics, output_path, title="Metric Comparison"):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    run_ids = list(metrics.keys())
    metric_names = list(metrics[run_ids[0]].keys())
    x = np.arange(len(metric_names)); width = 0.25
    for i, run_id in enumerate(run_ids):
        ax.bar(x + i * width, [metrics[run_id][m] for m in metric_names], width, label=run_id)
    ax.set_xlabel('Metric'); ax.set_ylabel('Value'); ax.set_title(title)
    ax.set_xticks(x + width); ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path); plt.close(fig)
