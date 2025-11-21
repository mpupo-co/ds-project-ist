import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib.pyplot import subplots
from pandas.api.types import (
    is_bool_dtype, is_numeric_dtype,
    is_categorical_dtype, is_string_dtype
)


# ---------- TYPE HELPERS ----------

def is_categorical_like(s):
    return (
        is_bool_dtype(s)
        or is_string_dtype(s)
        or is_categorical_dtype(s)
    )


def too_many_categories(s, max_cats=50):
    return s.nunique() > max_cats


def is_constant(s):
    return s.nunique() <= 1



# ---------- DECISION LOGIC ----------

def plot_type(s1, s2):
    """
    Decide which visualization to use for a pair of variables.
    Returns one of:
        'numeric_scatter'
        'jitter_scatter'
        'skip'
    """

    # Skip constant variables
    if is_constant(s1) or is_constant(s2):
        return "skip"

    # Skip boolean × boolean
    if is_bool_dtype(s1) and is_bool_dtype(s2):
        return "skip"

    # numeric × numeric → scatter
    if is_numeric_dtype(s1) and is_numeric_dtype(s2):
        return "numeric_scatter"

    # numeric × symbolic or categorical → jitter scatter
    if is_numeric_dtype(s1) and is_categorical_like(s2):
        if too_many_categories(s2):
            return "skip"
        return "jitter_scatter"

    if is_numeric_dtype(s2) and is_categorical_like(s1):
        if too_many_categories(s1):
            return "skip"
        return "jitter_scatter"

    # symbolic × symbolic → skip (no heatmaps allowed)
    return "skip"



# ---------- PLOT FUNCTIONS ----------

def plot_numeric_scatter(ax, s1, s2, v1, v2):
    ax.scatter(s1, s2, alpha=0.5)
    ax.set_title(f"{v1} × {v2}")
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)


def plot_jitter_scatter(ax, s1, s2, v1, v2):
    """Numeric × Categorical scatter (symbolic/boolean)."""
    if is_numeric_dtype(s1):
        num, cat = s1, s2
        swap = False
    else:
        num, cat = s2, s1
        swap = True

    c = cat.astype("category")
    jitter = (np.random.random(len(c)) - 0.5) * 0.1
    codes = c.cat.codes + jitter

    if not swap:
        ax.scatter(num, codes, alpha=0.5)
        ax.set_yticks(range(len(c.cat.categories)))
        ax.set_yticklabels(c.cat.categories)
        ax.set_xlabel(v1)
        ax.set_ylabel(v2)
    else:
        ax.scatter(codes, num, alpha=0.5)
        ax.set_xticks(range(len(c.cat.categories)))
        ax.set_xticklabels(c.cat.categories)
        ax.set_xlabel(v1)
        ax.set_ylabel(v2)

    ax.set_title(f"{v1} × {v2} (jitter)")


def plot_sparsity_pair(ax, data, v1, v2):
    s1, s2 = data[v1], data[v2]
    t = plot_type(s1, s2)

    if t == "numeric_scatter":
        plot_numeric_scatter(ax, s1, s2, v1, v2)

    elif t == "jitter_scatter":
        plot_jitter_scatter(ax, s1, s2, v1, v2)

    else:  # skip
        ax.axis("off")



# ---------- MAIN DRIVER ----------

def plot_sparsity_matrix(data: DataFrame, file_tag: str):
    vars = data.columns.tolist()
    if not vars:
        print("Sparsity: no variables.")
        return

    # find valid pairs
    pairs = []
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            v1, v2 = vars[i], vars[j]
            if plot_type(data[v1], data[v2]) != "skip":
                pairs.append((v1, v2))

    if not pairs:
        print("No meaningful plots.")
        return

    # layout
    n = len(pairs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axs = subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.array(axs).reshape(rows, cols)

    for (v1, v2), ax in zip(pairs, axs.flat):
        plot_sparsity_pair(ax, data, v1, v2)

    # disable unused axes
    for ax in axs.flat[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"images/{file_tag}_sparsity_study.png")
    print(f"Saved: images/{file_tag}_sparsity_study.png")
