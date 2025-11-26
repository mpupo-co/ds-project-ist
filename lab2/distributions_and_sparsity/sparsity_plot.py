import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib.pyplot import subplots


# ---------------------------------------------------------
# WRAPPER: builds the helper functions using user lists
# ---------------------------------------------------------

def make_type_helpers(numeric, symbolic, binary):

    def var_is_numeric(v):
        return v in numeric

    def var_is_symbolic(v):
        return v in symbolic

    def var_is_binary(v):
        return v in binary

    def var_is_categorical_like(v):
        return var_is_symbolic(v) or var_is_binary(v)

    return var_is_numeric, var_is_symbolic, var_is_binary, var_is_categorical_like



# ---------------------------------------------------------
# CONSTANT + CATEGORY COUNT HELPERS
# ---------------------------------------------------------

def too_many_categories(s, max_cats=50):
    return s.nunique() > max_cats

def is_constant(s):
    return s.nunique() <= 1



# ---------------------------------------------------------
# DECISION LOGIC (now uses passed-in type functions)
# ---------------------------------------------------------

def make_plot_type(var_is_numeric, var_is_binary, var_is_categorical_like):

    def plot_type(s1, s2, v1, v2):

        # Skip constant variables
        if is_constant(s1) or is_constant(s2):
            return "skip"

        # Skip boolean × boolean
        if var_is_binary(v1) and var_is_binary(v2):
            return "skip"

        # numeric × numeric → scatter
        if var_is_numeric(v1) and var_is_numeric(v2):
            return "numeric_scatter"

        # numeric × categorical → jitter scatter
        if var_is_numeric(v1) and var_is_categorical_like(v2):
            if too_many_categories(s2):
                return "skip"
            return "jitter_scatter"

        if var_is_numeric(v2) and var_is_categorical_like(v1):
            if too_many_categories(s1):
                return "skip"
            return "jitter_scatter"

        # symbolic × symbolic → skip
        return "skip"

    return plot_type



# ---------------------------------------------------------
# PLOT FUNCTIONS
# ---------------------------------------------------------

def make_plot_jitter_scatter(var_is_numeric):
    """We need a factory to capture var_is_numeric."""

    def plot_jitter_scatter(ax, s1, s2, v1, v2):

        # Determine which is numeric
        if var_is_numeric(v1):
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

    return plot_jitter_scatter



def plot_numeric_scatter(ax, s1, s2, v1, v2):
    ax.scatter(s1, s2, alpha=0.5)
    ax.set_title(f"{v1} × {v2}")
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)



# ---------------------------------------------------------
# MAIN DRIVER — THE ONLY PUBLIC FUNCTION
# ---------------------------------------------------------

def plot_sparsity_matrix(
        data: DataFrame,
        file_tag: str,
        numeric: list,
        symbolic: list,
        binary: list
):

    # Build type helpers using your lists
    (
        var_is_numeric,
        var_is_symbolic,
        var_is_binary,
        var_is_categorical_like
    ) = make_type_helpers(numeric, symbolic, binary)

    # Build decision logic
    plot_type = make_plot_type(var_is_numeric, var_is_binary, var_is_categorical_like)

    # Jitter plot depends on numeric list → build it
    plot_jitter_scatter = make_plot_jitter_scatter(var_is_numeric)

    vars = data.columns.tolist()

    # Find valid pairs
    pairs = []
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            v1, v2 = vars[i], vars[j]
            if plot_type(data[v1], data[v2], v1, v2) != "skip":
                pairs.append((v1, v2))

    if not pairs:
        print("No meaningful plots.")
        return

    # Layout
    n = len(pairs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axs = subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.array(axs).reshape(rows, cols)

    for (v1, v2), ax in zip(pairs, axs.flat):

        s1, s2 = data[v1], data[v2]
        t = plot_type(s1, s2, v1, v2)

        if t == "numeric_scatter":
            plot_numeric_scatter(ax, s1, s2, v1, v2)
        elif t == "jitter_scatter":
            plot_jitter_scatter(ax, s1, s2, v1, v2)
        else:
            ax.axis("off")

    # Disable unused axes
    for ax in axs.flat[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"images/{file_tag}_sparsity_study.png")
    print(f"Saved: images/{file_tag}_sparsity_study.png")