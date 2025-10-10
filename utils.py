import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score, precision_recall_curve


def calculate_optimal_threshold(y_true, y_prob, beta=2.0):
    """Find the optimal probability threshold for an F-beta score.

    This function iterates through a range of probability thresholds (from 0.0 to 1.0)
    to identify the threshold that maximizes the F-beta score.

    Args:
        y_true (array-like): True binary labels for the data.
        y_prob (array-like): Predicted probabilities for the positive class.
        beta (float, optional): The beta parameter for the F-beta score calculation.
            Defaults to 2.

    Returns:
        float: The probability threeshold that yields the highest F-beta score.
    """
    best_score = 0
    best_threshold = 0
    for t in np.arange(0.0, 1.0, 0.01):
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold


def plot_recall_precision_curve(y_true, y_prob, title, ref_threshold=0.5, ax=None):
    """Plot precision and recall scores against decision thresholds.

    This function calculates precision and recall for a range of thresholds
    and plots them. It highlights the precision and recall values at the
    specified reference threshold.

    Args:
        y_true (array-like): True binary labels for the data.
        y_prob (array-like): Predicted probabilities for the positive classes.
        ref_threshold (float, optional): A reference threshold to visually identify
            precision and recall at that given probability threshold. Defaults to 0.5
    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    ax.plot(thresholds, precision[:-1], "r--", label="Precision")
    ax.plot(thresholds, recall[:-1], "g--", label="Recall")

    ax.axvline(
        x=ref_threshold,
        color="blue",
        linestyle="--",
        label=f"Reference threshold:{ref_threshold}",
    )

    closest_threshold_idx = np.argmin(np.abs(thresholds - ref_threshold))

    precision_at_threshold = precision[closest_threshold_idx]
    recall_at_threshold = recall[closest_threshold_idx]

    ax.plot(
        ref_threshold,
        precision_at_threshold,
        "ro",
        label=f"Precision: {precision_at_threshold:.2f}",
    )
    ax.plot(
        ref_threshold,
        recall_at_threshold,
        "go",
        label=f"Recall: {recall_at_threshold:.2f}",
    )

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend(loc="best")

    return ax


def extract_hyperparams_from_cv_results(cv_results, param_map):
    """Extracts resampler and hyperparameter info from a cv_results_ DataFrame.

    This function processes a scikit-learn cv_results_ DataFrame to add three
    new columns: 'resampler_name', 'hyperparameter_name', and
    'hyperparameter_value'. It uses a mapping to identify the key hyperparameter
    for each resampling technique.

    Args:
        cv_results (pd.DataFrame): The DataFrame from a GridSearchCV or
            RandomizedSearchCV `.cv_results_` attribute.
        param_map (dict): A dictionary mapping resampler class names (e.g.,
            'SMOTE') to the specific hyperparameter name to extract (e.g.,
            'k_neighbors').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the three new columns.
    """
    df = cv_results.copy()

    df["resampler_name"] = df["param_resampler"].apply(
        lambda x: x if isinstance(x, str) else x.__class__.__name__
    )

    ignore_sampler_mask = ~df["resampler_name"].isin(param_map.keys())
    df.loc[ignore_sampler_mask, "resampler_name"] = pd.NA

    df["hyperparameter_name"] = df["resampler_name"].map(param_map)

    df["hyperparameter_value"] = pd.NA
    for param_name in param_map.values():
        mask = df["hyperparameter_name"] == param_name
        if param_name == "N/A":
            df.loc[mask, "hyperparameter_value"] = "N/A"
        else:
            df.loc[mask, "hyperparameter_value"] = df.loc[
                mask, f"param_resampler__{param_name}"
            ]

    return df


def plot_resampling_results(cv_results_df, resampler_params_map, title, plot_params):

    g = sns.catplot(data=cv_results_df, **plot_params, palette="pastel")

    max_scores = (
        (
            cv_results_df.groupby(["resampler_name"])[plot_params["y"]]
            .max()
            .reset_index()
        )
        .sort_values(by=plot_params["y"], ascending=False)
        .set_index("resampler_name")
    )

    best_resampler_name = max_scores.index[0]
    best_resampler_score = max_scores.iat[0, 0]

    for ax in g.axes.flatten():

        resampler_name = ax.get_title().split(" = ")[1]
        resampler_max_score = max_scores.at[resampler_name, plot_params["y"]]

        ax.set_title("")
        ax.text(
            0.5,
            1.1,
            resampler_name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            color="black",
            weight="bold",
            path_effects=[
                path_effects.Stroke(linewidth=11.5, foreground="white"),
                path_effects.Normal(),
            ],
        )

        ax.set_xlabel(
            resampler_params_map.get(resampler_name, "Parameter Value"),
            fontsize=11,
            color="black",
        )

        if best_resampler_name != resampler_name:
            ax.axhline(y=resampler_max_score, ls="--", color="red")
            ax.text(
                x=0.98,
                y=0.07,
                transform=ax.transAxes,
                s=f"Max Score for {resampler_name}: {resampler_max_score}",
                color="red",
                va="top",
                ha="right",
                fontsize=9,
            ).set_path_effects(
                [
                    path_effects.Stroke(linewidth=0.6, foreground="black"),
                    path_effects.Normal(),
                ]
            )

        ax.axhline(y=best_resampler_score, ls="--", color="green")
        ax.text(
            x=0.98,
            y=0.1,
            transform=ax.transAxes,
            s=f"BEST RESAMPLER: {best_resampler_name}: {best_resampler_score}",
            color="green",
            va="bottom",
            ha="right",
            fontsize=9,
        ).set_path_effects(
            [
                path_effects.Stroke(linewidth=0.6, foreground="black"),
                path_effects.Normal(),
            ]
        )

    g.figure.suptitle(
        title,
        y=1.03,
        fontsize=18,
        color="navy",
        path_effects=[
            path_effects.Stroke(linewidth=1, foreground="gray"),
            path_effects.Normal(),
        ],
    )
    sns.move_legend(g, "upper right")

    return g


def plot_cv_results_by_resampler(
    cv_results_df, resampler_params_map, title, plot_params
):

    df_performance = extract_hyperparams_from_cv_results(
        cv_results_df, resampler_params_map
    )

    g = plot_resampling_results(
        df_performance, resampler_params_map, title, plot_params
    )

    for ax in g.axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    return g
