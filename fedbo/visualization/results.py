"""
Utilities for downloading and plotting experiment data from Weights & Biases (wandb).

This script provides functions to:
- Fetch run data (history and configs) from the wandb API.
- Aggregate and process the data using pandas.
- Generate various multi-panel regret curve plots using matplotlib.
"""

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from wandb.apis.public.runs import Runs
from RWTHColors import ColorManager


# --- Globals & Configuration ---

# Define a path to a custom matplotlib style file.
# Note: __file__ works when running as a script, but might fail in
# interactive environments (e.g., Jupyter).
try:
    STYLE_PATH: Path = Path(__file__).parent / "style" / "fedbo.mplstyle"
    plt.style.use(str(STYLE_PATH))
except NameError:
    print("Could not find style file. Using default.")
    STYLE_PATH = None

# Initialize the wandb API
API: wandb.Api = wandb.Api()

# Mappings for pretty-printing project and model names in plots
PROJECT_NAMES: Dict[str, str] = {
    "ackley1d": "Ackley (1D)",
    "ackley5d": "Ackley (5D)",
    "ackley10d": "Ackley (10D)",
    "branin": "Branin (2D)",
    "hartmann6d": "Hartmann (6D)",
    "combination1d": "Combination (1D)",
    "combination5d": "Combination (5D)",
    "combination10d": "Combination (10D)",
    "cartpole": "Cartpole (5D)",
    "manytask10d": "Many-Task (10D)",
}

MODEL_NAMES_PLOT: Dict[str, Dict[str, str]] = {
    "fedbo.training.CentralizedTrainer.hydra_init": {
        "fedbo.models.MLP.hydra_init": "C-MLP",
        "fedbo.models.VBLL.hydra_init": "C-VBLL",
    },
    "fedbo.training.FederatedTrainer.hydra_init": {
        "fedbo.models.MLP.hydra_init": "FL-MLP (Ours)",
        "fedbo.models.VBLL.hydra_init": "FL-VBLL (Ours)",
        "fedbo.models.MultiLayerHeadMLP.hydra_init": "FL-MLH (Ours)",
    },
    "fedbo.training.LocalTrainer.hydra_init": {
        "fedbo.models.GaussianProcess.hydra_init": "L-GP",
        "fedbo.models.MLP.hydra_init": "L-MLP",
        "fedbo.models.VBLL.hydra_init": "L-VBLL",
    },
    "fedmany_trainer": {"fedmany_gp": "FMTBO", "fedmany_gp_EI_w_kt": "FMTBO"},
    "fts_de_trainer": {"fts_de_gp": "DP-FTS-DE"},
}


MODEL_NAMES: Dict[str, Dict[str, str]] = {
    "fedbo.training.CentralizedTrainer.hydra_init": {
        "fedbo.models.MLP.hydra_init": "C-MLP",
        "fedbo.models.VBLL.hydra_init": "C-VBLL",
    },
    "fedbo.training.FederatedTrainer.hydra_init": {
        "fedbo.models.MLP.hydra_init": "FL-MLP",
        "fedbo.models.VBLL.hydra_init": "FL-VBLL",
        "fedbo.models.MultiLayerHeadMLP.hydra_init": "FL-MLH",
    },
    "fedbo.training.LocalTrainer.hydra_init": {
        "fedbo.models.GaussianProcess.hydra_init": "L-GP",
        "fedbo.models.MLP.hydra_init": "L-MLP",
        "fedbo.models.VBLL.hydra_init": "L-VBLL",
    },
    "fedmany_trainer": {"fedmany_gp": "FMTBO", "fedmany_gp_EI_w_kt": "FMTBO"},
    "fts_de_trainer": {"fts_de_gp": "DP-FTS-DE"},
}


REVERSED_MODEL_NAMES: Dict[str, Tuple[str, str]] = {
    "C-MLP": ("fedbo.training.CentralizedTrainer.hydra_init", "fedbo.models.MLP.hydra_init"),
    "C-VBLL": ("fedbo.training.CentralizedTrainer.hydra_init", "fedbo.models.VBLL.hydra_init"),
    "FL-MLP": ("fedbo.training.FederatedTrainer.hydra_init", "fedbo.models.MLP.hydra_init"),
    "FL-VBLL": ("fedbo.training.FederatedTrainer.hydra_init", "fedbo.models.VBLL.hydra_init"),
    "L-GP": ("fedbo.training.LocalTrainer.hydra_init", "fedbo.models.GaussianProcess.hydra_init"),
    "L-MLP": ("fedbo.training.LocalTrainer.hydra_init", "fedbo.models.MLP.hydra_init"),
    "L-VBLL": ("fedbo.training.LocalTrainer.hydra_init", "fedbo.models.VBLL.hydra_init"),
    "FMTBO": ("fedmany_trainer", "fedmany_gp_EI_w_kt"),
    "FMTBO_2": ("fedmany_trainer", "fedmany_gp"),
    "DP-FTS-DE": ("fts_de_trainer", "fts_de_gp"),
    "FL-MLH": ("fedbo.training.FederatedTrainer.hydra_init", "fedbo.models.MultiLayerHeadMLP.hydra_init"),
}

CM = ColorManager()
MODEL_COLORS: Dict[str, str] = {
    # Centralized
    "C-MLP": CM.RWTHLila(),
    "C-VBLL": CM.RWTHBordeaux(),
    # Federated (Ours)
    "FL-MLP (Ours)": CM.RWTHBlau(),
    "FL-VBLL (Ours)": CM.RWTHGruen(),
    "FL-MLH (Ours)": CM.RWTHSchwarz(),
    # Local
    "L-GP": CM.RWTHMaiGruen(),
    "L-MLP": CM.RWTHViolett(),
    "L-VBLL": CM.RWTHMagenta(),
    # Baselines
    "FMTBO": CM.RWTHOrange(),
    "DP-FTS-DE": CM.RWTHRot(),
}
MODEL_COLORS["FL-MLP"] = MODEL_COLORS["FL-MLP (Ours)"]
MODEL_COLORS["FL-VBLL"] = MODEL_COLORS["FL-VBLL (Ours)"]


# --- Data Loading & Processing ---


def _get_nested_config(config: Dict[str, Any], key: str) -> Any:
    """
    Safely retrieves a nested value from a config dict using a dot-separated key.

    Args:
        config: The wandb config dictionary.
        key: A string key like "model.optimizer.lr".

    Returns:
        The nested value, or None if any part of the key is not found.
    """
    value = config
    try:
        for part in key.split("."):
            value = value[part]
        return value
    except (KeyError, TypeError):
        # Handles missing keys or trying to index into None
        # print(f"Warning: Could not find config key '{key}' in run.")
        return None


def load_wandb_data(
    entity: str,
    project: str,
    data_keys: List[str],
    config_keys: List[str],
) -> pd.DataFrame:
    """
    Loads run histories and specified config values from a wandb project.

    Args:
        entity: The wandb entity (user or team name).
        project: The wandb project name.
        data_keys: A list of metric keys to fetch from the run history.
        config_keys: A list of config keys to fetch (supports nesting, e.g., "model.lr").

    Returns:
        A pandas DataFrame containing all run data.
    """
    # Always include the step metric
    all_data_keys = list(data_keys) + ["regret_step"]

    runs: Runs = API.runs(f"{entity}/{project}")
    run_data = []

    print(f"Fetching data for {entity}/{project}...")
    for run in runs:
        history = run.history(keys=all_data_keys)
        history["run_id"] = run.id

        # Safely extract config keys
        for key in config_keys:
            history[key] = _get_nested_config(run.config, key)

        run_data.append(history)

    if not run_data:
        print(f"Warning: No runs found for project {project}.")
        return pd.DataFrame()

    return pd.concat(run_data, ignore_index=True)


def aggregate_df(df: pd.DataFrame, group_keys: List[str], metrics: List[str]) -> pd.DataFrame:
    """
    Groups a DataFrame by specified keys and aggregates metrics with mean and std.

    This function also flattens the resulting MultiIndex columns (e.g.,
    ('metric_name', 'mean') becomes 'metric_name_mean').

    Args:
        df: The input DataFrame from load_wandb_data.
        group_keys: List of config keys to group by.
        metrics: List of metric keys to aggregate.

    Returns:
        An aggregated DataFrame with flattened columns.
    """
    if df.empty:
        return pd.DataFrame()

    agg_keys = list(group_keys) + ["regret_step"]

    # Handle cases where keys might be missing from a run
    valid_agg_keys = [key for key in agg_keys if key in df.columns]
    missing_keys = set(agg_keys) - set(valid_agg_keys)
    if missing_keys:
        print(f"Warning: Missing keys in aggregate_df: {missing_keys}")

    if not valid_agg_keys:
        print("Error: No valid keys to group by.")
        return pd.DataFrame()

    grouped_df = df.groupby(valid_agg_keys)[metrics]
    aggr_df = grouped_df.agg(["mean", "std"]).reset_index()

    # Flatten the MultiIndex columns
    new_cols = []
    for col in aggr_df.columns.values:
        if col[0] in valid_agg_keys:
            new_cols.append(col[0])
        else:
            # col is a tuple like ('avg/simple_regret', 'mean')
            new_cols.append(f"{col[0]}_{col[1]}")
    aggr_df.columns = new_cols
    return aggr_df


# --- [NEW] Data Filtering Functions ---


def filter_df_by_model(df: pd.DataFrame, models_to_plot: List[str]) -> pd.DataFrame:
    """Filters a DataFrame to include only runs from the specified short names."""
    if df.empty:
        return df

    target_configs = []
    for short_name in models_to_plot:
        if short_name in REVERSED_MODEL_NAMES:
            target_configs.append(REVERSED_MODEL_NAMES[short_name])
        else:
            print(f"Warning: Model name '{short_name}' not found in REVERSED_MODEL_NAMES. Skipping.")

    if not target_configs:
        print("Warning: No valid models specified for filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    # Create a boolean mask for all valid configs
    mask = pd.Series(False, index=df.index)
    for trainer_target, model_target in target_configs:
        mask |= (df["trainer._target_"] == trainer_target) & (df["model._target_"] == model_target)

    return df[mask].reset_index(drop=True)


def _filter_df_list(df_list: List[pd.DataFrame], models_to_plot: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """Helper to filter a list of dataframes by model short names."""
    if not models_to_plot:
        return df_list

    filtered_list = []
    for df in df_list:
        if "trainer._target_" not in df.columns or "model._target_" not in df.columns:
            print(
                "Warning: Cannot filter by model. 'trainer._target_' or 'model._target_' "
                "not found in DataFrame columns. Ensure they are loaded in `config_keys`."
            )
            filtered_list.append(df)  # Append anyway, in case this is intended
        else:
            filtered_list.append(filter_df_by_model(df, models_to_plot))
    return filtered_list


def save_to_csv(plot_data: pd.DataFrame, project_key: str, config_name: str, metric_name: str, file_suffix: str = ""):
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    # Create a new DataFrame with regret_step, mean, and std
    plot_data_subset = plot_data[["regret_step", mean_col, std_col]].copy()
    plot_data_subset["max"] = plot_data_subset[mean_col] + plot_data_subset[std_col]
    plot_data_subset["min"] = plot_data_subset[mean_col] - plot_data_subset[std_col]
    plot_data_subset = plot_data_subset[["regret_step", mean_col, std_col, "max", "min"]]
    plot_data_subset.columns = ["regret_step", "mean", "std", "max", "min"]

    # Save the plot data subset to a file
    save_dir = Path(os.environ.get("FEDBO_RESULTS_DIR", "results/simulation_results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{project_key}_{config_name}"
    if file_suffix:
        filename += f"_{file_suffix}"
    filename = ''.join(c if c.isalnum() or c == '_' else '_' for c in filename)
    filename += ".csv"
    save_path = save_dir / filename
    plot_data_subset.to_csv(save_path, index=False)
    print(f"Saved plot data to {save_path}")

# --- Plotting Helper Functions ---


def _create_plot_grid(
    num_plots: int,
    ncols: int = 3,
    base_width: float = 2,
    base_height: float = 2,
    force_nrows: Optional[int] = None,
    **subplots_kwargs,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray], int, int]:
    """Creates and returns the fig, flat axes, nrows, and ncols."""
    if num_plots == 0:
        print("No dataframes to plot.")
        return None, None, 0, 0

    if force_nrows:
        nrows = force_nrows
    else:
        nrows = math.ceil(num_plots / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(base_width * ncols, base_height * nrows),
        squeeze=False,
        **subplots_kwargs,
    )
    return fig, axes.flatten(), nrows, ncols


def _finalize_plot_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    num_plots: int,
    legend_ncol: Optional[int] = None,
    nrows: Optional[int] = None,
):
    """Applies final touches: removes unused axes, adds legend, tight_layout."""
    # Iterate through the unused axes and remove them from the Figure.
    for i in range(num_plots, len(axes)):
        # axes[i].axis("off") # This line just turns off the box, doesn't remove space
        fig.delaxes(axes[i])  #

    # Get legend from first axis that has one
    handles, labels = [], []
    for ax in axes[:num_plots]:  # <--- MODIFIED: Only check axes that were used
        h, l = ax.get_legend_handles_labels()
        if l:
            handles, labels = h, l
            break

    if labels:
        ncol = len(labels) if legend_ncol is None else legend_ncol
        fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False)

    # Dynamically adjust top margin based on the number of rows
    if nrows == 1:
        # If we only have one row, we need a lot of top margin
        top_margin = 0.90
    elif nrows == 2:
        # A bit less for 2 rows
        top_margin = 0.95
    else:
        # Default for 3+ rows (use 0.95 or your old 0.97)
        top_margin = 0.97

    plt.tight_layout(rect=[0, 0, 1, top_margin])  # Adjust for legend
    return fig


def _set_outer_labels(
    ax: plt.Axes,
    plot_idx: int,
    nrows: int,
    ncols: int,
    xlabel: str = "Iteration",
    ylabel: str = "Avg. Simple Regret",
):
    """Sets x and y labels for plots on the outer grid."""
    if plot_idx % ncols == 0:  # First column
        ax.set_ylabel(ylabel)
    if plot_idx >= (nrows - 1) * ncols:  # Last row
        ax.set_xlabel(xlabel)


# --- Generic Plotting Function ---


def plot_regret_grid(
    df_list: List[pd.DataFrame],
    project_keys: List[str],
    metrics: List[str],
    config_keys: List[str],
    get_config_name: Callable[[pd.Series], str],
    ncols: int = 3,
    xlabel: str = "Iteration",
    ylabel: str = "Avg. Simple Regret",
    force_nrows: Optional[int] = None,
):
    """
    A generic function to plot a grid of regret curves.

    This handles the common pattern of:
    1. Creating a grid.
    2. Looping through (df, project_key).
    3. Aggregating data.
    4. Plotting one metric (mean + stddev) for different configs.
    5. Adding labels, legends, etc.

    Args:
        df_list: List of DataFrames, one per subplot.
        project_keys: List of project keys, one per subplot (for titles).
        metrics: List of metrics to aggregate. *Assumes one metric* for plotting.
        config_keys: List of config keys to group by.
        get_config_name: A function that takes a row from the aggregated DataFrame
                         and returns a string label for the legend.
        ncols: Number of columns in the plot grid.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    num_objectives = len(project_keys)
    fig, axes_flat, nrows, ncols = _create_plot_grid(
        num_objectives,
        ncols=ncols,
        sharex=False,
        sharey=False,
        force_nrows=force_nrows,  # <-- PASS IT HERE
    )
    if fig is None:
        return

    # Assumes we are plotting the first metric in the list
    metric_name = metrics[0]
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    for i, (df, project_key) in enumerate(zip(df_list, project_keys)):
        ax = axes_flat[i]
        aggr_df = aggregate_df(df, config_keys, metrics)

        if aggr_df.empty:
            print(f"Skipping empty aggregated dataframe for {project_key}")
            continue

        unique_configs = aggr_df[config_keys].drop_duplicates().sort_values(by=config_keys).reset_index(drop=True)

        
        for _, row in unique_configs.iterrows():
            config_name = get_config_name(row)

            # [MODIFIED] Get the color from our global map.
            color = MODEL_COLORS.get(config_name)

            # Create the filter dynamically
            plot_data_filter = pd.Series(True, index=aggr_df.index)
            for key in config_keys:
                # Handle potential NaN in config keys from outer join
                if pd.isna(row[key]):
                    plot_data_filter &= aggr_df[key].isna()
                else:
                    plot_data_filter &= aggr_df[key] == row[key]
            plot_data = aggr_df[plot_data_filter]

            # Create a new DataFrame with regret_step, mean, and std
            plot_data_subset = plot_data[["regret_step", mean_col, std_col]].copy()
            plot_data_subset["max"] = plot_data_subset[mean_col] + plot_data_subset[std_col]
            plot_data_subset["min"] = plot_data_subset[mean_col] - plot_data_subset[std_col]
            plot_data_subset = plot_data_subset[["regret_step", mean_col, std_col, "max", "min"]]
            plot_data_subset.columns = ["regret_step", "mean", "std", "max", "min"]

            # Save the plot data subset to a file
            save_dir = Path(os.environ.get("FEDBO_RESULTS_DIR", "results/simulation_results"))
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{project_key}_{config_name}"
            filename = ''.join(c if c.isalnum() or c == '_' else '_' for c in filename)
            filename += ".csv"
            save_path = save_dir / filename
            plot_data_subset.to_csv(save_path, index=False)

            # Plot mean line
            ax.plot(
                plot_data["regret_step"],
                plot_data[mean_col],
                label=config_name,
                linewidth=1.5,
                color=color,  # [MODIFIED] Pass the color
            )
            # Plot std deviation fill
            ax.fill_between(
                plot_data["regret_step"],
                np.clip(
                    plot_data[mean_col] - plot_data[std_col],
                    a_min=0,
                    a_max=None,
                ),
                plot_data[mean_col] + plot_data[std_col],
                alpha=0.2,
                color=color,  # [MODIFIED] Pass the color
            )

        project_name = PROJECT_NAMES.get(project_key, project_key)
        ax.text(
            0.95,
            0.95,
            project_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.minorticks_on()
        _set_outer_labels(ax, i, nrows, ncols, xlabel, ylabel)

    return _finalize_plot_grid(fig, axes_flat, num_objectives, nrows=nrows)


def plot_regret_grid_default_color(
    df_list: List[pd.DataFrame],
    project_keys: List[str],
    metrics: List[str],
    config_keys: List[str],
    get_config_name: Callable[[pd.Series], str],
    ncols: int = 3,
    xlabel: str = "Iteration",
    ylabel: str = "Avg. Simple Regret",
    force_nrows: Optional[int] = None,
):
    """
    A generic function to plot a grid of regret curves.

    This handles the common pattern of:
    1. Creating a grid.
    2. Looping through (df, project_key).
    3. Aggregating data.
    4. Plotting one metric (mean + stddev) for different configs.
    5. Adding labels, legends, etc.

    Args:
        df_list: List of DataFrames, one per subplot.
        project_keys: List of project keys, one per subplot (for titles).
        metrics: List of metrics to aggregate. *Assumes one metric* for plotting.
        config_keys: List of config keys to group by.
        get_config_name: A function that takes a row from the aggregated DataFrame
                         and returns a string label for the legend.
        ncols: Number of columns in the plot grid.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    num_objectives = len(project_keys)
    fig, axes_flat, nrows, ncols = _create_plot_grid(
        num_objectives,
        ncols=ncols,
        sharex=False,
        sharey=False,
        force_nrows=force_nrows,  # <-- PASS IT HERE
    )
    if fig is None:
        return

    # Assumes we are plotting the first metric in the list
    metric_name = metrics[0]
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    for i, (df, project_key) in enumerate(zip(df_list, project_keys)):
        ax = axes_flat[i]
        aggr_df = aggregate_df(df, config_keys, metrics)

        if aggr_df.empty:
            print(f"Skipping empty aggregated dataframe for {project_key}")
            continue

        unique_configs = aggr_df[config_keys].drop_duplicates().sort_values(by=config_keys).reset_index(drop=True)

        for _, row in unique_configs.iterrows():
            config_name = get_config_name(row)

            # [MODIFIED] Get the color from our global map.

            # Create the filter dynamically
            plot_data_filter = pd.Series(True, index=aggr_df.index)
            for key in config_keys:
                # Handle potential NaN in config keys from outer join
                if pd.isna(row[key]):
                    plot_data_filter &= aggr_df[key].isna()
                else:
                    plot_data_filter &= aggr_df[key] == row[key]
            plot_data = aggr_df[plot_data_filter]

            print("SAasdf")
            save_to_csv(plot_data, project_key, config_name, metric_name)

            # Plot mean line
            ax.plot(
                plot_data["regret_step"],
                plot_data[mean_col],
                label=config_name,
                linewidth=1.5,
            )
            # Plot std deviation fill
            ax.fill_between(
                plot_data["regret_step"],
                np.clip(
                    plot_data[mean_col] - plot_data[std_col],
                    a_min=0,
                    a_max=None,
                ),
                plot_data[mean_col] + plot_data[std_col],
                alpha=0.2,
            )

        project_name = PROJECT_NAMES.get(project_key, project_key)
        ax.text(
            0.95,
            0.95,
            project_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.minorticks_on()
        _set_outer_labels(ax, i, nrows, ncols, xlabel, ylabel)

    return _finalize_plot_grid(fig, axes_flat, num_objectives, nrows=nrows)


# --- Data for Mean Rank Plot ---


def get_final_regret_data(
    entity: str,
    projects: List[str],
    metric_to_get: str = "avg/simple_regret",
) -> Dict[str, List[Optional[float]]]:
    """
    Creates a dictionary of models and their final iteration regret values
    across a list of projects.

    Args:
        entity: The wandb entity (user or team name).
        projects: A list of project names to iterate over.
        metric_to_get: The wandb metric key to retrieve (e.g., "avg/simple_regret").

    Returns:
        A dictionary where keys are model short names and values are lists
        of the final metric value for each project. `None` is inserted if
        a model or project has no data.

        Example:
        {
          "FL-MLP": [0.1, 0.05, None, 0.2],
          "L-GP": [0.5, 0.3, 0.2, 0.4]
        }
    """
    # Initialize the result dictionary
    models_to_fetch = list(REVERSED_MODEL_NAMES.keys())
    data_dict: Dict[str, List[Optional[float]]] = {model_name: [] for model_name in models_to_fetch}

    # Define which keys are needed to identify a model
    config_keys = ["trainer._target_", "model._target_"]
    metrics_list = [metric_to_get]
    mean_col_name = f"{metric_to_get}_mean"

    print(f"Processing {len(projects)} projects for {len(models_to_fetch)} models...")

    for project in projects:
        print(f"  Processing project: {project}")

        # 1. Load data for the project
        df = load_wandb_data(entity, project, metrics_list, config_keys)
        if df.empty:
            print(f"    -> No data found for project.")
            # Append None for all models for this project
            for model_name in models_to_fetch:
                data_dict[model_name].append(None)
            continue

        # 2. Aggregate the data
        aggr_df = aggregate_df(df, config_keys, metrics_list)
        if aggr_df.empty or "regret_step" not in aggr_df.columns:
            print(f"    -> No aggregated data found.")
            for model_name in models_to_fetch:
                data_dict[model_name].append(None)
            continue

        # 3. Find the final iteration
        last_step = aggr_df["regret_step"].max()
        last_step_df = aggr_df[aggr_df["regret_step"] == last_step]

        # 4. Create a quick lookup for this project's results
        project_results: Dict[str, float] = {}
        for _, row in last_step_df.iterrows():
            try:
                trainer_target = row["trainer._target_"]
                model_target = row["model._target_"]
                # Convert (trainer, model) targets back to short name
                short_name = MODEL_NAMES[trainer_target][model_target]
                value = row[mean_col_name]
                project_results[short_name] = value
            except KeyError:
                # This model combo isn't in our MODEL_NAMES map
                continue

        # 5. Populate the main dict, inserting None for missing models
        for model_name in models_to_fetch:
            value_to_add = project_results.get(model_name, None)
            data_dict[model_name].append(value_to_add)
            if value_to_add is None:
                print(f"    -> No final value found for {model_name}.")

    print("Processing complete.")
    return data_dict


# --- Specific Plotting Functions ---


def plot_objective_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
    force_nrows: Optional[int] = None,
):
    """Plots avg simple regret vs. iteration for different models."""
    df_list = _filter_df_list(df_list, models_to_plot)  # [NEW]

    metrics = ["avg/simple_regret"]
    config_keys = ["trainer._target_", "model._target_"]

    def get_config_name(row: pd.Series) -> str:
        try:
            return MODEL_NAMES_PLOT[row["trainer._target_"]][row["model._target_"]]
        except KeyError:
            return f"{row['trainer._target_']} / {row['model._target_']}"

    return plot_regret_grid(df_list, projects, metrics, config_keys, get_config_name, force_nrows=force_nrows)


def plot_num_agents_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """Plots avg simple regret vs. iteration for different num_agents."""
    df_list = _filter_df_list(df_list, models_to_plot)

    metrics = ["avg/simple_regret"]
    config_keys = ["num_agents"]

    def get_config_name(row: pd.Series) -> str:
        return f"{int(row['num_agents'])} Agents"
    return plot_regret_grid_default_color(df_list, projects, metrics, config_keys, get_config_name)


def plot_num_layers_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """Plots avg simple regret vs. iteration for different num_hidden_layers."""
    df_list = _filter_df_list(df_list, models_to_plot)  # [NEW]

    metrics = ["avg/simple_regret"]
    config_keys = ["model.num_hidden_layers"]

    def get_config_name(row: pd.Series) -> str:
        return f"{int(row['model.num_hidden_layers'])} Layers"

    return plot_regret_grid_default_color(df_list, projects, metrics, config_keys, get_config_name)


def plot_num_head_layers_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """Plots avg simple regret vs. iteration for different num_hidden_layers."""
    df_list = _filter_df_list(df_list, models_to_plot)  # [NEW]

    metrics = ["avg/simple_regret"]
    config_keys = ["model.num_head_layers"]

    def get_config_name(row: pd.Series) -> str:
        return f"{int(row['model.num_head_layers'])} Head Layers"

    return plot_regret_grid_default_color(df_list, projects, metrics, config_keys, get_config_name)


def plot_num_hidden_features_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """Plots avg simple regret vs. iteration for different hidden_features."""
    df_list = _filter_df_list(df_list, models_to_plot)  # [NEW]

    metrics = ["avg/simple_regret"]
    config_keys = ["trainer._target_", "model._target_", "model.hidden_features"]

    def get_config_name(row: pd.Series) -> str:
        # try:
        #     m_name = MODEL_NAMES_PLOT[row["trainer._target_"]][row["model._target_"]]
        # except KeyError:
        #     m_name = "Unknown Model"
        num_feat = int(row["model.hidden_features"])
        return f"{num_feat} Features"

    return plot_regret_grid_default_color(df_list, projects, metrics, config_keys, get_config_name)


def plot_num_aggregations_data(
    df_list: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """Plots avg simple regret vs. iteration for different num_rounds."""
    df_list = _filter_df_list(df_list, models_to_plot)  # [NEW]

    metrics = ["avg/simple_regret"]
    config_keys = ["trainer.num_rounds"]

    def get_config_name(row: pd.Series) -> str:
        return f"{int(row['trainer.num_rounds'])} Rounds"

    return plot_regret_grid_default_color(df_list, projects, metrics, config_keys, get_config_name)


# --- More Complex Plotting Functions ---


# --- More Complex Plotting Functions (Refactored) ---


def plot_warm_start_data(
    df_list_agent_0: List[pd.DataFrame],
    df_list_agent_10: List[pd.DataFrame],
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
    force_nrows: Optional[int] = None,  # <-- [NEW] Add force_nrows
):
    """
    Plots simple regret for Agent 0 (solid) and Agent 10 (dashed).
    [MODIFIED] This function now draws its own two-line legend.
    """
    # [NEW] Filter both lists
    df_list_agent_0 = _filter_df_list(df_list_agent_0, models_to_plot)
    df_list_agent_10 = _filter_df_list(df_list_agent_10, models_to_plot)

    config_keys = ["trainer._target_", "model._target_"]
    num_objectives = len(df_list_agent_0)

    # [MODIFIED] Pass force_nrows
    fig, axes_flat, nrows, ncols = _create_plot_grid(num_objectives, sharex=False, sharey=False, force_nrows=force_nrows)
    if fig is None:
        return

    zip_data = zip(df_list_agent_0, df_list_agent_10, projects)
    for i, (df_agent_0, df_agent_10, project_key) in enumerate(zip_data):
        ax = axes_flat[i]

        aggr_df_agent_0 = aggregate_df(df_agent_0, config_keys, ["simple_regret/agent_0"])
        aggr_df_agent_10 = aggregate_df(df_agent_10, config_keys, ["simple_regret/agent_10"])

        if aggr_df_agent_0.empty:
            print(f"Skipping empty aggregated dataframe for {project_key} (Agent 0)")
            continue

        unique_configs = aggr_df_agent_0[config_keys].drop_duplicates().sort_values(by=config_keys).reset_index(drop=True)

        for _, row in unique_configs.iterrows():
            config_name = MODEL_NAMES_PLOT[row["trainer._target_"]][row["model._target_"]]

            color = MODEL_COLORS.get(config_name)

            # Filter data for this config
            plot_data_filter_0 = pd.Series(True, index=aggr_df_agent_0.index)
            for key in config_keys:
                plot_data_filter_0 &= aggr_df_agent_0[key] == row[key]
            plot_data_agent_0 = aggr_df_agent_0[plot_data_filter_0]

            plot_data_filter_10 = pd.Series(True, index=aggr_df_agent_10.index)
            for key in config_keys:
                plot_data_filter_10 &= aggr_df_agent_10[key] == row[key]
            plot_data_agent_10 = aggr_df_agent_10[plot_data_filter_10]

            save_to_csv(plot_data_agent_0, project_key, config_name, "simple_regret/agent_0", file_suffix="agent_0")
            save_to_csv(plot_data_agent_10, project_key, config_name, "simple_regret/agent_10", file_suffix="agent_10")

            # --- Plot for Agent 0 ---
            if not plot_data_agent_0.empty:
                ax.plot(
                    plot_data_agent_0["regret_step"],
                    plot_data_agent_0["simple_regret/agent_0_mean"],
                    # [MODIFIED] Label format
                    label=f"Cold-Start: {config_name}",
                    linewidth=1.5,
                    color=color,
                )
                ax.fill_between(
                    plot_data_agent_0["regret_step"],
                    np.clip(
                        plot_data_agent_0["simple_regret/agent_0_mean"] - plot_data_agent_0["simple_regret/agent_0_std"],
                        a_min=0,
                        a_max=None,
                    ),
                    plot_data_agent_0["simple_regret/agent_0_mean"] + plot_data_agent_0["simple_regret/agent_0_std"],
                    alpha=0.2,
                    color=color,
                )

            # --- Plot for Agent 10 ---
            if not plot_data_agent_10.empty:
                ax.plot(
                    plot_data_agent_10["regret_step"],
                    plot_data_agent_10["simple_regret/agent_10_mean"],
                    # [MODIFIED] Label format
                    label=f"Warm-Start: {config_name}",
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                )
                ax.fill_between(
                    plot_data_agent_10["regret_step"],
                    np.clip(
                        plot_data_agent_10["simple_regret/agent_10_mean"] - plot_data_agent_10["simple_regret/agent_10_std"],
                        a_min=0,
                        a_max=None,
                    ),
                    plot_data_agent_10["simple_regret/agent_10_mean"] + plot_data_agent_10["simple_regret/agent_10_std"],
                    alpha=0.2,
                    color=color,
                )

        # Set title and labels
        plot_project_key = project_key.split("-")[0]
        project_name = PROJECT_NAMES.get(plot_project_key, plot_project_key)
        ax.text(
            0.95,
            0.95,
            project_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.minorticks_on()
        _set_outer_labels(ax, i, nrows, ncols)

    # --- [NEW] Manual Legend Creation (replaces _finalize_plot_grid) ---

    # 1. [MODIFIED] Iterate through the unused axes and remove them
    for i in range(num_objectives, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    # 2. [MODIFIED] Collect all unique handles and labels from *used* axes
    all_handles_labels = {}
    for ax in axes_flat[:num_objectives]:  # <-- Only check axes that were plotted on
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_handles_labels:
                all_handles_labels[l] = h

    # 3. Separate into Agent 0 and Agent 10 lists
    agent_0_items = []
    agent_10_items = []
    for label, handle in all_handles_labels.items():
        if label.startswith("Cold-Start:"):
            agent_0_items.append((handle, label))
        elif label.startswith("Warm-Start:"):
            agent_10_items.append((handle, label))

    # 4. Sort them alphabetically by label
    # agent_0_items.sort(key=lambda item: item[1])
    # agent_10_items.sort(key=lambda item: item[1])

    # 5. Unzip them for the legend function
    if agent_0_items:
        agent_0_handles, agent_0_labels = zip(*agent_0_items)
    else:
        agent_0_handles, agent_0_labels = [], []

    if agent_10_items:
        agent_10_handles, agent_10_labels = zip(*agent_10_items)
    else:
        agent_10_handles, agent_10_labels = [], []

    # 6. Plot the two legends
    # We plot them just below the top of the figure (y=1.0)
    # `bbox_to_anchor=(0.5, 0.98)` centers it (x=0.5) and places it high (y=0.98)
    if agent_0_labels:
        fig.legend(
            agent_0_handles,
            agent_0_labels,
            loc="upper center",
            ncol=len(agent_0_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),  # First line
        )

    if agent_10_labels:
        fig.legend(
            agent_10_handles,
            agent_10_labels,
            loc="upper center",
            ncol=len(agent_10_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.93),  # Second line, just below the first
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def plot_manytask_data(df_agents: List[pd.DataFrame], models_to_plot: Optional[List[str]] = None, limit_y: bool = True):
    """Plots simple regret for each agent in a separate subplot."""
    df_agents = _filter_df_list(df_agents, models_to_plot)  # [NEW]

    config_keys = ["trainer._target_", "model._target_"]
    # Hardcoded values from original script
    y_max_vals = [
        1.6,
        500,
        20,
        450,
        20,
        3300,
        700,
        15000,
        20,
        1000,
        20,
        15,
        1000,
        650,
        2,
        15,
        600,
        3500,
    ]
    num_objectives = len(df_agents)

    fig, axes_flat, nrows, ncols = _create_plot_grid(num_objectives, sharex=False, sharey=False)
    if fig is None:
        return

    for i, df_agent in enumerate(df_agents):
        ax = axes_flat[i]
        metric = [f"simple_regret/agent_{i}"]
        mean_col = f"{metric[0]}_mean"
        std_col = f"{metric[0]}_std"

        aggr_df = aggregate_df(df_agent, config_keys, metric)

        if aggr_df.empty:
            print(f"Skipping empty aggregated dataframe for Agent {i}")
            continue

        unique_configs = aggr_df[config_keys].drop_duplicates().sort_values(by=config_keys).reset_index(drop=True)

        for _, row in unique_configs.iterrows():
            config_name = MODEL_NAMES_PLOT[row["trainer._target_"]][row["model._target_"]]

            # [MODIFIED] Get the base color for this model
            color = MODEL_COLORS.get(config_name)

            # Hardcoded filter from original script
            if (i + 1 in [10, 13]) and config_name in [
                "FMTBO",
                "DP-FTS-DE",
            ]:
                continue

            # Filter data for this config
            plot_data_filter = pd.Series(True, index=aggr_df.index)
            for key in config_keys:
                plot_data_filter &= aggr_df[key] == row[key]
            plot_data_agent = aggr_df[plot_data_filter]

            # Plot
            ax.plot(
                plot_data_agent["regret_step"],
                plot_data_agent[mean_col],
                label=config_name,
                linewidth=1.5,
                color=color,  # [MODIFIED]
            )
            ax.fill_between(
                plot_data_agent["regret_step"],
                np.clip(
                    plot_data_agent[mean_col] - plot_data_agent[std_col],
                    a_min=0,
                    a_max=None,
                ),
                plot_data_agent[mean_col] + plot_data_agent[std_col],
                alpha=0.2,
                color=color,  # [MODIFIED]
            )

        # Set title, labels, and limits
        ax.text(
            0.95,
            0.95,
            f"Agent {i+1}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.minorticks_on()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if limit_y and i < len(y_max_vals):
            ax.set_ylim(top=y_max_vals[i])
        _set_outer_labels(ax, i, nrows, ncols)

    return _finalize_plot_grid(fig, axes_flat, num_objectives, nrows=nrows)


# --- Load and Plot Example ---
def plot_wandb_objective_data(
    entity: str,
    projects: List[str],
    models_to_plot: Optional[List[str]] = None,
):
    """
    Main function to load and plot objective data for a list of projects.

    Args:
        entity: The wandb entity (user or team name).
        projects: A list of project names to fetch and plot.
        models_to_plot: Optional list of model short names to filter for.
    """
    metrics = ["avg/simple_regret"]
    config_keys = ["trainer._target_", "model._target_"]

    df_list = [load_wandb_data(entity, p, metrics, config_keys) for p in projects]

    # Filter out any empty dataframes that may have resulted from failed loads
    valid_data = [(df, p) for df, p in zip(df_list, projects) if not df.empty]
    if not valid_data:
        print("No valid data found for any project. Exiting.")
        return

    valid_dfs, valid_projects = zip(*valid_data)
    fig = plot_objective_data(list(valid_dfs), list(valid_projects), models_to_plot=models_to_plot)
    fig.show()