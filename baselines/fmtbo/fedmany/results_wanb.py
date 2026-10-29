import os
from pathlib import Path
import pandas as pd
import wandb
import numpy as np

# paths = [Path("results_fbo_combination_dim1_EI_w_kt")]
# paths = [Path("results_fbo_manytask_dim10_EI_w_kt")]
# paths = [Path("results_fbo_hartmann6d_shifted_dim6_EI_w_kt")]
# paths = [Path("results_fbo_combination_dim10_EI_w_kt_warm_start")]
# paths = [Path("results_fbo_cartpole_dim5_EI_w")]
paths = [Path("results_fbo_manytask_dim50_EI_w_kt")]
wandb_project = os.environ.get("WANDB_PROJECT", "manytask50d")
dim = 50
warm_start = False

conf_dim = dim
dim = dim if dim % 2 == 0 else dim + 1


def wandb_init(dir: Path, i: int):
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"), project=wandb_project,
        name=f"{dir.name}-Run-{i}", mode=os.environ.get("WANDB_MODE", "disabled"),
    )
    conf = {
        "params": f"{dir.name}",
        "trainer": {
            "_target_": "fedmany_trainer"
        },
        "model": {
            "_target_": "fedmany_gp_EI_w_kt"
        },
        "dim": conf_dim
    }
    run.config.update(conf)
    wandb.define_metric("regret_step")
    # Agent specific
    wandb.define_metric("instantaneous_regret/*", step_metric="regret_step")
    wandb.define_metric("simple_regret/*", step_metric="regret_step")
    wandb.define_metric("cumulative_regret/*", step_metric="regret_step")
    # Overall Metrics
    # Instananeous
    wandb.define_metric("avg/instantaneous_regret", step_metric="regret_step")
    wandb.define_metric("sum/instantaneous_regret", step_metric="regret_step")
    wandb.define_metric("worst/instantaneous_regret", step_metric="regret_step")
    wandb.define_metric("best/instantaneous_regret", step_metric="regret_step")
    # Simple
    wandb.define_metric("avg/simple_regret", step_metric="regret_step")
    wandb.define_metric("sum/simple_regret", step_metric="regret_step")
    wandb.define_metric("worst/simple_regret", step_metric="regret_step")
    wandb.define_metric("best/simple_regret", step_metric="regret_step")
    # Cumulative
    wandb.define_metric("avg/cumulative_regret", step_metric="regret_step")
    wandb.define_metric("sum/cumulative_regret", step_metric="regret_step")
    wandb.define_metric("worst/cumulative_regret", step_metric="regret_step")
    wandb.define_metric("best/cumulative_regret", step_metric="regret_step")
    return run


for path in paths:
    for i, subdir in enumerate(path.iterdir()):
        if not subdir.is_dir():
            continue
        files = sorted([f for f in subdir.iterdir()])
        values_df = pd.read_csv(files[0])
        best_y_df = pd.read_csv(files[1])

        value_cols = [col for col in values_df.columns if col.startswith(
            "client") or col.startswith("agent")]

        instantaneous_regret = pd.DataFrame({
            client: values_df[client] - best_y_df[f"{client}_best_y"].iloc[0]
            for client in value_cols
        })

        # --- Simple regret ---
        # Minimization → min so far - optimal
        simple_regret = pd.DataFrame({
            client: values_df[client].cummin() - best_y_df[f"{client}_best_y"].iloc[0]
            for client in value_cols
        })

        # --- Cumulative regret ---
        cumulative_regret = instantaneous_regret.cumsum()

        run = wandb_init(path, i)

        regret_step = 0
        iteration_regrets = {
            metric: {i: [] for i in range(len(values_df) - dim)}
            for metric in ["instantaneous_regret", "simple_regret", "cumulative_regret"]
        }

        step = 0
        for t in range(dim, len(values_df)):
            for j, client in enumerate(value_cols):
                log_data = {
                    f"instantaneous_regret/agent_{j}": instantaneous_regret[client].iloc[t],
                    f"simple_regret/agent_{j}": simple_regret[client].iloc[t],
                    f"cumulative_regret/agent_{j}": cumulative_regret[client].iloc[t],
                    "regret_step": step
                }
                run.log(log_data)

                iteration_regrets["instantaneous_regret"][step].append(
                    instantaneous_regret[client].iloc[t])
                iteration_regrets["simple_regret"][step].append(
                    simple_regret[client].iloc[t])
                iteration_regrets["cumulative_regret"][step].append(
                    cumulative_regret[client].iloc[t])
            step += 1

        for metric_key, step_dict in iteration_regrets.items():
            for step_key, values in step_dict.items():
                values_arr = np.array(values)
                log_dict = {
                    f"avg/{metric_key}": values_arr.mean(),
                    f"sum/{metric_key}": values_arr.sum(),
                    f"worst/{metric_key}": values_arr.max(),
                    f"best/{metric_key}": values_arr.min(),
                    "regret_step": step_key,
                }
                run.log(log_dict)

        if warm_start:
            warmstart_df = pd.read_csv(files[2])
            value_cols = [
                col for col in warmstart_df.columns if col.startswith("agent")]

            instantaneous_regret = pd.DataFrame({
                client: warmstart_df[client] - best_y_df[f"{client}_best_y"].iloc[0]
                for client in value_cols
            })
            # --- Simple regret ---
            # Minimization → min so far - optimal
            simple_regret = pd.DataFrame({
                client: warmstart_df[client].cummin(
                ) - best_y_df[f"{client}_best_y"].iloc[0]
                for client in value_cols
            })
            # --- Cumulative regret ---
            cumulative_regret = instantaneous_regret.cumsum()

            step = 25
            for t in range(dim, len(warmstart_df)):
                for client in value_cols:
                    log_data = {
                        f"instantaneous_regret/{client}": instantaneous_regret[client].iloc[t],
                        f"simple_regret/{client}": simple_regret[client].iloc[t],
                        f"cumulative_regret/{client}": cumulative_regret[client].iloc[t],
                        "regret_step": step
                    }
                    run.log(log_data)
                step += 1

        run.finish()
