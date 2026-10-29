import os
from pathlib import Path
import pandas as pd
import wandb
import numpy as np
import pickle

path = Path("results_dp_fts_de_manytask_50d")
wandb_project = os.environ.get("WANDB_PROJECT", "manytask50d")
conf_dim = 50
warm_start = False

# Warm start
# path = Path("results_dp_fts_de_cartpole_5d_warm_start")
# wandb_project = "cartpole-warm_start"
# conf_dim = 5
# warm_start = True


def wandb_init(dir: Path, i: int):
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"), project=wandb_project,
        name=f"{dir.name}-Run-{i}", mode=os.environ.get("WANDB_MODE", "disabled"),
    )
    conf = {
        "params": f"{dir.name}",
        "trainer": {
            "_target_": "fts_de_trainer"
        },
        "model": {
            "_target_": "fts_de_gp"
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


for i, subdir in enumerate(path.iterdir()):
    if not subdir.is_dir():
        continue

    agent_files = []
    for f in subdir.iterdir():
        if f.suffix == ".p":
            agent_files.append(f)
        elif f.suffix == ".csv":
            best_y_df = pd.read_csv(f)

    agent_files.sort(key=lambda p: int(p.stem.split("_")[1]))
    agent_data = [-1 * np.array(pickle.load(open(agent_f, "rb"))["all"]["values"])
                  for agent_f in agent_files]

    if warm_start:
        warmstart_data = agent_data.pop()

    values_df = pd.DataFrame(
        {f"agent_{i}": agent_data[i] for i in range(len(agent_data))})

    value_cols = [col for col in values_df.columns if col.startswith("agent")]

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
        metric: {i: [] for i in range(len(values_df))}
        for metric in ["instantaneous_regret", "simple_regret", "cumulative_regret"]
    }

    step = 0
    for t in range(len(values_df)):
        for client in value_cols:
            log_data = {
                f"instantaneous_regret/{client}": instantaneous_regret[client].iloc[t],
                f"simple_regret/{client}": simple_regret[client].iloc[t],
                f"cumulative_regret/{client}": cumulative_regret[client].iloc[t],
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
        warmstart_df = pd.DataFrame({f"agent_{len(agent_files)-1}": warmstart_data})

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
        for t in range(len(warmstart_df)):
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
