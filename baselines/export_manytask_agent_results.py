#!/usr/bin/env python3
"""Export agent-specific manytask results from completed seed logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


BASELINES_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIRS = {
    "FD-EMD": BASELINES_DIR / "fd_emd" / "logs",
    "IAF-FBO": BASELINES_DIR / "iaf_fbo" / "logs",
}
DEFAULT_OUTPUT_PATH = BASELINES_DIR / "manytask_agent_results.txt"
SEED_LOG_FILENAME = re.compile(r"^run_multitask_seed(?P<seed>\d+)_\d+\.out$")
RESULT_ROW = re.compile(
    r"^\s*(?P<client>\d+)\s+"
    r"(?P<task>manytask_\S+)\s+"
    r"(?P<best>\S+)\s+"
    r"(?P<regret>\S+)\s+"
    r"(?P<std>\S+)\s*$"
)
SEPARATOR_WIDTH = 110


@dataclass(frozen=True)
class SeedAgentResult:
    client_id: int
    task: str
    best_objective: float
    simple_regret: float


@dataclass(frozen=True)
class AgentSummary:
    client_id: int
    task: str
    mean_best_objective: float
    mean_simple_regret: float
    regret_std: float


def read_seed_agent_results(path: Path) -> list[SeedAgentResult]:
    """Read the final per-agent table printed by one manytask seed run."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(
            index
            for index, line in enumerate(lines)
            if "Client" in line
            and "Task" in line
            and "Mean best objective" in line
            and "Mean simple regret" in line
        )
    except StopIteration as error:
        raise ValueError(f"No final per-agent results table found in {path}.") from error

    results = []
    for line in lines[header_index + 1 :]:
        match = RESULT_ROW.match(line)
        if match is None:
            if results and line.strip().startswith("-"):
                break
            continue
        try:
            results.append(
                SeedAgentResult(
                    client_id=int(match.group("client")),
                    task=match.group("task"),
                    best_objective=float(match.group("best")),
                    simple_regret=float(match.group("regret")),
                )
            )
        except ValueError as error:
            raise ValueError(f"Non-numeric result in {path}: {line}") from error

    if not results:
        raise ValueError(f"No agent rows found in the final results table in {path}.")
    return results


def summarize_manytask_logs(log_dir: Path) -> tuple[list[AgentSummary], int]:
    """Aggregate each manytask agent's final performance across seed logs."""
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    seed_runs: dict[int, list[SeedAgentResult]] = {}
    for path in sorted(log_dir.glob("run_multitask_seed*.out")):
        match = SEED_LOG_FILENAME.match(path.name)
        if match is None:
            continue
        seed = int(match.group("seed"))
        if seed in seed_runs:
            raise ValueError(f"Multiple manytask logs found for seed {seed} in {log_dir}.")
        seed_runs[seed] = read_seed_agent_results(path)

    if not seed_runs:
        raise FileNotFoundError(f"No completed manytask seed logs found in {log_dir}.")

    ordered_runs = [seed_runs[seed] for seed in sorted(seed_runs)]
    reference = ordered_runs[0]
    reference_layout = [(result.client_id, result.task) for result in reference]
    for seed, run in zip(sorted(seed_runs), ordered_runs):
        layout = [(result.client_id, result.task) for result in run]
        if layout != reference_layout:
            raise ValueError(
                f"Agent ordering in seed {seed} does not match the other runs in "
                f"{log_dir}."
            )

    summaries = []
    for result_index, reference_result in enumerate(reference):
        best_values = [run[result_index].best_objective for run in ordered_runs]
        regrets = [run[result_index].simple_regret for run in ordered_runs]
        summaries.append(
            AgentSummary(
                client_id=reference_result.client_id,
                task=reference_result.task,
                mean_best_objective=fmean(best_values),
                mean_simple_regret=fmean(regrets),
                regret_std=pstdev(regrets),
            )
        )
    return summaries, len(ordered_runs)


def format_method_results(
    method: str, summaries: list[AgentSummary], num_seeds: int
) -> str:
    """Format one method using the table layout from the baseline main scripts."""
    if not summaries:
        raise ValueError("At least one agent summary is required.")

    separator = "-" * SEPARATOR_WIDTH
    lines = [
        method,
        f"Final performance per task over {num_seeds} seed(s)",
        separator,
        (
            f"{'Client':>6}  {'Task':<38}  {'Mean best objective':>20}"
            f"  {'Mean simple regret':>20}  {'Regret std':>14}"
        ),
        separator,
    ]
    for summary in summaries:
        lines.append(
            f"{summary.client_id:>6}  {summary.task:<38}"
            f"  {summary.mean_best_objective:>20.10g}"
            f"  {summary.mean_simple_regret:>20.10g}"
            f"  {summary.regret_std:>14.10g}"
        )
    lines.append(separator)
    overall_mean = fmean(summary.mean_simple_regret for summary in summaries)
    lines.append(
        f"{'Mean final simple regret across tasks and seeds':<88}"
        f"  {overall_mean:>20.10g}"
    )
    return "\n".join(lines)


def export_manytask_agent_results(
    log_dirs: dict[str, Path], output_path: Path = DEFAULT_OUTPUT_PATH
) -> Path:
    """Aggregate and export agent-specific manytask results for each method."""
    tables = []
    for method, log_dir in log_dirs.items():
        summaries, num_seeds = summarize_manytask_logs(log_dir)
        tables.append(format_method_results(method, summaries, num_seeds))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the agent-specific final manytask tables printed by the "
            "FD-EMD and IAF-FBO seed jobs."
        )
    )
    parser.add_argument(
        "--fd-emd-log-dir",
        type=Path,
        default=DEFAULT_LOG_DIRS["FD-EMD"],
        help="Directory containing FD-EMD manytask seed .out files.",
    )
    parser.add_argument(
        "--iaf-fbo-log-dir",
        type=Path,
        default=DEFAULT_LOG_DIRS["IAF-FBO"],
        help="Directory containing IAF-FBO manytask seed .out files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output text file (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_manytask_agent_results(
        {
            "FD-EMD": args.fd_emd_log_dir,
            "IAF-FBO": args.iaf_fbo_log_dir,
        },
        args.output,
    )
    print(f"Saved manytask agent-results table to {output_path}")


if __name__ == "__main__":
    main()
