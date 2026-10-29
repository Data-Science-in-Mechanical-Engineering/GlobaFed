import csv
import sys

import numpy as np
import pytest

import fd_emd.main as main_module
from fd_emd.algorithm import FDEMD, FDEMDConfig, _lcb
from fd_emd.main import (
    parse_args,
    print_final_results,
    regret_csv_filename,
    save_agent_regret_csv,
    save_regret_csv,
)
from fd_emd.objectives import create_objectives
from fd_emd.rbfn import RBFNetwork


def test_rbfn_round_trip_preserves_predictions():
    rng = np.random.default_rng(3)
    x = rng.random((20, 2))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2
    model = RBFNetwork.fit(x, y, num_centers=5, rng=rng)
    restored = RBFNetwork.from_vector(model.to_vector(), input_dim=2, num_centers=5)
    np.testing.assert_allclose(restored.predict(x), model.predict(x))


def test_rbfn_vector_contains_raw_paper_parameters():
    model = RBFNetwork(
        centers=np.array([[0.1], [0.8]]),
        widths=np.array([0.5, 2.0]),
        weights=np.array([3.0, -4.0]),
    )
    np.testing.assert_allclose(
        model.to_vector(),
        np.array([0.1, 0.8, 0.5, 2.0, 3.0, -4.0]),
    )


def test_sorted_averaging_orders_all_node_parameters_by_center_metric():
    model = RBFNetwork(
        centers=np.array([[0.8, 0.8], [0.1, 0.2], [0.4, 0.3]]),
        widths=np.array([8.0, 1.0, 4.0]),
        weights=np.array([80.0, 10.0, 40.0]),
    )

    sorted_model = model.sorted_by_center_metric()

    np.testing.assert_allclose(
        sorted_model.centers,
        np.array([[0.1, 0.2], [0.4, 0.3], [0.8, 0.8]]),
    )
    np.testing.assert_allclose(sorted_model.widths, np.array([1.0, 4.0, 8.0]))
    np.testing.assert_allclose(sorted_model.weights, np.array([10.0, 40.0, 80.0]))
    np.testing.assert_allclose(
        sorted_model.predict([[0.2, 0.7]]),
        model.predict([[0.2, 0.7]]),
    )


def test_lcb_matches_paper_equations_9_and_10():
    local = RBFNetwork(np.array([[0.0]]), np.array([1.0]), np.array([0.0]))
    global_model = RBFNetwork(np.array([[0.0]]), np.array([1.0]), np.array([0.0]))
    local = RBFNetwork(local.centers, local.widths, local.weights + 1.0)
    global_model = RBFNetwork(global_model.centers, global_model.widths, global_model.weights + 3.0)
    candidates = np.array([[0.2], [0.8]])

    predictions = np.vstack((local.predict(candidates), global_model.predict(candidates)))
    expected = predictions.mean(axis=0) - 2.0 * np.sqrt(
        np.mean((predictions - predictions.mean(axis=0)) ** 2, axis=0)
    )
    np.testing.assert_allclose(_lcb(local, global_model, candidates, 2.0), expected)


def test_l2_update_recalculates_centers_with_kmeans():
    rng = np.random.default_rng(9)
    x = rng.random((12, 2))
    y = x[:, 0] - x[:, 1]
    model = RBFNetwork.fit(x, y, num_centers=5, rng=rng)
    old_centers = model.centers.copy()
    updated_x = np.vstack((x, np.array([[3.0, 3.0]])))
    updated_y = np.append(y + 0.5, 0.5)
    updated = model.train_l2(
        updated_x,
        updated_y,
        epochs=5,
        learning_rate=0.06,
        rng=np.random.default_rng(10),
        kmeans_iterations=20,
    )

    assert not np.array_equal(updated.centers, old_centers)
    np.testing.assert_allclose(
        np.sum(updated.centers**2, axis=1),
        np.sort(np.sum(updated.centers**2, axis=1)),
    )
    assert not np.array_equal(updated.weights, model.weights)


def test_server_uses_random_global_before_first_round():
    rng = np.random.default_rng(4)
    objectives = create_objectives("sphere", 2, 2, rng)
    optimizer = FDEMD(
        objectives,
        FDEMDConfig(
            rounds=0,
            initial_points=5,
            num_centers=5,
            ensemble_size=2,
            auxiliary_size=10,
            population_size=4,
            generations=1,
            seed=4,
        ),
    )

    assert optimizer.last_diagnostics == {}
    assert all(
        not np.array_equal(optimizer.global_model.to_vector(), client.model.to_vector())
        for client in optimizer.clients
    )


@pytest.mark.parametrize(
    ("task", "dim", "expected_optimum"),
    [
        ("ackley_shifted", 3, None),
        ("branin_shifted", 2, None),
        ("hartmann6d_shifted", 6, None),
        ("cartpole", 5, -500.0),
    ],
)
def test_benchmark_tasks_are_available(task, dim, expected_optimum):
    objectives = create_objectives(task, dim, 2, np.random.default_rng(12))

    assert [objective.name for objective in objectives] == [f"{task}_0", f"{task}_1"]
    assert all(objective.dim == dim for objective in objectives)
    assert all(np.all(objective.lower < objective.upper) for objective in objectives)
    if expected_optimum is not None:
        assert all(objective.optimum == expected_optimum for objective in objectives)


@pytest.mark.parametrize(
    ("task", "dim", "required_dim"),
    [
        ("branin_shifted", 3, 2),
        ("hartmann6d_shifted", 5, 6),
        ("cartpole", 4, 5),
    ],
)
def test_fixed_dimension_benchmark_tasks_validate_dim(task, dim, required_dim):
    with pytest.raises(ValueError, match=rf"requires dim={required_dim}"):
        create_objectives(task, dim, 1, np.random.default_rng(13))


def test_shifted_benchmark_tasks_are_seeded_and_finite():
    first = create_objectives("ackley_shifted", 3, 2, np.random.default_rng(14))
    second = create_objectives("ackley_shifted", 3, 2, np.random.default_rng(14))
    point = np.full(3, 0.5)

    np.testing.assert_allclose(
        [objective(point) for objective in first],
        [objective(point) for objective in second],
    )
    assert first[0].optimum != first[1].optimum


def test_shifted_ackley_keeps_bounds_and_moves_optimum():
    expected_rng = np.random.default_rng(25)
    objectives = create_objectives(
        "ackley_shifted", 3, 2, np.random.default_rng(25)
    )

    for objective in objectives:
        x_shift = expected_rng.uniform(-3.0, 3.0)
        expected_rng.uniform(-3.0, 3.0)  # y shift
        np.testing.assert_allclose(objective.lower, -5.0)
        np.testing.assert_allclose(objective.upper, 10.0)
        np.testing.assert_allclose(objective.upper - objective.lower, 15.0)
        np.testing.assert_allclose(
            objective(np.full(3, (-x_shift + 5.0) / 15.0)),
            objective.optimum,
            atol=1e-12,
        )


def test_shifted_hartmann_keeps_bounds_and_moves_landscape():
    objectives = create_objectives(
        "hartmann6d_shifted", 6, 2, np.random.default_rng(26)
    )
    point = np.linspace(0.1, 0.9, 6)

    for objective in objectives:
        np.testing.assert_allclose(objective.lower, 0.0)
        np.testing.assert_allclose(objective.upper, 1.0)
        np.testing.assert_allclose(objective.upper - objective.lower, 1.0)
    centered_values = [
        objective(point) - objective.optimum for objective in objectives
    ]
    assert not np.isclose(
        centered_values[0],
        centered_values[1],
        atol=1e-12,
    )


def test_shifted_branin_keeps_bounds_and_moves_landscape():
    objectives = create_objectives(
        "branin_shifted", 2, 2, np.random.default_rng(27)
    )
    point = np.array([0.25, 0.75])

    for objective in objectives:
        np.testing.assert_allclose(objective.lower, 0.0)
        np.testing.assert_allclose(objective.upper, 10.0)
        np.testing.assert_allclose(objective.upper - objective.lower, 10.0)
    centered_values = [
        objective(point) - objective.optimum for objective in objectives
    ]
    assert not np.isclose(
        centered_values[0],
        centered_values[1],
        atol=1e-12,
    )


def test_combination_benchmark_task_is_finite():
    objective = create_objectives("combination", 1, 1, np.random.default_rng(15))[0]

    assert objective.name == "combination_0"
    assert np.isfinite(objective(np.array([0.5])))
    assert np.isfinite(objective.optimum)


def test_fd_emd_smoke_run_adds_one_point_per_round():
    rng = np.random.default_rng(5)
    objectives = create_objectives("sphere", 2, 3, rng)
    config = FDEMDConfig(
        rounds=2,
        initial_points=5,
        num_centers=5,
        ensemble_size=3,
        auxiliary_size=20,
        population_size=8,
        generations=2,
        kmeans_iterations=5,
        seed=5,
    )
    clients = FDEMD(objectives, config).run()
    assert all(len(client.x) == 7 for client in clients)
    assert all(np.isfinite(client.y).all() for client in clients)


def test_warm_started_client_joins_after_warm_start_round(tmp_path):
    objectives = create_objectives("sphere", 2, 3, np.random.default_rng(5))
    config = FDEMDConfig(
        rounds=3,
        initial_points=5,
        num_centers=5,
        ensemble_size=3,
        auxiliary_size=20,
        population_size=8,
        generations=2,
        kmeans_iterations=5,
        seed=5,
        warm_start_round=2,
    )
    clients = FDEMD(objectives, config).run()
    assert [len(client.x) for client in clients] == [8, 8, 6]

    path = save_agent_regret_csv(
        clients, task="sphere", dim=2, rounds=3, initial_points=5, seed=5, output_dir=tmp_path
    )
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert path.name == "sphere2d_FD_EMD_warm_start_5.csv"
    warm_rows = [row for row in rows if row["warm_start"] == "1"]
    assert [row["regret_step"] for row in warm_rows] == ["2"]
    assert {row["agent"] for row in warm_rows} == {"2"}
    assert len(rows) == 7


def test_parse_args_can_disable_wandb_entirely():
    args = parse_args(["--no-wandb"])

    assert args.no_wandb is True


def test_parse_args_accepts_csv_output_directory(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path)])

    assert args.output_dir == tmp_path


def test_main_does_not_import_wandb_logger_when_disabled(monkeypatch, tmp_path):
    output_dir = tmp_path / "fd-emd-results"
    args = parse_args([
        "--no-wandb",
        "--num-runs",
        "1",
        "--output-dir",
        str(output_dir),
    ])
    run_loggers = []
    saved_output_dirs = []

    class FakeFDEMD:
        def __init__(self, objectives, config):
            pass

        def run(self, logger=None):
            run_loggers.append(logger)
            return []

    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(main_module, "create_objectives", lambda *args, **kwargs: [])
    monkeypatch.setattr(main_module, "FDEMD", FakeFDEMD)
    monkeypatch.setattr(main_module, "print_final_results", lambda runs: None)
    monkeypatch.setattr(
        main_module,
        "save_regret_csv",
        lambda *args, **kwargs: saved_output_dirs.append(kwargs["output_dir"]),
    )
    monkeypatch.setitem(sys.modules, "fd_emd.wandb_logging", None)

    main_module.main()

    assert run_loggers == [None]
    assert saved_output_dirs == [output_dir]


def test_print_final_results_reports_each_task(capsys):
    rng = np.random.default_rng(7)
    objectives = create_objectives("sphere", 2, 2, rng)
    config = FDEMDConfig(
        rounds=0,
        initial_points=3,
        num_centers=3,
        ensemble_size=2,
        auxiliary_size=5,
        population_size=4,
        kmeans_iterations=2,
        seed=7,
    )
    clients = FDEMD(objectives, config).run()

    print_final_results([clients, clients])

    output = capsys.readouterr().out
    assert "Final performance per task over 2 seed(s)" in output
    assert "sphere_0" in output
    assert "sphere_1" in output
    assert "Mean final simple regret across tasks and seeds" in output


@pytest.mark.parametrize(
    ("task", "dim", "filename"),
    [
        ("ackley", 10, "ackley10d_FD_EMD.csv"),
        ("combination", 5, "combination5d_FD_EMD.csv"),
        ("cartpole", 5, "cartpole_FD_EMD.csv"),
        ("branin_shifted", 2, "branin_shifted_FD_EMD.csv"),
        ("hartmann6d_shifted", 6, "hartmann6d_shifted_FD_EMD.csv"),
    ],
)
def test_regret_csv_filename(task, dim, filename):
    assert regret_csv_filename(task, dim) == filename


def test_regret_csv_filename_appends_seed():
    assert regret_csv_filename("ackley", 10, 7) == "ackley10d_FD_EMD_7.csv"


def test_save_regret_csv_aggregates_mean_client_simple_regret(tmp_path):
    rng = np.random.default_rng(16)
    objectives = create_objectives(
        "sphere",
        2,
        2,
        rng,
        heterogeneous_shifts=False,
    )
    config = FDEMDConfig(
        rounds=2,
        initial_points=3,
        num_centers=3,
        ensemble_size=2,
        auxiliary_size=5,
        population_size=4,
        generations=1,
        kmeans_iterations=2,
        seed=16,
    )
    clients = FDEMD(objectives, config).run()

    output_path = save_regret_csv(
        [clients, clients],
        task="sphere",
        dim=2,
        initial_points=config.initial_points,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "sphere2d_FD_EMD.csv"
    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert list(rows[0]) == ["regret_step", "mean", "std", "max", "min"]
    assert [int(row["regret_step"]) for row in rows] == [0, 1]
    expected = [
        np.mean(
            [
                np.min(client.y[: config.initial_points + step + 1])
                - client.objective.optimum
                for client in clients
            ]
        )
        for step in range(config.rounds)
    ]
    np.testing.assert_allclose([float(row["mean"]) for row in rows], expected)
    np.testing.assert_allclose([float(row["std"]) for row in rows], 0.0)
    np.testing.assert_allclose([float(row["max"]) for row in rows], expected)
    np.testing.assert_allclose([float(row["min"]) for row in rows], expected)


def test_save_regret_csv_preserves_each_regret_step(tmp_path):
    clients = FDEMD(
        create_objectives(
            "sphere",
            2,
            2,
            np.random.default_rng(17),
            heterogeneous_shifts=False,
        ),
        FDEMDConfig(
            rounds=0,
            initial_points=2,
            num_centers=3,
            ensemble_size=2,
            auxiliary_size=5,
            population_size=4,
            generations=1,
            kmeans_iterations=2,
            seed=17,
        ),
    ).run()
    clients[0].y = np.array([9.0, 8.0, 6.0, 7.0, 3.0])
    clients[1].y = np.array([10.0, 9.0, 8.0, 5.0, 7.0])

    output_path = save_regret_csv(
        [clients],
        task="sphere",
        dim=2,
        initial_points=2,
        output_dir=tmp_path,
    )

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [int(row["regret_step"]) for row in rows] == [0, 1, 2]
    np.testing.assert_allclose(
        [float(row["mean"]) for row in rows],
        [7.0, 5.5, 4.0],
    )
