import csv
import importlib

import numpy as np
from torch import nn

from iaf_fbo.algorithm import (
    IAFFBO,
    IAFFBOConfig,
    acquisition_scores,
    competitive_swarm_optimize,
    pairwise_dataset,
)
from iaf_fbo.classifier import PairwiseClassifier, average_classifiers
from iaf_fbo.main import (
    parse_args,
    regret_csv_filename,
    save_agent_regret_csv,
    save_regret_csv,
)
from iaf_fbo.objectives import create_objectives


def test_classifier_is_a_pytorch_module_and_fit_updates_its_parameters():
    rng = np.random.default_rng(0)
    classifier = PairwiseClassifier.random(1, rng)
    pairs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float64,
    )
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    initial_parameters = classifier.to_vector().copy()

    losses = classifier.fit(
        pairs,
        labels,
        epochs=3,
        learning_rate=1e-2,
        batch_size=2,
        rng=rng,
    )

    assert isinstance(classifier, nn.Module)
    assert len(losses) == 3
    assert np.isfinite(losses).all()
    assert not np.array_equal(classifier.to_vector(), initial_parameters)


def test_classifier_vector_round_trip_preserves_predictions():
    rng = np.random.default_rng(1)
    classifier = PairwiseClassifier.random(3, rng)
    pairs = rng.random((8, 6))

    restored = PairwiseClassifier.from_vector(classifier.to_vector(), 3)

    np.testing.assert_allclose(
        restored.predict_proba(pairs),
        classifier.predict_proba(pairs),
    )


def test_pairwise_dataset_contains_balanced_reverse_comparisons():
    points = np.array([[0.0], [0.5], [1.0]])
    scores = np.array([1.0, 3.0, 2.0])

    pair_inputs, labels = pairwise_dataset(
        points, scores, np.random.default_rng(2)
    )

    assert pair_inputs.shape == (6, 2)
    assert labels.sum() == 3
    comparisons = {
        tuple(pair): label for pair, label in zip(pair_inputs.tolist(), labels)
    }
    for (first, second), label in comparisons.items():
        assert comparisons[(second, first)] == 1.0 - label


def test_pairwise_dataset_keeps_ties_and_labels_both_orders_zero():
    points = np.array([[0.0], [0.5], [1.0]])
    scores = np.array([1.0, 1.0, 2.0])

    pair_inputs, labels = pairwise_dataset(
        points, scores, np.random.default_rng(23)
    )

    comparisons = {
        tuple(pair): label for pair, label in zip(pair_inputs.tolist(), labels)
    }
    assert comparisons[(0.0, 0.5)] == 0.0
    assert comparisons[(0.5, 0.0)] == 0.0
    assert pair_inputs.shape == (6, 2)


class FixedPosterior:
    def predict(self, candidates, return_std):
        assert return_std is True
        return np.array([1.0, 3.0]), np.array([0.5, 0.25])


def test_acquisition_scores_are_utilities_for_minimization():
    candidates = np.zeros((2, 1))
    gp = FixedPosterior()

    ucb = acquisition_scores(
        gp, candidates, kind="ucb", beta=2.0, best_value=0.0
    )
    lcb = acquisition_scores(
        gp, candidates, kind="lcb", beta=2.0, best_value=0.0
    )
    ei = acquisition_scores(
        gp, candidates, kind="ei", beta=2.0, best_value=0.0
    )

    # CSO maximizes classifier preferences, so confidence bounds for the
    # minimized objective are converted into utilities. UCB is applied to -f,
    # while LCB on f is negated.
    np.testing.assert_allclose(ucb, [0.0, -2.5])
    np.testing.assert_allclose(lcb, [0.0, -2.5])
    improvement = -np.array([1.0, 3.0])
    std = np.array([0.5, 0.25])
    z = improvement / std
    from scipy.stats import norm

    np.testing.assert_allclose(
        ei, improvement * norm.cdf(z) + std * norm.pdf(z)
    )
    np.testing.assert_allclose(ucb, lcb)


def test_pairwise_subsampling_keeps_reverse_pairs_and_balance():
    points = np.arange(8, dtype=np.float64).reshape(-1, 1)
    scores = points[:, 0] ** 2

    pair_inputs, labels = pairwise_dataset(
        points,
        scores,
        np.random.default_rng(22),
        max_examples=10,
    )

    assert pair_inputs.shape == (10, 2)
    assert labels.sum() == 5
    comparisons = {
        tuple(pair): label for pair, label in zip(pair_inputs.tolist(), labels)
    }
    for (first, second), label in comparisons.items():
        assert comparisons[(second, first)] == 1.0 - label


def test_group_aggregation_is_elementwise_arithmetic_mean():
    first = PairwiseClassifier.random(2, np.random.default_rng(3))
    second = PairwiseClassifier.random(2, np.random.default_rng(4))

    averaged = average_classifiers([first, second], dim=2)

    np.testing.assert_allclose(
        averaged.to_vector(),
        (first.to_vector() + second.to_vector()) / 2.0,
    )


class FirstCoordinateComparator:
    def prefers(self, first, second):
        return first[:, 0] >= second[:, 0]


def test_cso_uses_classifier_and_stays_in_normalized_domain():
    proposal = competitive_swarm_optimize(
        FirstCoordinateComparator(),
        2,
        population_size=20,
        generations=15,
        phi=0.1,
        rng=np.random.default_rng(5),
    )

    assert proposal.shape == (2,)
    assert np.all((0.0 <= proposal) & (proposal <= 1.0))
    assert proposal[0] > 0.8


def test_smoke_run_adds_one_private_evaluation_per_client_and_round():
    objectives = create_objectives(
        "sphere",
        2,
        3,
        np.random.default_rng(6),
        heterogeneous_shifts=False,
    )
    config = IAFFBOConfig(
        rounds=2,
        initial_points=4,
        num_groups=2,
        server_sample_size=3,
        classifier_epochs=2,
        classifier_batch_size=16,
        population_size=8,
        generations=2,
        gp_restarts=0,
        seed=6,
    )

    optimizer = IAFFBO(objectives, config)
    clients = optimizer.run()

    assert all(len(client.x) == 6 for client in clients)
    assert all(np.isfinite(client.y).all() for client in clients)
    assert set(client.group_id for client in clients) <= {0, 1}
    assert len(optimizer.group_models) == 2


def test_warm_started_client_joins_after_warm_start_round(tmp_path):
    objectives = create_objectives(
        "sphere",
        2,
        3,
        np.random.default_rng(6),
        heterogeneous_shifts=False,
    )
    config = IAFFBOConfig(
        rounds=3,
        initial_points=4,
        num_groups=2,
        server_sample_size=3,
        classifier_epochs=2,
        classifier_batch_size=16,
        population_size=8,
        generations=2,
        seed=6,
        warm_start_round=2,
    )

    clients = IAFFBO(objectives, config).run()

    assert [len(client.x) for client in clients] == [7, 7, 5]
    path = save_agent_regret_csv(
        clients, task="sphere", dim=2, rounds=3, initial_points=4, seed=6, output_dir=tmp_path
    )
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert path.name == "sphere2d_IAF_FBO_warm_start_6.csv"
    warm_rows = [row for row in rows if row["warm_start"] == "1"]
    assert [(row["agent"], row["regret_step"]) for row in warm_rows] == [("2", "2")]
    assert len(rows) == 7


def test_aggregated_model_initializes_the_next_local_update():
    objectives = create_objectives(
        "sphere", 1, 2, np.random.default_rng(7), heterogeneous_shifts=False
    )
    optimizer = IAFFBO(
        objectives,
        IAFFBOConfig(
            rounds=0,
            initial_points=3,
            num_groups=1,
            server_sample_size=2,
            classifier_epochs=0,
            population_size=4,
            generations=0,
            seed=7,
        ),
    )
    optimizer._cluster_and_aggregate(0)
    expected = optimizer.group_models[0].to_vector().copy()

    optimizer._train_client(optimizer.clients[0])

    np.testing.assert_allclose(optimizer.clients[0].classifier.to_vector(), expected)


def test_gp_uses_untransformed_targets_and_isotropic_rbf():
    optimizer = IAFFBO(
        create_objectives(
            "sphere", 2, 1, np.random.default_rng(24), heterogeneous_shifts=False
        ),
        IAFFBOConfig(
            rounds=0,
            initial_points=4,
            server_sample_size=2,
            population_size=4,
            generations=0,
            seed=24,
        ),
    )
    client = optimizer.clients[0]

    gp = optimizer._fit_gp(client)

    np.testing.assert_allclose(gp.y_train_, client.y)
    assert gp.normalize_y is False
    assert np.ndim(gp.kernel_.k2.length_scale) == 0


def test_clustering_handles_identical_client_vectors():
    optimizer = IAFFBO(
        create_objectives(
            "sphere", 1, 3, np.random.default_rng(9), heterogeneous_shifts=False
        ),
        IAFFBOConfig(
            rounds=0,
            initial_points=3,
            num_groups=3,
            server_sample_size=2,
            population_size=4,
            generations=0,
            seed=9,
        ),
    )
    shared = optimizer.clients[0].classifier.copy()
    for client in optimizer.clients:
        client.classifier = shared.copy()

    optimizer._cluster_and_aggregate(0)

    assert set(optimizer.group_models) == {0}
    assert all(client.group_id == 0 for client in optimizer.clients)


def test_cli_and_csv_follow_baseline_conventions(tmp_path):
    assert parse_args(["--no-wandb"]).no_wandb is True
    assert parse_args(["--output-dir", str(tmp_path)]).output_dir == tmp_path
    assert regret_csv_filename("ackley", 10) == "ackley10d_IAF_FBO.csv"
    assert regret_csv_filename("ackley", 10, 3) == "ackley10d_IAF_FBO_3.csv"
    assert regret_csv_filename("cartpole", 5) == "cartpole_IAF_FBO.csv"

    clients = IAFFBO(
        create_objectives(
            "sphere", 1, 2, np.random.default_rng(8), heterogeneous_shifts=False
        ),
        IAFFBOConfig(
            rounds=1,
            initial_points=3,
            num_groups=1,
            server_sample_size=2,
            classifier_epochs=1,
            population_size=4,
            generations=1,
            seed=8,
        ),
    ).run()
    path = save_regret_csv(
        [clients],
        "sphere",
        1,
        initial_points=3,
        output_dir=tmp_path,
    )
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert list(rows[0]) == ["regret_step", "mean", "std", "max", "min"]


def test_main_saves_csv_after_every_completed_seed(monkeypatch, tmp_path):
    main_module = importlib.import_module("iaf_fbo.main")
    args = parse_args(
        [
            "--task",
            "sphere",
            "--dim",
            "1",
            "--num-clients",
            "1",
            "--num-runs",
            "3",
            "--rounds",
            "1",
            "--initial-points",
            "1",
            "--no-wandb",
            "--output-dir",
            str(tmp_path),
        ]
    )
    saved_runs = []

    class FakeOptimizer:
        def __init__(self, objectives, config):
            self.objectives = objectives

        def run(self):
            return self.objectives

    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        main_module,
        "create_objectives",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(main_module, "IAFFBO", FakeOptimizer)
    monkeypatch.setattr(main_module, "print_final_results", lambda runs: None)
    monkeypatch.setattr(
        main_module,
        "save_regret_csv",
        lambda runs, **kwargs: saved_runs.append((len(runs), kwargs["seed"])),
    )

    main_module.main()

    assert saved_runs == [(1, 0), (1, 1), (1, 2)]
