import itertools
import os
from scipy.io import loadmat
import jax.numpy as jnp
import numpy as np
from jaxtyping import Key
from typing import Type

from fedbo.data.objective_dataset import ObjectiveDataset
from fedbo.transformations import ObjectiveTransform, MinMaxScaling
from fedbo.data.objectives import (
    BaseObjective,
    TransformedObjective,
    GriewankObjective,
    RastriginObjective,
    AckleyObjective,
    SchwefelObjective,
    SphereObjective,
    RosenbrockObjective,
    WeierstrassObjective,
)


class ManyTaskDataset(ObjectiveDataset):
    task_iterator = itertools.count()
    tasks: Type[BaseObjective] = [
        GriewankObjective,
        RastriginObjective,
        AckleyObjective,
        RastriginObjective,
        AckleyObjective,
        SchwefelObjective,
        RastriginObjective,
        SphereObjective,
        AckleyObjective,
        RosenbrockObjective,
        AckleyObjective,
        WeierstrassObjective,
        RosenbrockObjective,
        RastriginObjective,
        GriewankObjective,
        WeierstrassObjective,
        RastriginObjective,
        SchwefelObjective,
    ]
    bounds: list[tuple[int, int]] = [
        (-100, 100),
        (-50, 50),
        (-50, 50),
        (-50, 50),
        (-50, 50),
        (-500, 500),
        (-50, 50),
        (-100, 100),
        (-50, 50),
        (-50, 50),
        (-50, 50),
        (-0.5, 0.5),
        (-50, 50),
        (-50, 50),
        (-100, 100),
        (-0.5, 0.5),
        (-50, 50),
        (-500, 500),
    ]
    transformations = [
        ("mat_files/CI_H.mat", "Rotation_Task1", "GO_Task1"),
        ("mat_files/CI_H.mat", "Rotation_Task2", "GO_Task2"),
        ("mat_files/CI_M.mat", "Rotation_Task1", "GO_Task1"),
        ("mat_files/CI_M.mat", "Rotation_Task2", "GO_Task2"),
        ("mat_files/CI_L.mat", "Rotation_Task1", "GO_Task1"),
        (None, None, None),
        ("mat_files/PI_H.mat", "Rotation_Task1", "GO_Task1"),
        ("mat_files/PI_H.mat", None, "GO_Task2"),
        ("mat_files/PI_M.mat", "Rotation_Task1", "GO_Task1"),
        (None, None, None),
        ("mat_files/PI_L.mat", "Rotation_Task1", "GO_Task1"),
        ("mat_files/PI_L.mat", "Rotation_Task2", "GO_Task2"),
        (None, None, None),
        ("mat_files/NI_H.mat", "Rotation_Task2", "GO_Task2"),
        ("mat_files/NI_M.mat", "Rotation_Task1", "GO_Task1"),
        ("mat_files/NI_M.mat", "Rotation_Task2", "GO_Task2"),
        ("mat_files/NI_L.mat", "Rotation_Task1", "GO_Task1"),
        (None, None, None),
    ]

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        key: Key,
        t: ObjectiveTransform = None,
    ):
        """
        Intialize ManyTask Dataset.
        Args:
            num_samples: The initial number of samples.
            input_dim: The dimension of the input features.
            key: A PRNG Key.
            t: Optional transformations for the objective function.
        """
        assert input_dim <= 50, "Maximal supported dim is 50!"
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        task = self.get_task(input_dim)
        t = ObjectiveTransform(x_transform_post=MinMaxScaling(task.bounds()))
        super().__init__(task, num_samples, key, t)

    def get_task(self, input_dim: int):
        """
        Returns a task based on the input dimension.
        Args:
            input_dim: The dimension of the input features.
        Returns:
            A task instance.
        """
        task_id = next(ManyTaskDataset.task_iterator)
        assert task_id < len(ManyTaskDataset.tasks), "Reached maximum number of tasks!"
        task = ManyTaskDataset.tasks[task_id](input_dim)
        bounds = ManyTaskDataset.bounds[task_id]
        t_name, rot_key, shift_key = ManyTaskDataset.transformations[task_id]
        if t_name is None:
            return task
        else:
            t_path = os.path.join(self.dir_path, t_name)
            mat = loadmat(t_path)
            rot_mat = (
                jnp.array(mat[rot_key][0:input_dim, 0:input_dim])
                if rot_key is not None
                else 1.0
            )
            shift_mat = (
                jnp.array(np.squeeze(mat[shift_key])[0:input_dim])
                if shift_key is not None
                else 0.0
            )
            return TransformedObjective(task, bounds, rot_mat, shift_mat)
