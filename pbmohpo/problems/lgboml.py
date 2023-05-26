from typing import Dict, List, Optional, Union

import ConfigSpace as CS
import lightgbm as lgb
import numpy as np
from openml.tasks import get_task
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from pbmohpo.problems.problem import Problem


class LgbOpenML(Problem):
    """
    LightGBM Tuning Problem.

    Tune LightGBM for a multiclass classification Problem.

    One call evaluates a cross-validation as defined in the OpenML task and computes the average accuracy per class.

    Objectives are the accuracies of each class (representing different missclassification costs of a DM)

    Parameters
    ----------
    task_id: int
        OpenML task ID
    seed: int, np.random.RandomState
        Seed passed to the problem
    """

    def __init__(
        self, task_id: int, seed: Optional[Union[np.random.RandomState, int]] = 42
    ) -> None:
        super().__init__(seed)
        self.task_id = task_id
        self.task = get_task(task_id)

    def get_config_space(self) -> CS.ConfigurationSpace:
        """
        Defines the tuning space for LightGBM

        Number of iterations (1, 100, log),
        Learning rate (0.001, 0.3, log),
        Bagging fraction (0, 1),
        Feature fraction (0, 1)

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            The configuration space of the problem
        """
        cs = CS.ConfigurationSpace(seed=self.seed)

        cs.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "num_iterations", lower=1, upper=100, log=True
            )
        )

        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "learning_rate", lower=0.001, upper=0.3, log=True
            )
        )

        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("bagging_fraction", lower=0, upper=1)
        )

        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("feature_fraction", lower=0, upper=1)
        )
        return cs

    def get_objective_names(self) -> List:
        """
        Get the names of the objectives, i.e. names of classes

        Returns
        -------
        List
            Names of objectives
        """
        return self.task.class_labels

    def __call__(
        self,
        x: CS.Configuration,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Dict:
        """
        Objective function.

        The call method implements the objective function that should be optimized.

        Parameters
        ----------
        seed: int, np.random.RandomState
            Optional seed used to call the objective function.

        Returns
        -------
        Dict
            Dictionary of named objective values
        """
        params = x.get_dictionary()
        params["num_threads"] = 1  # Do not paralellize lightgbm
        params["force_col_wise"] = True  # Removes some warnings

        params["objective"] = "multiclass"
        params["num_classes"] = self.n_objectives

        # num_iterations is passed to lgb.train seperatetly to remove a warning
        num_iterations = params["num_iterations"]
        del params["num_iterations"]

        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()
        X, y = self.task.get_X_and_y(dataset_format="dataframe")
        y = LabelEncoder().fit_transform(y)

        results = []

        for repeat_idx in range(n_repeats):
            for fold_idx in range(n_folds):
                for sample_idx in range(n_samples):
                    (
                        train_indicies,
                        test_indicies,
                    ) = self.task.get_train_test_split_indices(
                        repeat=repeat_idx, fold=fold_idx, sample=sample_idx
                    )

                    X_train = X.iloc[train_indicies]
                    y_train = y[train_indicies]
                    X_test = X.iloc[test_indicies]
                    y_test = y[test_indicies]

                    data = lgb.Dataset(X_train, y_train)
                    model = lgb.train(params, data, num_iterations)
                    preds = model.predict(X_test)
                    matrix = confusion_matrix(y_test, np.argmax(preds, axis=1))
                    results.append(matrix.diagonal() / matrix.sum(axis=1))

        result = np.mean(results, axis=0)
        result_dict = {k: v for k, v in zip(self.get_objective_names(), result)}
        return result_dict
