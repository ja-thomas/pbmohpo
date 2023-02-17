from pbmohpo.archive import (
    UtilityArchive,
    Evaluation,
    PreferenceArchive,
    PreferenceEvaluation,
)
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.optimizer import Optimizer, PreferenceOptimizer
from pbmohpo.problems.problem import Problem


class Benchmark:
    """
    Conduct a benchmark.

    Run an optimizer with a decision maker and problem for a given budget.

    Parameters
    ----------
    problem: Problem
        Problem to be optimized
    optimizer: Optimizer
        Used optimizer
    dm: DecisionMaker
        Decision maker that evaluates objective values from problems
    budget: int
        Number of tuning iterations to run
    """

    def __init__(
        self, problem: Problem, optimizer: Optimizer, dm: DecisionMaker, budget: int
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.dm = dm
        self.budget = budget

        self.is_preferential = issubclass(type(optimizer), PreferenceOptimizer)

        self.archive = PreferenceArchive() if self.is_preferential else UtilityArchive()

    def run(self) -> None:
        """
        Run the benchmark.

        Run the benchmark by conducting as many steps as given by the budget and
        populate the archive with the results
        """

        while len(self.archive) < self.budget:

            if self.is_preferential:
                first_config, second_config = self.optimizer.propose(self.archive)
                first_result = self._compute_utility_evaluation(first_config)
                second_result = self._compute_utility_evaluation(second_config)

                result = PreferenceEvaluation(
                    first=first_result,
                    second=second_result,
                    first_won=self.dm.compare(
                        first_result.objectives, second_result.objectives
                    ),
                )
            else:
                config = self.optimizer.propose(self.archive)
                result = self._compute_utility_evaluation(config)

            self.archive.data.append(result)

            max_util = self.archive.max_utility

            print(
                f"Running [{len(self.archive):{len(str(self.budget))}}|{self.budget}]: Best utility: {max_util}"
            )

    def _compute_utility_evaluation(self, config):
        objectives = self.problem(config)
        utility = self.dm._compute_utility(objectives)
        return Evaluation(config=config, objectives=objectives, utility=utility)
