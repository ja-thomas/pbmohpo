from pbmohpo.archive import (
    UtilityArchive,
    UtilityArchiveItem,
    DuelArchive,
    DuelArchiveItem,
)
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.optimizer import Optimizer
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
        self.archive = DuelArchive() if optimizer.is_preferential else UtilityArchive()

    def step(self) -> None:
        """
        Single experiment.

        Run the benchmark for one step from the available budget

        Return
        ------
        CS.Configuration:
            Proposed configuration
        """
        return self.optimizer.propose(self.archive)

    def run(self) -> None:
        """
        Run the benchmark.

        Run the benchmark by conducting as many steps as given by the budget and
        populate the archive with the results
        """
        if self.optimizer.is_preferential:
            self._duel_run()
        else:
            self._utility_run()

    def _compute_utility_archive_item(self, config):
        objectives = self.problem(config)
        utility = self.dm._compute_utility(objectives)
        return UtilityArchiveItem(config=config, objectives=objectives, utility=utility)

    def _utility_run(self) -> None:
        for i in range(self.budget):
            config = self.step()
            result = self._compute_utility_archive_item(config)
            self.archive.data.append(result)

            print(
                f"Running [{i:{len(str(self.budget))}}|{self.budget}]: Best utility: {self.archive.max_utility}"
            )

    def _duel_run(self) -> None:

        evals = 0
        while evals <= self.budget:

            first_config, second_config = self.step()
            first_result = self._compute_utility_archive_item(first_config)
            second_result = self._compute_utility_archive_item(second_config)

            result = DuelArchiveItem(
                first=first_result,
                second=second_result,
                first_won=self.dm.compare(
                    first_result.objectives, second_result.objectives
                ),
            )

            self.archive.data.append(result)

            evals += 2
