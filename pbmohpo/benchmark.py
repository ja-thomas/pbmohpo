from pbmohpo.archive import Archive, ArchiveItem
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
        self.archive = Archive()

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
        for i in range(self.budget):
            config = self.step()
            objectives = self.problem(config)
            utility = self.dm._compute_utility(objectives)

            result = ArchiveItem(config=config, objectives=objectives, utility=utility)

            self.archive.data.append(result)

            print(
                f"Running [{i:{len(str(self.budget))}}|{self.budget}]: Best utility: {self.archive.max_utility}"
            )
