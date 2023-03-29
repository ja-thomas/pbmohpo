from pbmohpo.archive import Archive, Evaluation
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
        self,
        problem: Problem,
        optimizer: Optimizer,
        dm: DecisionMaker,
        eval_budget: int,
        dm_budget: int,
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.dm = dm
        self.eval_budget = eval_budget
        self.dm_budget = dm_budget

        self.archive = Archive()

    def run(self) -> None:
        """
        Run the benchmark.

        Run the benchmark by conducting as many steps as given by the budget and
        populate the archive with the results
        """

        while True:
            if (
                self.optimizer.should_propose_config(self.archive)
                and len(self.archive.evaluations) < self.eval_budget
            ):
                print("Propose Configuration")
                config = self.optimizer.propose_config(self.archive)
                objectives = self.problem(config)
                utility = self.dm._compute_utility(objectives)
                result = Evaluation(
                    config=config, objectives=objectives, utility=utility
                )
                self.archive.evaluations.append(result)

            elif (
                not self.optimizer.should_propose_config(self.archive)
                and len(self.archive.comparisons) < self.dm_budget
            ):
                print("Propose Duel")
                c1, c2 = self.optimizer.propose_duel(self.archive)
                c1_won = self.dm.compare(
                    self.archive.evaluations[c1].objectives,
                    self.archive.evaluations[c2].objectives,
                )
                self.archive.comparisons.append([c1, c2] if c1_won else [c2, c1])
            else:
                break

            max_util = self.archive.max_utility
            print(
                f"Running [{len(self.archive.evaluations):{len(str(self.eval_budget))}}|{self.eval_budget}]: Best utility: {max_util}"
            )
