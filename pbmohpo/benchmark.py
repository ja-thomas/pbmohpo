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
    eval_budget: int
        Number of configurations that can be evaluated
    dm_budget: int
        Number of comparisons the DM can make
    eval_batch_size: int
        How many configurations to propose in one step
    dm_batch_size: int
        How many comparisons does the DM in one step
    """

    def __init__(
        self,
        problem: Problem,
        optimizer: Optimizer,
        dm: DecisionMaker,
        eval_budget: int,
        dm_budget: int,
        eval_batch_size: int = 1,
        dm_batch_size: int = 1,
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.dm = dm
        self.eval_budget = eval_budget
        self.dm_budget = dm_budget
        self.eval_batch_size = eval_batch_size
        self.dm_batch_size = dm_batch_size

        self.archive = Archive()

    def run(self) -> None:
        """
        Run the benchmark.

        Run the benchmark by conducting as many steps as given by the budget and
        populate the archive with the results
        """

        while len(self.archive.evaluations) < self.eval_budget or (
            self.optimizer.dueling and len(self.archive.comparisons) < self.dm_budget
        ):

            if len(self.archive.evaluations) < self.eval_budget:
                # Number of configurations to propose is either the batch size or the remaining budget
                n_eval = min(
                    self.eval_budget - len(self.archive.evaluations),
                    self.eval_batch_size,
                )

                configs = self.optimizer.propose_config(self.archive, n=n_eval)

                for config in configs:
                    objectives = self.problem(config)
                    utility = self.dm._compute_utility(objectives)
                    result = Evaluation(
                        config=config, objectives=objectives, utility=utility
                    )
                    self.archive.evaluations.append(result)
                max_util = self.archive.max_utility
                print(
                    f"Running Evaluations: [{len(self.archive.evaluations):{len(str(self.eval_budget))}}|{self.eval_budget}]: Best utility: {max_util}"
                )

            if (
                self.optimizer.dueling
                and len(self.archive.comparisons) < self.dm_budget
            ):
                duels = self.optimizer.propose_duel(self.archive, n=self.dm_batch_size)

                for c1, c2 in duels:
                    c1_won = self.dm.compare(
                        self.archive.evaluations[c1].objectives,
                        self.archive.evaluations[c2].objectives,
                    )
                    self.archive.comparisons.append([c1, c2] if c1_won else [c2, c1])

                print(
                    f"Running Duels: [{len(self.archive.comparisons):{len(str(self.dm_budget))}}|{self.dm_budget}]: Best utility: {max_util}"
                )
