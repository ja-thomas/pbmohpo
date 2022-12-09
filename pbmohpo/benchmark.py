from pbmohpo.problems.problem import Problem
from pbmohpo.optimizers.optimizer import Optimizer
from pbmohpo.decision_makers.decision_maker import Decision_maker


class Benchmark:
    def __init__(
        self, problem: Problem, optimizer: Optimizer, dm: Decision_maker, budget: int
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.dm = dm
        self.budget = budget
        self.archive = []

    def step(self) -> None:
        return self.optimizer.propose(self.archive)

    def run(self) -> None:
        for i in range(self.budget):
            config = self.step()
            objectives = self.problem(config)
            utility = self.dm._compute_utility(objectives)

            self.archive.append(
                {"config": config, "objectives": objectives, "utility": utility}
            )
            max_util = max([el["utility"] for el in self.archive])
            print(f"Running [{i}|{self.budget}]: Best utility: {max_util}")
