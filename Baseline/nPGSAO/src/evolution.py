"""The SGA implementation used by both datasets."""

import numpy as np


class GA:
    """Generational roulette SGA without elitist reinsertion."""

    def __init__(self, fit_evaluator, dim, high, ite, num_pop=15,
                 prob_cross=0.8, prob_mu=0.1, low=0):
        self.fit_evaluator = fit_evaluator
        self.num_pop = num_pop
        self.dim = dim
        self.prob_cross = prob_cross
        self.prob_mu = prob_mu
        self.low = low
        self.high = high
        self.pop = None
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.best = None
        self.best_fit = None
        self.final_best = None
        self.final_best_fit = None
        self.history = None
        self.evaluation_count = 0

    def _random_population(self):
        return self.fit_evaluator.random_population(self.num_pop)

    def _evaluate_population(self):
        for i in range(self.num_pop):
            self.pop[i] = self.fit_evaluator.repair_randomly(self.pop[i])
            self.fits[i] = 1.0 / self.fit_evaluator.fn(self.pop[i])
        self.evaluation_count += self.num_pop

    def uniform_cross(self, first, second):
        mask = np.random.randint(2, size=self.dim, dtype=bool)
        return (first * mask + second * ~mask,
                first * ~mask + second * mask)

    @staticmethod
    def roulette(values):
        threshold = np.random.uniform() * np.sum(values)
        cumulative = values[0]
        index = 0
        while threshold > cumulative and index < len(values) - 1:
            index += 1
            cumulative += values[index]
        return index

    def _mutate_one_gene(self, solution):
        child = np.copy(solution)
        child[np.random.randint(self.dim)] = np.random.randint(
            self.low, self.high
        )
        return child

    def ga(self):
        self.evaluation_count = 0
        self.pop = self._random_population()
        self._evaluate_population()
        best_index = int(np.argmax(self.fits))
        self.best = self.pop[best_index].copy()
        self.best_fit = float(self.fits[best_index])
        self.history = [self.best_fit]

        for _ in range(self.ite):
            children = []
            while len(children) < self.num_pop:
                first = self.pop[self.roulette(self.fits)].copy()
                second = self.pop[self.roulette(self.fits)].copy()
                if np.random.rand() < self.prob_cross:
                    first, second = self.uniform_cross(first, second)
                for child in (first, second):
                    if len(children) == self.num_pop:
                        break
                    if np.random.rand() < self.prob_mu:
                        child = self._mutate_one_gene(child)
                    children.append(child)

            self.pop = np.asarray(children, dtype=int)
            self._evaluate_population()
            best_index = int(np.argmax(self.fits))
            if self.fits[best_index] > self.best_fit:
                self.best = self.pop[best_index].copy()
                self.best_fit = float(self.fits[best_index])
            current_best = float(np.max(self.fits))
            self.history.append(current_best)

        best_index = int(np.argmax(self.fits))
        self.final_best = self.pop[best_index].copy()
        self.final_best_fit = float(self.fits[best_index])
        return 1.0 / self.final_best_fit
