
import numpy as np


def Jaccard(first, second, dim):
    first_sites, first_counts = np.unique(first, return_counts=True)
    second_sites, second_counts = np.unique(second, return_counts=True)
    _, first_pos, second_pos = np.intersect1d(
        first_sites, second_sites, assume_unique=True, return_indices=True
    )
    intersection = int(np.minimum(
        first_counts[first_pos], second_counts[second_pos]
    ).sum())
    union = len(first) + len(second) - intersection
    return intersection / union if union > 0 else 1.0


def cal_sim_mat(population, population_size, dim, sim_func=Jaccard):
    similarities = np.ones((population_size, population_size))
    for i in range(population_size):
        for j in range(i):
            value = sim_func(population[i], population[j], dim)
            similarities[i, j] = similarities[j, i] = value
    return similarities


def crossing(first, second, dim):
    mask = np.random.randint(2, size=dim, dtype=bool)
    return (first * mask + second * ~mask,
            first * ~mask + second * mask)


class nichingswarm:

    def __init__(self, fit_evaluator, num_pop=15, dim=20, w_low=0.4,
                 w_high=1.2, a1=2.0, a2=2.0, ite=500, low=0, high=1,
                 sim_thres=0.5, sim_func=Jaccard):
        self.fit_evaluator = fit_evaluator
        self.num_pop = num_pop
        self.dim = dim
        self.low = low
        self.high = high
        self.ite = ite
        self.sim_thres = sim_thres
        self.sim_func = sim_func
        self.pop = None
        self.fits = np.zeros(num_pop)
        self.pb = None
        self.pb_fit = None
        self.gb = None
        self.gb_fit = None
        self.final_best = None
        self.final_best_fit = None
        self.history = None
        self.evaluation_count = 0

    def _random_population(self):
        return self.fit_evaluator.random_population(self.num_pop)

    def _evaluate_population_without_repair(self):
        for i in range(self.num_pop):
            self.fits[i] = (
                1.0 / self.fit_evaluator.fn_without_repair(self.pop[i])
            )
        self.evaluation_count += self.num_pop

    def _mutate_one_gene(self, solution):
        child = np.copy(solution)
        child[np.random.randint(self.dim)] = np.random.randint(
            self.low, self.high
        )
        return child

    def nPGSAO(self):
        self.evaluation_count = 0
        self.pop = self._random_population()
        self._evaluate_population_without_repair()
        self.pb = self.pop.copy()
        self.pb_fit = self.fits.copy()
        best_index = int(np.argmax(self.fits))
        self.gb = self.pop[best_index].copy()
        self.gb_fit = float(self.fits[best_index])
        self.history = [self.gb_fit]
        crossover_probability = 0.8
        mutation_probability = 0.1

        for _ in range(self.ite):
            children = np.empty_like(self.pop)
            for i in range(self.num_pop):
                child = self.pop[i].copy()
                if np.random.rand() < crossover_probability:
                    partner = self.pop[np.random.randint(self.num_pop)]
                    child, _ = crossing(child, partner, self.dim)
                if np.random.rand() < mutation_probability:
                    child = self._mutate_one_gene(child)
                children[i] = child

            self.pop = children
            self._evaluate_population_without_repair()
            improved = self.fits > self.pb_fit
            self.pb[improved] = self.pop[improved]
            self.pb_fit[improved] = self.fits[improved]
            best_index = int(np.argmax(self.pb_fit))
            if self.pb_fit[best_index] > self.gb_fit:
                self.gb = self.pb[best_index].copy()
                self.gb_fit = float(self.pb_fit[best_index])
            self.history.append(float(np.max(self.fits)))

        best_index = int(np.argmax(self.fits))
        self.final_best = self.pop[best_index].copy()
        self.final_best_fit = float(self.fits[best_index])
        return 1.0 / self.final_best_fit
