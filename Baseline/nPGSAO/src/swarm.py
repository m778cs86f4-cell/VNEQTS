
import numpy as np


class PSO:

    def __init__(self, fit_evaluator, num_pop=15, dim=20, w_low=0.4,
                 w_high=1.2, a1=2.0, a2=2.0, ite=500, prob_mu=0.1,
                 low=0, high=1):
        self.fit_evaluator = fit_evaluator
        self.num_pop = num_pop
        self.dim = dim
        self.w_low = w_low
        self.w_high = w_high
        self.prob_mu = prob_mu
        self.a1 = a1
        self.a2 = a2
        self.low = low
        self.high = high
        self.pop = None
        self.vel = None
        self.fits = np.zeros(num_pop)
        self.ite = ite
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

    def _evaluate_population(self):
        for i in range(self.num_pop):
            self.pop[i] = self.fit_evaluator.repair_randomly(self.pop[i])
            self.fits[i] = 1.0 / self.fit_evaluator.fn(self.pop[i])
        self.evaluation_count += self.num_pop

    def pso(self):
        self.evaluation_count = 0
        self.pop = self._random_population()
        self.vel = (
            np.random.rand(self.num_pop, self.dim) * self.high
            - self.high / 2
        )
        self._evaluate_population()
        self.pb = self.pop.copy()
        self.pb_fit = self.fits.copy()
        best_index = int(np.argmax(self.fits))
        self.gb = self.pop[best_index].copy()
        self.gb_fit = float(self.fits[best_index])
        self.history = [self.gb_fit]
        inertia = (self.w_low + self.w_high) / 2.0

        for _ in range(self.ite):
            r1 = np.random.rand(self.num_pop, self.dim)
            r2 = np.random.rand(self.num_pop, self.dim)
            self.vel = (
                inertia * self.vel
                + self.a1 * r1 * (self.pb - self.pop)
                + self.a2 * r2 * (self.gb - self.pop)
            )
            self.vel = np.clip(self.vel, -self.high / 2, self.high / 2)
            self.pop = (
                (self.pop + self.vel.astype(int))
                % (self.high - self.low) + self.low
            )
            self._evaluate_population()

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

    def run(self):
        return self.pso()
