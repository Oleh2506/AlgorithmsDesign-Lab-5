import numpy as np
import random as rng
import warnings as wrn
import KnapsackProblem as kp
import math 

class KnapsackArtificialBeeColony:
    
    def __init__(self,
                 knapsack_problem: kp.KnapsackProblem,
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 min_max: str='min',
                 nan_protection: bool=True,
                 seed: int=None):

        boundaries = list(zip(knapsack_problem.lower, knapsack_problem.upper))
        self.wpr = knapsack_problem.wpr
        self.constraint_function = knapsack_problem.is_solution_acceptable
        self.boundaries = boundaries
        self.min_max_selector = min_max
        self.cost_function = knapsack_problem.calculate_total_profit
        self.nan_protection = nan_protection
        self.seed = seed

        self.max_iterations = int(max([iterations, 1]))
        if (iterations < 1):
            warn_message = 'Using the minimun value of iterations = 1'
            wrn.warn(warn_message, RuntimeWarning)

        self.employed_onlookers_count = int(max([(colony_size/2), 2]))
        if (colony_size < 4):
            warn_message = 'Using the minimun value of colony_size = 4'
            wrn.warn(warn_message, RuntimeWarning)

        if (scouts <= 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
            if (scouts < 0):
                warn_message = 'Negative scout count given, using default scout ' \
                    'count: colony_size * dimension = ' + str(self.scout_limit)
                wrn.warn(warn_message, RuntimeWarning)
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)

        self.scout_status = 0
        self.iteration_status = 0
        self.nan_status = 0

        if (self.seed is not None):
            rng.seed(self.seed)

        self.foods = [None] * self.employed_onlookers_count
        for i in range(len(self.foods)):
            _ABC_engine(self).generate_food_source(i)

        try:
            self.best_food_source = self.foods[np.nanargmax([food.fit for food in self.foods])]
        except:
            self.best_food_source = self.foods[0]
            warn_message = 'All food sources\'s fit resulted in NaN and beecolpy can got stuck ' \
                         'in an infinite loop during fit(). Enable nan_protection to prevent this.'
            wrn.warn(warn_message, RuntimeWarning)
           

    def fit(self):
        if (self.seed is not None):
            rng.seed(self.seed)

        for _ in range(self.max_iterations):

            _ABC_engine(self).employer_bee_phase()

            _ABC_engine(self).onlooker_bee_phase()
            
            _ABC_engine(self).memorize_best_solution()
            
            _ABC_engine(self).scout_bee_phase()
            
            self.iteration_status += 1

        return self.best_food_source.position


    def get_solution(self):
        assert (self.iteration_status > 0), 'fit() not executed yet!'
        return self.best_food_source.position


    def get_status(self):
        return self.iteration_status, self.scout_status, self.nan_status


class _FoodSource:
    def __init__(self, abc, engine):

        self.abc = abc
        self.engine = engine
        self.trial_counter = 0
        self.position = [rng.uniform(*self.abc.boundaries[i]) for i in range(len(self.abc.boundaries))]
        self.fit = self.engine.calculate_fit(self.position)


    def evaluate_neighbor(self, partner_position):

        j = rng.randrange(0, len(self.abc.boundaries))

        xj_new = self.position[j] + rng.uniform(-1, 1)*(self.position[j] - partner_position[j])
        
        xj_new = self.abc.boundaries[j][0] if (xj_new < self.abc.boundaries[j][0]) else \
            self.abc.boundaries[j][1] if (xj_new > self.abc.boundaries[j][1]) else xj_new

        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.abc.boundaries))]
        j = len(neighbor_position) - 1
        while not self.abc.constraint_function(neighbor_position):
            while j >= 0:
                if round(neighbor_position[j]) != 0:
                    neighbor_position[j] = 0
                    break
                else:
                    j -= 1

        neighbor_fit = self.engine.calculate_fit(neighbor_position)
        
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1



class _ABC_engine:
    def __init__(self, abc):
        self.abc = abc


    def check_nan_lock(self):
        if not(self.abc.nan_protection):
            if np.all([np.isnan(food.fit) for food in self.abc.foods]):
                raise Exception('All food sources\'s fit resulted in NaN and beecolpy got ' \
                                'stuck in an infinite loop. Enable nan_protection to prevent this.')


    def execute_nan_protection(self, food_index):
        while (np.isnan(self.abc.foods[food_index].fit) and self.abc.nan_protection):
            self.abc.nan_status += 1
            self.abc.foods[food_index] = _FoodSource(self.abc, self)


    def generate_food_source(self, index):
        self.abc.foods[index] = _FoodSource(self.abc, self)
        self.execute_nan_protection(index)


    def prob_i(self, actual_fit, max_fit):
        return 0.9*(actual_fit/max_fit) + 0.1


    def calculate_fit(self, evaluated_position):
        cost = self.abc.cost_function(evaluated_position)
        if (self.abc.min_max_selector == 'min'):
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else:
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return fit_value


    def food_source_dance(self, index):
        while True:
            d = int(rng.randrange(0, self.abc.employed_onlookers_count))
            if (d != index):
                break
        self.abc.foods[index].evaluate_neighbor(self.abc.foods[d].position)


    def employer_bee_phase(self):
        for i in range(len(self.abc.foods)):
            j = len(self.abc.foods[i].position) - 1
            while not self.abc.constraint_function(self.abc.foods[i].position):
                while j >= 0:
                    if round(self.abc.foods[i].position[j]) != 0:
                        self.abc.foods[i].position[j] = 0
                        break
                    else:
                        j -= 1

            self.abc.foods[i].fit = self.calculate_fit(self.abc.foods[i].position)

            self.food_source_dance(i)


    def onlooker_bee_phase(self):
        self.check_nan_lock()
        max_fit = np.nanmax([food.fit for food in self.abc.foods])
        onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
        p = 0
        i = 0
        while (p < self.abc.employed_onlookers_count):
            if (rng.uniform(0, 1) <= onlooker_probability[i]):
                p += 1
                self.food_source_dance(i)
                self.check_nan_lock()
                max_fit = np.nanmax([food.fit for food in self.abc.foods])
                if (self.abc.foods[i].fit != max_fit):
                    onlooker_probability[i] = self.prob_i(self.abc.foods[i].fit, max_fit)
                else:
                    onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
            i = (i+1) if (i < (len(self.abc.foods)-1)) else 0


    def scout_bee_phase(self):
        trial_counters = [food.trial_counter for food in self.abc.foods]
        if (max(trial_counters) > self.abc.scout_limit):
            trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
            i = trial_counters[rng.randrange(0, len(trial_counters))]
            self.generate_food_source(i)
            self.abc.scout_status += 1


    def memorize_best_solution(self):
        best_food_index = np.nanargmax([food.fit for food in self.abc.foods])
        if (self.abc.foods[best_food_index].fit >= self.abc.best_food_source.fit):
            self.abc.best_food_source = self.abc.foods[best_food_index]
