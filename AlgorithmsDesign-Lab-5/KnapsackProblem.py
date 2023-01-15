import random
import math
from sys import float_info

class KnapsackProblem:

    def __init__(self, 
                 capacity = 500,
                 number_of_unique_items = 100,
                 min_item_profit = 2,
                 max_item_profit = 30,
                 min_item_weight = 1,
                 max_item_weight = 20,
                 seed = None):

        self.capacity = capacity
        self.number_of_unique_items = number_of_unique_items
        self.min_item_profit = min_item_profit
        self.max_item_profit = max_item_profit
        self.min_item_weight = min_item_weight
        self.max_item_weight = max_item_weight

        if (seed == None):
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)

        self._random()
        self.lower = [0] * number_of_unique_items
        self.upper = [math.floor(capacity / self.wpr[i][1]) for i in range(number_of_unique_items)]
        # self.upper = [1] * number_of_unique_items
 
    def _random(self):
        weights = []
        profits = []
        relation_values = []

        for _ in range(self.number_of_unique_items):
            weights.append(random.randint(self.min_item_weight, self.max_item_weight))
            profits.append(random.randint(self.min_item_profit, self.max_item_profit))
            relation_values.append(profits[-1] / weights[-1])

        zipped_wpr = zip(weights, profits, relation_values)
        wpr = list(zipped_wpr)
        wpr.sort(key = lambda i: i[2], reverse = True)

        self.wpr = wpr

    def _calculate_total_weight(self, x: list):
        total_weight = 0
        for i in range(self.number_of_unique_items):
            total_weight += self.wpr[i][0] * round(x[i])

        return total_weight

    def is_solution_acceptable(self, x: list):
        solution_is_acceptable = True
        total_weight = self._calculate_total_weight(x)
        if total_weight > self.capacity:
            solution_is_acceptable = False

        return solution_is_acceptable

    def calculate_total_profit(self, x: list):
        total_profit = 0
        for i in range(self.number_of_unique_items):
            total_profit += self.wpr[i][1] * round(x[i])

        if not self.is_solution_acceptable(x):
            total_profit = float_info.min

        return total_profit

