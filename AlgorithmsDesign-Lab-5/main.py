import math
import numpy as np
from KnapsackProblem import *
from KnapsackArtificialBeeColony import *
import matplotlib.pyplot as plt

def main():
    currKnapsackProblem = KnapsackProblem(seed = 123)
    abc_obj = KnapsackArtificialBeeColony(currKnapsackProblem, 50, iterations = 200, min_max = 'max', seed = 123, scouts = 10)
    abc_obj.fit()
    solution = abc_obj.get_solution()

    print("Solution:")
    print(solution)
    print("Solution Total Profit: " + str(currKnapsackProblem.calculate_total_profit(solution)))
    print("Solution Total Weight: " + str(currKnapsackProblem._calculate_total_weight(solution)))
    #x1 = []
    #y1 = []
    #for i in range(100):
    #    abc_obj = KnapsackArtificialBeeColony(currKnapsackProblem, 4 + i * 4, iterations = 100, min_max = 'max', seed = 123, scouts = 5)
    #    abc_obj.fit()
    #    x1.append(4 + i * 4)

    #    solution = abc_obj.get_solution()
    #    y1.append(currKnapsackProblem.calculate_total_profit(solution))
    #    print(str(currKnapsackProblem.calculate_total_profit(solution)) + ' ' + str(currKnapsackProblem._calculate_total_weight(solution)) + " " + str(4 + i * 4))

    #plt.xlabel("Colony Size") 
    #plt.ylabel("Max Total Profit") 
    #plt.plot(x1, y1)
    #plt.savefig('ColonySize_vs_MaxTotalProfit.pdf', format = 'pdf')
    #abc_obj = KnapsackArtificialBeeColony(currKnapsackProblem, 148, iterations = 100, min_max = 'max', seed = 123, scouts = 5)
    #abc_obj.fit()
    #solution = abc_obj.get_solution()
    #print(str(currKnapsackProblem.calculate_total_profit(solution)) + ' ' + str(currKnapsackProblem._calculate_total_weight(solution)))
    #x2 = []
    #y2 = []
    #for i in range(30):
    #    abc_obj = KnapsackArtificialBeeColony(currKnapsackProblem, 148, iterations = 100, min_max = 'max', seed = 123, scouts = 5 + 5 * i)
    #    abc_obj.fit()
    #    x2.append(4 + i * 4)

    #    solution = abc_obj.get_solution()
    #    y2.append(currKnapsackProblem.calculate_total_profit(solution))
    #    print(str(currKnapsackProblem.calculate_total_profit(solution)) + ' ' + str(currKnapsackProblem._calculate_total_weight(solution)) + " " + str(5 + 5 * i))

    #plt.xlabel("Scouts Per Iteration") 
    #plt.ylabel("Max Total Profit") 
    #plt.plot(x2, y2)
    #plt.savefig('ScoutsPerIteration_vs_MaxTotalProfit.pdf', format = 'pdf')

if __name__ == "__main__":
    main()

