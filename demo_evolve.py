#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for binary function: f(x) = y*sim(2*pi*x) + x*cos(2*pi*y)
'''

from math import sin, cos, pi
import random

from gaft import GAEngine
from gaft.components import BinaryIndividual,DecimalIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Import evaluation function.
from cifar_train_eval_func import run



# Define population.
indv_template = DecimalIndividual(ranges=[(0, 2), (0, 2),(0, 2), (0, 2),(0, 2)], eps=1)
population = Population(indv_template=indv_template, size=8).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.



@engine.fitness_register
def fitness(indv):
    g_1, g_2, g_3, g_4, g_5  = indv.solution

    layer_1 = convert(g_1)
    layer_2 = convert(g_2)
    layer_3 = convert(g_3)
    layer_4 = convert(g_4)
    layer_5 = convert(g_5)
    hyper_list = [1,layer_1,layer_2, layer_3, layer_4, layer_5]

    #TODO: Convert to [1,2,4,8], negative feedback: model size, flops
    FLOPS = (40.6 + layer_1 * 114.94/64.0 
                  + layer_2 * 75.76/32.0 
                  + layer_3 * 151.26/32.0
                  + layer_4 * 75.6/32
                  + layer_5 * 150.99/32) / 49.4
    k = 1.5
    print(hyper_list)
    accuracy = run(hyper_list)
    fit = accuracy - 0.5 * FLOPS

    print("Accuracy= ",accuracy,"FLOPS= ",FLOPS,"Fitness= ",fit)
    return fit 

def convert(g):
    layer = 2 ** g
    return layer

if '__main__' == __name__:
    
    # print(indv_template)
    engine.run(ng=20)
