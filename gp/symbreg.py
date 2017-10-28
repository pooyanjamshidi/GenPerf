import operator
import math
import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import re

coeff_scale = 10
max_depth = 10

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("PerfModel", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand101", lambda: coeff_scale*random.randint(-1, 1)*random.random())
pset.renameArguments(ARG0='o1')
pset.renameArguments(ARG1='o2')
pset.renameArguments(ARG2='o3')
pset.renameArguments(ARG3='o4')
pset.renameArguments(ARG4='o5')
pset.renameArguments(ARG5='o6')

def create_indv():
    init_model = gp.PrimitiveTree.from_string("add(mul(o1,o2),mul(o3,o4))", pset)
    return init_model

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=max_depth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def KLdiv(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    div = np.sum(np.where(p != 0, p*np.log(p/q), 0))
    return div

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # traversing the syntax tree to discover options and interactions

    nodes, edges, labels = gp.graph(expr=individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pos = graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

    degrees = g.degree()
    primitives = nx.dfs_successors(g)


    L = g.size()
    #for i in range(L):
    #    if degrees[i] == 1:




    sqerrors = []
    for point in points:
        x1 = point[0]
        x2 = point[1]
        sqerrors.append((func(x1,x2) - x1 ** 4 - x1 ** 3 - x2 ** 2 - x2) ** 2)
    return math.fsum(sqerrors) / len(points),


toolbox.register("evaluate", evalSymbReg, points=[(2,4), (3,1), (1,1)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))


def main():
    random.seed(318)

    pop = toolbox.population(n=300)

    init_model = gp.PrimitiveTree.from_string("add(mul(o1,o2),mul(o3,o4))",pset)
    init_model.fitness = creator.FitnessMin

    perf_pop = []
    perf_pop.append(init_model)


    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(perf_pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(pop[1])
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()