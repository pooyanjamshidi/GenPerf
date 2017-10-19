import operator
import random
import math
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


prim = gp.PrimitiveSet("PRIMITIVES", 1)
prim.addPrimitive(operator.add, 2)
prim.addPrimitive(operator.sub, 2)
prim.addPrimitive(operator.mul, 2)
prim.addPrimitive(operator.truediv, 2)
prim.addPrimitive(operator.neg, 1)
prim.addPrimitive(operator.abs, 1)
prim.addEphemeralConstant("rand", lambda: random.randint(-1,1))
prim.renameArguments(ARG0='x')


creator.create("Fitness", base.Fitness)
creator.create("Genotype", gp.PrimitiveTree, fitness = creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset = prim, min_ = 1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Genotype)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset = prim)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points)

toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=prim)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(300)

    pop = toolbox.population(n = 1)
    hof = tools.HallOfFame(1)

    stat_fit = tools.Statistics(lambda ind: ind.fitness.value)
    stat_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.4, 0.1, 1000, stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == '__main__':
    main()


