import numpy as np
import re as regex
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import copy
from sys import stdout
from sympy import *

seed = 300
popsize = 10
ndim = 20
maxNumberOfOptions = 30
numberIterations = 1

probabilityOfMutatingCoefficient = 0.3
probabilityOfAddingFeature = 0.1
probabilityOfRemovingFeature = 0.1
probabilityOfAddingInteraction = 0.1
probabilityOfRemovingInteraction = 0.1
probabilityOfPairWiseInteraction = 0.8
probabilityOfThreeWiseInteraction = 0.2
probabilityOfSwitchingSign = 0.05
probabilityOfUniformCrossover = 0.1
probabilityOfBreeding = 0.5

standardDeviationForMutation = 10

# not implemented here
probabilityOfChangingCoefficientOfInfluencingOptions = 0.1
probabilityOfChangingCoefficientOfNonInfluencingOptions = 0.1

# goal
targetNumberOfInteractions = 10
targetNumberOfIndividualOptions = 5
numberOfNegativFeatures = 3
numberOfAbsolutCoefficientsAbove80 = 5
targetCorrelationHigh = 0.8
targetCorrelationLow = 0.2

class Model:
    def __init__(self, terms):
        self.allOptions = ["o" + str(i) for i in range(ndim)]
        self.individualOptions = []
        self.interactions = []
        for i in range(len(terms)):
            if terms[i].isInteraction():
                self.interactions.append(terms[i])
            else:
                self.individualOptions.append(terms[i])

    def evaluateModel(self, values):
        if len(values) != ndim:
            raise ValueError()

        vars = {}
        for i in range(ndim):
            idx = int(regex.findall("\d+$", self.allOptions[i])[0])
            vars[self.allOptions[i]] = values[idx]
        f = sympify(self.__str__())

        return f.subs(vars).evalf()


    def getInteractions(self):
        return self.interactions

    def getIndividualOptions(self):
        return self.individualOptions

    def getNumberOfInteractions(self):
        return len(self.interactions)

    def getNumberOfOptions(self):
        return len(self.individualOptions)

    def removeInteraction(self, position):
        self.interactions.pop(position)

    def removeIndividualOption(self, position):
        self.individualOptions.pop(position)

    def addOption(self, coefficient):
        if len(self.individualOptions) < ndim:
            self.individualOptions.append(Term(["o" + str(len(self.individualOptions) + 1)], coefficient))

    def addInteraction(self, term):
        self.interactions.append(term)

    def changeTerm(self, newTerm, position):
        if position < len(self.individualOptions):
            tempTerm = self.individualOptions[position]
            self.individualOptions[position] = newTerm
        else:
            position -= len(self.individualOptions)
            tempTerm = self.interactions[position]
            self.interactions[position] = newTerm
        return tempTerm

    def getTermByPosition(self, position):
        if position < len(self.individualOptions):
            return self.individualOptions[position]
        else:
            return self.interactions[position - len(self.individualOptions)]

    def __str__(self):
        str2 = ""
        Lo = len(self.individualOptions)
        Li = len(self.interactions)
        for i in range(len(self.individualOptions)):
            if i < Lo - 1:
                str2 += str(self.individualOptions[i]) + " + "
            else:
                str2 += str(self.individualOptions[i])
        str2 += " + "
        for i in range(len(self.interactions)):
            if i < Li - 1:
                str2 += str(self.interactions[i]) + " + "
            else:
                str2 += str(self.interactions[i])
        return str2


class Term:
    def __init__(self, options, coefficient):
        self.options = options
        self.coefficient = coefficient

    def __str__(self):
        str2 = str(self.coefficient) + " * "
        if len(self.options) > 1:
            for i in range(len(self.options)):
                if i < len(self.options)-1:
                    str2 += str(self.options[i]) + " * "
                else:
                    str2 += str(self.options[i])
        else:
            str2 += str(self.options[0])
        return str2

    def isIndividualOption(self):
        if len(self.options) == 1:
            return True
        else:
            return False

    def isInteraction(self):
        if len(self.options) == 1:
            return False
        else:
            return True

# KL divergence
def KLdiv(p, q):
   """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
   """
   p = np.asarray(p, dtype=np.float)
   q = np.asarray(q, dtype=np.float)
   div = np.sum(np.where(p != 0, p * np.log(p / q), 0))
   return div

# Mutation
def mutate(model):
    # remove interaction?
    if probabilityOfRemovingInteraction > random.uniform(0, 1) and model.getNumberOfInteractions() > 0:
        model.removeInteraction(max(0, random.randint(0, model.getNumberOfInteractions()) - 1))
    # remove option?
    if probabilityOfRemovingFeature > random.uniform(0, 1) and model.getNumberOfOptions() > 0:
        model.removeIndividualOption(max(0, random.randint(0, model.getNumberOfOptions()) - 1))

    # add interaction?
    if probabilityOfAddingInteraction > random.uniform(0, 1):
        if probabilityOfPairWiseInteraction > random.uniform(0, 1):
            # pairwise
            option1 = random.randint(0, model.getNumberOfOptions())
            option2 = random.randint(0, model.getNumberOfOptions())
            while (option1 == option2):
                option2 = random.randint(0, model.getNumberOfOptions())
            term = Term(["o" + str(option1), "o" + str(option2)], random.randint(-100, 100))
        else:
            # three-wise
            option1 = random.randint(0, model.getNumberOfOptions())
            option2 = random.randint(0, model.getNumberOfOptions())
            while (option1 == option2):
                option2 = random.randint(0, model.getNumberOfOptions())
            option3 = random.randint(0, model.getNumberOfOptions())
            while (option1 == option3 or option2 == option3):
                option3 = random.randint(0, model.getNumberOfOptions())
            term = Term(["o" + str(option1), "o" + str(option2), "o" + str(option3)], random.randint(-100, 100))
        model.addInteraction(term)

    # add option?
    if probabilityOfAddingFeature > random.uniform(0, 1):
        model.addOption(random.randint(-100, 100))

    # mutating the coefficient for options and interactions + sign switch
    for i in range(len(model.getIndividualOptions())):
        if probabilityOfMutatingCoefficient > random.uniform(0, 1):
            model.getIndividualOptions()[i].coefficient += np.random.normal(0, standardDeviationForMutation)
        if probabilityOfSwitchingSign > random.uniform(0, 1):
            model.getIndividualOptions()[i].coefficient *= -1
    for i in range(len(model.getInteractions())):
        if probabilityOfMutatingCoefficient > random.uniform(0, 1):
            model.getInteractions()[i].coefficient += np.random.normal(0, standardDeviationForMutation)
        if probabilityOfSwitchingSign > random.uniform(0, 1):
            model.getInteractions()[i].coefficient *= -1

    return model


def uniformCrossover(parent_a, parent_b):
    # First we initialize our children based on the parents
    child_a = parent_a
    child_b = parent_b
    # Next, we randomly choose an index
    a_modelTerms = parent_a.getNumberOfOptions() + parent_a.getNumberOfInteractions()
    b_modelTerms = parent_b.getNumberOfOptions() + parent_b.getNumberOfInteractions()
    if a_modelTerms <= b_modelTerms:
        smallerModelTerms = a_modelTerms
    else:
        smallerModelTerms = b_modelTerms
    for i in range(smallerModelTerms):
        if probabilityOfUniformCrossover >= random.uniform(0, 1):
            temp = parent_a.changeTerm(parent_b.getTermByPosition(i), i)
            _ = parent_b.changeTerm(temp, i)
    return child_a, child_b


def crossover(parent_a, parent_b):
    return uniformCrossover(parent_a, parent_b)


def breed(allModels, allFitnesses):
    newModelList = []
    for i in range(int(len(allModels) / 2)):
        parent_a = selectParent(allModels, allFitnesses)
        parent_b = selectParent(allModels, allFitnesses)
        if probabilityOfBreeding > random.uniform(0, 1):
            child_a, child_b = crossover(copy.deepcopy(parent_a), copy.deepcopy(parent_b))
        else:
            child_a = copy.deepcopy(parent_a)
            child_b = copy.deepcopy(parent_b)
        newModelList.append(child_a)
        newModelList.append(child_b)
        # Q.append(mutate(child_b, probability= 1/(len(child_b)*10)))
    return newModelList


def fitnessProportionateSelection(allModels, allFitnesses):
    # this is zero based, so we start from index 1
    for i in range(1, len(allModels)):
        allFitnesses[i] = allFitnesses[i] + allFitnesses[i - 1]
    n = random.uniform(0, allFitnesses[len(allModels) - 1])
    # this could be done smarter... how?
    for i in range(1, len(allModels)):
        if allFitnesses[i - 1] < n and n <= allFitnesses[i]:
            return allModels[i]
    return allModels[0]


def selectParent(allModels, allFitnesses):
    return fitnessProportionateSelection(allModels, allFitnesses)


def assessFitness(model, sourceModel = None, weights = None):
    # compute klDivergence not implemented

    n = 1000
    xTest = np.random.randint(2, size = (n, ndim))
    yTestSource = np.zeros(n)
    yTestTarget = np.zeros(n)

    for i in range(n):
        yTestSource[i] = sourceModel.evaluateModel(xTest[i, :])
        yTestTarget[i] = model.evaluateModel(xTest[i, :])

    corr = abs(np.corrcoef(yTestSource, yTestTarget)[1,0])
    if corr < targetCorrelationLow:
        correlationDissimilarity = 1
    else:
        correlationDissimilarity = targetCorrelationLow / corr

    perfdistSimilarity = 1 / (1 + KLdiv(yTestSource, yTestTarget))
    interactionSimilarity = 1 / (1 + abs(model.getNumberOfInteractions() - targetNumberOfInteractions))
    optionSimilarity = 1 / (1 + abs(model.getNumberOfOptions() - targetNumberOfIndividualOptions))
    negativeOptions = 0
    for i in range(len(model.getIndividualOptions())):
        if model.getIndividualOptions()[i].coefficient < 0:
            negativeOptions += 1
    negativeSimilarity = 1 / (1 + abs(negativeOptions - numberOfNegativFeatures))
    highCoefficients = 0
    for i in range(len(model.getIndividualOptions())):
        if abs(model.getIndividualOptions()[i].coefficient) > 80:
            highCoefficients += 1
    for i in range(len(model.getInteractions())):
        if abs(model.getInteractions()[i].coefficient) > 80:
            highCoefficients += 1
    influencingSimilarity = 1 / (1 + abs(highCoefficients - numberOfAbsolutCoefficientsAbove80))
    if weights != None:
        fitness = np.average([interactionSimilarity, optionSimilarity, negativeSimilarity, influencingSimilarity, correlationDissimilarity, perfdistSimilarity])
    else:
        fitness = np.average([interactionSimilarity, optionSimilarity, negativeSimilarity, influencingSimilarity, correlationDissimilarity, perfdistSimilarity], weights=weights)

    return fitness


def genetic_algorithm(allModels, startingModel, iterations=100):
    best = None
    bestFitness = None
    bestHistory = []
    generation = 1
    while (generation <= iterations):
        allFitness = []
        for i in range(len(allModels)):
            allModels[i] = mutate(allModels[i])
        print("i2")
        # assessing the fitness of all models
        for i in range(len(allModels)):
            fitness = assessFitness(allModels[i], startingModel)
            allFitness.append(fitness)
            # allIndividuals.append((generation,fitness))
            if best == None or fitness > bestFitness:
                best = allModels[i]
                bestFitness = fitness
                bestHistory.append((allModels[i], bestFitness))
            print(i)
        allModels = breed(allModels, allFitness)
        generation += 1
        stdout.write("\r%d" % generation)
        stdout.flush()
    stdout.write("\n")
    return best, bestFitness, bestHistory, allModels


def main():
    np.random.seed(seed)

    given_model = []
    size = random.randint(5, 10)
    individualOptions = int(size * 0.8)
    interactions = size - individualOptions
    for i in range(size):
        if i < individualOptions:
            term = Term(["o" + str(i)], random.randint(-100, 100))
            given_model.append(term)
        else:
            option1 = random.randint(0, individualOptions - 1)
            option2 = random.randint(0, individualOptions - 1)
            while (option1 == option2):
                option2 = random.randint(0, individualOptions - 1)
            term = Term(["o" + str(option1), "o" + str(option2)], random.randint(-100, 100))
            given_model.append(term)
    startingModel = Model(given_model)

    allModels = [copy.deepcopy(startingModel) for i in range(popsize)]
    best, bestFitness, bestHistory, allModels = genetic_algorithm(allModels, startingModel)
    print(best)
    print(bestFitness)


if __name__ == '__main__':
    main()
