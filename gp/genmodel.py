import numpy as np
from scipy import stats
import re as regex
import random
import copy
from sympy import *
import csv
from gp.lib import *

# generic properties
seed = 300
popsize = 10
maxNumberOfOptions = 10
numberIterations = 1000

# mutation properties
probabilityOfMutatingCoefficient = 0.8
probabilityOfAddingFeature = 0.2
probabilityOfRemovingFeature = 0.1
probabilityOfAddingInteraction = 0.2
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
targetNumberOfInteractions = 5
targetNumberOfIndividualOptions = 12
numberOfNegativFeatures = 3
numberOfAbsolutCoefficientsAbove80 = 5
targetCorrelationHigh = 0.8
targetCorrelationLow = 0.2


class Model:
    def __init__(self, terms, ndim):
        self.ndim = ndim
        self.allOptions = ["o" + str(i) for i in range(ndim)]
        self.constant = 0
        self.individualOptions = []
        self.interactions = []
        self.name = ""
        for i in range(len(terms)):
            if terms[i].isInteraction():
                self.interactions.append(terms[i])
            elif terms[i].isIndividualOption():
                self.individualOptions.append(terms[i])
            elif terms[i].isConstant():
                self.constant = float(terms[i].options[0])

    def evaluateModel(self, xTest):
        if xTest.shape[1] != self.ndim:
            raise ValueError()

        L = xTest.shape[0]
        r = np.zeros(L)
        f = sympify(self.__str__())
        vars = {}

        for i in range(L):
            for j in range(self.ndim):
                idx = int(regex.findall("\d+$", self.allOptions[j])[0])
                vars[self.allOptions[j]] = xTest[i, idx]
            r[i] = f.subs(vars).evalf()

        return r

    def evaluateModelFast(self, xTest):
        Lo = len(self.individualOptions)
        Li = len(self.interactions)
        A = xTest

        M = np.zeros(self.ndim + Li)

        for i in range(Lo):
            M[self.allOptions.index(self.individualOptions[i].options[0].replace(" ", ""))] = self.individualOptions[
                i].coefficient

        for i in range(Li):
            options = self.interactions[i].options
            coeff = self.interactions[i].coefficient
            M[self.ndim + i] = coeff

            A = np.append(A, A[:, self.allOptions.index(options[0].replace(" ", "")):self.allOptions.index(
                options[0].replace(" ", "")) + 1], axis=1)
            for idx in range(1, len(options)):
                A[:, self.ndim + i] = A[:, self.ndim + i] * A[:, self.allOptions.index(options[idx].replace(" ", ""))]

        r = np.dot(A, M) + self.constant

        return r

    def simplifyModel(self):
        Lo = self.getNumberOfOptions()
        Li = self.getNumberOfInteractions()
        options2remove = []
        for i in range(1, Lo):
            currentOption = self.individualOptions[i]
            for j in range(i):
                if self.individualOptions[j].options[0].replace(" ", "") == currentOption.options[0].replace(" ", ""):
                    self.individualOptions[j].coefficient = self.individualOptions[
                                                                j].coefficient + currentOption.coefficient
                    options2remove.append(i)
                    break

        interactions2remove = []
        for i in range(1, Li):
            currentInteraction = self.interactions[i]
            for j in range(i):
                if len(self.interactions[j].options) == len(currentInteraction.options):
                    equalOptions = 0
                    for k in range(len(self.interactions[j].options)):
                        for l in range(len(currentInteraction.options)):
                            if self.interactions[j].options[k] == currentInteraction.options[l]:
                                equalOptions += 1
                                break
                    if equalOptions == len(self.interactions[j].options):
                        self.interactions[j].coefficient = self.interactions[j].coefficient + currentInteraction.coefficient
                        interactions2remove.append(i)
                        break

        for i in sorted(options2remove, reverse=True):
            self.individualOptions.pop(i)

        for i in sorted(interactions2remove, reverse=True):
            self.interactions.pop(i)

    def getInteractions(self):
        return self.interactions

    def getIndividualOptions(self):
        return self.individualOptions

    def getNumberOfInteractions(self):
        return len(self.interactions)

    def getNumberOfOptions(self):
        return len(self.individualOptions)

    def removeInteraction(self, position):
        if len(self.interactions) >= 1:
            self.interactions.pop(position)

    def removeIndividualOption(self, position):
        if len(self.individualOptions) > 1:  # for a model to be valid, at least one individual option is needed
            self.individualOptions.pop(position)

    def addOption(self, coefficient):
        if len(self.individualOptions) < self.ndim:
            for i in range(self.ndim):
                proposedOption = "o" + str(i)
                shouldBeAdded = True
                for j in range(len(self.individualOptions)):
                    if proposedOption == self.individualOptions[j].options[0].replace(" ", ""):
                        shouldBeAdded = False
                        break
                if shouldBeAdded:
                    self.individualOptions.append(Term(coefficient, [proposedOption]))
                    break
        self.simplifyModel()

    def addInteraction(self, term):
        self.interactions.append(term)
        self.simplifyModel()

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

        if Li > 0:
            str2 += " + "
        for i in range(len(self.interactions)):
            if i < Li - 1:
                str2 += str(self.interactions[i]) + " + "
            else:
                str2 += str(self.interactions[i])

        if self.constant != 0:
            str2 += " + " + str(self.constant)
        return str2


class Term:
    def __init__(self, coefficient, options="1"):  # The default value is for the constant term
        self.coefficient = coefficient
        self.options = options

    def __str__(self):
        str2 = str(self.coefficient) + " * "
        if len(self.options) > 1:
            for i in range(len(self.options)):
                if i < len(self.options) - 1:
                    str2 += str(self.options[i]) + " * "
                else:
                    str2 += str(self.options[i])
        else:
            str2 += str(self.options[0])
        return str2

    def isConstant(self):
        if len(self.options) == 1 and self.options[0].replace(" ", "").replace('.', '', 1).isdigit():
            return True
        else:
            return False

    def isIndividualOption(self):
        if len(self.options) == 1 and not self.options[0].replace(" ", "").replace('.', '', 1).isdigit():
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
            option1 = random.randint(0, model.getNumberOfOptions() - 1)
            option2 = random.randint(0, model.getNumberOfOptions() - 1)
            while (option1 == option2):
                option2 = random.randint(0, model.getNumberOfOptions() - 1)
            term = Term(random.randint(-100, 100), ["o" + str(option1), "o" + str(option2)])
        else:
            # three-wise
            option1 = random.randint(0, model.getNumberOfOptions() - 1)
            option2 = random.randint(0, model.getNumberOfOptions() - 1)
            while (option1 == option2):
                option2 = random.randint(0, model.getNumberOfOptions() - 1)
            option3 = random.randint(0, model.getNumberOfOptions() - 1)
            while (option1 == option3 or option2 == option3):
                option3 = random.randint(0, model.getNumberOfOptions() - 1)
            term = Term(random.randint(-100, 100), ["o" + str(option1), "o" + str(option2), "o" + str(option3)])
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


def evaluate2csv(model, xTest):
    n = len(xTest)
    yTest = model.evaluateModelFast(xTest)

    with open(model.name + ".csv", "w") as csvfile:
        fieldnames = ["y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(n):
            writer.writerow({"y": yTest[i]})


def assessFitness(model, yTestSource=None, yTestTarget=None, weights=None):
    # compute klDivergence not implemented

    # calculating kernel densities for source and target

    # kernel_s = stats.gaussian_kde(yTestSource)
    # kernel_t = stats.gaussian_kde(yTestTarget)

    corr = abs(np.corrcoef(yTestSource, yTestTarget)[1, 0])
    if corr < targetCorrelationLow:
        correlationDissimilarity = 1
    else:
        correlationDissimilarity = targetCorrelationLow / corr

    # kl = KLdiv(kernel_s.pdf(yTestSource), kernel_t.pdf(yTestTarget))
    # if not np.isnan(kl):
    #     perfdistSimilarity = 1 / (1 + kl)
    # else:
    #     perfdistSimilarity = 0
    interactionSimilarity = 1 / (1 + abs(model.getNumberOfInteractions() - targetNumberOfInteractions))
    optionSimilarity = 1 / (1 + abs(model.getNumberOfOptions() - targetNumberOfIndividualOptions))

    # These other fitness factors can be added if necessary for the problem

    # negativeOptions = 0
    # for i in range(len(model.getIndividualOptions())):
    #     if model.getIndividualOptions()[i].coefficient < 0:
    #         negativeOptions += 1
    # negativeSimilarity = 1 / (1 + abs(negativeOptions - numberOfNegativFeatures))
    # highCoefficients = 0
    # for i in range(len(model.getIndividualOptions())):
    #     if abs(model.getIndividualOptions()[i].coefficient) > 80:
    #         highCoefficients += 1
    # for i in range(len(model.getInteractions())):
    #     if abs(model.getInteractions()[i].coefficient) > 80:
    #         highCoefficients += 1
    # influencingSimilarity = 1 / (1 + abs(highCoefficients - numberOfAbsolutCoefficientsAbove80))

    if weights != None:
        fitness = np.average([interactionSimilarity, optionSimilarity, correlationDissimilarity])
    else:
        fitness = np.average([interactionSimilarity, optionSimilarity, correlationDissimilarity],
                             weights=weights)

    if np.isnan(fitness):
        print("fitness is nan")
    return fitness


def genetic_algorithm(allModels, startingModel, iterations=100):
    best = None
    bestFitness = None
    bestHistory = []
    generation = 1

    n = 1000
    ndim = startingModel.ndim
    xTest = np.random.randint(2, size=(n, ndim))

    yTestSource = startingModel.evaluateModelFast(xTest)

    while (generation <= iterations):
        allFitness = []
        for i in range(len(allModels)):
            allModels[i] = mutate(allModels[i])
        # assessing the fitness of all models
        for i in range(len(allModels)):

            yTestTarget = allModels[i].evaluateModelFast(xTest)

            fitness = assessFitness(allModels[i], yTestSource, yTestTarget)
            allFitness.append(fitness)
            # allIndividuals.append((generation,fitness))
            if best == None or fitness > bestFitness:
                best = allModels[i]
                bestFitness = fitness
                bestHistory.append((allModels[i], bestFitness))
            print(i, allFitness, bestFitness)
        allModels = breed(allModels, allFitness)
        generation += 1
        print("%d\n" % generation)
        # stdout.flush()
    # stdout.write("\n")
    return best, bestFitness, bestHistory, allModels


def genModel():
    given_model = []
    size = random.randint(5, 10)
    individualOptions = int(size * 0.8)
    interactions = size - individualOptions
    for i in range(size):
        if i < individualOptions:
            term = Term(random.randint(-100, 100), ["o" + str(i)])
            given_model.append(term)
        else:
            option1 = random.randint(0, individualOptions - 1)
            option2 = random.randint(0, individualOptions - 1)
            while (option1 == option2):
                option2 = random.randint(0, individualOptions - 1)
            term = Term(random.randint(-100, 100), ["o" + str(option1), "o" + str(option2)])
            given_model.append(term)
    generatedModel = Model(given_model, individualOptions)
    return generatedModel


def genModelfromString(txtModel):
    terms = regex.split("[+-]\s+", txtModel)
    generatedModel = []
    for i in range(len(terms)):
        term = regex.split("[*]", terms[i])
        if len(term) == 1 and term[0].replace('.', '', 1).isdigit():  # this is the constant term
            coeff = float(term)
            generatedModel.append(Term(coeff))
        else:
            coeff = 1
            idx = -1
            for index in range(len(term)):
                if term[index].replace('.', '', 1).isdigit():
                    coeff = float(term[index])
                    idx = index

            if idx != -1:  # we have a explicit coefficient, i.e., 2*o1 instead of o1
                term.pop(idx)
            generatedModel.append(Term(coeff, term))

    return generatedModel


def main():
    np.random.seed(seed)

    # Generate the starting model
    # startingModel = genModel()

    n = 1000
    ndim = 20

    perf_model_txt = "21 + 2.1*o1 + 4.2*o2 + 0.1*o3 + 100*o4 + 2*o5 + 0.1*o6 + o7 + o8 + o9 + o10 + 23*o1*o3 + 2*o4*o7 + o8*o9*o10"
    perf_model = genModelfromString(perf_model_txt)
    startingModel = Model(perf_model, ndim=ndim)
    startingModel.name = "source"

    # Generate response data for the source model

    xTest = np.random.randint(2, size=(n, ndim))
    # yTestSource2 = np.zeros(n)

    # tic()
    # yTestSource1 = startingModel.evaluateModelFast(xTest)
    # print(toc())
    #
    # tic()
    # yTestSource2 = startingModel.evaluateModel(xTest)
    # print(toc())

    evaluate2csv(startingModel, xTest)

    allModels = [copy.deepcopy(startingModel) for i in range(popsize)]
    best, bestFitness, bestHistory, allModels = genetic_algorithm(allModels, startingModel, numberIterations)
    print(best)
    print(bestFitness)

    best.name = "target"
    evaluate2csv(best, xTest)


if __name__ == '__main__':
    main()
