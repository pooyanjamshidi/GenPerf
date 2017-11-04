# GenPerf

## Intorduction
GenPerf is a symbolic regression tool that generates synthetic data for 
evaluating transfer learning approaches. At a high level, GenPerf receives 
a source performance model that has been specified with the 
[SPLConqueror](http://fosd.de/SPLConqueror) format. GenPerf then
generates genetic mutations of the source model to create a target 
performance model that has some similarities to the source model. 
The similarities are based on the core insights in our 
[empirical study](https://arxiv.org/abs/1709.02280) are are based on 
several criteria: 
* Similarity in terms of correlation of source and target response
* Similarity in terms of number of inlfuential options
* Similarity in terms of number of option interactions
* Similarity in terms of performance distribution

![GenPerf Architecture](https://github.com/pooyanjamshidi/GenPerf/blob/master/docs/architecture1.png)

```python
# goal
targetNumberOfInteractions = 10
targetNumberOfIndividualOptions = 5
numberOfNegativFeatures = 3
numberOfAbsolutCoefficientsAbove80 = 5
targetCorrelationHigh = 0.8
targetCorrelationLow = 0.2
``` 
   
### Generating source performance model using [Thor](https://github.com/se-passau/thor-avm/tree/master/Thor)

Thor is a genetic generator for realistic attributed performance models. 
GenPerf uses Thor to generate a realistic source performance model using 
measurements of real world configurable systems. The detailed process of generating 
source performance model has been described [here](https://github.com/se-passau/thor-avm/tree/master/Thor/Tutorial).
Few examples of generated performance models for configurable systems such as LLVM and Lrzip are as follows:

```xml
LLVM: 207 * time_passes + 16 * gvn + 16 * licm + 12 * instcombine + 14 * inline + 3,5 * time_passes * Num1 * Num2 + 5,5 * gvn * licm * Num1 * Num1 + -3,7 * instcombine * inline * Num2

Lrzip: 43838 * level + 2218747 * compressionZpaq + 288311 * compressionLrzip + 191662 * compressionBzip2 + 34718 * compressionGzip + 11946 * encryption + 6676 * compression + 3433850 * compressionZpaq * level9 + 836940 * compressionLrzip * level8 + 720098 * compressionLrzip * level7 + 3415670 * compressionZpaq * level8 + 485719 * compressionLrzip * level9 + -1597534 * compressionZpaq * level1 + -1597084 * compressionZpaq * level3 + -1596575 * compressionZpaq * level2 + 111344 * compressionGzip * level9 + 102375 * compressionGzip * level8 + 59973 * compressionGzip * level7 + -129840 * compressionLrzip * level2 + -128920 * compressionLrzip * level1 + 42831 * compressionGzip * level6 + 21313 * compressionGzip * level5 + -55078 * compressionLrzip * level3 + 43656 * compressionLrzip * level6 + -37020 * compressionBzip2 * level1 + 3,5 * Num1 * Num2 + 4 * Num3 + 5 * Num4 * Num4
```   


## Setting the genetic mutation parameters
```python
# mutation properties
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
```

## Usage

### 1- Specify the source model

```python
perf_model_txt = "2*o1 + 3*o1*o2 + 4*o2"
perf_model = genModelfromString(perf_model_txt)
startingModel = Model(perf_model)
startingModel.name = "test"
```

### 2- Mutate

```python
allModels = [copy.deepcopy(startingModel) for i in range(popsize)]
best, bestFitness, bestHistory, allModels = genetic_algorithm(allModels, startingModel, numberIterations)
print(best)
print(bestFitness)
```

### 3- Generate data

```python
n = 1000
xTest = np.random.randint(2, size=(n, ndim))
evaluate2csv(best, xTest)
```

### 4- Run

```python
python genmodel.py
```

## Contact

If you notice a bug, want to request a feature, or have a question or feedback, please send an email to the tool maintainers:

* [Pooyan Jamshidi](https://github.com/pooyanjamshidi), pooyan.jamshidi@gmail.com
* [Norbert Siegmund](https://github.com/nsiegmun), norbert.siegmund@uni-weimar.de