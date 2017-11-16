from gp.genmodel import *

ndim = 20
n = 10000

source = "21 + 2.1*o1 + 4.2*o2 + 0.1*o3 + 100*o4 + 2*o5 + 0.1*o6 + o7 + o8 + o9 + o10 + 23*o1*o3 + 2*o4*o7 + o8*o9*o10"
target = "10.18 * o1  + 48.03 * o2  + 10.08 * o3  + 77.99 * o5  + 39.78 * o6  + 7.52 * o7  + -6.26 * o8  + -3.09 * o9  + 35.30 * o10  + 30.66 * o0 + 11.82 * o11 + 15.91 * o12 + -31.67 * o1 * o3  + 27.04 * o8 * o9 * o10 + -22.48 * o10 * o11 + 22.60 * o8 * o4 + -10.99 * o2 * o1 + 21.0"

xTest = np.random.randint(2, size=(n, ndim))

source_model = genModelfromString(source)
target_model = genModelfromString(target)

sourceModel = Model(source_model, ndim=ndim)
targetModel = Model(target_model, ndim=ndim)

sourceModel.name = "source1"
targetModel.name = "target1"


evaluate2csv(sourceModel, xTest)
evaluate2csv(targetModel, xTest)
