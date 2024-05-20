#Logistic Regression (Binary Classification)
import numpy as np
import matplotlib.pyplot as plt
import pylab

def read_sample(fileName):

    with open(fileName) as f:
        content = [line.strip().split(',') for line in f]

    dataSet = []
    y =[]
    for i in range(len(content)):
        item = []
        item.append(1.0)
        item.append(float(content[i][0]))
        item.append(float(content[i][1]))
        dataSet.append(item)

        if (content[i][4] == 'Iris-setosa'):
            y.append([0.0])
        else:
            y.append([1.0])

    #normalization
    x1s = [i[1] for i in dataSet]
    x2s = [i[2] for i in dataSet]
    std1 = np.std(np.array(x1s))
    mean1 = np.mean(np.array(x1s))
    std2 = np.std(np.array(x2s))
    mean2 = np.mean(np.array(x2s))

    for i in range(len(dataSet)):
        dataSet[i][1] = (dataSet[i][1] - mean1) / (std1)
        dataSet[i][2] = (dataSet[i][2] - mean2) / (std2)

    maxf1 = max(dataSet, key=lambda x: x[1])[1]
    minf1 = min(dataSet, key=lambda x: x[1])[1]
    maxf2 = max(dataSet, key=lambda x: x[2])[2]
    minf2 = min(dataSet, key=lambda x: x[2])[2]

    for i in range(len(dataSet)):
        dataSet[i][1] = (dataSet[i][1] - minf1) / (maxf1 - minf1)
        dataSet[i][2] = (dataSet[i][2] - minf2) / (maxf2 - minf2)

    xTrain = dataSet[0:40] + dataSet[50:90]
    #print('xtrain=',xTrain)
    xTest = dataSet[40:50] + dataSet[90:100]
    #print('xtest=',xTest)
    yTrain = y[0:40] + y[50:90]
    yTest = y[40:50] + y[90:100]

    return np.array(xTrain) , np.array(xTest), np.array(yTrain), np.array(yTest), dataSet, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_ascent(X, y, teta, alpha, iterations, epsilon):

    preTeta = teta
    for _ in range(iterations):
        teta = teta + ((alpha) * (X.T @ ( y - sigmoid(X @ teta) )))
        diffTeta = teta - preTeta
        normDiffTeta = np.linalg.norm(diffTeta)
        if normDiffTeta < epsilon:
          return teta
        preTeta = teta
    return teta

def test(X, teta):
    return np.round(sigmoid(X @ teta))

def accuracy(x, y, teta):
    yPer = test(x, teta)
    return( float(sum(yPer == y))/ float(len(y)) ) * 100

def plot(dataSet, y, teta):

    for row in dataSet:
        del row[0]
    
    x0 = [row[0] for row in dataSet]
    #print('len x0=',len(x0))
    #print('x0=',x0)
    x1 = ( (teta[1] * x0) + teta[0] ) / -teta[2]
    #print('len x1=',len(x1))
    #print('x1=',x1)
    print("decision boundary y = %f * x + %f " % ( -1 * teta[1] / teta[2], -1 * teta[0] / teta[2] ) )

    plt.scatter(*zip(*dataSet[0:40]), color = "#ff4d4d")   # y = 0, train
    plt.scatter(*zip(*dataSet[40:50]), color = "#ff8080")  # y = 0, test
    plt.scatter(*zip(*dataSet[50:90]), color = "#00ace6")  # y = 1, train
    plt.scatter(*zip(*dataSet[90:100]), color = "#80dfff") # y = 1, test   
    plt.plot(x0, x1, color = "#9933ff")
    #print('x.shape',x.shape)

    plt.show()

def main():
    xTrain, xTest, yTrain, yTest, dataSet, y = read_sample('iris.data')

    maxIteration = 1000
    rate = 0.03
    epsilon = 0.01
    n = np.size(xTrain,1)
    initialTeta = np.zeros((n,1))
    teta = gradient_ascent(xTrain, yTrain, initialTeta, rate, maxIteration, epsilon)
    accuracyPercentTest = accuracy(xTest, yTest, teta)
    accuracyPercentTrain = accuracy(xTrain, yTrain, teta)

    print("teta0: %f \nteta1: %f \nteta2: %f" % (teta[0][0], teta[1][0], teta[2][0]))
    print("accuracy train", accuracyPercentTrain)
    print("accuracy test", accuracyPercentTest)
    print("xTrain.shape",xTrain.shape)
    print("yTrain.shape",yTrain.shape)
    plot(dataSet, y, teta)
    #print(dataSet[0:40])
    print(dataSet[40:50])

main()
