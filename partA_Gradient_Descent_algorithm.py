import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab

def read_data(name):
  f = open(name)
  X = []
  Y = []
  first = True
  for line in f:
    if first:
      first = False
      continue
    data_line = line.rstrip().split(',')
    features = []
    features.append(1.0)
    features.append(float(data_line[0]))
    X.append(features)
    Y.append([float(data_line[1])])

  #normalization
  x1s = [i[1] for i in X]
  std = np.std(np.array(x1s))
  mean = np.mean(np.array(x1s))

  for i in range(len(X)):
    X[i][1] = (X[i][1] - mean) / (std)
  
  return np.array(X), np.array(Y), X, Y

def j_function(t0, t1, X, Y):
  zValue = []
  for i in range(len(t0)):
    zRow = []
    for j in range(len(t0[i])):
      teta = np.array([[t0[i][j]], [t1[i][j]]])
      error = (X @ teta) - Y
      z = (0.5 * ((error.T) @ error))[0][0]
      zRow += [z]
    zValue += [zRow]

  return zValue

def gradient_decent(X, y, teta, alpha, iterations, epsilon):

  alpha = alpha / len(y)
  preTeta = teta
  for i in range(iterations):
    teta = teta - ((alpha) * (X.T @ ( (X @ teta) - y )))
    diffTeta = teta - preTeta
    normDiffTeta = np.linalg.norm(diffTeta)
    if normDiffTeta < epsilon:
      print ("numbrt of iterations", i)
      return teta  
    preTeta = teta

  print ("numbrt of iterations", i)
  return teta

def plot(dataX, dataY):
  fig = plt.figure()
  ax = plt.axes(projection="3d")

  theta_zero = np.linspace(1, 34, 200)
  theta_one = np.linspace(1, 100, 200)

  theta_zero, theta_one = np.meshgrid(theta_zero, theta_one)

  j = np.array(j_function(theta_one, theta_zero, dataX, dataY))

  fig = plt.figure(figsize=plt.figaspect(0.3))

  j_plot = fig.add_subplot(1, 3, 1, projection='3d')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.view_init(0, 0)
  j_plot.set_xlabel('Ɵ_zero')
  j_plot.set_ylabel('Ɵ_one')
  j_plot.set_zlabel("J")

  j_plot = fig.add_subplot(1, 3, 2, projection='3d')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.view_init(0, 90)
  j_plot.set_xlabel('Ɵ_zero')
  j_plot.set_ylabel('Ɵ_one')
  j_plot.set_zlabel("J")
  j_plot.set_title("title")

  j_plot = fig.add_subplot(1, 3, 3, projection='3d')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.plot_surface(theta_zero, theta_one, j, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  j_plot.view_init(45, 45)
  j_plot.set_xlabel('Ɵ_zero')
  j_plot.set_ylabel('Ɵ_one')
  j_plot.set_zlabel("J")
  plt.show()

def calMSE(X, Y, teta):
  error = (X @ teta) - Y
  return((((error.T) @ error)[0][0]) ** 0.5) / len(Y)

def main():
  npX, npY, X, Y = read_data('Data-Train.csv')
  testNpX, testNpY, testX, testY = read_data('Data-Test.csv')
  p = []
  for i in range(len(X)):
    l = []
    l.append(X[i][1])
    l.append(Y[i][0])
    p.append(l)
  
  test = []
  for i in range(len(testX)):
    l = []
    l.append(testX[i][1])
    l.append(testY[i][0])
    test.append(l)

  init = [[0.0], [1.0]]

  teta = gradient_decent(npX, npY, np.array(init), 0.03, 1000, 0.01)

  plt.scatter(*zip(*p), color = "#00ace6")   
  x0 = [row[1] for row in npX]
  x1 = ( (teta[1] * x0) + teta[0] ) 
  plt.plot(x0, x1, color = "#ff4d4d")
  plt.show()

  plt.scatter(*zip(*test), color = "#9933ff")   
  x0 = [row[1] for row in testNpX]
  x1 = ( (teta[1] * x0) + teta[0] ) 
  plt.plot(x0, x1, color = "#ff4d4d")
  plt.show()

  print("teta0: %f \nteta1: %f" % (teta[0], teta[1]))
  print("min squere error data train: ", calMSE(npX, npY, teta))
  print("min squere error data test: ", calMSE(testNpX, testNpY, teta))
  
  plot(npX, npY)

main()