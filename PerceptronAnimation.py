import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Data set

data = np.genfromtxt('E:/Python/ML/Data_set/perceptron_toydata.txt', delimiter= '\t')
X, y = data[:, :2], data[:, 2]
y = y.astype(np.int64)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)

# Shuffling & tran/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y =  X[shuffle_idx], y[shuffle_idx]

plt.scatter( X[y == 0, 0], X[y == 0, 1], label= 'class 0', marker= 'o')
plt.scatter( X[y == 1, 0], X[y == 1, 1], label= 'class 1', marker= 's')
plt.title('Training set')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim([-3,3]) 
plt.ylim([-3,3])
plt.legend()
plt.show()

## Defining the Perceptron model
class Perceptron():
     def __init__(self, num_features):
          self.num_features = num_features
          self.weights = np.zeros((num_features, 1), dtype = np.float64)
          self.bias = np.zeros(1, dtype = np.float64)
          
     def forward(self, x):
          linear = np.dot(x, self.weights) + self.bias
          predictions = np.where(linear > 0., 1, 0)
          return predictions
     
     def backward(self, x, y):
          predictions = self.forward(x)
          errors = y - predictions
          return errors
     
     def train(self, x, y, epochs): 
          for e in range(epochs):
               
               for i in range(y.shape[0]):
                    errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                    self.weights += (errors * x[i]).reshape(self.num_features, 1)
                    self.bias += errors
     
     def evaluate(self, x, y):
          predictions = self.forward(x).reshape(-1)
          accuracy = np.sum(predictions == y) / y.shape[0]
          return accuracy
     
 ## Training the Perceptron
all_weights = []
all_biases = []

ppn = Perceptron(num_features= 2)

acc = 0

for epoch in range(10): 
     
     for i in range(X.shape[0]):
          
          all_weights.append(ppn.weights.copy())
          all_biases.append(ppn.bias.copy())
          ppn.train(X[i].reshape(1, -1), y[i].reshape(-1), epochs= 1)
          
          acc = ppn.evaluate(X,  y)
          
          if acc == 1.0:
               break
          
     if acc == 1.0: 
          all_weights.append(ppn.weights.copy())
          all_biases.append(ppn.bias.copy())
          break

## Plotting the boundary

import imageio

scatter_highlight_defaults = {'c': 'r',
                              'edgecolor': 'k',
                              'alpha': 1.0,
                              'linewidths': 2,
                              'marker': 'o',
                              's': 150}

def plot(i):
     
     fig, ax = plt.subplots()
     w, b = all_weights[i], all_biases[i]
     
     x_min = -20
     y_min = ( (-w[0] * x_min) - b[0] / w[1] )
     
     x_max = 20
     y_max = ( (-w[0] * x_max) - b[0] / w[1] )
     
     ax.set_xlim([-5., 5])
     ax.set_ylim([-5., 5]) 
     
     ax.set_xlabel('Iteration %d' % i)
     
     ax.plot([x_min, x_max], [y_min, y_max], color = 'k')
     
     ax.scatter(X[y==0, 0], X[y==0, 1], label= 'class 0', marker= 'o')
     ax.scatter(X[y==1, 0], X[y==1, 1], label= 'class 1', marker= 's')
     
     ax.scatter(X[i][0], X[i][1], ** scatter_highlight_defaults)
     
     fig.canvas.draw()
     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype= 'uint8')
     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
     
     return image

kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
imageio.mimsave('training.gif', [plot(i) for i in range(len(all_weights))], fps=1);