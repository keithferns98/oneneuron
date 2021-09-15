import numpy as np

class Perceptron:
  def __init__(self,lr,epochs):
    self.weights=np.random.randn(3)*1e-4
    print(f'initial weights before training{self.weights}')
    self.lr=lr
    self.epochs=epochs

  def activationFunction(self,inputs,weights):
    z=np.dot(inputs,weights)
    return np.where(z>0,1,0)  
  def fit(self,X,y):
    self.X=X
    self.y=y

    X_with_bias=np.c_[self.X,-np.ones((len(self.X),1))] #bias concates with the inputs
    print(f'X with bias vals:\n{X_with_bias}')


    for epoch in range(self.epochs):
        print('--'*10)
        print(f'for epochs: {epoch}')
        y_hat=self.activationFunction(X_with_bias,self.weights)
        print(f'predicted value after forward pass: {y_hat}')
        self.error=self.y-y_hat
        print(f'error: \n {self.error}')

        self.weights=self.weights + self.lr* np.dot(X_with_bias.T,self.error)
        print(f'Updated weights {self.weights} after epoch {epoch}')
        print("####"*10)

  def predict(self,X):
    X_with_bias=np.c_[X,-np.ones((len(X),1))]
    return self.activationFunction(X_with_bias,self.weights)
  def total_loss(self):
    total_loss1=np.sum(self.error)
    print(f'total loss: {total_loss1}')
    return total_loss1