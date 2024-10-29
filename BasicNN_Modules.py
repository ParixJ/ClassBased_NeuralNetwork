import numpy as np


class Dense:

    def __init__(self,n_input,n_neuron,w_initializer='glorot'):
        np.random.seed(142)
        if w_initializer == 'glorot':
            self.fan_avg = n_neuron + n_input
            self.weight = np.random.uniform(-3/self.fan_avg,3/self.fan_avg,size=[n_input,n_neuron])
        elif w_initializer == 'lecun':
            self.weight = np.random.normal(-3/n_input,3/n_input,size=[n_input,n_neuron])
        else :
            print(ValueError('No such initializer supported by model, Available initializers include glorot and lecun initialization.'))
        self.bias = np.zeros((1,n_neuron))

    def forward(self,x):
        z = np.dot(x,self.weight) + self.bias
        return z
    
class Softmax:

    def __init__(self):
        pass

    def activate(self,z):
        exp_vals = np.exp(z - np.max(z,axis=1,keepdims=True))
        norm = exp_vals / np.sum(exp_vals,axis=1,keepdims=True)
        return norm

class Sigmoid:

    def __init__(self):
        pass

    def activate(self,z):
        a = 1 / (1 + np.exp(-z))
        d = a*(1-a)
        return a,d
    
class Tanh:

    def __init__(self):
        pass
    
    def activate(self,x):
        a = np.tanh(x)
        d = 1 - np.tanh(x)**2
        return a,d

class MSE:

    def __init__(self):
        pass

    def error(error,ytrue,ypred):
        error = 1/2 * (ytrue - ypred)**2
        der = ypred - ytrue
        return error,der

class Categorical_Cross_Entropy:
    
    def __init__(self):
        pass
    
    def error(self,ytrue,ypred):
        error = np.log(-ypred[range(len(ypred)), ytrue])
        return error