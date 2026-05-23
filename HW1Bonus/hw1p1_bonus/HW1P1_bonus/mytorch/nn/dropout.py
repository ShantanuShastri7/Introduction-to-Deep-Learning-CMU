# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p
        self.keep_probab = 1-p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            self.mask = np.random.binomial(1, self.keep_probab, size=x.shape)
            self.mask = self.mask/self.keep_probab

            return x*self.mask
            
        else:
            return x
		
    def backward(self, delta):
        
        return self.mask*delta