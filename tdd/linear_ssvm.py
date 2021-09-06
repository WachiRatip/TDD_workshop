import numpy as np

class logHingeLoss():
    #setting initial parameters
    def __init__(self, mu=0.05, C=1, max_iter=1000, tol=0.001):
        assert mu > 0
        self.mu=mu
        assert C > 0
        self.C = C
        assert max_iter > 0 and isinstance(max_iter, int)
        self.max_iter=max_iter
        assert tol > 0
        self.tol=tol
    
    def fit(self, x, t):
        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1
        
        # initail variables k, stop, w_0
        k = 0
        update = True
        w = np.random.rand(len(x[0])+1)
        
        # set up matrices
        X = np.ones((len(x),len(x[0])+1))
        X[:,:-1] = x.copy()
        
        while k < (self.max_iter) and update:
            # compute gradient
            _, p, grad = self.gradient(w, X, t)
            if np.isnan(np.sum(p)):
                raise ValueError("Some component in approximately loss functions values (p) is not a number.")
            
            # check stopping criterion
            if np.linalg.norm(grad)<=self.tol:
                update = False
            else:
                # update step
                self.eta = 1/(k+1)
                w -= self.eta*grad
            
            k += 1
        
        self.final_iter = k
        self._coef = w[:-1]
        self._intercept = w[-1]
        
        return self
    
    def gradient(self, w, X, t):
        u = (1-t*np.matmul(X,w))
        p = np.exp(u/self.mu)/(1+np.exp(u/self.mu))
        grad = w - self.C*(np.mean(p*t*X.T,axis=1))
        return u,p,grad

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)