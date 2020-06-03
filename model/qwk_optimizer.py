# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:19:31 2020

@author: Stephan
"""

import scipy as sp
from sklearn.metrics import cohen_kappa_score


class KappaOptimizer(object):
    ''' Kappaoptimizer for tensorflow
    
    
        References:
        # inspired by https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
                      https://www.kaggle.com/fanconic/panda-inference-for-effnetb0-regression/comments
    
    '''
    def __init__(self, thresholds=[0.5, 1.5, 2.5, 3.5, 4.5]):
        # initial optimizers
        self.thresholds = thresholds
        # define score function:
        self.func = self.quad_kappa
    
    
    def predict(self, preds_raw):
        return self._predict(self.thresholds, preds_raw)

    
    @classmethod
    def _predict(cls, thresholds, preds_raw):
        y_hat=preds_raw

        for i,pred in enumerate(y_hat):
            if   pred < thresholds[0]: y_hat[i] = 0
            elif pred < thresholds[1]: y_hat[i] = 1
            elif pred < thresholds[2]: y_hat[i] = 2
            elif pred < thresholds[3]: y_hat[i] = 3
            elif pred < thresholds[4]: y_hat[i] = 4
            else: y_hat[i] = 5
        return y_hat.astype('int')
    
    
    def quad_kappa(self, preds_raw, y_true):
        return self._quad_kappa(self.thresholds, preds_raw, y_true)

    
    @classmethod
    def _quad_kappa(cls, thresholds, preds_raw, y_true):
        y_hat = cls._predict(thresholds, preds_raw)
        
        return cohen_kappa_score(y_true, y_hat, weights='quadratic')

    
    def fit(self, preds_raw, y_true):
        ''' maximize quad_kappa '''
        neg_kappa = lambda thresholds: -self._quad_kappa(thresholds, preds_raw, y_true)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.thresholds, method='nelder-mead',
                                       options={'maxiter':100, 'fatol':1e-20, 'xatol':1e-20})
        self.thresholds = opt_res.x