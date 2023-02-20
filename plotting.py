#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def biplot(pc,pca_comp,names=None,labels=None):
    fig = plt.figure()
    xs = pc[:,0]
    ys = pc[:,1]
    coeff = np.multiply(1.0,pca_comp.T)
    n = coeff.shape[0]

    if labels is None:
        plt.scatter(xs ,ys, c = 'k') #without scaling
    else:
        plt.scatter(xs ,ys, c = labels) #without scaling

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if names is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, names[i], color = 'g', ha = 'center', va = 'center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    return 0