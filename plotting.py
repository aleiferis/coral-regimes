#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

def biplot(pc,pca_comp,names=None,labels=None):
    fig = plt.figure()
    xs = pc[:,0]
    ys = pc[:,1]
    coeff = np.multiply(1.0,pca_comp.T)
    n = coeff.shape[0]

    if labels is None:
        plt.scatter(xs ,ys, c = 'k') #without scaling
    else:
        # plt.scatter(xs ,ys, c = labels, label=labels) #without scaling
        plt.figure()
        sns.scatterplot(x=xs,y=ys,hue=labels,
                        palette = sns.color_palette ("Set1", len(np.unique(labels))),
                        data = pc,legend="full")


    scalefactor = math.sqrt((max(abs(xs)))**2 + (max(abs(xs)))**2)/math.sqrt((max(abs(coeff[:,0])))**2 + (max(abs(coeff[:,1])))**2)
    print(scalefactor)
    for i in range(n):    
        plt.arrow(0, 0, scalefactor*coeff[i,0], scalefactor*coeff[i,1],color = 'r',alpha = 0.5)
        if names is None:
            plt.text(scalefactor*coeff[i,0]* 1.15, scalefactor*coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(scalefactor*coeff[i,0]* 1.15, scalefactor*coeff[i,1] * 1.15, names[i], color = 'g', ha = 'center', va = 'center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.legend()
    plt.grid()
    return 0