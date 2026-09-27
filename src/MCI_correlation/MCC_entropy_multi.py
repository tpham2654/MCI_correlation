# MCC entropy
import pandas as pd
import numpy as np
from MCC_entropy import MCC_entropy

def MCC_entropy_multi(x, y):

    # Multiscale Symmetrized Cumulative Copula Entropy for multiple y's

    scores = []
       
    for f in y.index:
        scores.append(MCC_entropy(x, y.loc[f, :]))

    return(scores)
