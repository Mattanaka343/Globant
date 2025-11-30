import pandas as pd
import numpy as np


def discretize(df,disc_col):
    disc_vals = np.zeros(df[disc_col].shape,dtype = int)
    for i,entry in enumerate(df[disc_col].values):
        if entry == 0:
            disc_vals[i] = 0
        elif entry > 0 and entry <= 2:
            disc_vals[i] = 1
        elif entry > 2 and entry <4:
            disc_vals[i] = 2
        else:
            disc_vals[i] = 3
    return disc_vals

            
def encode(df, enc_col):
    enc_vals = np.zeros(df[enc_col].shape)
    for i,entry in enumerate(df[enc_col].values):
        if entry == 'Jr':
            enc_vals[i] = 1
        elif entry == 'Ssr':
            enc_vals[i] = 2
        elif entry == 'Sr Level 1':
            enc_vals[i] = 3
        elif entry == 'Sr Level 2':
            enc_vals[i] = 4
        else:
            enc_vals[i] = 5
    return enc_vals