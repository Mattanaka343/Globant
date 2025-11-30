import numpy as np
import pandas as pd
from scipy import stats

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


def MarkovChain(data, indicator, form = 'team', discrete = 'round'):
    if form == 'team':
        data = data[data['team_name'] == indicator].sort_values('date')
        data = data.groupby('date', as_index = False)['engagement'].mean()
    elif form == 'personal':
        data = data[data['name'] == indicator].sort_values('date')
        data = data.groupby('date',as_index = False)['engagement'].mean()

    if discrete == 'round':
        data['engagement'] = np.ceil(data['engagement'].values)

    elif discrete == 'classes':
        data['engagement'] = discretize(data,'engagement')
        
    vals = sorted(data['engagement'].unique())
    dim = len(vals)

    P = np.zeros((dim,dim))

    for i, origin in enumerate(vals):
        num = sum(data.iloc[k]['engagement'] == origin for k in range(len(data)-1))
        if num != 0:
            for j, dest in enumerate(vals):
                counter = 0
                for k in range(data.shape[0]-1):
                    if data.iloc[k]['engagement'] == origin and data.iloc[k+1]['engagement'] == dest:
                        counter +=1
                P[i,j] = counter/num
    return P, vals

def HasLimit(chain):
    Lambda, Q =  np.linalg.eig(chain)
    if any( np.abs(i)>1  or (np.abs(i)== 1 and i != 1) for i in Lambda):
        print('The chain has no limit distribution')
        return None
    else:
        for i, L in enumerate(Lambda):
            if abs(L) < 1:
                Lambda[i] = 0
            else:
                Lambda[i] = 1
        Lambda = np.diag(Lambda)
        Q_inv = np.linalg.inv(Q)

        return Q
    pass



def Simulate(Chain):
    pass



    



