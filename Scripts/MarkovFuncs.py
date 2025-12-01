import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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
    temp = data.copy(deep = True)
    if form == 'team':
        temp = temp[temp['team_name'] == indicator].sort_values('date')
        temp = temp.groupby('date', as_index = False)['engagement'].mean()
    elif form == 'personal':
        temp = temp[temp['name'] == indicator].sort_values('date')
        temp = temp.groupby('date',as_index = False)['engagement'].mean()

    if discrete == 'round':
        temp['engagement'] = np.ceil(temp['engagement'].values)

    elif discrete == 'classes':
        temp['engagement'] = discretize(temp,'engagement')
        
    vals = sorted(temp['engagement'].unique())
    dim = len(vals)

    P = np.zeros((dim,dim))

    for i, origin in enumerate(vals):
        num = sum(temp.iloc[k]['engagement'] == origin for k in range(len(temp)-1))
        if num != 0:
            for j, dest in enumerate(vals):
                counter = 0
                for k in range(temp.shape[0]-1):
                    if temp.iloc[k]['engagement'] == origin and temp.iloc[k+1]['engagement'] == dest:
                        counter +=1
                P[i,j] = counter/num
    last = temp['engagement'].iloc[-1]
    return P, vals, last

def HasLimit(chain):
    Lambda, Q =  np.linalg.eig(chain)
    if any( np.abs(i)>1  or (np.abs(i)== 1 and i != 1) for i in Lambda.round(3)):
        print('The chain has no limit distribution')
        return None
    else:
        for i, L in enumerate(Lambda.round(3)):
            if abs(L) < 1:
                Lambda[i] = 0
            else:
                Lambda[i] = 1
        Lambda = np.diag(Lambda)
        Q_inv = np.linalg.inv(Q)

        return Q@Lambda@Q_inv




def Simulate(chain,start,vals,iterations = 50,label= ''):
    np.random.seed(42)
    x = [i for i in range (iterations +1)]
    y = [start]
    for i in range(1,iterations+1):
        idx = vals.index(y[i-1])
        probs = chain[idx]
        new = np.random.choice(vals,size = 1, p = probs)
        y.append(new[0])
    plt.plot(x,y);
    plt.title(f'Simulación de engagement de {label}')
    plt.xlabel('Días a partir del último registro')
    plt.ylabel('Engagement')
    plt.savefig(f'../Figures/SimulaciónEngagement{label.replace(' ','_').replace('(','').replace(')','')}')
    plt.show()




    



