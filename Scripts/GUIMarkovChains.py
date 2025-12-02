import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px 


st.set_page_config(
    page_title = 'Creador de Cadenas de Markov',
    page_icon = '⛓️',
    layout = 'wide'
)

@st.cache_data
def load():
    try:
        data = pd.read_csv('../Data/data_globant_clean.csv')
        return data
    except Exception as e:
        st.error(f'Error al cargar los datos {e}')
        return None


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

def MarkovChain(data, indicator, form = 'Por Equipos', discrete = 'Redondeo'):
    temp = data.copy(deep = True)
    if form == 'Por Equipos':
        temp = temp[temp['team_name'] == indicator].sort_values('date')
        temp = temp.groupby('date', as_index = False)['engagement'].mean()
    elif form == 'Personal':
        temp = temp[temp['name'] == indicator].sort_values('date')
        temp = temp.groupby('date',as_index = False)['engagement'].mean()

    if discrete == 'Redondeo':
        temp['engagement'] = np.ceil(temp['engagement'].values)

    elif discrete == 'Clases':
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
    x = [i for i in range (iterations +1)]
    y = [start]
    for i in range(1,iterations+1):
        idx = vals.index(y[i-1])
        probs = chain[idx]
        new = np.random.choice(vals,size = 1, p = probs)
        y.append(new[0])

    fig = px.line(
        x=x,
        y=y,
        title=f"Simulación de engagement de {label}",
        labels={
            "x": "Días a partir del último registro",
            "y": "Engagement"
        }
    )
    st.plotly_chart(fig)

data = load()

if data is not None:
    st.title('Simulador de engagement con cadenas de Markov ⛓️')
    st.markdown('---')

    with st.sidebar:
        st.header('🔍 Define tu cadena')

        form = st.selectbox(
            '¿Cómo se construirá la cadena?',
            options = ['Personal','Por Equipos'],
            index = 0
        )

        if form == 'Personal':
            indicator = st.selectbox(
                '¿Para quién será la cadena?',
                options = list(data['name'].unique()),
                index = 0
            )
        else: 
            indicator = st.selectbox(
                '¿Para qué esuipo se realizará la cadena?',
                options = list(data['team_name'].unique()),
                index = 0
            )
        
        with st.expander('Características adicionales'):
            disc = st.selectbox(
                '¿Cómo se discretizarán los datos?',
                options = ['Redondeo','Clases'],
                index = 0
            )

            sim = st.number_input(
                'Cantidad de días simulados',
                min_value = 0,
                value = 50,
                placeholder = 'Ej: 50'                                
                )
        
        bot = st.button(
            '🚀 Simular',
            use_container_width = True,
            type = 'primary'
        )

        st.markdown('---')
        st.header("📊 Estadísticas")
        st.metric('Cantidad de Personas:', len(data["name"].unique()))
        st.metric('Cantidad de Equipos:', len(data["team_name"].unique()))
        st.metric('Promedio global del Engagement:', data["engagement"].mean().round(3))

    if bot:
        col1, col2 = st.columns(2)
        P, vals, last = MarkovChain(data,indicator,form=form,discrete=disc)
        st.header('Resultados de la simulación')
        Simulate(P,last,vals, iterations=sim, label=indicator)

        with col1:
           st.header('Matriz de transición')
           display_mat = pd.DataFrame(data=P.round(3), columns=vals, index = vals)
           st.dataframe(display_mat)

        with col2:
            lim_dist = HasLimit(P)
            if lim_dist is not None:
                st.header('Distribución Límite')
                display_dist = pd.DataFrame(data=lim_dist, index=vals,columns=vals).iloc[0:1]
                st.dataframe(display_dist)
            else:
                st.markdown('Esta matriz no poseé una distribución límite')
    else:
        st.header("👋 ¡Bienvenido!")
        st.markdown("""
        Este programa te ayuda a generar una cadena de Markov que modele el engagement de un 
        empleado o de un equipo junto con una simulación de este engagement para *n* días.  
        Para realizar esta tarea se utilizan los siguientes parámetros:

        **- Forma:** Cómo se construirá la cadena: para una persona o para un equipo.  
        **- Identificador:** El nombre de la persona o el equipo para el que se realizará la cadena.  
        **- Modelo de Discretización:** La forma en que se discretiza el engagement.  

        Si se selecciona **Redondeo**, se toma el entero superior.  
        Si se selecciona **Clases**, se discretiza así:

        - **Ausencia (0):** si *engagement = 0*  
        - **Bajo (1):** si *0 < engagement ≤ 2*  
        - **Medio (2):** si *2 < engagement < 4*  
        - **Alto (3):** si *4 ≤ engagement*

        **- Cantidad de Iteraciones:** Número de días después del último registro a simular.

        ---

        ### **Cómo usar**
        1. Selecciona los parámetros.
        2. Presiona el botón.
        3. Observa los resultados.
        """)

        

        

        