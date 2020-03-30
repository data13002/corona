import os
import numpy as np
import yaml
import datetime

import pandas as pd

import dash 
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from scipy.integrate import solve_ivp

#Répertoire de sauvegarde des fichiers bruts
PROCESSED_DIR = '../data/processed/'

#Table principale
ALL_DATA_FILE = 'all_data.csv'

ENV_FILE='../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

#Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE= os.path.join(ROOT_DIR, params['directories']['processed'], params['files']['all_data'])

#Lecture du fihcier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df:_df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020,3,10)]
              )


countries=[{'label':c, 'value': c} for c in epidemie_df['Country/Region'].unique()]
app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',
                    options=countries
                )

            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label':'Confirmed', 'value': 'Confirmed'},
                        {'label':'Deaths', 'value': 'Deaths'},
                        {'label':'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ])
        ]),
        dcc.Tab(label='Map', children=[
            html.H6(['The map:']),
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )     
        ]),
        dcc.Tab(label='Modelisation', children=[
            
            html.Div([
                dcc.Dropdown(
                    id='country3',
                    options=countries
                ),
                dcc.Input(
                    id= 'beta_i',placeholder='beta', type='number'),
                dcc.Input(
                    id='gamma_i',placeholder='gamma', type='number') ,
                dcc.Input(
                    id='pop_i',placeholder='population', type='number') ,
                
                #html.Hr(pppp),
                html.Div(id="number-out"),
                dcc.Graph(id='graph2'),
            ]),
        ]),
    ]),
])

@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country','value'),
        Input('country2','value'),
        Input('variable','value'),
        
    ]
)
def update_graph(country, country2, variable):
    print(country)
    
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable:'sum'}).reset_index()
    else:
            graph_df=(epidemie_df[epidemie_df['Country/Region'] == country]
              .groupby(['Country/Region', 'day'])
              .agg({variable:'sum'})
              .reset_index()
             )
    if country2 is not None:
        graph2_df=(epidemie_df[epidemie_df['Country/Region'] == country2]
              .groupby(['Country/Region', 'day'])
              .agg({variable:'sum'})
              .reset_index()
             )              
    return {
        'data':[
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day','value'),
    ]
)
def update_map(map_day):
    day= epidemie_df['day'].sort_values(ascending=True).unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Confirmed':'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )

    return {
        'data':[
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + '(' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed']/ 1_000, 10)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True)
        )
       
    }

#def get_country(self, country):
#    return (epidemie_df[epidemie_df['Country/Region']==country]
#           .groupby(['Country/Region', 'day'])
#           .agg({'Confirmed': 'sum', 'Deaths':'sum','Recovered':'sum'})
#           .reset_index()
#           )

#Monkey Patch pd.Dataframe
#pd.DataFrame.get_country = get_country  

beta_opt,gamma_opt = [0.01,0.1]
def SIR(t,y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])



def get_country(self, country3):
    return (epidemie_df[epidemie_df['Country/Region'] == country3]
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
            .reset_index()
           )
pd.DataFrame.get_country = get_country


@app.callback(
    Output('graph2', 'figure'),
    [
        Input('beta_i', 'value'),
        Input('gamma_i','value'),
        Input('pop_i', 'value'),
        Input('country3','value'),

       
    ]
)

def plot_epidemia(solution, infected, beta_i, gamma_i, country3):
    print(country3)
    

    #pop_i=pop
    
    if beta_i is None:
        beta=beta_opt
    else:
        beta=beta_i
        
        
    if gamma_i is not None:
        gamma=gamma_opt
    else: 
        gamma=gamma_i
        
        
    if country3 is None:
        country_df=(epidemie_df
           .groupby( 'day')
           .agg({'Confirmed': 'sum', 'Deaths':'sum','Recovered':'sum'})
           .reset_index()
           )
        country_df['infected'] = country_df['Confirmed'].diff()
        infected=country_df.loc[2:].head()
        solution = solve_ivp(SIR,[0,40],[51_470_000,1,0],t_eval=np.arange(0,40,1))
        
    else: 
        country_df= (epidemie_df[epidemie_df['Country/Region']==country3]
           .groupby(['Country/Region', 'day'])
           .agg({'Confirmed': 'sum', 'Deaths':'sum','Recovered':'sum'})
           .reset_index()
           )
        country_df['infected'] = country_df['Confirmed'].diff()
        infected=country_df.loc[2:].head()
        solution = solve_ivp(SIR,[0,40],[51_470_000,1,0],t_eval=np.arange(0,40,1))
    
    return{
        'data':[
            dict(
                x=solution.t,
                y=solution.y[0],
                type='line',
                name='Susceptible'
            )
        ] + ([
            dict(
                x=solution.t,
                y=solution.y[1],
                type='line',
                name='infected'
            )
        ]) + ([
            dict(
                x=solution.t,
                y=solution.y[2],
                type='line',
                name='Removed'
            )
        ]) + ([
            dict(
                x=infected.reset_index(drop=True).index,
                y=infected,
                type='line',
                name='original'
            )
        ])
            }   




        
if __name__ == '__main__':
    app.run_server(debug=True)