#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor


# In[2]:


col_names=['FE_SPEED_COMPRESSION_A1','FE_TOTAL_RESIN','FE_PRESS_FACTOR_ACTUAL','FE_SAWDUST_RATIO',
'FE_CALCULATED_DENSITY_MAT','FE_PANEL_DENSITY','FE_THICKNESS_RATIO','IB_avg']
col_names2=['FE_SPEED_COMPRESSION_A1','FE_TOTAL_RESIN','FE_PRESS_FACTOR_ACTUAL','FE_SAWDUST_RATIO',
'FE_CALCULATED_DENSITY_MAT','FE_PANEL_DENSITY','FE_THICKNESS_RATIO']


# In[3]:




df = pd.read_csv('data/exam1.csv', sep=',')

#df1 = pd.DataFrame(df_0, columns=col_names)
df1 = df[col_names]

# We change the most important features ranges to make them look like actual figures

df1= df1.replace([np.inf, -np.inf], np.nan)
df2 = df1.fillna(method='ffill').astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(df2.drop('IB_avg', axis=1), df2['IB_avg'], test_size=0.33, random_state=7)


# In[ ]:



# In[3]:


#model = Ridge(alpha=0.01)
#model = xgb.XGBRegressor()
model = RandomForestRegressor()

model.fit(X_train, y_train)

df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=col_names2)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)


# In[6]:




fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)



slider_1_label = 'FE_SPEED_COMPRESSION_A1'
slider_1_min = math.floor( df2[slider_1_label].min())
slider_1_mean = round( df2[slider_1_label].mean())
slider_1_max = round(df2[slider_1_label].max())

slider_2_label = 'FE_TOTAL_RESIN'
slider_2_min = math.floor( df2[slider_2_label].min())
slider_2_mean = round(df2[slider_2_label].mean())
slider_2_max =  round(df2[slider_2_label].max())

slider_3_label = 'FE_PRESS_FACTOR_ACTUAL'
slider_3_min = math.floor( df2[slider_3_label].min())
slider_3_mean = round(df2[slider_3_label].mean())
slider_3_max =  round(df2[slider_3_label].max())

slider_4_label = 'FE_SAWDUST_RATIO'
slider_4_min = math.floor( df2[slider_4_label].min())
slider_4_mean = round(df2[slider_4_label].mean())
slider_4_max =  round(df2[slider_4_label].max())

#slider_5_label ='FE_EXTERNAL_CHIP_RATIO'
#slider_5_min = math.floor( df2[slider_5_label].min())
#slider_5_mean = round(df2[slider_5_label].mean())
#slider_5_max =  round(df2[slider_5_label].max(),2)

slider_6_label = 'FE_CALCULATED_DENSITY_MAT'
slider_6_min = math.floor( df2[slider_6_label].min())
slider_6_mean = round(df2[slider_6_label].mean())
slider_6_max =  round(df2[slider_6_label].max())

slider_7_label = 'FE_PANEL_DENSITY'
slider_7_min = math.floor( df2[slider_7_label].min())
slider_7_mean = round(df2[slider_7_label].mean())
slider_7_max =  round(df2[slider_7_label].max())

slider_8_label = 'FE_THICKNESS_RATIO'
slider_8_min = math.floor( df2[slider_8_label].min())
slider_8_mean = round(df2[slider_8_label].mean())
slider_8_max =  round(df2[slider_8_label].max())

# In[7]:




app = dash.Dash()

app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        
                        html.H1(children="Simulation TEST"),
                        
                        
                        dcc.Graph(figure=fig_features_importance),
                        
                        
                        html.H4(children=slider_1_label),

                        
                        dcc.Slider(
                            id='X1_slider',
                           min=0.0,
                           max=0.35,
                           value=0.5,
                           step=None,
                           marks={opacity: f'{opacity:.1f}' for opacity in [0.0, 0.1, 0.2, 0.3]}
                            ),

                       
                        html.H4(children=slider_2_label),

                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=10.0,
                            value=slider_2_mean,
                            marks={opacity: f'{opacity:.1f}' for opacity in [0, 20.5, 50.6, 60.1,80.3,85.4,109.4]}
                        ),

                        html.H4(children=slider_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=0.1,
                            value=slider_3_mean,
                            marks={i: '{} '.format(i) for i in range(slider_3_min, slider_3_max+1)}
                        ),
                        
                        html.H4(children=slider_4_label),

                        dcc.Slider(
                            id='X4_slider',
                            min=slider_4_min,
                            max=slider_4_max,
                            step=0.1,
                            value=slider_4_mean,
                            marks={i: '{} '.format(i) for i in range(slider_4_min, slider_4_max+1)}
                        ),
                        
                        #html.H4(children=slider_3_label),

                        #dcc.Slider(
                         #   id='X5_slider',
                         #   min=slider_5_min,
                         #   max=slider_5_max,
                         #   step=0.1,
                         #   value=slider_5_mean,
                         #   marks={i: '{} '.format(i) for i in range(slider_5_min, slider_5_max+1)}
                        #),
                        
                        html.H4(children=slider_6_label),

                        dcc.Slider(
                            id='X6_slider',
                            min=0,
                            max=0.1,
                            step=0.001,
                            value=slider_6_mean,
                            marks={opacity: f'{opacity:.1f}' for opacity in [0.0, 0.01, 0.02, 0.1]}
                        ),
                        
                        html.H4(children=slider_7_label),

                        dcc.Slider(
                            id='X7_slider',
                            min=slider_7_min,
                            max=slider_7_max,
                            step=0.1,
                            value=slider_7_mean,
                            marks={i: '{} '.format(i) for i in range(slider_7_min, slider_7_max+1)}
                        ),
                        
                        html.H4(children=slider_8_label),

                        dcc.Slider(
                            id='X8_slider',
                            min=slider_8_min,
                            max=1.28,
                            step=0.1,
                            value=slider_8_mean,
                            marks={opacity: f'{opacity:.1f}' for opacity in [0.0, 0.1, 0.7 ,1.2]},
                        ),
                        html.H2(id="prediction_result"),

                    ])


# In[ ]:



@app.callback(Output(component_id="prediction_result",component_property="children"),

              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"),Input("X4_slider","value"),
              Input("X6_slider","value"),Input("X7_slider","value"),Input("X8_slider","value")])


def update_prediction(X1, X2, X3,X4,X6,X7,X8):

    
    input_X = np.array([X1,
                       X2,
                      X3,
                       X4,X6,X7,X8
                      ]).reshape(1,-1)        
    
    
    prediction = model.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Prediccion: {}".format(round(prediction,2))

if __name__ == "__main__":
    #app.run_server()
    app.run_server(host='0.0.0.0', port=8080)
    
    



# In[2]:


#!pip freeze > requirements.txt


# In[ ]:




