import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from scipy.stats import shapiro
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
# import statsmodels.tsa.api as smtsa
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest
import datetime  # Usar la biblioteca estándar datetime
plt.rcParams['text.usetex'] = False  # Desactivar el uso de LaTeX
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, close, subplots, title
# import requests
# import io
from statsmodels.tsa import stattools
import seaborn as sns
# from pmdarima import arima
# from pmdarima import datasets
# from pmdarima import utils
import warnings
import ta
from ta import trend
from statsmodels.tsa import seasonal
import os
import pickle
warnings.simplefilter(action="ignore", category=FutureWarning)
# from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import ExponentialSmoothing
# from sklearn.metrics import mean_squared_error
import pandas_datareader as pdr
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.metrics import r2_score
from statsmodels.graphics.gofplots import qqplot
import dash
from dash import dcc, html, dash_table
from statsmodels.tsa.stattools import adfuller
import base64
import io
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# Cargar los datos
data_pel = pd.read_csv('PEL.csv', parse_dates=["Date"], index_col="Date")
data_nvda = pd.read_csv('NVD.csv', parse_dates=["Date"], index_col="Date")

series_dict = {
    'Petróleo': data_pel,
    'NVIDIA': data_nvda
}

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Función para crear gráficos de descomposición aditiva
def create_decomposition_plot(data, selected_series):
    decompose_model = seasonal_decompose(data['Adj Close'], model='additive', period=20)
    fig, axarr = plt.subplots(4, sharex=True)
    fig.set_size_inches(7, 12)

    data['Adj Close'].plot(ax=axarr[0], color='b', linestyle='-')
    axarr[0].set_title(f'{selected_series}')

    pd.Series(data=decompose_model.trend, index=data.index).plot(ax=axarr[1], color='r', linestyle='-')
    axarr[1].set_title(f'Tendencia {selected_series}')

    pd.Series(data=decompose_model.seasonal, index=data.index).plot(ax=axarr[2], color='g', linestyle='-')
    axarr[2].set_title(f'Estacionalidad {selected_series}')

    pd.Series(data=decompose_model.resid, index=data.index).plot(ax=axarr[3], color='k', linestyle='-')
    axarr[3].set_title(f'Residual {selected_series}')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f'data:image/png;base64,{img}'

# Función para crear gráficos de ACF y PACF
def create_acf_pacf_figures(data, selected_series):
    # Gráfico de ACF
    fig_acf = plt.figure(figsize=(10, 6))
    plot_acf(data['Adj Close'], lags=40)
    plt.title(f'Autocorrelación (ACF) de {selected_series}')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_acf = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig_acf)

    # Gráfico de PACF
    fig_pacf = plt.figure(figsize=(10, 6))
    plot_pacf(data['Adj Close'], lags=40)
    plt.title(f'Autocorrelación Parcial (PACF) de {selected_series}')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_pacf = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig_pacf)

    return f'data:image/png;base64,{img_acf}', f'data:image/png;base64,{img_pacf}'

# Funciones para el rolling y sin rolling de las predicciones
def arima_sin_rolling(test, modelo):
    forecast_steps = len(test)
    forecast_values = modelo.forecast(steps=forecast_steps).values
    return forecast_values

def arima_rolling(history, test, best_order):
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=best_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])  # Append the observed value to history
    return predictions

# Crear las gráficas de predicciones para 7, 14, 21 y 28 días
def create_forecast_plot(train, test, yhat, dates_train, dates_test, criterio):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Horizonte de 7 días
    sns.lineplot(ax=axs[0, 0], x=dates_train[-100:], y=train[-100:], label="Train", color='#00008B')
    sns.lineplot(ax=axs[0, 0], x=dates_test[:7], y=test[:7], label="Test", color='#9A32CD')
    sns.lineplot(ax=axs[0, 0], x=dates_test[:7], y=yhat[:7], label="Forecast", color='#EE7600')
    axs[0, 0].set_title(f"Horizonte de 7 días - {criterio}")
    axs[0, 0].tick_params(axis='x', rotation=45)

    # Horizonte de 14 días
    sns.lineplot(ax=axs[0, 1], x=dates_train[-100:], y=train[-100:], label="Train", color='#00008B')
    sns.lineplot(ax=axs[0, 1], x=dates_test[:14], y=test[:14], label="Test", color='#9A32CD')
    sns.lineplot(ax=axs[0, 1], x=dates_test[:14], y=yhat[:14], label="Forecast", color='#EE7600')
    axs[0, 1].set_title(f"Horizonte de 14 días - {criterio}")
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Horizonte de 21 días
    sns.lineplot(ax=axs[1, 0], x=dates_train[-100:], y=train[-100:], label="Train", color='#00008B')
    sns.lineplot(ax=axs[1, 0], x=dates_test[:21], y=test[:21], label="Test", color='#9A32CD')
    sns.lineplot(ax=axs[1, 0], x=dates_test[:21], y=yhat[:21], label="Forecast", color='#EE7600')
    axs[1, 0].set_title(f"Horizonte de 21 días - {criterio}")
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Horizonte de 28 días
    sns.lineplot(ax=axs[1, 1], x=dates_train[-100:], y=train[-100:], label="Train", color='#00008B')
    sns.lineplot(ax=axs[1, 1], x=dates_test[:28], y=test[:28], label="Test", color='#9A32CD')
    sns.lineplot(ax=axs[1, 1], x=dates_test[:28], y=yhat[:28], label="Forecast", color='#EE7600')
    axs[1, 1].set_title(f"Horizonte de 28 días - {criterio}")
    axs[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_forecast = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return f'data:image/png;base64,{img_forecast}'

# Definir el layout de la app
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-principal', children=[
        dcc.Tab(label='Página Principal', value='tab-principal'),
        dcc.Tab(label='ARIMA', value='tab-ARIMA'),
        dcc.Tab(label='AIC', value='tab-AIC'),
        dcc.Tab(label='BIC', value='tab-BIC'),
        dcc.Tab(label='HQIC', value='tab-HQIC'),
    ]),
    
    # Dropdown para seleccionar la serie de tiempo
    html.Div([
        html.Label("Seleccione una serie de tiempo:"),
        dcc.Dropdown(
            id='dropdown-series',
            options=[{'label': 'Petróleo', 'value': 'Petróleo'},
                     {'label': 'NVIDIA', 'value': 'NVIDIA'}],
            value='Petróleo',  # Valor por defecto
            clearable=False
        )
    ], style={'margin': '20px'}),
    
    html.Div(id='tabs-content')
])

# Callback para cambiar el contenido según la pestaña seleccionada
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('dropdown-series', 'value')]
)
def render_content(tab, selected_series):
    data = series_dict[selected_series]

    if tab == 'tab-principal':
        fig_candlestick = go.Figure(data=[
            go.Candlestick(x=data.index,
                           open=data['Open'],
                           high=data['High'],
                           low=data['Low'],
                           close=data['Close'])
        ])
        fig_candlestick.update_layout(
            title=f"Precios de {selected_series}",
            xaxis_title="Fecha",
            yaxis_title=f"{selected_series} USD",
            xaxis_rangeslider_visible=False
        )

        summary_table = dash_table.DataTable(
            id='summary-table',
            columns=[{'name': col, 'id': col} for col in data.describe().columns],
            data=data.describe().reset_index().to_dict('records'),
            style_header={'backgroundColor': 'blue', 'color': 'white'},
            style_cell={'textAlign': 'center', 'fontSize': '14px', 'padding': '5px'}
        )

        decomposition_img = create_decomposition_plot(data, selected_series)
        acf_img, pacf_img = create_acf_pacf_figures(data, selected_series)

        return html.Div([
            html.H3(f'Visualización de {selected_series}'),
            dcc.Graph(id='precio-graph', figure=fig_candlestick),
            html.H4(f'Resumen estadístico de {selected_series}'),
            summary_table,
            html.H4(f'Descomposición Aditiva de {selected_series}'),
            html.Img(src=decomposition_img),
            html.H4(f'Autocorrelación (ACF) de {selected_series}'),
            html.Img(src=acf_img),
            html.H4(f'Autocorrelación Parcial (PACF) de {selected_series}'),
            html.Img(src=pacf_img)
        ])
    
    elif tab == 'tab-ARIMA':
        # Contenido para ARIMA
        if selected_series == 'NVIDIA':
            fixed_scores = pd.DataFrame({
                'lags': [3, 4, 4],
                'dif': [1, 1, 1],
                'ave': [4, 4, 4],
                'aic': [5056.34, 5057.07, 5108.79],
                'bic': [5101.63, 5108.02, 5142.75],
                'hqic': [5072.92, 5075.72, 5121.22]
            })
        elif selected_series == 'Petróleo':
            fixed_scores = pd.DataFrame({
                'lags': [0, 1, 2],
                'dif': [1, 1, 1],
                'ave': [0, 1, 2],
                'aic': [795.12, 810.34, 835.67],
                'bic': [798.52, 814.32, 839.01],
                'hqic': [796.49, 811.90, 837.10]
            })

        score_table = dash_table.DataTable(
            id='score-table',
            columns=[{'name': i, 'id': i} for i in fixed_scores.columns],
            data=fixed_scores.to_dict('records'),
            style_header={'backgroundColor': 'blue', 'color': 'white'},
            style_cell={'textAlign': 'center', 'fontSize': '14px', 'padding': '5px'}
        )
        
        decomposition_img = create_decomposition_plot(data, selected_series)
        
        return html.Div([
            html.H4(f'Puntajes Fijos para {selected_series} - AIC, BIC y HQIC'),
            score_table,
            html.H4('Descomposición Aditiva de la serie seleccionada'),
            html.Img(src=decomposition_img)
        ])
    
    elif tab == 'tab-AIC':
        # Contenido para AIC
        return html.Div([
            html.H4('Predicción basada en AIC'),
            dcc.Dropdown(
                id='dropdown-rolling-aic',
                options=[{'label': 'Con Rolling', 'value': 'con'},
                         {'label': 'Sin Rolling', 'value': 'sin'}],
                value='sin',  # Valor por defecto
                clearable=False
            ),
            html.Div(id='aic-forecast-graph')
        ])

    elif tab == 'tab-BIC':
        return html.Div([
            html.H4('Predicción basada en BIC'),
            dcc.Dropdown(
                id='dropdown-rolling-bic',
                options=[{'label': 'Con Rolling', 'value': 'con'},
                         {'label': 'Sin Rolling', 'value': 'sin'}],
                value='sin',
                clearable=False
            ),
            html.Div(id='bic-forecast-graph')
        ])

    elif tab == 'tab-HQIC':
        return html.Div([
            html.H4('Predicción basada en HQIC'),
            dcc.Dropdown(
                id='dropdown-rolling-hqic',
                options=[{'label': 'Con Rolling', 'value': 'con'},
                         {'label': 'Sin Rolling', 'value': 'sin'}],
                value='sin',
                clearable=False
            ),
            html.Div(id='hqic-forecast-graph')
        ])

# Callback para actualizar las gráficas de AIC
@app.callback(
    Output('aic-forecast-graph', 'children'),
    [Input('dropdown-rolling-aic', 'value')]
)
def update_aic_graph(rolling_option):
    # Asignación de variables para AIC
    train = data_nvda['Adj Close'][:2127]
    test = data_nvda['Adj Close'][2127:2157]
    dates_train = data_nvda.index[:2127]
    dates_test = data_nvda.index[2127:2157]

    model_fit_aic = ARIMA(train, order=(4, 1, 4)).fit()

    if rolling_option == 'sin':
        yhat = arima_sin_rolling(test, model_fit_aic)
    else:
        yhat = arima_rolling(train.tolist(), test.tolist(), (4, 1, 4))
    
    return html.Img(src=create_forecast_plot(train, test, yhat, dates_train, dates_test, "AIC"))

# Callback para actualizar las gráficas de BIC
@app.callback(
    Output('bic-forecast-graph', 'children'),
    [Input('dropdown-rolling-bic', 'value')]
)
def update_bic_graph(rolling_option):
    train = data_nvda['Adj Close'][:2127]
    test = data_nvda['Adj Close'][2127:2157]
    dates_train = data_nvda.index[:2127]
    dates_test = data_nvda.index[2127:2157]

    model_fit_bic = ARIMA(train, order=(4, 1, 4)).fit()

    if rolling_option == 'sin':
        yhat = arima_sin_rolling(test, model_fit_bic)
    else:
        yhat = arima_rolling(train.tolist(), test.tolist(), (4, 1, 4))

    return html.Img(src=create_forecast_plot(train, test, yhat, dates_train, dates_test, "BIC"))

# Callback para actualizar las gráficas de HQIC
@app.callback(
    Output('hqic-forecast-graph', 'children'),
    [Input('dropdown-rolling-hqic', 'value')]
)
def update_hqic_graph(rolling_option):
    train = data_nvda['Adj Close'][:2127]
    test = data_nvda['Adj Close'][2127:2157]
    dates_train = data_nvda.index[:2127]
    dates_test = data_nvda.index[2127:2157]

    model_fit_hqic = ARIMA(train, order=(4, 1, 4)).fit()

    if rolling_option == 'sin':
        yhat = arima_sin_rolling(test, model_fit_hqic)
    else:
        yhat = arima_rolling(train.tolist(), test.tolist(), (4, 1, 4))

    return html.Img(src=create_forecast_plot(train, test, yhat, dates_train, dates_test, "HQIC"))


# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)
