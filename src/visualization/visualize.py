from dash import *
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.read_csv("../data/raw/train.csv")

app.layout = html.Div(children=[
    html.H1('Insurance Sell'),
    dcc.Graph()
])

