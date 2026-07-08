# =============================================
# DASHBOARD: Heart Disease Prediction Dashboard
# Compatible with Python 3.14
# =============================================

# ---- TEMPORARY FIX FOR PYTHON 3.14 ----
import pkgutil
if not hasattr(pkgutil, "find_loader"):
    import importlib.util
    def find_loader(name):
        spec = importlib.util.find_spec(name)
        return spec.loader if spec else None
    pkgutil.find_loader = find_loader
# ---------------------------------------

# ---- IMPORT LIBRARIES ----
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---- LOAD DATA ----
data = pd.read_csv("D:/imp/azx/projects/pyspark_heartdisease/heart_disease_dataset.csv")

# Example coefficients from your trained logistic regression model
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
coefficients = [-0.0061, -1.9040, 0.7259, -0.0251, -0.0046, 0.1133, 0.1233,
                0.0246, -0.9796, -0.5242, 0.6879, -0.7635, -0.9291]

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})
coef_df['Impact'] = coef_df['Coefficient'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')

# ---- CREATE VISUALIZATIONS ----
bar_chart = px.bar(
    coef_df.sort_values(by='Coefficient'),
    x='Coefficient',
    y='Feature',
    color='Impact',
    color_discrete_map={'Increases Risk': 'red', 'Decreases Risk': 'blue'},
    orientation='h',
    title='Feature Importance — Logistic Regression (Heart Disease)'
)

corr_matrix = data.corr(numeric_only=True)
heatmap = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Heatmap")

pie_chart = px.pie(
    data,
    names='target',
    title='Heart Disease Distribution (0 = No Disease, 1 = Disease)',
    color='target',
    color_discrete_map={0: 'skyblue', 1: 'red'}
)

# ---- MODEL SETUP FOR PREDICTION ----
X = data[features]
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ---- DASH APP SETUP ----
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Heart Disease Prediction Dashboard", className="text-center text-primary my-4"),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="Overview", value="tab1"),
        dcc.Tab(label="Model Insights", value="tab2"),
        dcc.Tab(label="Prediction", value="tab3"),
    ]),
    
    html.Div(id="tab-content")
])

# ---- CALLBACK TO SWITCH TABS ----
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab1":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Dataset Overview", className="text-primary mb-3"),
                    html.Ul([
                        html.Li(f"Total Records: {data.shape[0]}"),
                        html.Li(f"Total Features: {data.shape[1] - 1}"),
                        html.Li("Target Variable: 'target' (1 = Heart Disease, 0 = No Disease)")
                    ]),
                    html.P("This dataset contains patient health indicators used to predict heart disease.", className="text-muted mt-2")
                ], width=4, style={"padding": "20px"}),

                dbc.Col([
                    html.H4("Class Distribution", className="text-primary mb-3"),
                    dcc.Graph(
                        figure=pie_chart.update_layout(
                            margin=dict(t=30, b=0, l=0, r=0),
                            height=350,
                            legend=dict(orientation="h", y=-0.2),
                        ),
                        style={"height": "400px"}
                    )
                ], width=8, style={"padding": "20px"})
            ], justify="around", align="center")
        ])


    elif tab == "tab2":
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=bar_chart)], width=6),
                dbc.Col([dcc.Graph(figure=heatmap)], width=6)
            ])
        ])

    elif tab == "tab3":
        return html.Div([
            html.H4("Heart Disease Probability Prediction"),
            html.P("Enter patient details to estimate disease probability:"),

            dbc.Row([
                dbc.Col([dcc.Input(id='age', type='number', placeholder='Age', style={'width': '100%', 'margin': '5px'})]),
                dbc.Col([dcc.Input(id='chol', type='number', placeholder='Cholesterol', style={'width': '100%', 'margin': '5px'})]),
                dbc.Col([dcc.Input(id='thalach', type='number', placeholder='Max Heart Rate', style={'width': '100%', 'margin': '5px'})])
            ]),
            
            dbc.Row([
                dbc.Col([dcc.Input(id='trestbps', type='number', placeholder='Resting BP', style={'width': '100%', 'margin': '5px'})]),
                dbc.Col([dcc.Input(id='oldpeak', type='number', placeholder='ST Depression', style={'width': '100%', 'margin': '5px'})]),
                dbc.Col([dcc.Input(id='sex', type='number', placeholder='Sex (1=Male, 0=Female)', style={'width': '100%', 'margin': '5px'})])
            ]),
            
            html.Button('Predict', id='predict-btn', n_clicks=0, className="btn btn-primary mt-3"),
            html.H4(id='prediction-output', className="mt-4 text-success")
        ])

# ---- CALLBACK FOR PREDICTION ----
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('chol', 'value'),
    State('thalach', 'value'),
    State('trestbps', 'value'),
    State('oldpeak', 'value'),
    State('sex', 'value')
)
def predict_heart_disease(n_clicks, age, chol, thalach, trestbps, oldpeak, sex):
    if n_clicks > 0 and None not in [age, chol, thalach, trestbps, oldpeak, sex]:
        # Create input vector (simplified subset of features)
        input_data = pd.DataFrame([[age, sex, 0, trestbps, chol, 0, 0, thalach, 0, oldpeak, 0, 0, 0]],
                                  columns=features)
        scaled_input = scaler.transform(input_data)
        prob = model.predict_proba(scaled_input)[0][1]
        prediction = "High Risk of Heart Disease" if prob > 0.5 else "Low Risk of Heart Disease"
        return f"Predicted Probability: {prob:.2f} → {prediction}"
    return ""


# ---- RUN THE DASHBOARD ----
if __name__ == '__main__':
    app.run(debug=True, port=8050)