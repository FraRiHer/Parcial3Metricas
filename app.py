import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np

# --- 1. Cargar el modelo previamente entrenado ---
# Asegúrate de que este archivo esté en la misma carpeta que app.py
MODEL_PATH = 'final_weather_classifier_model.joblib'
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}. Asegúrate de que '{MODEL_PATH}' esté en la misma carpeta que app.py.")
    exit() # Salir si el modelo no se encuentra
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit() # Salir si hay otro error al cargar

# --- 2. Definir el orden de las características originales ---
# Esto es CRUCIAL y debe coincidir con el orden de las columnas
# con las que se entrenó el preprocesador dentro de tu pipeline.
original_features_ordered = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
    'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season',
    'Visibility (km)', 'Location'
]

# --- 3. Definir las opciones para los dropdowns categóricos ---
# ¡IMPORTANTE! AJUSTA ESTOS VALORES SEGÚN LOS DATOS ÚNICOS DE TU DATASET ORIGINAL.
# Puedes obtenerlos de tu DataFrame original, por ejemplo: df['Cloud Cover'].unique().tolist()
cloud_cover_options = ['Clear', 'Partly Cloudy', 'Cloudy', 'Overcast']
season_options = ['Winter', 'Spring', 'Summer', 'Autumn'] # Ejemplo: 'Autumn', 'Spring', 'Summer', 'Winter'
location_options = ['Urban', 'Rural', 'Coastal', 'Mountain'] # Ejemplo: 'Mountain', 'Rural', 'Coastal', 'Urban'

# --- 4. Inicializar la aplicación Dash ---
# Se usa un tema de Bootstrap para mejorar la apariencia.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server # Necesario para el despliegue con Gunicorn

# --- 5. Definir el layout (la estructura visual) del dashboard ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("☁️ Predicción del Tipo de Clima ☀️❄️🌧️", className="text-center my-4"))),

    dbc.Row([
        # Columna para inputs numéricos
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Variables Numéricas"),
                dbc.CardBody([
                    dbc.Label("Temperatura (°C):"),
                    dbc.Input(id='temp-input', type='number', value=10, placeholder="Ej: 25.5"), # Valor inicial y placeholder
                    dbc.Label("Humedad (%):", className="mt-2"),
                    dbc.Input(id='humidity-input', type='number', value=60, placeholder="Ej: 70"),
                    dbc.Label("Velocidad del Viento (km/h):", className="mt-2"),
                    dbc.Input(id='wind-input', type='number', value=15, placeholder="Ej: 10"),
                    dbc.Label("Precipitación (%):", className="mt-2"),
                    dbc.Input(id='precip-input', type='number', value=5, placeholder="Ej: 0"),
                    dbc.Label("Presión Atmosférica (hPa):", className="mt-2"),
                    dbc.Input(id='pressure-input', type='number', value=1012, placeholder="Ej: 1010"),
                    dbc.Label("Índice UV:", className="mt-2"),
                    dbc.Input(id='uv-input', type='number', value=3, placeholder="Ej: 5"),
                    dbc.Label("Visibilidad (km):", className="mt-2"),
                    dbc.Input(id='visibility-input', type='number', value=10, placeholder="Ej: 8"),
                ])
            ])
        ], md=6), # md=6 significa que en pantallas medianas o más grandes, esta columna ocupa la mitad del ancho

        # Columna para inputs categóricos y el resultado de la predicción
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Variables Categóricas"),
                dbc.CardBody([
                    dbc.Label("Cobertura Nubosa:"),
                    dcc.Dropdown(id='cloud-input', options=[{'label': i, 'value': i} for i in cloud_cover_options], value=cloud_cover_options[0]), # Valor inicial
                    dbc.Label("Estación:", className="mt-2"),
                    dcc.Dropdown(id='season-input', options=[{'label': i, 'value': i} for i in season_options], value=season_options[0]),
                    dbc.Label("Ubicación:", className="mt-2"),
                    dcc.Dropdown(id='location-input', options=[{'label': i, 'value': i} for i in location_options], value=location_options[0]),
                ])
            ]),
            # Botón para activar la predicción
            html.Div(className="d-grid gap-2 mt-4", children=[
                dbc.Button('Predecir Clima', id='predict-button', n_clicks=0, color="primary", size="lg")
            ]),
            # Tarjeta para mostrar el resultado de la predicción
            dbc.Card(
                dbc.CardBody([
                    html.H4("Tipo de Clima Predicho:", className="card-title"),
                    html.Div(id='prediction-output', # Aquí se mostrará el resultado
                             className="lead",
                             style={'fontSize': '2rem', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'})
                ]), className="mt-4 text-center bg-light" # Estilos adicionales
            )
        ], md=6)
    ]),
    # Pie de página simple
    dbc.Row(dbc.Col(html.P("Este es un POC (Proof of Concept) para demostrar la funcionalidad del modelo de clasificación de clima.", className="text-center text-muted mt-5"))),
], fluid=True) # fluid=True hace que el contenedor use todo el ancho disponible

# --- 6. Definir el callback para la lógica de predicción ---
# Este decorador conecta los inputs y outputs del layout con la función Python
@app.callback(
    Output('prediction-output', 'children'), # El elemento a actualizar (el Div con id 'prediction-output')
    Input('predict-button', 'n_clicks'),    # El elemento que dispara el callback (el botón)
    # State permite tomar valores de otros inputs cuando el Input se activa, sin disparar el callback por sí mismos
    State('temp-input', 'value'),
    State('humidity-input', 'value'),
    State('wind-input', 'value'),
    State('precip-input', 'value'),
    State('cloud-input', 'value'),
    State('pressure-input', 'value'),
    State('uv-input', 'value'),
    State('season-input', 'value'),
    State('visibility-input', 'value'),
    State('location-input', 'value'),
    prevent_initial_call=True # Evita que el callback se ejecute cuando la app carga por primera vez
)
def update_prediction(n_clicks, temp, humidity, wind, precip, cloud, pressure, uv, season, visibility, location):
    # El callback se ejecuta solo si el botón ha sido clickeado al menos una vez
    if n_clicks > 0:
        # Crear un DataFrame con los datos de entrada.
        # El orden de las columnas DEBE ser el mismo que `original_features_ordered`.
        input_data_dict = {
            'Temperature': [temp],
            'Humidity': [humidity],
            'Wind Speed': [wind],
            'Precipitation (%)': [precip],
            'Cloud Cover': [cloud],
            'Atmospheric Pressure': [pressure],
            'UV Index': [uv],
            'Season': [season],
            'Visibility (km)': [visibility],
            'Location': [location]
        }
        # Crear el DataFrame asegurando el orden correcto de las columnas
        input_data = pd.DataFrame(input_data_dict)[original_features_ordered]


        # Asegurar tipos de datos numéricos para las columnas correspondientes
        # El preprocesador (SimpleImputer) manejará los NaNs si algún campo numérico está vacío
        numeric_cols_from_input = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                                   'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
        for col in numeric_cols_from_input:
            # Si el valor es None (campo vacío), se convertirá a NaN, que el imputer manejará.
            # Si no es None, se intenta convertir a numérico.
            if input_data[col].iloc[0] is not None:
                 input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
            else:
                input_data[col] = np.nan # Asegurar que sea NaN si es None

        try:
            # Realizar la predicción usando el pipeline cargado
            # El pipeline se encarga del preprocesamiento (escalado, one-hot encoding, etc.)
            prediction_array = model_pipeline.predict(input_data)
            prediction = prediction_array[0] # Obtener el valor de la predicción

            # Obtener probabilidades (opcional, pero bueno para mostrar confianza)
            if hasattr(model_pipeline, "predict_proba"):
                prediction_proba_array = model_pipeline.predict_proba(input_data)
                # Encontrar la probabilidad de la clase predicha
                # Obtener el índice de la clase predicha en la lista de clases del modelo
                predicted_class_index = list(model_pipeline.classes_).index(prediction)
                confidence = prediction_proba_array[0][predicted_class_index] * 100
                confidence_text = f"(Confianza: {confidence:.2f}%)"
            else:
                confidence_text = ""


            # Mapear predicción a un emoji para un output más visual
            emoji_map = {
                "Rainy": "🌧️ Lluvioso",
                "Sunny": "☀️ Soleado",
                "Cloudy": "☁️ Nublado",
                "Snowy": "❄️ Nevado"
            }
            
            # Devolver el resultado formateado
            return f"{emoji_map.get(prediction, prediction)} {confidence_text}"

        except Exception as e:
            # Si ocurre un error durante la predicción, mostrar un mensaje de error
            return f"Error al hacer la predicción: {str(e)}"
            
    return "" # Si no hay clicks, no mostrar nada o un mensaje inicial

# --- 7. Ejecutar la aplicación Dash ---
if __name__ == '__main__':
    # debug=True es útil para desarrollo, ya que actualiza la app automáticamente
    # y muestra mensajes de error detallados en el navegador.
    # Para producción, se suele poner debug=False.
    app.run_server(debug=True)
