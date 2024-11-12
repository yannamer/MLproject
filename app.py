from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

app = Flask(__name__)

# Charger le dataset initial pour les valeurs par défaut
energy_data = pd.read_csv('./Energy_consumption.csv')

# Configurer les colonnes et le pipeline de prétraitement
categorical_features = ['HVACUsage', 'LightingUsage', 'RenewableEnergy', 'DayOfWeek', 'Holiday']
numerical_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy']

numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# OneHotEncoder avec `handle_unknown='ignore'` pour éviter les erreurs de catégories inconnues
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Convertisseur avec une étape supplémentaire pour convertir en dense
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
], sparse_threshold=0)  # Conversion automatique en dense

# Définir le modèle HistGradientBoosting avec les meilleurs paramètres
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingRegressor(learning_rate=0.1, max_depth=10, max_iter=200, random_state=0))
])

# Entraîner le modèle sur le dataset complet
X = energy_data.drop(columns=['EnergyConsumption', 'Timestamp'])
y = energy_data['EnergyConsumption']
pipeline.fit(X, y)

# Page principale avec le formulaire
@app.route('/')
def index():
    return render_template('index.html')

# Route pour générer les prévisions comparatives
@app.route('/forecast', methods=['POST'])
def forecast():
    # Récupérer les valeurs du formulaire
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    square_footage = float(request.form['square_footage'])
    occupancy = int(request.form['occupancy'])
    hvac_usage = request.form['hvac_usage']
    renewable_energy = request.form['renewable_energy']

    # Générer une série de données futures représentatives pour l'année
    days_in_year = 365
    amplitude = 15  # Variation de la température autour de la moyenne
    baseline_temp = temperature

    seasonal_temperature = [baseline_temp + amplitude * np.sin(2 * np.pi * day / days_in_year) for day in range(days_in_year)]
    seasonal_humidity = [humidity + np.random.normal(0, 5) for _ in range(days_in_year)]

    # Créer un DataFrame avec les valeurs répétées pour chaque jour de l'année avec la température normale
    year_data_normal_temp = pd.DataFrame({
        'Temperature': seasonal_temperature,
        'Humidity': seasonal_humidity,
        'SquareFootage': [square_footage] * days_in_year,
        'Occupancy': [occupancy] * days_in_year,
        'HVACUsage': [hvac_usage] * days_in_year,
        'LightingUsage': ['Yes'] * days_in_year,
        'RenewableEnergy': [renewable_energy] * days_in_year,
        'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] * (days_in_year // 7) + ['Monday'] * (days_in_year % 7),
        'Holiday': ['Yes'] * days_in_year
    })

    # Prédire la consommation énergétique pour chaque jour de l'année avec la température normale
    daily_predictions_normal = pipeline.predict(year_data_normal_temp)
    total_annual_consumption_normal = int(daily_predictions_normal.sum())

    # Créer un DataFrame avec la température fixée à 0
    year_data_zero_temp = year_data_normal_temp.copy()
    year_data_zero_temp['Temperature'] = [0] * days_in_year  # Température fixée à 0

    # Prédire la consommation énergétique pour chaque jour de l'année avec la température à 0
    daily_predictions_zero = pipeline.predict(year_data_zero_temp)
    total_annual_consumption_zero = int(daily_predictions_zero.sum())

    # Calculer la différence et l'économie en euros
    consumption_difference = total_annual_consumption_normal - total_annual_consumption_zero
    savings_euros = round(consumption_difference * 0.2516, 2)

    # Renvoyer les résultats à l'utilisateur
    return render_template(
        'forecast.html',
        total_annual_consumption_normal=total_annual_consumption_normal,
        total_annual_consumption_zero=total_annual_consumption_zero,
        consumption_difference=consumption_difference,
        savings_euros=savings_euros
    )

# API de sensibilité à la température
@app.route('/temperature_sensitivity', methods=['POST'])
def temperature_sensitivity():
    # Utiliser les valeurs de la dernière saisie du formulaire
    temperature = float(request.form.get('temperature', 20.0))
    humidity = float(request.form.get('humidity', 50.0))
    square_footage = float(request.form.get('square_footage', 1500))
    occupancy = int(request.form.get('occupancy', 4))
    hvac_usage = request.form.get('hvac_usage', "On")
    renewable_energy = request.form.get('renewable_energy', "No")

    # Définir la plage de températures autour de la température saisie
    temperature_range = np.arange(temperature - 5, temperature + 6, 1)
    
    # Créer les prévisions pour chaque température de la plage
    sensitivities = []
    for temp in temperature_range:
        year_data_temp_variation = pd.DataFrame({
            'Temperature': [temp] * 365,
            'Humidity': [humidity] * 365,
            'SquareFootage': [square_footage] * 365,
            'Occupancy': [occupancy] * 365,
            'HVACUsage': [hvac_usage] * 365,
            'LightingUsage': ['Yes'] * 365,
            'RenewableEnergy': [renewable_energy] * 365,
            'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] * (365 // 7) + ['Monday'] * (365 % 7),
            'Holiday': ['No'] * 365
        })

        # Prédiction pour cette température
        daily_predictions = pipeline.predict(year_data_temp_variation)
        total_consumption = int(daily_predictions.sum())
        sensitivities.append({
            "temperature": temp,
            "total_annual_consumption": total_consumption
        })

    # Retourner les prévisions pour chaque variation de température
    return jsonify(sensitivities=sensitivities)

if __name__ == '__main__':
    app.run(debug=True, port=5001)



