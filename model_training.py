import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib

# Load the prepared data from previous script
routes_df = pd.read_csv('ev_routes.csv')
stations_df = pd.read_csv('ev_charging_stations.csv')

# Function to prepare features (same as previous script)
def prepare_features(routes_df, stations_df):
    # [Same implementation as in previous script]
    routes_df['Nearest_Station_Lat'] = None
    routes_df['Nearest_Station_Lon'] = None
    routes_df['Nearest_Station_Capacity'] = None
    routes_df['Nearest_Station_Availability'] = None

    for idx, route in routes_df.iterrows():
        stations_df['Distance'] = np.sqrt(
            (stations_df['Latitude'] - route['Start_Latitude'])**2 +
            (stations_df['Longitude'] - route['Start_Longitude'])**2
        )
        
        nearest_station = stations_df.loc[stations_df['Distance'].idxmin()]
        
        routes_df.at[idx, 'Nearest_Station_Lat'] = nearest_station['Latitude']
        routes_df.at[idx, 'Nearest_Station_Lon'] = nearest_station['Longitude']
        routes_df.at[idx, 'Nearest_Station_Capacity'] = int(nearest_station['Capacity'].replace(' kW', ''))
        routes_df.at[idx, 'Nearest_Station_Availability'] = nearest_station['Availability']

    features = [
        'Start_Latitude', 'Start_Longitude',
        'End_Latitude', 'End_Longitude',
        'Route_Distance_km', 'Battery_Capacity_kWh',
        'Initial_Charge_Percent', 'Energy_Consumption_Rate_kWh_km',
        'Nearest_Station_Lat', 'Nearest_Station_Lon',
        'Nearest_Station_Capacity', 'Nearest_Station_Availability'
    ]

    X = routes_df[features]
    y_charging = routes_df['Charging_Needed'].astype(int)
    y_distance = routes_df['Route_Distance_km']

    return X, y_charging, y_distance, routes_df

# Prepare the data
X, y_charging, y_distance, routes_df = prepare_features(routes_df, stations_df)

# Split the data
X_train, X_test, y_charging_train, y_charging_test, y_distance_train, y_distance_test = train_test_split(
    X, y_charging, y_distance, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Charging Need Prediction Model (Classification)
charging_model = RandomForestClassifier(n_estimators=100, random_state=42)
charging_model.fit(X_train_scaled, y_charging_train)

# Predict and evaluate charging need model
y_charging_pred = charging_model.predict(X_test_scaled)
print("Charging Need Prediction Report:")
print(classification_report(y_charging_test, y_charging_pred))

# 2. Route Distance Prediction Model (Regression)
distance_model = RandomForestRegressor(n_estimators=100, random_state=42)
distance_model.fit(X_train_scaled, y_distance_train)

# Predict and evaluate distance prediction model
y_distance_pred = distance_model.predict(X_test_scaled)
print("\nRoute Distance Prediction Metrics:")
print(f"Mean Squared Error: {mean_squared_error(y_distance_test, y_distance_pred)}")
print(f"R-squared Score: {r2_score(y_distance_test, y_distance_pred)}")

# Save models and scaler
joblib.dump(charging_model, 'charging_model.joblib')
joblib.dump(distance_model, 'distance_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')

# Create prediction function
def predict_route_optimization(start_lat, start_lon, end_lat, end_lon,
                                battery_capacity, initial_charge,
                                energy_consumption_rate,
                                stations_df):
    # Prepare input features
    input_features = pd.DataFrame({
        'Start_Latitude': [start_lat],
        'Start_Longitude': [start_lon],
        'End_Latitude': [end_lat],
        'End_Longitude': [end_lon],
        'Route_Distance_km': [np.sqrt((end_lat - start_lat)**2 + (end_lon - start_lon)**2) * 111],
        'Battery_Capacity_kWh': [battery_capacity],
        'Initial_Charge_Percent': [initial_charge],
        'Energy_Consumption_Rate_kWh_km': [energy_consumption_rate]
    })

    # Find nearest charging station
    stations_df['Distance'] = np.sqrt(
        (stations_df['Latitude'] - start_lat)**2 +
        (stations_df['Longitude'] - start_lon)**2
    )
    nearest_station = stations_df.loc[stations_df['Distance'].idxmin()]

    # Add station features
    input_features['Nearest_Station_Lat'] = nearest_station['Latitude']
    input_features['Nearest_Station_Lon'] = nearest_station['Longitude']
    input_features['Nearest_Station_Capacity'] = int(nearest_station['Capacity'].replace(' kW', ''))
    input_features['Nearest_Station_Availability'] = nearest_station['Availability']

    # Scale features
    input_scaled = scaler.transform(input_features)

    # Predict charging need and route distance
    charging_need = charging_model.predict(input_scaled)
    estimated_distance = distance_model.predict(input_scaled)

    return {
        'Charging_Need': bool(charging_need[0]),
        'Estimated_Route_Distance': estimated_distance[0],
        'Nearest_Charging_Station': {
            'Latitude': nearest_station['Latitude'],
            'Longitude': nearest_station['Longitude'],
            'Name': nearest_station['Station Name'],
            'Capacity': nearest_station['Capacity'],
            'Availability': nearest_station['Availability']
        }
    }

# Example usage (commented out)
# result = predict_route_optimization(
#     start_lat=37.7, start_lon=-122.4,
#     end_lat=37.8, end_lon=-122.5,
#     battery_capacity=80, initial_charge=70,
#     energy_consumption_rate=0.3,
#     stations_df=stations_df
# )
# print(result)
