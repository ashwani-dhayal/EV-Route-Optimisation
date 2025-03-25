import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the datasets
routes_df = pd.read_csv('ev_routes.csv')
stations_df = pd.read_csv('ev_charging_stations.csv')

# Prepare features for the model
def prepare_features(routes_df, stations_df):
    # Add station location features to routes
    routes_df['Nearest_Station_Lat'] = None
    routes_df['Nearest_Station_Lon'] = None
    routes_df['Nearest_Station_Capacity'] = None
    routes_df['Nearest_Station_Availability'] = None

    # Find nearest charging station for each route
    for idx, route in routes_df.iterrows():
        # Calculate distances to all stations
        stations_df['Distance'] = np.sqrt(
            (stations_df['Latitude'] - route['Start_Latitude'])**2 +
            (stations_df['Longitude'] - route['Start_Longitude'])**2
        )
        
        # Find the nearest station
        nearest_station = stations_df.loc[stations_df['Distance'].idxmin()]
        
        routes_df.at[idx, 'Nearest_Station_Lat'] = nearest_station['Latitude']
        routes_df.at[idx, 'Nearest_Station_Lon'] = nearest_station['Longitude']
        routes_df.at[idx, 'Nearest_Station_Capacity'] = int(nearest_station['Capacity'].replace(' kW', ''))
        routes_df.at[idx, 'Nearest_Station_Availability'] = nearest_station['Availability']

    # Select features for the model
    features = [
        'Start_Latitude', 'Start_Longitude',
        'End_Latitude', 'End_Longitude',
        'Route_Distance_km', 'Battery_Capacity_kWh',
        'Initial_Charge_Percent', 'Energy_Consumption_Rate_kWh_km',
        'Nearest_Station_Lat', 'Nearest_Station_Lon',
        'Nearest_Station_Capacity', 'Nearest_Station_Availability'
    ]

    X = routes_df[features]
    y = routes_df['Charging_Needed'].astype(int)  # Convert to binary

    return X, y, routes_df

# Prepare the data
X, y, routes_df = prepare_features(routes_df, stations_df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data Preparation Complete:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\nFeature Columns:")
print(X.columns.tolist())
