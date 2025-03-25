import numpy as np
import pandas as pd
import networkx as nx
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. Generate Synthetic Charging Stations
def generate_charging_stations(num_stations=50):
    """
    Generate synthetic EV charging stations
    
    Parameters:
    - num_stations: Number of charging stations to generate
    
    Returns:
    - DataFrame with charging station details
    """
    # San Francisco Bay Area approximate coordinates
    lat_min, lat_max = 37.6, 37.8
    lon_min, lon_max = -122.5, -122.3

    charging_stations = pd.DataFrame({
        'Latitude': np.random.uniform(lat_min, lat_max, num_stations),
        'Longitude': np.random.uniform(lon_min, lon_max, num_stations),
        'Station Name': [f'Station_{i}' for i in range(num_stations)],
        'Capacity': np.random.choice(['50 kW', '75 kW', '100 kW', '150 kW'], num_stations),
        'Availability': np.random.uniform(0.7, 1.0, num_stations)
    })
    
    return charging_stations

# 2. Generate Synthetic Routes
def generate_routes(num_routes=100):
    """
    Generate synthetic EV route data
    
    Parameters:
    - num_routes: Number of routes to generate
    
    Returns:
    - DataFrame with route details
    """
    # San Francisco Bay Area approximate coordinates
    lat_min, lat_max = 37.6, 37.8
    lon_min, lon_max = -122.5, -122.3

    routes = []
    for _ in range(num_routes):
        # Generate start and end points
        start_lat = np.random.uniform(lat_min, lat_max)
        start_lon = np.random.uniform(lon_min, lon_max)
        end_lat = np.random.uniform(lat_min, lat_max)
        end_lon = np.random.uniform(lon_min, lon_max)
        
        # Calculate Haversine distance (approximate)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            
            # Convert degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            return distance

        route_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)

        # Generate EV-specific parameters
        battery_capacity = np.random.uniform(50, 100)  # Battery capacity in kWh
        initial_charge = np.random.uniform(50, 100)  # Initial charge percentage
        energy_consumption_rate = np.random.uniform(0.2, 0.4)  # kWh/km

        # Calculate estimated energy required
        estimated_energy_required = route_distance * energy_consumption_rate

        # Determine if charging is needed
        charging_needed = estimated_energy_required > (battery_capacity * initial_charge / 100)

        routes.append({
            'Start_Latitude': start_lat,
            'Start_Longitude': start_lon,
            'End_Latitude': end_lat,
            'End_Longitude': end_lon,
            'Route_Distance_km': route_distance,
            'Battery_Capacity_kWh': battery_capacity,
            'Initial_Charge_Percent': initial_charge,
            'Energy_Consumption_Rate_kWh_km': energy_consumption_rate,
            'Estimated_Energy_Required_kWh': estimated_energy_required,
            'Charging_Needed': charging_needed
        })

    return pd.DataFrame(routes)

# 3. Main Execution
def main():
    # Generate Charging Stations
    charging_stations = generate_charging_stations()
    
    # Generate Routes
    routes = generate_routes()

    # Save to CSV
    charging_stations.to_csv('ev_charging_stations.csv', index=False)
    routes.to_csv('ev_routes.csv', index=False)

    # Print dataset information
    print("Dataset Generation Complete!")
    print(f"Charging Stations Generated: {len(charging_stations)}")
    print(f"Routes Generated: {len(routes)}")

    # Preview Datasets
    print("\nCharging Stations Preview:")
    print(charging_stations.head())
    print("\nRoutes Preview:")
    print(routes.head())

    # Basic Statistics
    print("\nCharging Stations Statistics:")
    print(charging_stations.describe())
    
    print("\nRoutes Statistics:")
    print(routes.describe())

# Run the script
if __name__ == "__main__":
    main()