from model_training import predict_route_optimization, stations_df

result = predict_route_optimization(
    start_lat=37.7, start_lon=-122.4,
    end_lat=37.8, end_lon=-122.5,
    battery_capacity=80,
    initial_charge=70,
    energy_consumption_rate=0.3,
    stations_df=stations_df
)
print(result)
