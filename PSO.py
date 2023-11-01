# Particle Swarm Optimization (PSO)
import random
import numpy as np
import pandas as pd

# Define data
data = pd.read_excel(r"C:\CI homework\homework 4\AirQualityUCI.xlsx")

# Define the column names you want to select
selected_feature_columns = ['Column2', 'Column5', 'Column7', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13']
target_column = 'Column4'

# Select and format the data using column names
selected_features = data[selected_feature_columns].values
target = data[target_column].values


def standardize_data(data):
    # Calculate the mean and standard deviation for each feature
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=0)  # Use ddof=0 for population standard deviation
    
    # Avoid division by zero by replacing stds with 1.0 if they are zero
    stds[stds == 0] = 1.0
    
    # Standardize the data using broadcasting
    standardized_data = (data - means) / stds
    
    return standardized_data

# Assuming 'selected_features' contains your data
selected_features_standardized = standardize_data(selected_features)
    
def initialize_weights(input_size, hidden_layers, nodes, output_size):
    weights = []
    layer_size = input_size
    for _ in range(hidden_layers):
        layer_weights = np.random.uniform(-0.1, 0.1, size=(nodes, layer_size))
        weights.append(layer_weights)
        layer_size = nodes
    output_weights = np.random.uniform(-0.1, 0.1, size=(output_size, layer_size))
    weights.append(output_weights)
    return weights

def feedforward(inputs, weights):
    for layer_weights in weights:
        inputs = np.maximum(0, np.dot(layer_weights, inputs))
    return inputs

def mean_absolute_error(y_true, y_pred):
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)
    return mae

def objective_function(params, X, y):
    num_hidden_layers, num_nodes = params
    input_size = X.shape[1]
    output_size = 1  # Since target is a single value
    weights = initialize_weights(input_size, num_hidden_layers, num_nodes, output_size)

    # Compute predictions for all input data in a more efficient way
    y_pred = feedforward(X.T, weights)  # Transpose X to have features along rows
    y_pred = y_pred[0]  # Access the first (and only) row

    loss = np.mean(np.abs(y - y_pred))  # Calculate MAE directly using NumPy functions
    return loss

def random_search(objective_function, num_searches, input_size, output_size):
    best_result = None
    best_mae = float('inf')

    for _ in range(num_searches):
        num_hidden_layers = random.randint(1, 6)
        num_nodes = random.randint(1, 60)
        params = (num_hidden_layers, num_nodes)
        mae = objective_function(params, selected_features, target)
        
        if mae < best_mae:
            best_mae = mae
            best_result = params
    
    return best_result, best_mae

# Initialize best result and best MAE
best_result = None
best_mae = float('inf')

# Number of random searches
num_random_searches = 300

for _ in range(num_random_searches):
    # Generate random parameters for the search
    num_hidden_layers = random.randint(1, 6)
    num_nodes = random.randint(1, 60)
    params = (num_hidden_layers, num_nodes)
    
    # Evaluate the objective function with the random parameters
    mae = objective_function(params, selected_features, target)
    
    # Update best result if the current result is better
    if mae < best_mae:
        best_mae = mae
        best_result = params

optimal_params = best_result

# Print the results
optimal_num_hidden_layers, optimal_num_nodes = optimal_params
print("Optimal Number of Hidden Layers:", optimal_num_hidden_layers)
print("Optimal Number of Nodes per Layer:", optimal_num_nodes)
print("Minimum MAE:", best_mae)

def generate_future_data(days_ahead, initial_data, volatility=6):
    future_data = []
    last_data_point = initial_data

    for _ in range(days_ahead):
        # Generate random values for each feature
        random_values = [random.normalvariate(0, volatility) for _ in last_data_point]
        
        # Create the next data point by adding the random values to the last data point
        future_data_point = [x + y for x, y in zip(last_data_point, random_values)]
        future_data.append(future_data_point)
        
        # Update the last data point
        last_data_point = future_data_point

    return future_data

# Generate future data for 5 days and 10 days ahead
initial_data = selected_features[-1]  # Use the last available data point as the initial data
future_data_5_days = generate_future_data(5, initial_data)
future_data_10_days = generate_future_data(10, initial_data)

# Use the trained neural network to make predictions for the generated future data
optimal_weights = initialize_weights(len(selected_features[0]), optimal_num_hidden_layers, optimal_num_nodes, 6)

def predict_future_data(future_data, weights):
    predicted_data = [feedforward(data_point, weights)[0] for data_point in future_data]
    return predicted_data

predicted_benzene_5_days = predict_future_data(future_data_5_days, optimal_weights)
predicted_benzene_10_days = predict_future_data(future_data_10_days, optimal_weights)


def print_and_use_predicted_results(future_data_5_days, future_data_10_days, predicted_benzene_5_days, predicted_benzene_10_days):
    print("Generated Future Data for 5 Days Ahead:")
    for data_point in future_data_5_days:
        print(data_point)

    print("\nGenerated Future Data for 10 Days Ahead:")
    for data_point in future_data_10_days:
        print(data_point)

    print("\nPredicted Benzene Concentration for 5 Days Ahead:")
    for prediction in predicted_benzene_5_days:
        print(prediction)

    print("\nPredicted Benzene Concentration for 10 Days Ahead:")
    for prediction in predicted_benzene_10_days:
        print(prediction)

    # You can also return the results if you want to use them in your code
    return future_data_5_days, future_data_10_days, predicted_benzene_5_days, predicted_benzene_10_days

# Call the function to print and use the predicted results
(
    future_data_5_days,
    future_data_10_days,
    predicted_benzene_5_days,
    predicted_benzene_10_days,
) = print_and_use_predicted_results(future_data_5_days, future_data_10_days, predicted_benzene_5_days, predicted_benzene_10_days)
