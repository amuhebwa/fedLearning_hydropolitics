import code
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from allModels import create_model
from utils import selected_regions_Ids
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

def create_dataset_forecast(_dataset, n_steps_in: int, n_steps_out: int):
    X, y = list(), list()
    for i in range(len(_dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(_dataset):
            break
        seq_x, seq_y = _dataset[i:end_ix, :-1], _dataset[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_dimensionless_flow(row, epsilon=1e-4):
    dimensionless_flow = row['streamflow'] / (row['total_precipitation_sum'] *row['slp_dg_sav'] * row['ria_ha_usu'] + epsilon)
    return dimensionless_flow

# apply log transform to cols
def transform_cols(v):
    return np.log10(np.sqrt(v) + 0.1)

# load and process each of the region's dataset.
def load_and_process_dataset(src_path, num_of_years):
    dataset_df = pd.read_csv(src_path)
    all_columns = dataset_df.columns.values
    stat_features = [col for col in dataset_df.columns if dataset_df[col].nunique() == 1]
    dynamic_features = [col for col in dataset_df.columns if (dataset_df[col].nunique() > 1) & (col !='date')]

    # format streamflow
    dataset_df = dataset_df[~dataset_df['streamflow'].isna()]

    dataset_df['original_streamflow'] = dataset_df['streamflow'].copy()
    dataset_df['streamflow'] = dataset_df.apply(create_dimensionless_flow, axis=1)

    dataset_df['streamflow'] = dataset_df['streamflow'].apply(transform_cols)

    # process static data
    static_df = dataset_df[stat_features]
    static_df += (np.random.rand(*static_df.shape)) * 0.01  # add a small amount of noise to the data
    static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
    static_df = pd.DataFrame(static_df, columns=stat_features)
    dataset_df[stat_features] = static_df[stat_features]

    # drop date column
    dataset_df.drop(columns=['date', 'original_streamflow'], inplace=True)
    '''
    For optimization, we might need to use only 10 years of data.
    Once we have a stronger machine, we will use all the data.
    '''
    dataset_df = dataset_df.iloc[-(365*num_of_years):, :]

    # process dynamic data
    scalers = dict()
    for _, current_column in enumerate(dynamic_features):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        dataset_df[current_column] = (current_scaler.fit_transform(dataset_df[current_column].values.reshape(-1, 1))).ravel()
    return dataset_df, scalers
if __name__=="__main__":


    """
    Since we are running these experiments in Parallel,
    we need to pass the index of the station id  from each of the regions.
    """
    print("station index should be passed from the terminal and should be between 0 and 99")

    station_index = 0
    experiment_name= "allRegions"
    base_dir = "/gypsum/eguide/projects/amuhebwa/Federated_Learning_HydroPolitics"
    data_dir = f"{base_dir}/all_regions_cleaned_data"

    train_regions = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb']
    num_of_models = len(train_regions)
    number_of_train_rounds = 3

    # *************************
    # for testing, we will use just 2 years of data.
    # *************************
    number_of_years = 2

    batch_size = 32
    lookback_days, forecast_days = 270, 1
    epochs = 5
    num_of_features = 0

    # Get data path
    data_paths, station_ids = [], []
    for region in train_regions:
        region_stations = selected_regions_Ids[region]
        current_id = str(region_stations[station_index])
        station_ids.append(current_id)
        print(f"Region {region} : Station {current_id}")
        data_path = f"{data_dir}/{region}/stationId_{region}_{current_id}.csv"
        data_paths.append(data_path)



    # load and process each of the region's dataset.
    # dataset_df, scalers = load_and_process_dataset(data_paths[0], number_of_years)
    # load and process the rest of the regions
    train_datasets = []
    for data_path in data_paths:
        current_dataset_df, _ = load_and_process_dataset(data_path, number_of_years)
        train_datasets.append(current_dataset_df)

    # update number of features
    num_of_features = len(train_datasets[0].columns)-1 # exclude streamflow
    # create a new model
    master_model = create_model(lookback_days, forecast_days, num_of_features)
    list_of_models = [create_model(lookback_days, forecast_days, num_of_features) for _ in range(num_of_models)]

    for train_round in range(number_of_train_rounds):
        print(f"Training iteration {train_round+1} of {number_of_train_rounds}")
        for i, current_model in enumerate(list_of_models):
            print(f"Training model {i+1} of {num_of_models}")
            current_df = train_datasets[i]
            train_region = train_regions[i]
            x_train, y_train = create_dataset_forecast(current_df.to_numpy(), lookback_days, forecast_days)

            # Define callbacks
            model_callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
            # Train the model
            current_model.fit(x_train, y_train, validation_split = 0.3, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, callbacks=model_callbacks)
            # add the model to the list of models
            list_of_models[i] = current_model
            del current_df, x_train, y_train, current_model

        # Compute the average of the weights
        print("Computing the average of the weights")
        average_weights = []
        for layer_weights in zip(*[model.get_weights() for model in list_of_models]):
            layer_average = np.mean(layer_weights, axis=0)
            average_weights.append(layer_average)

        # Set the average weights to be the new model weights
        print()
        for model in list_of_models:
            model.set_weights(average_weights)

        # at the end of each round, update the master model with the average weights
        print("Updating the master model with the average weights")
        master_model.set_weights(average_weights)

    # Save the final_model.
    str_station_ids = "_".join(station_ids)
    model2save = f"{base_dir}/trained_models/{experiment_name}_stationIds_{str_station_ids}.keras"
    # delete the model if it exists
    if os.path.exists(model2save):
        os.remove(model2save)
    master_model.save(model2save)
