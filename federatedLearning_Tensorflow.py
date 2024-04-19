import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from allModels import *

from helper_functions import *
from utils import *
import argparse
tf.random.set_seed(1234)
np.random.seed(1234)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--station_index', required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    station_index = args.station_index
    station_index = int(station_index)
    experiment_name = args.experiment_name  # could be allRegions, regionsA, regionsB, regionsC, regionsD, regionsE, ....
    experiment_name = str(experiment_name)

    base_dir = "/gypsum/eguide/projects/amuhebwa/Federated_Learning_HydroPolitics"
    data_dir = f"{base_dir}"
    train_regions = [*selected_regions_Ids.keys()]
    num_of_models = len(train_regions)
    number_of_train_rounds = 10

    # depending on the kind of experiment, we will eliminate one region.
    if experiment_name == "regionsA":
        train_regions.remove("camels")
    elif experiment_name == "regionsB":
        train_regions.remove("camelsbr")
    elif experiment_name == "regionsC":
        train_regions.remove("camelscl")
    elif experiment_name == "regionsD":
        train_regions.remove("camelsgb")
    elif experiment_name == "regionsE":
        train_regions.remove("lamah")
    else:
        pass

    # *************************
    # for testing, we will use just 2 years of data.
    # *************************
    number_of_years = 14

    batch_size = 32
    lookback_days, forecast_days = 270, 1
    epochs = 100
    num_of_features = 0

    data_paths_dict = dict()
    station_ids = []
    for region in train_regions:
        region_stations = selected_regions_Ids[region]
        current_id = str(region_stations[station_index])
        station_ids.append(current_id)
        print(f"Region {region} : Station {current_id}")
        data_path = f"{data_dir}/train_dataset/{current_id}.csv"
        # data_paths.append(data_path)
        data_paths_dict[region] = data_path
    str_station_ids = "_".join(station_ids)
    # load and process each of the region's dataset.
    # load and process the rest of the regions
    complete_datasets_dict = dict()
    for region in train_regions:
        data_path = data_paths_dict[region]
        print(data_path)
        current_dataset_df, _ = load_and_process_dataset(data_path, number_of_years)
        complete_datasets_dict[region] = current_dataset_df
        num_of_features = len(current_dataset_df.columns) - 1  # exclude streamflow

    # update number of features
    # create a new model
    master_model = create_model(lookback_days, forecast_days, num_of_features)
    _models = [create_model(lookback_days, forecast_days, num_of_features) for _ in range(num_of_models)]
    models_dict = dict(zip(train_regions, _models))

    """
    We need to store results for each of the models and the global model, at each iternation. 
    """
    model_type, regions_list, iterations_list, nse_list, kge_list, rbias_list, nrmse_list = [], [], [], [], [], [], []
    test_dataset_dict = dict()
    for train_round in range(number_of_train_rounds):
        print(f"Training iteration {train_round + 1} of {number_of_train_rounds}")
        for train_region, current_model in models_dict.items():
            print(f"Training model for region {train_region}")
            current_df = complete_datasets_dict[train_region]
            # split the data into train and test
            train_size = int(len(current_df) * 0.7)
            test_size = len(current_df) - train_size
            train_df, test_df = current_df.iloc[0:train_size], current_df.iloc[train_size:len(current_df)]
            x_train, y_train = create_dataset_forecast(train_df.to_numpy(), lookback_days, forecast_days)
            x_test, y_test = create_dataset_forecast(test_df.to_numpy(), lookback_days, forecast_days)
            # save the test data
            if train_region not in test_dataset_dict.keys():
                test_dataset_dict[train_region] = (x_test, y_test)

            checkpoint_path = f'{base_dir}/checkpoints/{experiment_name}_{str_station_ids}_model.h5'
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            # Define callbacks
            model_callbacks = [EarlyStopping(monitor='val_loss', patience=10), ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)]
            # Train the model
            current_model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, callbacks=model_callbacks)
            # Load the best model
            current_model.load_weights(checkpoint_path)

            # ===========================================================================================
            predictions = current_model.predict(x_test)
            y_test = y_test.ravel()
            predictions = predictions.ravel()
            nse = calculate_NSE(y_test, predictions)
            kge = calculate_KGE(y_test, predictions)
            rbias = calculate_RBIAS(y_test, predictions)
            nrmse = calculate_NRMSE(y_test, predictions)
            print(f" Current model for {train_region} : NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")
            # add the results to the list
            model_type.append("Local")
            regions_list.append(train_region)
            iterations_list.append(train_round)
            nse_list.append(nse)
            kge_list.append(kge)
            rbias_list.append(rbias)
            nrmse_list.append(nrmse)
            # print("===> Question for Aggrey: Do we need to get the inverse of the scaling for streamflow????")
            # ===========================================================================================
            del current_df, train_df, test_df, x_test, y_test, x_train, y_train, predictions

            models_dict[train_region] = current_model
        # Compute the average of the weights
        print("Computing the average of the weights")
        # get the values of models_dict
        list_of_models = list(models_dict.values())
        average_weights = average_lstm_weights(list_of_models)

        print("Updating the models with the average weights")
        for _region, _current_model in models_dict.items():
            _current_model.set_weights(average_weights)
            # We need to recomplie the model after updating the weights
            optzr = tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-3, momentum=0.9, centered=False)
            _current_model.compile(loss=tf.keras.losses.Huber(), optimizer=optzr)
            models_dict[_region] = _current_model
        # at the end of each round, update the master model with the average weights
        print("Updating the master model with the average weights")
        master_model.set_weights(average_weights)
        # We need to recomplie the model after updating the weights
        optzr = tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-3, momentum=0.9, centered=False)
        master_model.compile(loss=tf.keras.losses.Huber(), optimizer=optzr)
        # Evaluate the master model
        print("Evaluating the master model")
        for _train_region in train_regions:
            test_dataset = test_dataset_dict[_train_region]
            x_test, y_test = test_dataset
            predictions = master_model.predict(x_test)
            y_test = y_test.ravel()
            predictions = predictions.ravel()
            nse = calculate_NSE(y_test, predictions)
            kge = calculate_KGE(y_test, predictions)
            rbias = calculate_RBIAS(y_test, predictions)
            nrmse = calculate_NRMSE(y_test, predictions)
            print(f" NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")
            model_type.append("Global")
            regions_list.append(_train_region)
            iterations_list.append(train_round)
            nse_list.append(nse)
            kge_list.append(kge)
            rbias_list.append(rbias)
            nrmse_list.append(nrmse)
            del test_dataset, x_test, y_test, predictions

    # Save the final_model.
    str_station_ids = "_".join(station_ids)
    model2save = f"{base_dir}/trained_models/{experiment_name}_stationIds_{str_station_ids}.h5"
    # delete the model if it exists
    if os.path.exists(model2save):
        os.remove(model2save)
    master_model.save(model2save)
    print(f"Model saved to {model2save}")
    # Save the results
    results_df = pd.DataFrame({"Model": model_type, "Region": regions_list, "Iteration": iterations_list, "NSE": nse_list, "KGE": kge_list, "RBIAS": rbias_list, "NRMSE": nrmse_list})
    results_df.to_csv(f"{base_dir}/prediction_results/{experiment_name}_stationIds_{str_station_ids}.csv", index=False)
