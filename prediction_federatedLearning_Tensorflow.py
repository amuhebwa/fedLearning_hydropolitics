import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import gc
import glob
from helper_functions import *
from utils import *

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
    test_regions = [*test_regions_Ids.keys()]
    model_path = f"{base_dir}/trained_models/{experiment_name}_stationIds*.h5"
    models_list = glob.glob(model_path)
    number_of_years = 14
    batch_size = 32
    lookback_days, forecast_days = 270, 1
    epochs = 100
    num_of_features = 0

    data_paths_dict = dict()
    test_region = test_regions.pop()
    current_stations = test_regions_Ids[test_region]
    for station_id in current_stations:
        data_path = f"{data_dir}/train_dataset/{station_id}.csv"
        data_paths_dict[station_id] = data_path

    # load and process each of the region's dataset.
    # load and process the rest of the regions
    complete_datasets_dict = dict()
    for station_id in data_paths_dict:
        data_path = data_paths_dict[station_id]
        print(data_path)
        current_dataset_df, _ = load_and_process_dataset(data_path, number_of_years)
        complete_datasets_dict[station_id] = current_dataset_df
        num_of_features = len(current_dataset_df.columns) - 1  # exclude streamflow

    model_path = models_list[station_index]
    model_name = model_path.split('/').pop().split('.')[0]
    # load keras model
    trained_model = load_model(model_path)
    expt_name_list, station_id_list, nse_list, kge_list, rbias_list, nrmse_list = [], [], [], [], [], []
    for test_station_id in complete_datasets_dict:
        test_dataset = complete_datasets_dict[test_station_id]
        x_test, y_test = create_dataset_forecast(test_dataset.to_numpy(), lookback_days, forecast_days)
        predictions = trained_model.predict(x_test)
        predictions = predictions.ravel()
        y_test = y_test.ravel()
        nse = calculate_NSE(y_test, predictions)
        kge = calculate_KGE(y_test, predictions)
        rbias = calculate_RBIAS(y_test, predictions)
        nrmse = calculate_NRMSE(y_test, predictions)
        expt_name_list.append(experiment_name)
        station_id_list.append(test_station_id)
        nse_list.append(nse)
        kge_list.append(kge)
        rbias_list.append(rbias)
        nrmse_list.append(nrmse)
        print(f" Current model for {test_station_id} : NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")
        # add the results to the list
        del test_dataset, x_test, y_test, predictions
        gc.collect()
    results_df = pd.DataFrame({"Experiment": expt_name_list, "StationId": station_id_list, "NSE": nse_list, "KGE": kge_list, "RBIAS": rbias_list, "NRMSE": nrmse_list})
    results_df.to_csv(f"{base_dir}/prediction_results/Heldout_{model_name}.csv", index=False)
