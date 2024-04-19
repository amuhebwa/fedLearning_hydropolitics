import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset  # Import DateOffset
import ee
import code

ee.Initialize()



column_names = [
    "Albedo_inst", "AvgSurfT_inst", "CanopInt_inst", "ECanop_tavg", "ESoil_tavg","Evap_tavg","LWdown_f_tavg","Lwnet_tavg","PotEvap_tavg","Psurf_f_inst","Qair_f_inst","Qg_tavg","Qh_tavg","Qle_tavg",
    "Qs_acc","Qsb_acc","Qsm_acc","Rainf_f_tavg","Rainf_tavg","RootMoist_inst","SWE_inst","SWdown_f_tavg","SnowDepth_inst","Snowf_tavg","SoilMoi0_10cm_inst","SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst","SoilTMP0_10cm_inst","SoilTMP10_40cm_inst","SoilTMP40_100cm_inst","SoilTMP100_200cm_inst","Swnet_tavg","Tair_f_inst","Tveg_tavg", "Wind_f_inst"
]

# Function to download climate data for a station
def download_climate_data(row, image_collection, column_names, data_storage_dir):
    startDate = datetime(2000, 1, 1)
    endDate = datetime(2014, 12, 31)
    step = DateOffset(months=1)  # Use DateOffset instead of relativedelta
    comid = str(row.gauge_id)

    print('----> starting downloads for comid:', comid, ' <----')

    # Create a date range
    date_range = pd.date_range(start=startDate, end=endDate, freq=step)

    df_list = []

    for i in range(len(date_range) - 1):
        start_date = date_range[i]
        end_date = date_range[i + 1]
        climate_data = image_collection.filterDate(start_date, end_date)
        print(f"Start Date: {start_date}, End Date: {end_date}")

        point = {'type': 'Point', 'coordinates': [float(row.gauge_lon), float(row.gauge_lat)]}

        try:
            info_dict = climate_data.getRegion(point, 20000).getInfo()
            header = info_dict[0]
            data = np.array(info_dict[1:])
            iTime = header.index('time')
            time_days = [datetime.fromtimestamp(i / 1000) for i in (data[:, iTime].astype(int))]
            iBands = [header.index(b) for b in column_names]
            df_arr = data[:, iBands].astype(np.float64)
            temp_df = pd.DataFrame(data=df_arr, index=time_days, columns=column_names)
            temp_df = temp_df.resample('D').mean()
            temp_df = temp_df.interpolate()
            temp_df = temp_df.dropna().reset_index()
            temp_df = temp_df.rename(columns={'index': 'Date'})
            temp_df['Date'] = pd.to_datetime(temp_df['Date'])
            temp_df['Date'] = temp_df['Date'].dt.date
            df_list.append(temp_df)
        except Exception as e:
            print(e)
            pass

    if len(df_list) != 0:
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        output_file_path = os.path.join(data_storage_dir, f"stationId_{str(comid)}.csv")
        final_df.to_csv(output_file_path, index=False)
        print(f"Data saved to: {output_file_path}")
        del column_names, image_collection


if __name__=="__main__":
    base_dir = '/gypsum/eguide/projects/amuhebwa/Federated_Learning_HydroPolitics'
    stations_coords = pd.read_csv(f"{base_dir}/subset_of_all_stations.csv")
    # stations_coords = stations_coords.sample(frac=1, random_state=12345).reset_index(drop=True)
    model_name = "GLDAS"
    data_storage_dir = f"{base_dir}/climate_datasets"

    if not os.path.exists(data_storage_dir):
        os.makedirs(data_storage_dir)

    # image_collection = ee.ImageCollection("NASA/GLDAS/V20/NOAH/G025/T3H")
    # Loop through stations and download climate data
    image_collection = ee.ImageCollection("NASA/GLDAS/V20/NOAH/G025/T3H")
    for i, row in stations_coords.iterrows():
        # image_collection = ee.ImageCollection("NASA/GLDAS/V20/NOAH/G025/T3H")
        download_climate_data(row, image_collection, column_names, data_storage_dir)
        time.sleep(10)
