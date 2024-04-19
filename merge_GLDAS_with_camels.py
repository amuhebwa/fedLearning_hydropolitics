import code

import pandas as pd
import numpy as np
import glob
import os

camels_columns = [
    'date', 'aridity', 'frac_snow', 'moisture_index', 'seasonality', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'sgr_dk_sav', 'glc_pc_s06', 'glc_pc_s07', 'nli_ix_sav',
    'glc_pc_s04', 'glc_pc_s05', 'glc_pc_s02', 'glc_pc_s03', 'glc_pc_s01', 'pet_mm_syr', 'glc_pc_s08', 'glc_pc_s09', 'tbi_cl_smj', 'crp_pc_sse', 'glc_pc_s22', 'glc_pc_s20', 'glc_pc_s21', 'wet_pc_sg1',
    'wet_pc_sg2', 'pac_pc_sse', 'clz_cl_smj', 'gwt_cm_sav', 'glc_pc_s17', 'glc_pc_s18', 'hft_ix_s93', 'glc_pc_s15', 'ire_pc_sse', 'glc_pc_s16', 'glc_pc_s13', 'prm_pc_sse', 'glc_pc_s14', 'glc_pc_s11',
    'glc_pc_s12', 'glc_pc_s10', 'kar_pc_sse', 'slp_dg_sav', 'glc_pc_s19', 'for_pc_sse', 'lit_cl_smj', 'cls_cl_smj', 'pre_mm_syr', 'pnv_pc_s01', 'pnv_pc_s04', 'pnv_pc_s05', 'pnv_pc_s02', 'rdd_mk_sav',
    'pnv_pc_s03', 'pnv_pc_s08', 'pnv_pc_s09', 'pnv_pc_s06', 'pnv_pc_s07', 'wet_cl_smj', 'snw_pc_syr', 'pnv_pc_s11', 'pnv_pc_s12', 'pnv_pc_s10', 'pnv_pc_s15', 'pnv_pc_s13', 'pnv_pc_s14', 'cmi_ix_syr',
    'wet_pc_s08', 'wet_pc_s09', 'slt_pc_sav', 'wet_pc_s02', 'wet_pc_s03', 'wet_pc_s01', 'hdi_ix_sav', 'wet_pc_s06', 'wet_pc_s07', 'wet_pc_s04', 'wet_pc_s05', 'fec_cl_smj', 'glc_cl_smj', 'swc_pc_syr',
    'hft_ix_s09', 'soc_th_sav', 'gdp_ud_sav', 'cly_pc_sav', 'ppd_pk_sav', 'ero_kh_sav', 'aet_mm_syr', 'ari_ix_sav', 'tmp_dc_syr', 'tec_cl_smj', 'fmh_cl_smj', 'pnv_cl_smj', 'run_mm_syr', 'ele_mt_sav',
    'urb_pc_sse', 'lka_pc_sse', 'inu_pc_smx', 'pst_pc_sse', 'dis_m3_pyr', 'lkv_mc_usu', 'rev_mc_usu', 'ria_ha_usu', 'riv_tc_usu', 'pop_ct_usu', 'dor_pc_pva', 'area_fraction_used_for_aggregation',
    'streamflow',
]


def compute_normalized_streamflow(row, epsilon=1e-6):
    """
    Compute normalized discharge for a given row.
    """
    # denominator = row['TotalPcpRate'] * row['slope'] * row['uparea'] + epsilon
    slope = row['slp_dg_sav']
    total_pcp_rate = row['Rainf_f_tavg']
    denominator = total_pcp_rate * slope + epsilon
    return row['streamflow'] / denominator

def transform_cols(v):
    return np.log10(np.sqrt(v) + 0.1)

if __name__ == "__main__":
    base_dir = "/gypsum/eguide/projects/amuhebwa/Federated_Learning_HydroPolitics"
    save_dir = f"{base_dir}/train_dataset"
    epsilon = 1e-6

    dataset_df = pd.read_csv("subset_of_all_stations.csv")
    # dictionary for storing region names and their corresponding station ids
    region2station_ids = dict()
    for i, row in dataset_df.iterrows():
        gauge_id = row['gauge_id']
        region_name = gauge_id.split('_')[0]
        climate_data_path = f"{base_dir}/climate_datasets/stationId_{gauge_id}.csv"
        camels_data_path = f"{base_dir}/all_regions_cleaned_data/{region_name}/stationId_{gauge_id}.csv"

        # check if both climate and camels data paths exist using os.path.exists
        if  os.path.exists(climate_data_path) and os.path.exists(camels_data_path):
            # load the climate data
            climate_df = pd.read_csv(climate_data_path)
            # load the camels data
            camels_df = pd.read_csv(camels_data_path)
            camels_df = camels_df[camels_columns]
            # rename 'date' to 'Date'
            camels_df = camels_df.rename(columns={'date': 'Date'})
            # convert Date to datetime
            camels_df['Date'] = pd.to_datetime(camels_df['Date'])
            climate_df['Date'] = pd.to_datetime(climate_df['Date'])
            # merge the two datasets
            merged_df = pd.merge(climate_df, camels_df, on='Date', how='inner')
            merged_df = merged_df[~merged_df['streamflow'].isna()]
            merged_df.insert(0, 'original_streamflow', merged_df['streamflow'].values)
            '''
            create a dimensionless streamflow
            '''
            slope = merged_df['slp_dg_sav']
            total_pcp_rate = merged_df['Rainf_f_tavg']
            merged_df['streamflow'] = merged_df['streamflow'] / (total_pcp_rate * slope + epsilon)
            #merged_df['streamflow'] = merged_df['streamflow'].apply(lambda x: np.log10(np.sqrt(x) + 0.1))
            merged_df['streamflow'] = merged_df['streamflow'].apply(transform_cols)
            name2save = f"{save_dir}/{gauge_id}.csv"
            if len(merged_df) > 365:
                merged_df.to_csv(name2save, index=False)
                region2station_ids[region_name] = region2station_ids.get(region_name, []) + [gauge_id]
                print(f"Saved {name2save}")
    # loop through the dictionary and print the region name and the corresponding station ids
    #for region_name, station_ids in region2station_ids.items():
    #    print(f"Region {region_name} : {station_ids}")
    print(region2station_ids)


