import code
import os
import numpy as np
import pandas as pd



if __name__=="__main__":
    base_dir = "/gypsum/eguide/projects/amuhebwa/Federated_Learning_HydroPolitics/all_regions_cleaned_data"
    """
    All regions. 
    """
    # regions = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']
    data_regions = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']
    test_regions = ['hysets', 'lamah']
    train_regions = [region for region in data_regions if region not in test_regions]

    number_of_stations_to_sample = 100

    regions_dict = dict()
    for region in train_regions:
        print(f"Processing {region} region")
        region_dir = f"{base_dir}/{region}"
        region_files = os.listdir(region_dir)
        # sample 100 files from each region
        region_files = np.random.choice(region_files, number_of_stations_to_sample, replace=False)
        station_ids = [region_file.split('_').pop().split('.')[0] for region_file in region_files]
        regions_dict[region] = station_ids
    code.interact(local=locals())
