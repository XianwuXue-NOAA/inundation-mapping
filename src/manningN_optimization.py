
import json
import argparse
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import os
from os.path import join
import re
import sys
import traceback
import warnings
import csv
import numpy as np

import geopandas as gpd
import pandas as pd
from scipy.optimize import minimize
import random

from run_test_case import Test_Case
from tools_shared_variables import (
    AHPS_BENCHMARK_CATEGORIES,
    MAGNITUDE_DICT,
    PREVIOUS_FIM_DIR,
    TEST_CASES_DIR,
)

fim_dir = "/home/rdp-user/outputs/mno_11010004_cal_off_0710/" 
huc = "11010004" 
mannN_file_aibased = "/efs-drives/fim-dev-efs/fim-data/inputs/rating_curve/variable_roughness/ml_outputs_v1.01.parquet"

# *********************************************************
def create_master_metrics_csv(fim_version): #prev_metrics_csv, 
    """
    This function searches for and collates metrics into a single CSV file that can queried database-style.
        The CSV is an input to eval_plots.py.
        This function automatically looks for metrics produced for official versions and loads them into
            memory to be written to the output CSV.

    Args:
        master_metrics_csv_output (str)    : Full path to CSV output.
                                                If a file already exists at this path, it will be overwritten.
        dev_versions_to_include_list (list): A list of non-official FIM version names.
                                                If a user supplied information on the command line using the
                                                -dc flag, then this function will search for metrics in the
                                                "testing_versions" library of metrics and include them in
                                                the CSV output.
    """


    # Default to processing all possible versions in PREVIOUS_FIM_DIR.
    config = "DEV"
    # Specify which results to iterate through
    if config == 'DEV':
        iteration_list = [
            'official',
            'testing',
        ]  # iterating through official model results AND testing model(s)
    else:
        iteration_list = ['official']  # only iterating through official model results

    prev_versions_to_include_list = []
    dev_versions_to_include_list = []

    prev_versions_to_include_list = os.listdir(PREVIOUS_FIM_DIR)
    if config == 'DEV':  # development fim model results
        dev_versions_to_include_list = [fim_version]

    # Construct header
    metrics_to_write = [
        'true_negatives_count',
        'false_negatives_count',
        'true_positives_count',
        'false_positives_count',
        'contingency_tot_count',
        'cell_area_m2',
        'TP_area_km2',
        'FP_area_km2',
        'TN_area_km2',
        'FN_area_km2',
        'contingency_tot_area_km2',
        'predPositive_area_km2',
        'predNegative_area_km2',
        'obsPositive_area_km2',
        'obsNegative_area_km2',
        'positiveDiff_area_km2',
        'CSI',
        'FAR',
        'TPR',
        'TNR',
        'PND',
        'PPV',
        'NPV',
        'ACC',
        'Bal_ACC',
        'MCC',
        'EQUITABLE_THREAT_SCORE',
        'PREVALENCE',
        'BIAS',
        'F1_SCORE',
        'TP_perc',
        'FP_perc',
        'TN_perc',
        'FN_perc',
        'predPositive_perc',
        'predNegative_perc',
        'obsPositive_perc',
        'obsNegative_perc',
        'positiveDiff_perc',
        'masked_count',
        'masked_perc',
        'masked_area_km2',
    ]

    # Create table header
    additional_header_info_prefix = ['version', 'nws_lid', 'magnitude', 'huc']
    list_to_write = [
        additional_header_info_prefix
        + metrics_to_write
        + ['full_json_path']
        + ['flow']
        + ['benchmark_source']
        + ['extent_config']
        + ["calibrated"]
    ]

    # add in composite of versions (used for previous FIM3 versions)
    if "official" in iteration_list:
        composite_versions = [v.replace('_ms', '_comp') for v in prev_versions_to_include_list if '_ms' in v]
        prev_versions_to_include_list += composite_versions

    # Iterate through 5 benchmark sources
    for benchmark_source in ['ble', 'nws', 'usgs', 'ifc', 'ras2fim']:
        benchmark_test_case_dir = os.path.join(TEST_CASES_DIR, benchmark_source + '_test_cases')
        test_cases_list = [d for d in os.listdir(benchmark_test_case_dir) if re.match(r'\d{8}_\w{3,7}', d)]

        if benchmark_source in ['ble', 'ifc', 'ras2fim']:
            magnitude_list = MAGNITUDE_DICT[benchmark_source]

            # Iterate through available test cases
            for each_test_case in test_cases_list:
                try:
                    # Get HUC id
                    int(each_test_case.split('_')[0])
                    huc = each_test_case.split('_')[0]

                    # Update filepaths based on whether the official or dev versions should be included
                    for iteration in iteration_list:
                        if (
                            iteration == "official"
                        ):  # and str(pfiles) == "True": # "official" refers to previous finalized model versions
                            versions_to_crawl = os.path.join(
                                benchmark_test_case_dir, each_test_case, 'official_versions'
                            )
                            versions_to_aggregate = prev_versions_to_include_list

                        if (
                            iteration == "testing"
                        ):  # "testing" refers to the development model version(s) being evaluated
                            versions_to_crawl = os.path.join(
                                benchmark_test_case_dir, each_test_case, 'testing_versions'
                            )
                            versions_to_aggregate = dev_versions_to_include_list

                        # Pull version info from filepath
                        for magnitude in magnitude_list:
                            for version in versions_to_aggregate:
                                if '_ms' in version:
                                    extent_config = 'MS'
                                elif ('_fr' in version) or (version == 'fim_2_3_3'):
                                    extent_config = 'FR'
                                else:
                                    extent_config = 'COMP'
                                if "_c" in version and version.split('_c')[1] == "":
                                    calibrated = "yes"
                                else:
                                    calibrated = "no"
                                version_dir = os.path.join(versions_to_crawl, version)
                                magnitude_dir = os.path.join(version_dir, magnitude)

                                # Add metrics from file to metrics table ('list_to_write')
                                if os.path.exists(magnitude_dir):
                                    magnitude_dir_list = os.listdir(magnitude_dir)
                                    for f in magnitude_dir_list:
                                        if '.json' in f:
                                            flow = 'NA'
                                            nws_lid = "NA"
                                            sub_list_to_append = [version, nws_lid, magnitude, huc]
                                            full_json_path = os.path.join(magnitude_dir, f)
                                            if os.path.exists(full_json_path):
                                                stats_dict = json.load(open(full_json_path))
                                                for metric in metrics_to_write:
                                                    sub_list_to_append.append(stats_dict[metric])
                                                sub_list_to_append.append(full_json_path)
                                                sub_list_to_append.append(flow)
                                                sub_list_to_append.append(benchmark_source)
                                                sub_list_to_append.append(extent_config)
                                                sub_list_to_append.append(calibrated)

                                                list_to_write.append(sub_list_to_append)
                except ValueError:
                    pass

        # Iterate through AHPS benchmark data
        if benchmark_source in AHPS_BENCHMARK_CATEGORIES:
            test_cases_list = os.listdir(benchmark_test_case_dir)

            for each_test_case in test_cases_list:
                try:
                    # Get HUC id
                    int(each_test_case.split('_')[0])
                    huc = each_test_case.split('_')[0]

                    # Update filepaths based on whether the official or dev versions should be included
                    for iteration in iteration_list:
                        if iteration == "official":  # "official" refers to previous finalized model versions
                            versions_to_crawl = os.path.join(
                                benchmark_test_case_dir, each_test_case, 'official_versions'
                            )
                            versions_to_aggregate = prev_versions_to_include_list

                        if (
                            iteration == "testing"
                        ):  # "testing" refers to the development model version(s) being evaluated
                            versions_to_crawl = os.path.join(
                                benchmark_test_case_dir, each_test_case, 'testing_versions'
                            )
                            versions_to_aggregate = dev_versions_to_include_list

                        # Pull model info from filepath
                        for magnitude in ['action', 'minor', 'moderate', 'major']:
                            for version in versions_to_aggregate:
                                if '_ms' in version:
                                    extent_config = 'MS'
                                elif ('_fr' in version) or (version == 'fim_2_3_3'):
                                    extent_config = 'FR'
                                else:
                                    extent_config = 'COMP'
                                if "_c" in version and version.split('_c')[1] == "":
                                    calibrated = "yes"
                                else:
                                    calibrated = "no"

                                version_dir = os.path.join(versions_to_crawl, version)
                                magnitude_dir = os.path.join(version_dir, magnitude)

                                if os.path.exists(magnitude_dir):
                                    magnitude_dir_list = os.listdir(magnitude_dir)
                                    for f in magnitude_dir_list:
                                        if '.json' in f and 'total_area' not in f:
                                            nws_lid = f[:5]
                                            sub_list_to_append = [version, nws_lid, magnitude, huc]
                                            full_json_path = os.path.join(magnitude_dir, f)
                                            flow = ''
                                            if os.path.exists(full_json_path):
                                                # Get flow used to map
                                                flow_file = os.path.join(
                                                    benchmark_test_case_dir,
                                                    'validation_data_' + benchmark_source,
                                                    huc,
                                                    nws_lid,
                                                    magnitude,
                                                    'ahps_'
                                                    + nws_lid
                                                    + '_huc_'
                                                    + huc
                                                    + '_flows_'
                                                    + magnitude
                                                    + '.csv',
                                                )
                                                if os.path.exists(flow_file):
                                                    with open(flow_file, newline='') as csv_file:
                                                        reader = csv.reader(csv_file)
                                                        next(reader)
                                                        for row in reader:
                                                            flow = row[1]

                                                # Add metrics from file to metrics table ('list_to_write')
                                                stats_dict = json.load(open(full_json_path))
                                                for metric in metrics_to_write:
                                                    sub_list_to_append.append(stats_dict[metric])
                                                sub_list_to_append.append(full_json_path)
                                                sub_list_to_append.append(flow)
                                                sub_list_to_append.append(benchmark_source)
                                                sub_list_to_append.append(extent_config)
                                                sub_list_to_append.append(calibrated)
                                                list_to_write.append(sub_list_to_append)
                except ValueError:
                    pass

    df_to_write = pd.DataFrame(list_to_write)
    df_to_write.columns = df_to_write.iloc[0]
    df_to_write = df_to_write[1:]

    return df_to_write


# *********************************************************
def run_test_cases(fim_version): #prev_metrics_csv, 
    """
    This function
    """

    print("================================")
    print("Start synthesize test cases")
    start_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print(f"started: {dt_string}")
    print()

    # Default to processing all possible versions in PREVIOUS_FIM_DIR.
    fim_version = fim_version #"all"

    # Create a list of all test_cases for which we have validation data
    archive_results = False
    benchmark_category = "all"
    all_test_cases = Test_Case.list_all_test_cases(
        version=fim_version,
        archive=archive_results,
        benchmark_categories=[] if benchmark_category == "all" else [benchmark_category],
    )    
    model = "GMS"
    job_number_huc = 1
    overwrite=True
    verbose=False
    calibrated = False
    job_number_branch = 5
    # Set up multiprocessor
    with ProcessPoolExecutor(max_workers=job_number_huc) as executor:
        # Loop through all test cases, build the alpha test arguments, and submit them to the process pool
        executor_dict = {}
        for test_case_class in all_test_cases:
            # if not os.path.exists(test_case_class.fim_dir):
            #     continue
            alpha_test_args = {
                'calibrated': calibrated,
                'model': model,
                'mask_type': 'huc',
                'overwrite': overwrite,
                'verbose': verbose,
                'gms_workers': job_number_branch,
            }
            try:
                future = executor.submit(test_case_class.alpha_test, **alpha_test_args)
                executor_dict[future] = test_case_class.test_id
            except Exception as ex:
                print(f"*** {ex}")
                traceback.print_exc()
                sys.exit(1)

        # # Send the executor to the progress bar and wait for all MS tasks to finish
        # progress_bar_handler(
        #     executor_dict, True, f"Running {model} alpha test cases with {job_number_huc} workers"
        # )
    metrics_df = create_master_metrics_csv(fim_version)

    print("================================")
    print("End synthesize test cases")

    end_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print(f"ended: {dt_string}")

    # Calculate duration
    time_duration = end_time - start_time
    print(f"Duration: {str(time_duration).split('.')[0]}")
    print()

    return metrics_df


# *********************************************************
def initialize_mannN_ai(fim_dir, huc, mannN_file_aibased):

    log_text = f'Initializing manningN for HUC: {huc}\n'
    
    # Load AI-based manning number data
    ai_mannN_data = pd.read_parquet(mannN_file_aibased, engine='pyarrow')
    # aib_mannN_data.columns
    ai_mannN_data_df = ai_mannN_data[["COMID", "owp_roughness"]]

    # Clip ai_manningN for HUC8
    fim_huc_dir = join(fim_dir, huc)

    path_nwm_streams = join(fim_huc_dir, "nwm_subset_streams.gpkg")
    nwm_stream = gpd.read_file(path_nwm_streams)

    wbd8_path = join(fim_huc_dir, 'wbd.gpkg')
    wbd8 = gpd.read_file(wbd8_path, engine="pyogrio", use_arrow=True)
    nwm_stream_clp = nwm_stream.clip(wbd8)

    ai_mannN_data_df_huc = ai_mannN_data_df.merge(nwm_stream_clp, left_on='COMID', right_on='ID')
    mannN_ai_df = ai_mannN_data_df_huc.drop_duplicates(subset=['COMID'], keep='first')
    mannN_ai_df.index = range(len(mannN_ai_df))
    mannN_ai_df = mannN_ai_df.drop(columns=['ID', 'order_'])
    mannN_ai_df = mannN_ai_df.rename(columns={'COMID': 'feature_id'})

    # Initializing optimized manningN
    # MaxN = 0.5 # MinN = 0.01 #AI-N min=0.01 #max=0.35
    mannN_ai_df['channel_ratio_optz'] = [random.uniform(0.025, 50) for _ in range(len(mannN_ai_df))] #[0.025,50]
    mannN_ai_df['overbank_ratio_optz'] = [random.uniform(0.025, 50) for _ in range(len(mannN_ai_df))] #[0.025,50]
    mannN_ai_df['channel_n_optz'] = mannN_ai_df['owp_roughness']# *mannN_ai_df['channel_ratio_optz']
    mannN_ai_df['overbank_n_optz'] = mannN_ai_df['owp_roughness']# *mannN_ai_df['overbank_ratio_optz']ManningN
    mannN_ai_df['ManningN'] = mannN_ai_df['owp_roughness']# *mannN_ai_df['overbank_ratio_optz']

    # ch_lower_optz = 0.01
    # ch_upper_optz = 0.20
    # ob_lower_optz = 0.01
    # ob_upper_optz = 0.50
    # mannN_ai_df['channel_n_optz'] = mannN_ai_df['channel_n_optz'].clip(lower=ch_lower_optz, upper=ch_upper_optz)
    # mannN_ai_df['overbank_n_optz'] = mannN_ai_df['overbank_n_optz'].clip(lower=ob_lower_optz, upper=ob_upper_optz)
    # mannN_ai_df.columns

    initial_mannN_df = mannN_ai_df[['feature_id', 'ManningN']]

    return initial_mannN_df
    

# *********************************************************
def update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df): #, mannN_values

    fim_huc_dir = join(fim_dir, huc)
    # Get hydro_table from each branch
    ht_all_branches_path = []
    branches = os.listdir(join(fim_huc_dir, 'branches'))
    for branch in branches:
        ht_full = join(fim_huc_dir, 'branches', str(branch), f'hydroTable_{branch}.csv')
        if os.path.isfile(ht_full):
            ht_all_branches_path.append(ht_full)

    # Update hydro_table with updated Q and n
    for ht_path in ht_all_branches_path: #[0:1]
        # try:
        ht_name = os.path.basename(ht_path)
        branch = ht_name.split(".")[0].split("_")[-1]
        ht_df = pd.read_csv(ht_path, dtype={'feature_id': 'int64'}, low_memory=False) #

        ## Check the Stage_bankfull exists in the hydro_table (channel ratio column that the user specified) 
        # drop these cols (in case optz_mann was previously performed)
        if 'manningN_optz' in ht_df.columns:
            ht_df = ht_df.drop(
                ['manningN_optz', 'Discharge(cms)_optzN', 'optzN_on', 'discharge_cms', 'ManningN'],
                axis=1,
            )
        ## Merge (crosswalk) the df of Manning's n with the SRC df
        ht_df = ht_df.merge(mannN_fid_df, how='left', on='feature_id')

        ## Calculate composite Manning's n using the channel geometry ratio attribute given by user
        ht_df['manningN_optz'] = ht_df['ManningN']
        ## Define the channel geometry variable names to use from the hydroTable
        hydr_radius = 'HydraulicRadius (m)'
        wet_area = 'WetArea (m2)'

        ## Calculate Q using Manning's equation
        ht_df['Discharge(cms)_optzN'] = (
            ht_df[wet_area]
            * pow(ht_df[hydr_radius], 2.0 / 3)
            * pow(ht_df['SLOPE'], 0.5)
            / ht_df['manningN_optz']
        )
        optzN_on = True
        ht_df['optzN_on'] = optzN_on
        ht_df['discharge_cms'] = ht_df['Discharge(cms)_optzN']
        ht_df['overbank_n'] = ht_df['manningN_optz']
        ht_df['channel_n'] = ht_df['manningN_optz']
        ht_df.to_csv(ht_path, index=False)

        # except Exception as ex:
        #     summary = traceback.StackSummary.extract(traceback.walk_stack(None))
        #     print(
        #         'WARNING: ' + str(huc) + '  branch id: ' + str(branch) + " updadting hydro_table failed for some reason"
        #     )
        #     # log_text += (
        #     #     'ERROR --> '
        #     #     + str(huc)
        #     #     + '  branch id: '
        #     #     + str(branch)
        #     #     + " updating hydro_table failed (details: "
        #     #     + (f"*** {ex}")
        #     #     + (''.join(summary.format()))
        #     #     + '\n'
        #     # )


# *********************************************************
def objective_function(mannN_values, fim_dir, huc, mannN_fid_df):
    # This function update hydrotable with mannN and Q,
    # Run alpha test, and defines the objective function

    mannN_fid_df['ManningN'] = mannN_values
    print(mannN_fid_df['ManningN'])

    # Create a dataframe, update mannN columns with the new mannN values and ddd feature_ids, 
    print(f'Updating mannN_values for HUC: {huc}\n')
    print(mannN_values[0:20])

    print(f'Updating hydro-tables for each branch for HUC: {huc}\n')
    update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df) #, mannN_values

    print(f'Running Alphat Test for HUC: {huc}\n')
    # Call synthesize_test_cases script and run them
    fim_version = os.path.basename(os.path.dirname(fim_dir))
    synth_test_df = run_test_cases(fim_version) # pfim_csv, 
    # Read BLE test cases if HUC8 has them **************************************************************************************************
    # 100-year flood 
    false_neg_100 = synth_test_df["FN_perc"][1] # min
    false_pos_100 = synth_test_df["FP_perc"][1] # min

    # # 500-year flood
    # false_neg_500 = synth_test_df["FN_perc"][2] # min
    # false_pos_500 = synth_test_df["FP_perc"][2] # min

    # Calculate metrics (error) for objective function
    error_mannN = false_neg_100 + false_pos_100 #+ false_neg_500 + false_pos_500

    # log_text += 'Completed: ' + str(huc)
    print(error_mannN, mannN_fid_df['channel_n_optz'][0:20])

    return error_mannN


# *********************************************************
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Impliment user provided Manning's n values for in-channel vs. overbank flow. "
        "Recalculate Manning's eq for discharge"
    )
    parser.add_argument('-fim_dir', '--fim-dir', help='FIM output dir', required=True, type=str)
    parser.add_argument('-huc', '--huc', help='HUC8 Number', required=True, type=str)
    parser.add_argument(
        '-mann',
        '--mannN_file_aibased',
        help="Path to a csv file containing initial Manning's n values by featureid",
        required=True,
        type=str,
    )

    args = vars(parser.parse_args())

    fim_dir = args['fim_dir']
    huc = args['huc']
    mannN_file_aibased = args['mannN_file_aibased']
    
    # *********************************************************
    # huc = "11010004"
    initial_mannN_ai_df = initialize_mannN_ai(fim_dir, huc, mannN_file_aibased)    

    print(f'Updating hydro-tables for each branch for HUC: {huc}\n')

    # Define the initial values for mannN
    mannN_init = initial_mannN_ai_df['ManningN'].values
    mannN_fid_df = initial_mannN_ai_df.copy()

    # Define the bounds for mannN (assuming it's a bounded optimization problem)
    bounds = [(0.01, 0.5) for _ in range(len(mannN_init))]

    # Define rest of arguments for objective_function
    obj_func_args = (fim_dir, huc, mannN_fid_df)

    # Define the constraints
    # alpha_metrics = alpha_test_metrics_analysis(synth_test_path)
    # constraints = (
    #     {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[0] - 0},  # a > 0
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[0]},  # a < 100
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[1] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[1]},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[2] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[2]},    
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[3] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[3]},    
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[4] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[4]},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[5] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[5]},    
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[6] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[6]},    
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[7] - 0},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[7]},

    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[0] - alpha_test_metrics_analysis(synth_test_path)[1] - alpha_test_metrics_analysis(synth_test_path)[2] - alpha_test_metrics_analysis(synth_test_path)[3]},
    #     # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[4] - alpha_test_metrics_analysis(synth_test_path)[5] - alpha_test_metrics_analysis(synth_test_path)[6] - alpha_test_metrics_analysis(synth_test_path)[7]},
    # )
    print(mannN_init[0:20])
    # Run the optimization using the SLSQP algorithm
    res = minimize(objective_function, mannN_init, method="SLSQP", bounds=bounds, args=obj_func_args) #, constraints=constraints

    print(res.x, res.fun)
