
import json
import argparse
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
import os
from os.path import join
import re
import sys
import traceback
import warnings

import geopandas as gpd
import pandas as pd
from scipy.optimize import minimize
import random

from utils.run_test_case import Test_Case
from utils.tools_shared_variables import (
    AHPS_BENCHMARK_CATEGORIES,
    MAGNITUDE_DICT,
    PREVIOUS_FIM_DIR,
    TEST_CASES_DIR,
)

# *********************************************************
def create_master_metrics_csv(prev_metrics_csv, fim_version):
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
    iteration_list = ['official', 'testing']
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

    # If previous metrics are provided: read in previously compiled metrics and join to calcaulated metrics
    if prev_metrics_csv is not None:
        prev_metrics_df = pd.read_csv(prev_metrics_csv)

        # Put calculated metrics into a dataframe and set the headers
        df_to_write_calc = pd.DataFrame(list_to_write)
        df_to_write_calc.columns = df_to_write_calc.iloc[0]
        df_to_write_calc = df_to_write_calc[1:]

        # Join the calculated metrics and the previous metrics dataframe
        df_to_write = pd.concat([df_to_write_calc, prev_metrics_df], axis=0)

    else:
        df_to_write = pd.DataFrame(list_to_write)
        df_to_write.columns = df_to_write.iloc[0]
        df_to_write = df_to_write[1:]

    # Save aggregated compiled metrics ('df_to_write') as a CSV
    # df_to_write.to_csv(master_metrics_csv_output, index=False)

    return df_to_write


# *********************************************************
def run_test_cases(prev_metrics_csv, fim_version):
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

    # Check whether a previous metrics CSV has been provided and, if so, make sure the CSV exists
    if prev_metrics_csv is not None:
        if not os.path.exists(prev_metrics_csv):
            print(f"Error: File does not exist at {prev_metrics_csv}")
            sys.exit(1)
        else:
            print(f"Metrics will be combined with previous metric CSV: {prev_metrics_csv}")
            print()
    else:
        print("ALERT: A previous metric CSV has not been provided (-pcsv) - this is optional.")
        print()

    model = "GMS"
    for test_case_class in all_test_cases:
        if not os.path.exists(test_case_class.fim_dir):
            continue
        
        overwrite=True,
        verbose=False,
        alpha_test_args = {
            'calibrated': False,
            'model': model,
            'mask_type': 'huc',
            'overwrite': overwrite,
            'verbose': verbose,
            'gms_workers': 1,
        }

        test_case_class.alpha_test(**alpha_test_args)

        metrics_df = create_master_metrics_csv(prev_metrics_csv, fim_version)

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
def initialize_mannN_ai (fim_dir, huc, mannN_file_aibased):

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
    mannN_ai_df['overbank_n_optz'] = mannN_ai_df['owp_roughness']# *mannN_ai_df['overbank_ratio_optz']
    # ch_lower_optz = 0.01
    # ch_upper_optz = 0.20
    # ob_lower_optz = 0.01
    # ob_upper_optz = 0.50
    # mannN_ai_df['channel_n_optz'] = mannN_ai_df['channel_n_optz'].clip(lower=ch_lower_optz, upper=ch_upper_optz)
    # mannN_ai_df['overbank_n_optz'] = mannN_ai_df['overbank_n_optz'].clip(lower=ob_lower_optz, upper=ob_upper_optz)
    # mannN_ai_df.columns

    initial_mannN_df = mannN_ai_df[['feature_id', 'channel_n_optz', 'overbank_n_optz']]

    return initial_mannN_df
    

# *********************************************************
def update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df):

    log_text = f'Updating hydro_table with new set of manningN for HUC: {huc}\n'

    fim_huc_dir = join(fim_dir, huc)

    # Get src_full from each branch
    src_all_branches_path = []
    branches = os.listdir(join(fim_huc_dir, 'branches'))
    for branch in branches:
        src_full = join(fim_huc_dir, 'branches', str(branch), f'src_full_crosswalked_{branch}.csv')
        if os.path.isfile(src_full):
            src_all_branches_path.append(src_full)

        # Update src with updated Q and n
        for src_path in src_all_branches_path: #[0:1]
            try:
                src_name = os.path.basename(src_path)
                branch = src_name.split(".")[0].split("_")[-1]
                log_text += f'  Branch: {branch}\n'

                log_text += f'Reading the src_full_crosswalked.csv: HUC{str(huc)}, branch id: {str(branch)}\n'
                src_df = pd.read_csv(src_path, dtype={'feature_id': 'int64'}, low_memory=False) #

                ## Check the Stage_bankfull exists in the src (channel ratio column that the user specified) 
                if "Stage_bankfull" not in src_df.columns:
                    print(
                        'WARNING --> '
                        + str(huc)
                        + '  branch id: '
                        + str(branch)
                        + src_path
                        + ' does not contain the specified channel ratio column: '
                        + 'Stage_bankfull'
                    )
                    print('Skipping --> ' + str(huc) + '  branch id: ' + str(branch))
                    log_text += (
                        'WARNING --> '
                        + str(huc)
                        + '  branch id: '
                        + str(branch)
                        + src_path
                        + ' does not contain the specified channel ratio column: '
                        + 'Stage_bankfull'
                        + '\n'
                    )
                else:            
                    # drop these cols (in case optz_mann was previously performed)
                    if 'manningN_optz' in src_df.columns:
                        src_df = src_df.drop(
                            ['channel_n_optz', 'overbank_n_optz', 'manningN_optz', 'Discharge(cms)_optzN', 'optzN_on'],
                            axis=1,
                        )
                    ## Merge (crosswalk) the df of Manning's n with the SRC df
                    src_df = src_df.merge(mannN_fid_df, how='left', on='feature_id')

                    ## Calculate composite Manning's n using the channel geometry ratio attribute given by user
                    src_df['manningN_optz'] = (src_df['Stage_bankfull'] * src_df['channel_n_optz']) + (
                        (1.0 - src_df['Stage_bankfull']) * src_df['overbank_n_optz']
                    )

                    ## Define the channel geometry variable names to use from the src
                    hydr_radius = 'HydraulicRadius (m)'
                    wet_area = 'WetArea (m2)'

                    ## Calculate Q using Manning's equation
                    src_df['Discharge(cms)_optzN'] = (
                        src_df[wet_area]
                        * pow(src_df[hydr_radius], 2.0 / 3)
                        * pow(src_df['SLOPE'], 0.5)
                        / src_df['manningN_optz']
                    )

                    optzN_on = True
                    src_df['optzN_on'] = optzN_on
                    # ## Use the default discharge column when optzN is not being applied
                    # src_df['Discharge(cms)_optzN'] = np.where(
                    #     src_df['optzN_on'] == False, src_df['Discharge (m3s-1)'], src_df['Discharge(cms)_optzN']
                    # )  # reset the discharge value back to the original if optzN_on=false
                    # src_df['manningN_optz'] = np.where(
                    #     src_df['optzN_on'] == False, src_df['ManningN'], src_df['manningN_optz']
                    # )  # reset the ManningN value back to the original if optzN_on=false

                    ## Output new SRC with bankfull column
                    # if output_suffix != "":
                    #     src_path = os.path.splitext(src_path)[0] + output_suffix + '.csv'
                    src_df.to_csv(src_path, index=False)

                    ## Output new hydroTable with updated discharge and ManningN column
                    src_df_trim = src_df[
                        ['HydroID', 'Stage', 'optzN_on', 'manningN_optz', 'Discharge(cms)_optzN']
                    ]
                    src_df_trim = src_df_trim.rename(columns={'Stage': 'stage'})
                    # create a copy of vmann modified ManningN (used to track future changes)
                    src_df_trim['ManningN'] = src_df_trim['manningN_optz']
                    # create a copy of vmann modified discharge (used to track future changes)
                    src_df_trim['discharge_cms'] = src_df_trim['Discharge(cms)_optzN']

                    # Read hydro_table file
                    htable_name = f'hydroTable_{branch}.csv'
                    htable_filename = join(fim_huc_dir, 'branches', branch, htable_name)
                    df_htable = pd.read_csv(htable_filename, dtype={'HUC': str}, low_memory=False)

                    ## drop the previously modified discharge columns to be replaced with updated version
                    # df_htable.columns
                    df_htable = df_htable.drop(
                        columns=['optzN_on', 'discharge_cms', 'ManningN', 'Discharge(cms)_optzN', 'manningN_optz'],
                        errors='ignore',
                    #    inplace=True
                    )
                    df_htable = df_htable.merge(
                        src_df_trim, how='left', left_on=['HydroID', 'stage'], right_on=['HydroID', 'stage']
                    )
                    # # reset the ManningN value back to the original if vmann=false
                    # df_htable['optzN_on'] = np.where(
                    #     df_htable['LakeID'] > 0, False, df_htable['optzN_on']
                    # )

                    ## Output new hydroTable csv
                    # if output_suffix != "":
                    #     htable_filename = os.path.splitext(htable_filename)[0] + output_suffix + '.csv'
                    df_htable.to_csv(htable_filename, index=False)

            except Exception as ex:
                summary = traceback.StackSummary.extract(traceback.walk_stack(None))
                print(
                    'WARNING: ' + str(huc) + '  branch id: ' + str(branch) + " updadting hydro_table failed for some reason"
                )
                log_text += (
                    'ERROR --> '
                    + str(huc)
                    + '  branch id: '
                    + str(branch)
                    + " updating hydro_table failed (details: "
                    + (f"*** {ex}")
                    + (''.join(summary.format()))
                    + '\n'
                )
            log_text += 'Completed: Hydro-table updated with new mannN and Q for ' + str(huc)

    return log_text


# *********************************************************
def objective_function(mannN_values, *obj_func_args): #, fim_dir, huc, pfim_csv, ## projectDir, synth_test_path, 
    # This function update hydrotable with mannN and Q,
    # Run alpha test, and defines the objective function

    # Create a dataframe, update mannN columns with the new mannN values and ddd feature_ids, 
    mannN_fid_df = initialize_mannN_ai(fim_dir, huc, mannN_file_aibased)
    mannN_fid_df['channel_n_optz'] = mannN_values
    mannN_fid_df['overbank_n_optz'] = mannN_values

    log_text = update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df)

    print(f'Running Alphat Test for HUC: {huc}\n')
    log_text += f'Running Alphat Test for HUC: {huc}\n'

    # Call synthesize_test_cases script and run them
    # toolDir = os.path.join(projectDir, "tools")
    mannN_optz = os.path.basename(os.path.dirname(fim_dir))

    # os.system(f"python3 {toolDir}/synthesize_test_cases.py -c DEV -e GMS -v {mannN_optz} -jh 2 -jb 3 -m {synth_test_path} -o -pcsv {pfim_csv}")

    # # Load alpha test metrics (synth_test_cvs)
    # synth_test_df = pd.read_csv(synth_test_path)

    synth_test_df = create_master_metrics_csv(pfim_csv, mannN_optz)
    # Read BLE test cases if HUC8 has them **************************************************************************************************
    # 100-year flood 
    false_neg_100 = synth_test_df["FN_perc"][0] # min
    false_pos_100 = synth_test_df["FP_perc"][0] # min

    # 500-year flood
    false_neg_500 = synth_test_df["FN_perc"][1] # min
    false_pos_500 = synth_test_df["FP_perc"][1] # min

    # Calculate metrics (error) for objective function
    error_mannN = false_neg_100 + false_pos_100 + false_neg_500 + false_pos_500

    log_text += 'Completed: ' + str(huc)
    print("first run completed")

    return error_mannN #, log_text


# *********************************************************
def alpha_test_metrics_analysis(synth_test_path):

    # Load synth_test_cvs
    synth_test_df = pd.read_csv(synth_test_path)
    # Read BLE test cases if HUC8 has them **************************************************************************************************
    # 100-year flood 
    true_neg_100 = synth_test_df["TN_perc"][0]
    false_neg_100 = synth_test_df["FN_perc"][0] # min
    true_pos_100 = synth_test_df["TP_perc"][0]
    false_pos_100 = synth_test_df["FP_perc"][0] # min

    # 500-year flood
    true_neg_500 = synth_test_df["TN_perc"][1]
    false_neg_500 = synth_test_df["FN_perc"][1] # min
    true_pos_500 = synth_test_df["TP_perc"][1]
    false_pos_500 = synth_test_df["FP_perc"][1] # min

    alpha_metrics = [true_neg_100, false_neg_100, true_pos_100, false_pos_100,
                     true_neg_500, false_neg_500, true_pos_500, false_pos_500]

    return alpha_metrics #, log_text


# *********************************************************
def constraint1(synth_test_path):

    # Load synth_test_cvs
    synth_test_df = pd.read_csv(synth_test_path)

    # 100-year flood 
    true_neg_100 = synth_test_df["TN_perc"][0]

    return true_neg_100 - 100


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
    parser.add_argument(
        '-pcsv',
        '--pfim_csv',
        help="Path to a csv file containing alpha metrics for previous fim",
        required=True,
        type=bool,
    )
    # parser.add_argument(
    #     '-synth_path',
    #     '--synth_test_path',
    #     help="Path to a csv file to save alpha test metrics",
    #     required=True,
    #     type=str,
    # )
    # parser.add_argument(
    #     '-projDir',
    #     '--projectDir',
    #     help="Path to the project directory (dev)",
    #     required=True,
    #     type=str,
    # )
    # parser.add_argument(
    #     '-j',
    #     '--number-of-jobs',
    #     help='OPTIONAL: number of workers (default=8)',
    #     required=False,
    #     default=8,
    #     type=int,
    # )

    args = vars(parser.parse_args())

    fim_dir = args['fim_dir']
    huc = args['huc']
    mannN_file_aibased = args['mannN_file_aibased']
    pfim_csv = args['pfim_csv']
    # synth_test_path = args['synth_test_path']
    # projectDir = args['projectDir']
    # number_of_jobs = args['number_of_jobs']
    
    # *********************************************************
    # huc = "11010004"
    initial_mannN_ai_df = initialize_mannN_ai(fim_dir, huc, mannN_file_aibased)

    # Define the initial values for mannN
    mannN_init = initial_mannN_ai_df['channel_n_optz'].values

    # Define the bounds for mannN (assuming it's a bounded optimization problem)
    bounds = [(0.01, 0.5) for _ in range(len(mannN_init))]

    # Define rest of arguments for objective_function
    obj_func_args = (fim_dir, huc, pfim_csv) #projectDir, synth_test_path, 

    # Define the constraints
    # alpha_metrics = alpha_test_metrics_analysis(synth_test_path)

    # print(alpha_metrics)

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

    # Run the optimization using the SLSQP algorithm
    res = minimize(objective_function, mannN_init, method="SLSQP", bounds=bounds, args=obj_func_args) #, constraints=constraints

    alpha_metrics = alpha_test_metrics_analysis(synth_test_path)

    print(alpha_metrics)
    
    print(res.x, res.fun)
