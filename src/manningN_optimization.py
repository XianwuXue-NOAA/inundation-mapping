
import argparse
import datetime as dt
import multiprocessing
import os
from os.path import isdir, isfile, join
import re
import sys
import subprocess
from dotenv import load_dotenv
from collections import deque
from multiprocessing import Pool
import traceback
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.tools import sjoin
from scipy.optimize import minimize
import random

from utils.shared_variables import DOWNSTREAM_THRESHOLD, ROUGHNESS_MAX_THRESH, ROUGHNESS_MIN_THRESH

# mannN_file_aibased = "/efs-drives/fim-dev-efs/fim-data/inputs/rating_curve/variable_roughness/ml_outputs_v1.01.parquet"
# fim_dir = "/home/rdp-user/outputs/mno_11010004_cal_off2/"
# projectDir = "/home/rdp-user/projects/dev-roughness-optimization/" #os.getenv('projectDir')
# huc = "11010004"
# optzN_on = True
# # output_suffix = "" #optz_mannN
# synth_test_path = "/efs-drives/fim-dev-efs/fim-home/heidi.safa/roughness_optimization/alphatest_metrics_optz_mannN2.csv"
# pfim_csv = "/efs-drives/fim-dev-efs/fim-data/previous_fim/fim_4_5_2_0/fim_4_5_2_0_metrics.csv"

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
def update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df, optzN_on):

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
                src_df = pd.read_csv(src_path, dtype={'feature_id': 'int64'}) #low_memory=False

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
                    df_htable = pd.read_csv(htable_filename, dtype={'HUC': str})

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
def objective_function(mannN_values, *obj_func_args): #, fim_dir, huc, projectDir, synth_test_path, pfim_csv, optzN_on
    # This function update hydrotable with mannN and Q,
    # Run alpha test, and defines the objective function

    # Create a dataframe, update mannN columns with the new mannN values and ddd feature_ids, 
    mannN_fid_df = initialize_mannN_ai(fim_dir, huc, mannN_file_aibased)
    mannN_fid_df['channel_n_optz'] = mannN_values
    mannN_fid_df['overbank_n_optz'] = mannN_values

    log_text = update_hydrotable_with_mannN_and_Q(fim_dir, huc, mannN_fid_df, optzN_on)

    log_text += f'Running Alphat Test for HUC: {huc}\n'

    # Call synthesize_test_cases script and run them
    toolDir = os.path.join(projectDir, "tools")
    mannN_optz = os.path.basename(os.path.dirname(fim_dir))
    os.system(f"python3 {toolDir}/synthesize_test_cases.py -c DEV -e GMS -v {mannN_optz} -jh 2 -jb 3 -m {synth_test_path} -o -pcsv {pfim_csv}")

    # Load alpha test metrics (synth_test_cvs)
    synth_test_df = pd.read_csv(synth_test_path)
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
    parser.add_argument(
        '-optzN_on',
        '--optzN_on',
        help="Switch between mannN_optimization off or on (True or False)",
        required=True,
        type=str,
    )
    parser.add_argument(
        '-synth_path',
        '--synth_test_path',
        help="Path to a csv file to save alpha test metrics",
        required=True,
        type=str,
    )
    parser.add_argument(
        '-projDir',
        '--projectDir',
        help="Path to the project directory (dev)",
        required=True,
        type=str,
    )
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
    optzN_on = args['optzN_on']
    pfim_csv = args['pfim_csv']
    synth_test_path = args['synth_test_path']
    projectDir = args['projectDir']
    # number_of_jobs = args['number_of_jobs']
    
    # *********************************************************
    # huc = "11010004"
    initial_mannN_ai_df = initialize_mannN_ai(fim_dir, huc, mannN_file_aibased)

    # Define the initial values for mannN
    mannN_init = initial_mannN_ai_df['channel_n_optz'].values

    # Define the bounds for mannN (assuming it's a bounded optimization problem)
    bounds = [(0.01, 0.5) for _ in range(len(mannN_init))]

    # Define rest of arguments for objective_function
    obj_func_args = (fim_dir, huc, projectDir, synth_test_path, pfim_csv, optzN_on)

    # Define the constraints
    alpha_metrics = alpha_test_metrics_analysis(synth_test_path)

    constraints = (
        {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[0] - 0},  # a > 0
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[0]},  # a < 100
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[1] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[1]},
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[2] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[2]},    
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[3] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[3]},    
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[4] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[4]},
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[5] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[5]},    
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[6] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[6]},    
        # {'type': 'ineq', 'fun': lambda synth_test_path: alpha_test_metrics_analysis(synth_test_path)[7] - 0},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[7]},

        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[0] - alpha_test_metrics_analysis(synth_test_path)[1] - alpha_test_metrics_analysis(synth_test_path)[2] - alpha_test_metrics_analysis(synth_test_path)[3]},
        # {'type': 'ineq', 'fun': lambda synth_test_path: 100 - alpha_test_metrics_analysis(synth_test_path)[4] - alpha_test_metrics_analysis(synth_test_path)[5] - alpha_test_metrics_analysis(synth_test_path)[6] - alpha_test_metrics_analysis(synth_test_path)[7]},
    )

    # Run the optimization using the SLSQP algorithm
    res = minimize(objective_function, mannN_init, method="SLSQP", bounds=bounds, args=obj_func_args, constraints=constraints)
    print(res.x, res.fun)
