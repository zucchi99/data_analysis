
import sys
sys.path.insert(0, 'src')
from functions import *

params = DEFAULT_PARAMETERS

notebooks = [
    '0_extend__df_raw', 
    '1_explore__df_ext', 
    '2_estimate_factors', 
    ###'3_add_concentrations', 
    '4_simulate_flux_ARIMA', 
    '5a_add_initial_flux', 
    '5b_estimate_flux_slope', 
    '5c_estimate_flux_min', 
    '6_estimate_flux',
    '7_load_uppaal_series',
    '8_compare_estimated_vs_uppaal',
]

input_data_path = {
    '0_extend__df_raw' :                PATH_SENSORS_DATA_RAW_UF,
    '1_explore__df_ext' :               PATH_SENSORS_DATA_EXT_UF_V1,
    '2_estimate_factors' :              PATH_SENSORS_DATA_EXT_UF_V1,
    '4_simulate_flux_ARIMA':            PATH_SENSORS_DATA_EXT_UF_V1,
    ###'3_add_concentrations':          PATH_SENSORS_DATA_EXT_UF_V2,
    '5a_add_initial_flux':              PATH_SENSORS_DATA_EXT_UF_V3,
    '5b_estimate_flux_slope':           PATH_SENSORS_DATA_EXT_UF_V3,
    '5c_estimate_flux_min':             PATH_SENSORS_DATA_EXT_UF_V3,
    '6_estimate_flux':                  PATH_SENSORS_DATA_EXT_UF_V3,
    '7_load_uppaal_series':             PATH_UPPAAL_DATA_RAW,
    '8_compare_estimated_vs_uppaal':    PATH_UPPAAL_DATA_EXT,
}

def merge_data_files() :
    df_all = pd.DataFrame()
    i = 0
    output_file = 'ALL_DATA.csv'
    output_file_path = f"{PATH_SENSORS_DATA_EXT_UF_V1}/{output_file}"
    for f in listdir_by_extension(PATH_SENSORS_DATA_EXT_UF_V1) :
        if (not f in [output_file]) : 
            # read file
            f = PATH_SENSORS_DATA_EXT_UF_V1 + '/' + f
            df_cur = pd.read_csv(f)
            # drop outliers
            if drop_initial_final_off_rows :
                df_cur = drop_initial_final_rows(df_cur, log=False)
                df_cur, _ = get_df_ON_OFF(df_cur)
            if drop_off_rows :
                args = { 'log' : False }
                if drop_outliers :
                    df_cur, _ = remove_outliers(df_cur, cols=['res tot [1/m]'],    drop_fun=drop_outliers_far_median, args=args)
                    df_cur, _ = remove_outliers(df_cur, cols=['prs feed_2 [kPa]'], drop_fun=drop_outliers_far_neighbours, args=args)
                    df_cur, _ = remove_outliers(df_cur, cols=['flux [L/m^2h]'],    drop_fun=drop_outliers_out_range, args=args)
                    df_cur, _ = remove_outliers(df_cur, cols=['flux [L/m^2h]'],    drop_fun=drop_initial_jumps, args=args)
            # concatenate
            df_cur['file_idx'] = i
            #df_cur.loc[len(df_cur)] = empty_separator_row
            df_all = pd.concat([df_all, df_cur])
            i += 1
    df_all = df_all.drop('index', axis=1).reset_index(drop=True)
    # update tmp groups
    new_group = 0
    df_all.loc[0, 'new group'] = new_group
    for i in range(1, len(df_all)) :
        file_cur = df_all.loc[i, 'file_idx']
        file_prv = df_all.loc[i-1, 'file_idx']
        group_cur = df_all.loc[i, 'TMP group']
        group_prv = df_all.loc[i-1, 'TMP group']
        new_group += (1 if ((file_cur != file_prv) or (group_cur != group_prv)) else 0)
        df_all.loc[i, 'new group'] = new_group
        #print(f'files: {file_prv} -> {file_cur}, group: {group_prv} -> {group_cur}, new_group:{new_group}')
    df_all = df_all.drop('TMP group', axis=1).rename(columns={'new group' : 'TMP group'})
    # write to file
    df_all.to_csv(output_file_path, index=False)

def update_parameters_json(args=DEFAULT_PARAMETERS) :
    with open(FILE_PARAMETERS, 'w') as fid:
        json.dump(args, fid, indent=2)

def init_chars(n, char='#') :
    return ''.join([char for _ in range(n)])

def run_all() : 
    j = 0
    for notebook in notebooks[:2] :
        f_ipynb = f'src/{notebook}.ipynb'
        n_chars = 1
        print("\n###################################################")
        print(f"{init_chars(n_chars)} STEP: {j}  <=  <notebook_idx>")
        n_chars = 2
        print(f"{init_chars(n_chars)} notebook: '{f_ipynb}'")
        data_path = input_data_path.get(notebook, None)
        file_idx        = -1
        file_idx_uppaal = -1
        path_html = f'{PATH_NOTEBOOK_OUTPUT}{notebook}'
        os.makedirs(path_html, exist_ok=True)
        # one execution per file        # one execution per input file
        # input files are ordered by date (alphabetically order with file names 'yyyy-mm-dd .*')
        # last input file may be ALL_DATA.csv
        for f_input in listdir_by_extension(data_path) :
            file_idx        += 1
            file_idx_uppaal += 1
            f_html = f'{path_html}/{f_input}-{file_idx}.html'
            n_chars = 3
            print(f"{init_chars(n_chars)} STEP: {j}.{file_idx}  <=  <notebook_idx>.<concentration_idx>")
            n_chars = 4
            #print(f"{init_chars(n_chars)} input data path: '{data_path}'")
            print(f"{init_chars(n_chars)} input data file: '{f_input}'")
            print(f"{init_chars(n_chars)} output notebook file: '{f_html}'")
            params['file_idx']        = file_idx
            params['file_idx_uppaal'] = file_idx_uppaal
            if notebook == '8_compare_estimated_vs_uppaal' :
                if read_from_json(FILE_DATA_SIMULATIONS_ASSOC, f_input) == None :
                    print(f"{init_chars(n_chars)} WARNING: for UPPAAL's simulation file '{f_input}' missing the associated real-data file, add association in file '{FILE_DATA_SIMULATIONS_ASSOC}'")
                    continue
            # one execution per file and per tmp
            if notebook in ['4_simulate_flux_ARIMA'] :
                if f_input != 'ALL_DATA.csv' :
                    tmps_idxs = TMP_INTERVALS[f_input]
                    k = 0
                    for (tmp_idx, cur_idxs) in tmps_idxs.items() :
                        f_html_2 = re.sub("\.html$", f"-{k}.html", f_html)
                        n_chars = 5
                        print(f"{init_chars(n_chars)} STEP: {j}.{file_idx}.{k}  <=  <notebook_idx>.<concentration_idx>.<transembrane_idx>")
                        n_chars = 6
                        print(f"{init_chars(n_chars)} current data rows: '{cur_idxs}'")
                        #print(f"{init_chars(n_chars)} input data file: {f_input}")
                        print(f"{init_chars(n_chars)} output notebook file: '{f_html_2}'")
                        params['tmp_idx'] = tmp_idx
                        update_parameters_json(params)
                        run_notebook(f_ipynb, f_html_2)
                        k += 1
            else :
                update_parameters_json(params)
                run_notebook(f_ipynb, f_html)
        #if notebook == '0_extend__df_raw' :
        #    merge_data_files()
        j += 1

def run_notebook(notebook_file, output_file):
    """Pass arguments to a Jupyter notebook, run it and convert to html."""
    # Run the notebook
    subprocess.call([
        'jupyter-nbconvert',
        '--log-level=ERROR', # silent mode
        '--execute',
        '--to', 'html',
        '--output', output_file,
        notebook_file
    ])


def read_from_json(json_file, key) : 
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get(key, None)


def create_directories() :
    for dir_path in ALL_PATHS :
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def clear_all_output_directories() : 
    # recursive delete!!!
    for dir_path in NON_RAW_PATHS :
        shutil.rmtree(dir_path, ignore_errors=True)

#############################
# MAIN

# delete all output directories
# clear_all_output_directories()

# create any missing directory
create_directories()

# reset parameters with default values
update_parameters_json()

# read parameters
params = read_parameters()

# set notebook global parameters
params['plot_scatterplot_matrix'] = False

# run pipeline
run_all()

# reset parameters with default values
update_parameters_json()