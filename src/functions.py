from constants import *

fun_k_n = (lambda J, dJ, J_min, n : - dJ / (J**(2-n) * (J - J_min)) )

def partition_column_in_groups(df, series_col='TMP [kPa]', group_col='TMP group', max_distance_from_average=12.5) :
    cur_group = 0
    first_value = df[series_col].values[0]
    #cur_min = first_value
    #cur_max = first_value
    mov_avg = first_value
    df[group_col] = 0
    for i, row in df.iterrows() : 
        cur_val = row[series_col]
        # A) use bounds
        # cur_min = min(cur_min, cur_val)
        # cur_max = max(cur_max, cur_val)
        # B) use moving average
        mov_avg = (mov_avg + cur_val) / 2
        if (abs(mov_avg - cur_val) > max_distance_from_average)  :
            # current data point too far from previous data points
            #cur_min = cur_val
            #cur_max = cur_val
            mov_avg = cur_val
            cur_group += 1
        df.loc[i, group_col] = cur_group
    print(f"added column: '{group_col}', computed by checking if the current point is near the moving average of the previous points (max_distance_from_average={max_distance_from_average})")
    return df

def generate_concentration_lines(df, x_axis='time [m]') : 
    conc_shifted = df['initial feed concentration [g/L]'].shift(1)
    df['changed concentration'] = (df['initial feed concentration [g/L]'] != conc_shifted)
    df_changed_conc = df[df['changed concentration'] == True][[x_axis, 'initial feed concentration [g/L]']]
    conc_lines = {}
    for _, row in df_changed_conc.iterrows() :
        x = row[x_axis].astype(int)
        conc = row['initial feed concentration [g/L]']
        conc_lines[x] = f"feed conc = {conc:.2f} [g/L]"
    for (k,v) in conc_lines.items() :
        print(f"{k:4d}: {v}")
    conc_lines_GREATER_ZERO = { k : v for k, v in conc_lines.items() if v != 'feed conc = 0.00 [g/L]'}
    return conc_lines, conc_lines_GREATER_ZERO

def get_input_file(in_folder, in_file_idx, log=True) :
    in_files = [ (in_folder, file) for file in sorted(os.listdir(in_folder)) if re.match(".*\.csv", file) ]
    cur_file = in_files[in_file_idx][1]
    file_path = in_folder + cur_file

    if log :
        print("file list:")
        for i in range(len(in_files)) :
            print(i, in_files[i][0] + in_files[i][1])
        print(f"\ninput file:\n{file_path}")

    return cur_file, file_path
    
def estimate_initial_resistance(df, y_col='res tot est at 20° [1/m]', factors=['log(1+time [m])']) :
    x = df[factors]
    y = df[y_col]
    model, y_pred, intercept, coeffs = call_linear_model(x, y, summary=False)
    #print(f"\np-values:\n{model.pvalues}\n")
    #print(get_error_stats(y, y_pred, y_col))
    #print()
    #compute_error_metrics(y, y_pred)
    return model, y_pred, intercept, coeffs

def listdir_by_extension(path, ext='csv') :
    return sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and re.match(f".*\.{ext}$", f)])

def get_experiment_data(file_name) :
    df_experiments = pd.read_excel(FILE_EXPERIMENTS_METADATA)
    cur_file_name = re.sub('.csv', '', file_name)
    # filter only the current experiment data
    cur_experiment = df_experiments[df_experiments['data file'] == cur_file_name]
    check_df_size_after_filter(cur_experiment, FILE_EXPERIMENTS_METADATA, 'experiments')
    return cur_experiment

def get_membrane_data(cur_experiment) :
    df_membranes = pd.read_excel(FILE_MEMBRANES_METADATA)
    membrane_model = cur_experiment['membrane model'].values[0]
    # filter only the used membrane data
    cur_membrane = df_membranes[df_membranes['membrane model'] == membrane_model]
    check_df_size_after_filter(cur_membrane, FILE_MEMBRANES_METADATA, 'memebranes')
    return cur_membrane

def check_df_size_after_filter(df_filtered, in_file, about) :
    if len(df_filtered) != 1 :
        if len(df_filtered) == 0 :
            print(f"ERROR: no associated {about} metadata found, please fill the data in the input file {in_file}!")
        if len(df_filtered)  > 1 :
            print(f"ERROR: more than one {about} metadata found, please keep only one in the input file {in_file}! Matching rows:")
            print(df_filtered)
        sys.exit(1)

def add_initial_flux(df, INITIAL_VISCOSITY, tmp_col='TMP est [kPa]', flux_col='flux at 20° [L/m^2h]', flux_used='flux at 20°', res_col='res tot est at 20° [1/m]', time_col='time [m]') :
    previous_time_col = f'{time_col} prev'
    previous_res_col = f'{res_col} prev'
    is_flux_steady_col = f'is {flux_used} steady'
    flux_diff_col = f'd/dt {flux_used}'
    flux_min_col = f'{flux_used} min [L/m^2h]'
    # only use real data (to obtain the resistance of the last row of the previous series)
    df_start_end = df[df['is forecast'] == 0].copy()
    # for the first series after clean water do NOT use last clear water resistance (under-estimated), rather estimate it with a lm with few following resistances
    first_fouled_idx = df_start_end[df_start_end['initial feed concentration [g/L]'] > 0].index[0]
    file_idx = df_start_end.loc[first_fouled_idx, 'file_idx']
    tmp_idx  = df_start_end.loc[first_fouled_idx, 'tmp_idx']
    df_first_fouled = df_start_end[(df_start_end['file_idx'] == file_idx) & (df_start_end['tmp_idx'] == tmp_idx)][:5].copy()
    df_first_fouled['log(1+time [m])'] = df_first_fouled['time [m]'].apply(lambda x : math.log(1 + x))
    _, _, FIRST_RES_EST, _ = estimate_initial_resistance(df_first_fouled)
    print((f"Note: for file_idx:{file_idx}, tmp_idx:{tmp_idx}, which is the first series after the clear water, the initial resistance a t=0 is estimated with:\n"
        f"total_resistance = a + b * log(1+time), using as training dataset the following 5 resistances recorded with time t=[1,5] [m].\n"
        f"Then, res_tot at t=0 is set equal to the intercept a={FIRST_RES_EST:.3E}"))
    # filter only first and last row of each series (time == 1) or (time == MAX)
    df_start_end[previous_time_col] = df_start_end[time_col].shift(-1)
    df_start_end = df_start_end[(df_start_end[time_col] == 1) | (df_start_end[previous_time_col] == 1)]
    # copy resistance at time=MAX to the next series time=1 
    df_start_end[previous_res_col] = df_start_end[res_col].shift(1)
    #print(df_start_end[['file_idx', 'tmp_idx', 'time [m]', previous_res_col]])
    df_start_end.loc[first_fouled_idx, previous_res_col] = FIRST_RES_EST
    #print(f"{first_fouled_idx} ==> {FIRST_RES_EST}")
    #print(df_start_end[['file_idx', 'tmp_idx', 'time [m]', previous_res_col]])
    # drop rows with time=2,...,MAX now useless
    df_start_end = df_start_end[(df_start_end[time_col] == 1)]
    df_minute_1 = df_start_end
    # duplicate rows of time=1 also for time=0
    df_minute_0 = df_minute_1.copy()
    df_minute_0[time_col] = 0
    # for time=0 use the resistance of time=MAX of the previous series
    df_minute_0 = df_minute_0.drop(res_col, axis=1).rename(columns={previous_res_col : res_col})
    # for time=1 keep the resistance as it is
    df_minute_1 = df_minute_1.drop(previous_res_col, axis=1)
    # move up flags only to time=0 (automatically done before by duplicating the time=1 row)
    df_minute_1['increased TMP'] = 0
    df_minute_1['decreased TMP'] = 0
    # MOST IMPORTANT PART
    # use the last resistance of the previous series to calculate the INITIAL FLUX
    df_minute_0[flux_col] = 1000.0 * df_minute_0[tmp_col] / ((INITIAL_VISCOSITY / (1000.0 * 3600.0)) * df_minute_0[res_col])
    # drop the very first row (the first series does not have a precedent value for the resistance thus the flux is NaN)
    df_minute_0 = df_minute_0.dropna(subset=[flux_col])
    # add the flag to mark the row
    df_minute_0['is initial'] = 1
    df_minute_1['is initial'] = 0
    # concat the rows of time=0 and time=1
    df_initialized = pd.concat([df_minute_0, df_minute_1]).sort_values(by=['date', 'file_idx', 'tmp_idx', time_col]).reset_index(drop=True)
    # add for t=1 the d/dt flux and k(n) columns
    df_initialized[is_flux_steady_col] = df_initialized.apply(lambda x : abs(x[flux_col] - x[flux_min_col]) < 0.1, axis=1).astype(int)
    df_initialized[flux_diff_col] = df_initialized[flux_col].diff()
    df_initialized[flux_diff_col] = df_initialized.apply(lambda x : x[flux_diff_col] if x[time_col] == 1 else np.nan, axis=1)
    df_initialized = df_initialized.drop([previous_time_col], axis=1)
    it = iter(ALL_MAX_K_N)
    for n in ALL_N : 
        MAX_K_N = next(it)
        df_initialized[f'k(n={n})'] = df_initialized.apply(lambda x : np.nan if x[is_flux_steady_col] == 1 else fun_k_n(x[flux_col], x[flux_diff_col], x[flux_min_col], n), axis = 1)
        df_initialized[f'k(n={n})'] = df_initialized.apply(lambda x : max(-MAX_K_N, min(MAX_K_N, x[f'k(n={n})'])) if pd.notna(x[f'k(n={n})']) else np.nan, axis = 1) # 
    #print(df_initialized[['file_idx', 'tmp_idx', flux_col, flux_diff_col, flux_min_col, 'k(n=2)']]) #'time [m]',
    # add the new rows with time=0 and time=1
    df['is initial'] = 0
    # drop from the df the old time=1 row
    df = df[df[time_col] > 1]
    df = pd.concat([df, df_initialized]).sort_values(by=['date', 'file_idx', 'tmp_idx', time_col]).reset_index(drop=True)
    # recompute index
    df = df.reset_index(drop=True)
    df['index'] = range(0, len(df))
    return df

def read_parameters() :
    with open(FILE_PARAMETERS, 'r') as fid:
        params = json.load(fid)
    print("parameters:")
    for (k,v) in params.items() :
        print(f"{k}: {v}")
    return params

def min_max_scaler(c) :
    return (c - c.min()) / (c.max() - c.min())

def apply_coefficients(df, coeffs, out_col) : 
    x_cols       = coeffs['x']
    intercept    = coeffs['intercept']
    coefficients = coeffs['coefficients']
    df[out_col]  = intercept
    for i in range(len(x_cols)) :
        df[out_col] += coefficients[i] * df[x_cols[i]]
    return df

def read_estimated_coefficients_from_json(json_file, key) : 
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data[key]

def write_estimated_coefficients_to_json(json_file, key, x_cols, intercept, coefficients) :
    data = {}
    if os.path.isfile(json_file) :
        with open(json_file, 'r') as f :
            data = json.load(f)
    data[key] = {
        'x'            : x_cols,
        'intercept'    : intercept,
        'coefficients' : list(coefficients)
    }
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def get_initial_concentration_coefficients(df, lm_coeffs, col='permeate concentration [g/L]') :
    x_col = f"log {col}"
    df[x_col] = df[col].apply(lambda x : math.log(x))
    cur_conc_coeffs = {}
    for sample_type in ['retentate', 'permeate'] :
        intercept, coeffs = lm_coeffs[sample_type]
        coeffs = np.array(coeffs)
        cur_conc_coeffs[sample_type] = (intercept + coeffs * df.loc[0, x_col])[0]
    return cur_conc_coeffs

def add_resistance_smooth_and_percentages(df, res_col='res tot [1/m]', out_cols=('res tot smooth [1/m]', 'res [%]', 'd/dt res [%]'), frac=0.15, min=MIN_RES, max=MAX_RES) : 
    df[out_cols[0]] = smooth_data_lowess(df['time [m]'], df[res_col], frac)
    df[out_cols[1]] = min_max_scaler(df[out_cols[0]], min=min, max=max)
    df[out_cols[2]] = df[out_cols[1]].diff()
    return df

def predict_concentration_given_coeff(df, conc_coeffs, types=['retentate', 'permeate'], dt_res_col='d/dt res [%]') :
    for sample_type in types :
        conc_est   = f'{sample_type} concentration est [g/L]'
        conc_real  = f'{sample_type} concentration [g/L]'
        # ideally find coeff s.t. ratio = 1
        conc_ratio = f'{sample_type} concentration est ratio' 
        # initial conc is given
        df[conc_est] = df[conc_real]
        for i in range(1, len(df)) :
            # use variation in resistance as variation in concentration
            df.loc[i, conc_est] = df.loc[i-1, conc_est] * (1 + (conc_coeffs[sample_type] * df.loc[i, dt_res_col]))
        df[conc_ratio] = df[conc_est] / df[conc_real] 
    return df

def min_max_scaler(c, min='default', max='default') :
    min = c.min() if min == 'default' else min
    max = c.max() if max == 'default' else max
    return (c - min) / (max - min)

def compute_error_metrics(y_real, y_pred, log=True) :
    err_metrics = {
        "R^2"   : metrics.r2_score(y_real, y_pred),
        "RMSE"  : metrics.root_mean_squared_error(y_real, y_pred),
        "MAE"   : metrics.mean_absolute_error(y_real, y_pred),
        "MedAE" : metrics.median_absolute_error(y_real, y_pred),
        "maxAE" : metrics.max_error(y_real, y_pred),
        "MAPE"  : metrics.mean_absolute_percentage_error(y_real, y_pred),
        "maxAPE" : max(abs((y_real - y_pred) / y_real))
    }
    if log : 
        print("Error metrics:")
        for (k,v) in err_metrics.items() :
            # f'{message:{fill}{align}{width}}'
            print(f"{k:{' '}{'<'}{6}} {v:8.4f}")
    return err_metrics

def get_error_stats(y, y_pred, y_lbl) :
    df = pd.DataFrame()
    df[y_lbl] = y
    df['absolute error'] = abs(y - y_pred)
    df['relative error'] = df['absolute error'] / abs(y)
    df['squared error']  = df['absolute error'] ** 2
    return get_summary(df, [y_lbl, 'absolute error', 'relative error', 'squared error'], stats=["min", "median", "mean",  "max", "var", "std"], conf_int=None)

def get_summary(df, cols, group_cols=[], stats=["min", "median", "mean",  "max"], transpose='default', conf_int=0.95) :
    transpose = (group_cols != []) if transpose == 'default' else transpose == 'y'
    all_cols = cols + group_cols
    x = df[all_cols]
    if group_cols != [] :
        x = x.groupby(group_cols)
    if conf_int is not None:
        stats = list(set(stats + ['mean', 'std']))
    x = x.agg(stats)
    if conf_int is not None:
        x.loc[f'ci_{conf_int}_low']  = None
        x.loc[f'ci_{conf_int}_high'] = None
        n = len(df)
        t_value = scipy.stats.t.ppf(1 - (1 - conf_int) / 2.0, n - 1)
        for i in range(len(cols)) :
            cur_x = x.iloc[:,i]
            # Calculate the margin of error
            me = t_value * cur_x['std'] / (n ** 0.5)
            # Calculate the lower and upper bounds of the confidence interval   
            x.loc[f'ci_{conf_int}_low'].iloc[i]  = cur_x['mean'] - me
            x.loc[f'ci_{conf_int}_high'].iloc[i] = cur_x['mean'] + me
    if transpose :
        x = np.transpose(x)
    return x

def get_y_dict_min_max_cols(df, y_col:str, min_max_cols:list) :
    y_dict = {}
    y_dict[y_col] = df[y_col]
    y_ax_lbl = [y_col, 'factor [%]']
    title    = f'{y_col} vs its factors'
    lbls = []
    for c in min_max_cols :
        col = df[c]
        c_name_perc = re.sub('\[.*\]$', '[%]', c)
        df[c_name_perc] = min_max_scaler(col)
        y_dict[c_name_perc] = df[c_name_perc]
        lbls.append(c_name_perc)
    return y_dict, lbls, y_ax_lbl, title


def check_stationarity(df_train, lags='default', SIGNIF_LVL=0.05, show_plots=True) :
    lags = (math.floor(len(df_train) / 2) - 1) if lags=='default' else lags
    print('Checking stationarity with Augmented Dickey-Fuller test (ADF)')
    adf_test = tsatools.adfuller(df_train)
    pval = adf_test[1]
    print(f'p-value: {pval}')
    x = (f'is stationary at a {100*SIGNIF_LVL}% significance level? ')
    is_significant = pval < SIGNIF_LVL
    if is_significant :
        print(x + "yes")
    else :
        print(x + "no, try differencing")
    print(f"test statistic:     {adf_test[0]}")
    print(f" 1% critical value: {adf_test[4]['1%']}")
    print(f" 5% critical value: {adf_test[4]['5%']}")
    print(f"10% critical value: {adf_test[4]['10%']}")
    if show_plots :
        print("lags:", lags)
        fig, axs = plot.subplots(1, 2, figsize=(18, 5))
        for i in range(2) :
            axs[i].set_xlabel('lags')
            axs[i].set_ylabel('correlation')
        acf = tsaplots.plot_acf(df_train, ax=axs[0])
        pacf = tsaplots.plot_pacf(df_train, lags=lags, ax=axs[1])
    return adf_test, is_significant

def change_col_offset(df, col_name, start_from=1) :
    return (df[col_name] - df.loc[0, col_name] + start_from)

def add_TMP_levels(df, col='TMP [kPa]', levels=[-math.inf,100,200,300,400,math.inf]) :
    it = iter(levels)
    l = next(it)
    while (l != math.inf) :
        r = next(it)
        df[f'is TMP in [{l}, {r})'] = df[col].apply(lambda tmp : 1 if ((l <= tmp) and (tmp < r)) else 0)
        l = r
    return df

def add_column_if_missing(df, c='is_outlier', v=False) : 
    if not c in df.columns :
        df[c] = v
    return df

# def find_TMP_groups(df) :
# TODO

def identify_all_outliers(df, drop_off_rows=True, drop_outliers=True, log=True) :
    args = {}
    args['log'] = log
    df = invoke__identify_outliers(df, cols=['is_ON'],            drop_fun=indentify_outliers__machine_OFF,    args=args)
    df = invoke__identify_outliers(df, cols=['res tot [1/m]'],    drop_fun=indentify_outliers__far_median,     args=args)
    df = invoke__identify_outliers(df, cols=['prs feed_2 [kPa]'], drop_fun=indentify_outliers__far_neighbours, args=args)
    df = invoke__identify_outliers(df, cols=['flux [L/m^2h]'],    drop_fun=indentify_outliers__out_range,      args=args)
    #df = invoke__identify_outliers(df, cols=['flux [L/m^2h]'],    drop_fun=indentify_outliers__initial_jumps)
    if drop_off_rows :
        df = df[df['is_ON'] == True].reset_index(drop=True)
    if drop_outliers :
        # note: outliers include also off rows
        df = df[df['is_outlier'] == False].reset_index(drop=True)
    return df

def invoke__identify_outliers(df, drop_fun, cols=['res tot [1/m]'], args:dict={}) :
    df = add_column_if_missing(df)
    log = args.get('log', True)
    n_bef = len(df[df['is_outlier'] == True])
    for c in cols :
        args['column'] = c
        df = drop_fun(df, args)
        n_aft = len(df[df['is_outlier'] == True])
        n_new = n_aft - n_bef
        if log and n_new > 0:
            print(f'found {n_new} new outliers after checking column {c} by function {drop_fun.__name__}')
            print("total number of outliers:", n_bef, "->", n_aft)
        n_bef = n_aft
    return df

def indentify_outliers__machine_OFF(df, args:dict={}) :
    MIN_MINUTES_ON = args.get('MIN_MINUTES_ON', 5)
    df = add_column_if_missing(df)
    # define as outliers all INITIAL rows until the machine is ON for >= 5 min
    counter = 0
    i = 0
    while (counter < MIN_MINUTES_ON and i < len(df)) :
        counter = (counter + 1) if (df.loc[i, 'is_ON'] == True) else 0
        i += 1
    outliers = list(range(0, i - counter))
    df.loc[outliers, 'is_outlier'] = True
    # define as outliers all FINAL rows since machine is not ON for >= 5 min
    counter = 0
    i = 0
    while (counter < MIN_MINUTES_ON and i < len(df)) :
        counter = (counter + 1) if (df.loc[len(df)-i-1, 'is_ON'] == True) else 0
        i += 1
    outliers = list(range(len(df) - i + counter, len(df)))
    df.loc[outliers, 'is_outlier'] = True
    # define as outliers any OFF row
    df['is_outlier'] = ((df['is_outlier']) | (~df['is_ON']))
    return df

#def indentify_outliers__initial_jumps(df, c, args:dict={}) :
#    until = args.get('until', 2)
#    max_ratio = args.get('max ratio', 1.2)
#    i = 0
#    idx = -1
#    for i in range(until) :
#        cur = abs(df.loc[i, c])
#        nxt = abs(df.loc[i+1, c])
#        if max(cur, nxt) / min(cur, nxt) > max_ratio :
#            idx = i
#        i += 1
#    # if idx == -1 then to_drop=[] otherwise to_drop=[0,..,idx+1]
#    to_drop = [ i for i in range(0, idx+1)]
#    return df, to_drop 

def indentify_outliers__out_range(df, args:dict={}) :
    c = args['column']
    min = args.get('min', 0)
    max = args.get('max', math.inf)
    df['is_outlier'] = ((df['is_outlier']) | (df[c] < min) | (df[c] > max))
    return df

def get_outliers_free_df(df) :
    # use only the clean dataset to seek for outliers 
    df_bckp = df
    df = df[df['is_outlier'] == False].reset_index()
    return df_bckp, df

def restore_previous_index(df, index_col='level_0', drop_index_col=True) :
    # restore previous index
    df.index = df[index_col]
    df.index.name = None
    if drop_index_col :
        df = df.drop(columns=[index_col])
    return df

def indentify_outliers__far_neighbours(df, args:dict={}) :
    df_bckp, df = get_outliers_free_df(df)
    # seek for outliers 
    c = args['column']
    interval = args.get('interval', 5)
    max_ratio = args.get('max ratio', 1.25)
    half_int = int(interval/2)
    for i in range(half_int, len(df)-half_int) :
        left_median = abs(stats.median(df[c][i-half_int:i]))
        right_median = abs(stats.median(df[c][i+1:i+half_int+1]))
        x = abs(df.loc[i, c])
        if (x < left_median and x < right_median) or (x > left_median and x > right_median) :
            if (max(x, left_median) / min(x, left_median) > max_ratio) and (max(x, right_median) / min(x, right_median) > max_ratio) :
                #print(f"to drop: {i}, l_median: {left_median}, r_median: {right_median}, value: {x}")
                df.loc[i, 'is_outlier'] = True
    df = restore_previous_index(df)
    # add the new found outliers to the original df
    df_bckp['is_outlier'] = (df_bckp['is_outlier'] | df['is_outlier'])
    return df_bckp

def indentify_outliers__far_median(df, args:dict={}) :
    # use only the clean dataset to seek for outliers 
    df_bckp = df
    df = df[df['is_outlier'] == False]
    # seek for outliers 
    c = args['column']
    group_by_col = args.get('group_by', None) # default: None
    max_ratio    = args.get('max ratio', 100) # default: 100
    groups = [1] # default only one group
    median = abs(stats.median(df[c]))
    if group_by_col is not None :
        groups = df[group_by_col].drop_duplicates().values
    for g in groups :
        if group_by_col is not None :
            median = df[df[group_by_col] == g][[group_by_col, c]].groupby(group_by_col).median().values[0][0]
        df[f'abs({c})'] = abs(df[c])
        df['to_drop'] = df[f'abs({c})'].apply(lambda x : (max(x, median) / min(x, median)) > max_ratio)
        if group_by_col is not None :
            df['to_drop'] = ((df['to_drop']) & (df[group_by_col] == g))
        df['is_outlier'] = ((df['is_outlier']) | (df['to_drop']))
        df = df.drop(columns=[f'abs({c})', 'to_drop'])
    # add the new found outliers to the original df
    df_bckp['is_outlier'] = (df_bckp['is_outlier'] | df['is_outlier'])
    return df_bckp

def get_concentration_lines(df, time_col, conc_col='initial feed concentration [g/L]', conc_type='feed', log=True) :
    conc_shifted = df[conc_col].shift(1)
    df['changed concentration'] = (df[conc_col] != conc_shifted)
    changed_conc = df[df['changed concentration'] == True][[time_col, conc_col]]
    conc_lines = {}
    for _, row in changed_conc.iterrows() :
        x = row[time_col].astype(int)
        conc = row[conc_col]
        conc_lines[x] = f"{conc_type} conc = {conc:.2f} [g/L]"
    if log :
        for (k,v) in conc_lines.items() :
            print(f"{k:4d}: {v}")
    conc_lines_GREATER_ZERO = { k : v for k, v in conc_lines.items() if v != '{conc_type} conc = 0.00 [g/L]'}
    return conc_lines, conc_lines_GREATER_ZERO

def call_linear_model(x, y, fit_intercept=True, summary=True, check_vif=True) :
    if check_vif :
        factors = x.columns
        MAX_LEN = max([ len(c) for c in factors])
        if len(factors) > 1 :
            for i in range(len(factors)) :
                c = factors[i]
                vif = outliers_influence.variance_inflation_factor(x, i)
                warn = ""
                if vif >= 10 :
                    warn = 'Severe multicollinearity, the model coefficients can be poorly estimated'
                elif vif >= 4 :
                    warn = 'Some multicollinearity'
                else :
                    warn = 'No multicollinearity at all'
                print("Variance Inflation Factor (VIF)")
                print(f" - {c:<{MAX_LEN}} -> {vif:5.2f} ==> {warn}")
        #else :
            #print(f" - The model has just one variable, {factors[0]}, thus there can't be multicollinerity.")
        print()

    constant_already_present = False
    if fit_intercept :
        n_cols = len(x.columns)
        x = sm.add_constant(x) # adding a constant if not already present
        # if a const column was already present then it is not added
        constant_already_present = (n_cols == len(x.columns))
    model = sm.OLS(y, x).fit()
    intercept = None
    i = 0
    params = list(model.params)
    if fit_intercept and (not constant_already_present) : 
        intercept = params[0]
        i += 1
    elif constant_already_present :
        intercept = None
    coefficients = params[i:]
    y_pred = model.predict(x)
    if summary :
        print('Intercept:   ', intercept)
        print('Coefficients:', coefficients)
        print(model.summary())
    return model, y_pred, intercept, coefficients

def predict_y(x, intercept, coeffs) :
    x = sm.add_constant(x, has_constant='add')
    y_pred = np.reshape((intercept * x.iloc[:,[0]]).to_numpy(dtype='float64'), -1)
    for i in range(len(coeffs)) :
        y_pred = y_pred + np.reshape((coeffs[i] * x.iloc[:,[i+1]]).to_numpy(dtype='float64'), -1)
    return y_pred

def smooth_data_lowess(x, y, pct):
    #frac = (5*span / len(y))
    return sm.nonparametric.lowess(y, x, frac=pct, return_sorted=False)

def add_cross_flow_velocity_and_estimated_retentate_flow(df, file_name) :
    # CALCULATING CROSSFLOW VELOCITY 
    # TODO CHECK CORRECTNESS OF COMPUTATION !
    cur_experiment = get_experiment_data(file_name)
    cur_membrane   = get_membrane_data(cur_experiment)
    # input data
    mbn_num         = cur_experiment['number of membranes used'].values[0]
    mbn_channels    = cur_membrane['number of channels'].values[0]
    mbn_channel_dim = cur_membrane['channel dimension [m]'].values[0]
    mbn_section     = mbn_channels * mbn_num * (mbn_channel_dim/2)**2 * math.pi
    mbn_diam        = mbn_num * cur_membrane['diameter [m]'].values[0]
    mbn_len         = cur_membrane['length [m]'].values[0]
    # output columns
    vlct_crsflow    = 'vlct crsflow [m/s]'
    est_flow        = 'flow retentate est by vlct crsflow [L/h]'
    # computation
    df[vlct_crsflow] = (df['flow feed [L/h]'] / (1000 * 3600.0)) / mbn_section
    df[est_flow]     = 1.25 * 0.25 * math.pi * (3600 * df[vlct_crsflow]) * mbn_diam**2 * mbn_len * mbn_channels * mbn_num
    print(f"added column: '{vlct_crsflow}', estimated using a formula based on membranes data and feed flow")
    print(f"added column: '{est_flow}', estimated by using a formula based on membranes data and cross-flow velocity")
    return df

def add_estimate_retentate_pressure(df, target_col='prs retentate [kPa]', group_by_cols=['TMP group']) :
    # estimate the retentate pressure using the median value of each TMP group
    target_col_est = re.sub(r'(\[.*\])$', r'est \1', target_col)
    df_tmp = pd.DataFrame(df[group_by_cols + [target_col]].groupby(group_by_cols).median()[target_col]).reset_index().rename(columns={target_col : target_col_est})
    df = df.join(df_tmp.set_index(group_by_cols), on=group_by_cols)
    print(f"added column: '{target_col_est}', estimated by taking the median value of its group")
    return df

def plot_time_series_subplots(x_series, y_ss, y_ax_lbl, title, x_format=None, s=3, figsize=(20,7), rows=1, cols=1, concentration_lines=None, is_outlier=None):
    fig, axs = plot.subplots(rows, cols, figsize=figsize)
    plot.title(title)
    i = 0
    j = 0
    for (chart_title, y_series) in y_ss.items() :
        cur_ax = axs[j] if rows == 1 else axs[i, j]
        (x_ax_lbl, x) = x_series[i*cols + j]
        j = (j + 1) % cols
        i = (i + 1) if (j == 0) else i
        cur_ax.set_xlabel(x_ax_lbl)
        cur_ax.set_ylabel(y_ax_lbl)
        cur_ax.set_title(chart_title)
        k = 0
        for (y_legend, y) in y_series.items() :
            plot_time_series(x, y, y_legend, x_format=x_format, s=s, cur_ax=cur_ax, color=COLOR_CYCLE[k], grid=True, concentration_lines=concentration_lines, is_outlier=is_outlier)
            concentration_lines=None
            k += 1
            cur_ax.legend()
        

def plot_time_series_2_axis(x, y_series, x_ax_lbl, y_ax_lbl, title, x_format=None, secondary_y=[], s=3, figsize=(20,7), color=COLOR_CYCLE, loc='best', concentration_lines=None, is_outlier=None):
    fig, ax1 = plot.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    i = 0
    j = 0
    for (y_legend, y) in y_series.items() :
        cur_ax = ax2 if (y_legend in secondary_y) else ax1
        plot_time_series(x, y, y_legend, x_format=x_format, s=s, cur_ax=cur_ax, color=color[i], grid=(cur_ax == ax1 and j == 0), concentration_lines=concentration_lines, is_outlier=is_outlier)
        concentration_lines=None
        i += 1
        if cur_ax == ax1 :
            j += 1
    if is_outlier is not None :
        ax2.plot([], [], linestyle='', color='black', marker='x', markersize=s+2, label='outlier')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=loc)
    ax1.set_xlabel(x_ax_lbl)
    ax1.set_ylabel(y_ax_lbl[0])
    ax2.set_ylabel(y_ax_lbl[1])
    plot.title(title)
    return ax1, ax2

def plot_time_series_1_axis(x, y_series, x_ax_lbl, y_ax_lbl, title, x_format=None, s=3, figsize=(20,7), cur_ax=plot, color=COLOR_CYCLE, concentration_lines=None, is_outlier=None):
    i = 0
    if cur_ax == plot and figsize is not None:
        plot.figure(figsize=figsize)
    for (y_legend, y) in y_series.items() :
        plot_time_series(x, y, y_legend, x_format=x_format, s=s, cur_ax=cur_ax, color=color[i], grid=(i == 0), concentration_lines=concentration_lines, is_outlier=is_outlier)
        i += 1
        concentration_lines=None
    if is_outlier is not None :
        cur_ax.plot([], [], linestyle='', color='black', marker='x', markersize=s+2, label='outlier')
    if cur_ax == plot :
        if (len(y_series) > 1) :
            plot.legend()
        plot.xlabel(x_ax_lbl)
        plot.ylabel(y_ax_lbl)
        plot.title(title)
    else:
        if (len(y_series) > 1) :
            cur_ax.legend()
        cur_ax.set_xlabel(x_ax_lbl)
        cur_ax.set_ylabel(y_ax_lbl)
        cur_ax.set_title(title)
    return cur_ax

#def add_outlier_patch_legend(ax) :
#    Patch = matplotlib.patches.Patch
#    handles, labels = ax.get_legend_handles_labels()
#    labels.append('outlier')
#    handles.append(Patch(color='black', marker='x'))  
#    ax.legend(handles=[red_patch])

def plot_time_series(x, y, y_legend, x_format=None, s=3, cur_ax=plot, color=COLOR_CYCLE[0], grid=True, concentration_lines=None, is_outlier=None) :
    if concentration_lines is not None :
        i = len(COLOR_CYCLE) - 1
        for (line_x,lbl) in concentration_lines.items() :
            cur_ax.axvline(x=line_x, label=lbl, color=COLOR_CYCLE[i], linewidth=3)
            i -= 1
    cur_ax.plot(x, y, label=y_legend, linestyle='--', marker='o', color=color, markersize=s)
    if is_outlier is not None :
        df_clean = pd.DataFrame(data={'x' : x, 'y': y, 'is_outlier': is_outlier})[is_outlier == True]
        cur_ax.plot(df_clean['x'], df_clean['y'], linestyle='', marker='x', color='black', markersize=s+2)
    if x_format != None :
        xformatter = mdates.DateFormatter(x_format)
        tmp = (cur_ax.gcf().axes[0]) if cur_ax == plot else cur_ax
        tmp.xaxis.set_major_formatter(xformatter)
    if grid :
        # toggle the enabling of the grid, by default is false ==> call it just once if you want it
        cur_ax.grid()

def check_types(df) :
    for c in df.columns:
        print("{:<25} --> {}".format(c, type(df[c][0])))

def kelvin_celsius_converter(temp, t_from='C') :
    offset = 273.15
    offset = offset if t_from == 'C' else (-offset)
    return temp + offset

def calc_viscosity(temperature_C, pressure_Pa=101325, element='Water') :
    out_param = 'V'      # viscosity     [Pa s]
    in1_param = 'T'      # temperature   [K]
    in1_value = kelvin_celsius_converter(temperature_C)
    in2_param = 'P'      # pressure      [Pa]
    in2_value = pressure_Pa # atmospheric pressure
    element   = 'Water'
    return  CP.PropsSI(out_param, in1_param, in1_value, in2_param, in2_value, element)

def change_column_index(cols, col_from, idx_to=6) :
    idx_from = 0
    while (idx_from < len(cols) and cols[idx_from] != col_from) :
        idx_from = idx_from + 1
    if (idx_from >= len(cols)) :
        return -1
    if (idx_from > idx_to) :
        return (cols[:idx_to] + cols[idx_from:idx_from+1] + cols[idx_to:idx_from] + cols[idx_from+1:])
    else :
        return (cols[:idx_from] + cols[idx_from+1:idx_to] + cols[idx_from:idx_from+1] + cols[idx_to:])

def change_unit_measure(df, unit_measures=UNIT_MEASURES) :
    for c_from in df.columns :
        for (unit_from, (unit_to, conv_fun)) in unit_measures.items() :
            pattern = "(.*)" + re.escape(f"[{unit_from}]")
            if re.match(pattern, c_from) :
                c_to = re.sub(pattern, r"\1[" + unit_to + "]", c_from)
                df.rename(columns={c_from : c_to}, inplace=True)
                df[c_to] = conv_fun(df[c_to])
    return df

# to be called inside a map or an apply:
# df.apply(reset_cols_if_is_OFF, axis=1, args=('is_ON', 'flow feed [L/h]'))
def reset_cols_if_is_OFF(x, is_on, c) :
    return 0 if x[is_on] == False else x[c]