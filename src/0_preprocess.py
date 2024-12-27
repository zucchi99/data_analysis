from functions import *

# read parameters
params = read_parameters()
file_idx = params['file_idx']
reset_columns_when_OFF = params['reset_columns_when_OFF']
log = params['log']
if log :
    print(params)

# define input, output paths
in_folder = PATH_SENSORS_DATA_RAW_UF
out_folder = PATH_SENSORS_DATA_EXT_UF_V1
cur_file, file_path = get_input_file(in_folder=in_folder, in_file_idx=file_idx, log=log)
out_file_path = out_folder + cur_file

# read input data
df_experiments = pd.read_excel(FILE_EXPERIMENTS_METADATA)
df_membranes = pd.read_excel(FILE_MEMBRANES_METADATA)
df = pd.read_csv(file_path)

df.rename(columns=UF_COLUMNS, inplace=True)
df['datetime'] = pd.to_datetime(df["date"] + " " + df["time"], format='%Y/%m/%d %H:%M:%S')
df['time span [s]'] = df['datetime'].diff().dt.total_seconds()
df['time span [s]'] = df['time span [s]'].apply(lambda x : 60 if (58 <= x and x <= 62) else x) # 1-2 secs errors are removed
# df['tank liters [L]'] = df['tank liters [%]'] * FEED_TANK_CAPACITY_LITERS / 100
df.loc[0, 'time span [s]'] = 0.0
DATE = df.loc[0, 'datetime'].date().isoformat()
df = df[change_column_index(df.columns.tolist(), 'TMP [bar]', 7)] 
df = df[change_column_index(df.columns.tolist(), 'datetime', 0)] 
df = df[change_column_index(df.columns.tolist(), 'time span [s]', 1)]
# df = df[change_column_index(df.columns.tolist(), 'tank liters [L]', 11)]
df = change_unit_measure(df)
df = df.drop(columns=['date', 'time', 'millisecond [ms]'])


