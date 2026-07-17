import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import glob
plt.style.use('default')

data_im_loc = 'C:/Users/ikeva/z.datasets/IMERG_PR_daily_africa/daily_precip_imerg_v06_*.nc'
data_im = sorted(glob.glob(data_im_loc))
target_years = np.arange(2003, 2025, 1)
data_im_targets = [f for f in data_im if any(f'_{year}.nc' in f for year in target_years)]
ds_pr =  xr.open_mfdataset(data_im_targets, combine='nested', concat_dim='time', join='override')
ds_pr = ds_pr.sel(lon=slice(-20,20), lat=slice(0,20))

def convert_time(time_array):
    converted_time = []
    for t in time_array:
        if isinstance(t, np.datetime64) or isinstance(t, pd.Timestamp):  # No conversion needed
            converted_time.append(pd.Timestamp(t))
        else:
            converted_time.append(t)
    return np.array(converted_time, dtype="datetime64[ns]")

normalized_time = convert_time(ds_pr['time'].values)
ds_pr = ds_pr.assign_coords(time=normalized_time)

regions_monthly = {'sah-w': [[14, 18, -18, 0], [6, 7, 8, 9]],
                   'sah-e': [[14, 18, 0, 15], [6, 7, 8, 9]],
                   'sud-sah-w': [[12, 14, -18, 0], [5, 6, 7, 8, 9]],
                   'sud-sah-e': [[12, 14, 0, 15], [5, 6, 7, 8, 9]],
                   'sud-w': [[8, 12, -18, 0], [4, 5, 6, 7, 8, 9, 10]],
                   'sud-e': [[8, 12, 0, 15], [4, 5, 6, 7, 8, 9, 10]],
                   'guin-w': [[4, 8, -18, 0], [3, 4, 5, 6, 7, 8, 9, 10]],
                   'guin-e': [[4, 8, 0, 15], [3, 4, 5, 6, 7, 8, 9, 10]]}

mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-18,15), lat=slice(4,18))
monthly = ds_pr.resample(time="1MS").sum(dim="time")
pr_monthly_domain = monthly['pr'].sel(lon=slice(-18,15), lat=slice(4,18))
ls_aligned = ls.reindex_like(pr_monthly_domain, method='nearest')
land_mask = (ls_aligned <= 50)
pr_monthly_land = pr_monthly_domain.where(land_mask, other=np.nan)


#%% -- mean regional PR plot --

def regional_mean(ds, lat0, lat1, lon0, lon1):
    return ds.sel(
        lat=slice(lat0, lat1),
        lon=slice(lon0, lon1)
    ).mean(dim=("lat", "lon"))

pr_regs = {}
for region in regions_monthly.keys():
    lat0, lat1, lon0, lon1 = regions_monthly[region][0]
    pr_reg  = regional_mean(ds_pr.pr.sel(lon=slice(-18,15), lat=slice(4,18)).where(land_mask, other=np.nan), lat0, lat1, lon0, lon1)
    pr_regs[region] = pr_reg

selr = 'sah-w'
selrn = 'Sahelian-W'
selt = 2024
plt.style.use('ggplot')

plt.figure(figsize=(8,4))
pr_regs[selr].sel(time=slice(f'{selt}-02-01', f'{selt}-11-30')).plot(color='tab:blue')
plt.axhline(1, color='r', linestyle=':', label='1-mm wet threshold')
plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=10))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
for label in plt.gca().get_xticklabels(which='minor'):
    label.set(fontsize=6)
plt.gca().get_xaxis().set_tick_params(which='major', pad=10)
plt.ylim(0,27)
plt.xlabel("")
plt.ylabel("Regional-mean precipitation")
plt.legend(loc='upper right')
plt.grid(False)
plt.title(f"{selrn}; {selt}")
plt.show()


#%% -- PR anomalies per month --

regional_anoms_monthly = {}
for region, (bbox, months) in regions_monthly.items():
    months = months[:1]
    lat_min, lat_max, lon_min, lon_max = bbox
    pr_reg = pr_monthly_land.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    ).mean(dim=("lat", "lon"))
    pr_reg = pr_reg.sel(time=pr_reg["time.month"].isin(months))
    
    clim = pr_reg.groupby("time.month").mean("time")
    anoms = pr_reg.groupby("time.month") - clim
    anom_idx = anoms / anoms.std("time")
    regional_anoms_monthly[region] = anom_idx
    

def regional_anoms_to_csv(regional_anoms_monthly, filename):
    records = []

    for region, da in regional_anoms_monthly.items():
        df = da.to_dataframe().reset_index()
        df = df.rename(columns={'time': 'time_dim'})
        df['year']  = df['time_dim'].dt.year
        df['month'] = df['time_dim'].dt.month

        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        value_col    = numeric_cols[0]

        for _, row in df.iterrows():
            records.append({
                'year':    int(row['year']),
                'region':  region,
                'month':   int(row['month']),
                'anomaly': round(float(row[value_col]), 2),
            })

    df_out = (pd.DataFrame(records)
                .sort_values(['region', 'year', 'month'])
                .reset_index(drop=True))
    df_out.to_csv(filename)
    return filename

csv_file = regional_anoms_to_csv(regional_anoms_monthly, filename='monthly_rs_anoms_2003-2024.csv')


#%% -- correlation heatmap PRa in 1st & 2nd RS month --

filepath_anoms = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv'
df_all = pd.read_csv(filepath_anoms, index_col="year")

climate_indices1 = ["ENSO_NDJ", "ENSO_DJF", "ENSO_JFM", "ENSO_FMA", "ENSO_MAM", 
                   "AMM_NDJ", "AMM_DJF", "AMM_JFM", "AMM_FMA", "AMM_MAM",
                   "DUST_NDJ", "DUST_DJF", "DUST_JFM", "DUST_FMA", "DUST_MAM",
                   "OM_N_NDJ", "OM_N_DJF", "OM_N_JFM", "OM_N_FMA", "OM_N_MAM",
                   "OM_S_NDJ", "OM_S_DJF", "OM_S_JFM", "OM_S_FMA", "OM_S_MAM",
                   "IOD_NDJ", "IOD_DJF"	, "IOD_JFM", "IOD_FMA", "IOD_MAM",
                   "NAO_NDJ", "NAO_DJF"	, "NAO_JFM", "NAO_FMA", "NAO_MAM"]
climate_indices2 = ["ENSO_NDJ", "ENSO_DJF", "ENSO_JFM", "ENSO_FMA", "ENSO_MAM", "ENSO_AMJ", 
                   "AMM_NDJ", "AMM_DJF", "AMM_JFM", "AMM_FMA", "AMM_MAM", "AMM_AMJ",
                   "DUST_NDJ", "DUST_DJF", "DUST_JFM", "DUST_FMA", "DUST_MAM", "DUST_AMJ",
                   "OM_N_NDJ", "OM_N_DJF", "OM_N_JFM", "OM_N_FMA", "OM_N_MAM", "OM_N_AMJ",
                   "OM_S_NDJ", "OM_S_DJF", "OM_S_JFM", "OM_S_FMA", "OM_S_MAM", "OM_S_AMJ",
                   "IOD_NDJ", "IOD_DJF"	, "IOD_JFM", "IOD_FMA", "IOD_MAM", "IOD_AMJ",
                   "NAO_NDJ", "NAO_DJF"	, "NAO_JFM", "NAO_FMA", "NAO_MAM", "NAO_AMJ"]
regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']

df_indices1  = df_all.loc[2003:2024][climate_indices1]
df_indices2  = df_all.loc[2003:2024][climate_indices2]
df_pranoms1 = df_all.loc[2003:2024][[f'{reg}-1stmonth_anom' for reg in regions]].rename(columns={f'{reg}-1stmonth_anom': f'{reg}' for reg in regions})
df_pranoms2 = df_all.loc[2003:2024][[f'{reg}-2ndmonth_anom' for reg in regions]].rename(columns={f'{reg}-2ndmonth_anom': f'{reg}' for reg in regions})

region_seasons1 = {
    "sah-w":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sah-e":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-sah-w": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-sah-e": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-w":     ["NDJ", "DJF", "JFM"],
    "sud-e":     ["NDJ", "DJF", "JFM"],
    "guin-w":    ["NDJ", "DJF"],
    "guin-e":    ["NDJ", "DJF"]
}

region_seasons2 = {
    "sah-w":     ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"],
    "sah-e":     ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"],
    "sud-sah-w": ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-sah-e": ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-w":     ["NDJ", "DJF", "JFM", "FMA"],
    "sud-e":     ["NDJ", "DJF", "JFM", "FMA"],
    "guin-w":    ["NDJ", "DJF", "JFM"],
    "guin-e":    ["NDJ", "DJF", "JFM"]
}
north_regs = {"sah-w", "sah-e", "sud-sah-w", "sud-sah-e"}

from scipy import stats

def valid_indices_for_region(region, all_index_cols):
    allowed_seasons = region_seasons1.get(region, [])
    # allowed_seasons = region_seasons2.get(region, [])
    out = []
    for col in all_index_cols:
        season_suffix = col.split("_")[-1]
        if season_suffix not in allowed_seasons:
            continue
        if col.startswith("OM_N") and region not in north_regs:
            continue
        if col.startswith("OM_S") and region in north_regs:
            continue
        out.append(col)
    return out

def rs1_index_association(df_onset, df_indices):
    df_onset = df_onset.copy()
    df_indices = df_indices.copy()

    df_onset.index = pd.Index(df_onset.index.astype(int), name="year")
    df_onset = df_onset.apply(pd.to_numeric, errors="coerce")
    df_indices = df_indices.apply(pd.to_numeric, errors="coerce")

    region_names = list(df_onset.columns)
    index_names = list(df_indices.columns)

    r_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)

    for reg in region_names:
        valid_seasons = set(valid_indices_for_region(reg, index_names))
        for idx in index_names:
            if idx not in valid_seasons:
                continue
            tmp = pd.concat(
                [df_indices[idx].rename("x"), df_onset[reg].rename("y")],
                axis=1
            ).dropna()

            if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                continue

            r, p = stats.pearsonr(tmp["x"], tmp["y"])
            r_matrix.loc[idx, reg] = r
            p_matrix.loc[idx, reg] = p


    r_matrix.to_csv("x.correlations/2003-2024/corr_matrix_1stmnth.csv")
    # r_matrix.to_csv("x.correlations/2003-2024/corr_matrix_2ndmnth.csv")
    return r_matrix, p_matrix

r_matrix, p_matrix = rs1_index_association(df_pranoms1, df_indices1)
# r_matrix, p_matrix = rs1_index_association(df_pranoms2, df_indices2)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

def plot_correlation_heatmap(r_matrix, p_matrix):
    fig = plt.figure(figsize=(20,5))
    plt.gca().set_facecolor('white')
    annot = p_matrix.copy().astype(object)
    annot[:] = ''
    annot[p_matrix < 0.05] = '*'

    ax = sns.heatmap(
        r_matrix.T.astype(float),
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=annot.T,
        annot_kws={'color': 'k'},
        fmt='',
        linewidths=1.3,
        cbar_kws={'label': 'Pearson r'}
    )
    plt.gca().tick_params(axis='x', rotation=90, labelsize=16)
    plt.gca().tick_params(axis='y', rotation=0, labelsize=18)
    ax.figure.axes[-1].yaxis.label.set_size(16)
    ax.figure.axes[-1].tick_params(labelsize=16)
    plt.title('Correlation RS1 anomaly with climate indices', fontsize=24, y=1.02)
    # plt.title('Correlation precipitation anomaly in second RS month with climate indices', fontsize=16, y=1.01)
    plt.grid(False)
    fig.tight_layout()
    plt.savefig("x.correlations/2003-2024/corr_heatmap_1stmnth_vs_idx.png")
    # plt.savefig("x.correlations/2003-2024/corr_heatmap_2ndmnth_vs_idx.png")
    plt.close()

plot_correlation_heatmap(r_matrix, p_matrix)

