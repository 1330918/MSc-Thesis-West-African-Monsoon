# -- dry spells database --
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
plt.style.use('ggplot')

data_im_loc = 'C:/Users/ikeva/z.datasets/IMERG_PR_daily_africa/daily_precip_imerg_v06_*.nc'
data_im = sorted(glob.glob(data_im_loc))
target_years = np.arange(2003, 2025, 1)
data_im_targets = [f for f in data_im if any(f'_{year}.nc' in f for year in target_years)]
ds_im =  xr.open_mfdataset(data_im_targets, combine='nested', concat_dim='time', join='override')

def convert_time(time_array):
    converted_time = []
    for t in time_array:
        if isinstance(t, np.datetime64) or isinstance(t, pd.Timestamp):
            converted_time.append(pd.Timestamp(t))
        else:
            converted_time.append(t)
    return np.array(converted_time, dtype="datetime64[ns]")

normalized_time = convert_time(ds_im['time'].values)
ds_im = ds_im.assign_coords(time=normalized_time)

mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-18,15), lat=slice(4,18))
pr = ds_im['pr']
ls_aligned = ls.reindex_like(pr, method='nearest')
land_mask = (ls_aligned <= 50)
pr_land = pr.where(land_mask, other=np.nan)

GRIDBOX_CENTERS = {
    'sah-w': [
        [(16.5, -17.5), (16.5, -12.5), (16.5, -7.5), (16.5, -2.5)], 
        [(15.5, -17.5), (15.5, -12.5), (15.5, -7.5), (15.5, -2.5)] 
    ],
    'sah-e': [
        [(16.5, 2.5), (16.5, 7.5), (16.5, 12.5)],
        [(15.5, 2.5), (15.5, 7.5), (15.5, 12.5)]
    ],
    'sud-sah-w': [[(13.5, -17.5), (13.5, -12.5), (13.5, -7.5), (13.5, -2.5)]],
    'sud-sah-e': [[(13.5, 2.5), (13.5, 7.5), (13.5, 12.5)]],
    'sud-w': [
        [(11.5, -17.5), (11.5, -12.5), (11.5, -7.5), (11.5, -2.5)],
        [(9.5, -17.5), (9.5, -12.5), (9.5, -7.5), (9.5, -2.5)]
    ],
    'sud-e': [
        [(11.5, 2.5), (11.5, 7.5), (11.5, 12.5)],
        [(9.5, 2.5), (9.5, 7.5), (9.5, 12.5)]
    ],
    'guin-w': [
        [(7.5, -17.5), (7.5, -12.5), (7.5, -7.5), (7.5, -2.5)],
        [(5.5, -17.5), (5.5, -12.5), (5.5, -7.5), (5.5, -2.5)] 
    ],
    'guin-e': [
        [(7.5, 2.5), (7.5, 7.5), (7.5, 12.5)], 
        [(5.5, 2.5), (5.5, 7.5), (5.5, 12.5)]
    ]
}

DRY_THRESHOLDS = {
    'sah-w': [1, 5],     # <1 mm in 5 days
    'sah-e': [1, 5],  
    'sud-sah-w': [1, 7], # <1 mm in 7 days
    'sud-sah-e': [1, 7],
    'sud-w': [2, 7],     # <2 mm in 7 days
    'sud-e': [2, 7],
    'guin-w': [5, 7],    # <5 mm in 7 days
    'guin-e': [5, 7] 
}

SEASON_SPLIT_DOY = {
    'sah-w':     221,  # Aug 10
    'sah-e':     221,
    'sud-sah-w': 213,  # Aug 1
    'sud-sah-e': 213,
    'sud-w':     203,  # Jul 22
    'sud-e':     203,
    'guin-w':    198,  # Jul 17
    'guin-e':    198,
}

regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']
region_names = ['Sahelian-W','Sahelian-E', 'Sudano-Sahelian-W', 'Sudano-Sahelian-E', 'Sudanian-W', 'Sudanian-E', 'Guinean-W', 'Guinean-E']

# extract detected onsets and cessations
df_dates = pd.read_csv('C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv', index_col='year')
actual_onset_doy = {reg: df_dates[f'{reg}-doy_os'].dropna().to_dict()  for reg in regions}
actual_cess_doy  = {reg: df_dates[f'{reg}-doy_cess'].dropna().to_dict() for reg in regions}


#%% -- spell detection --

def build_monsoon_mask(time_index, onset_doy_dict, cess_doy_dict):
    mask = np.zeros(len(time_index), dtype=bool)
    for t, timestamp in enumerate(time_index):
        year = pd.to_datetime(timestamp).year
        onset = onset_doy_dict.get(year)
        cess  = cess_doy_dict.get(year)
        doy   = pd.to_datetime(timestamp).dayofyear
        mask[t] = (doy >= onset) and (doy <= cess)
    return mask

def detect_daily_dry_spells(da, region, lat_c, lon_c, dry_thresh, dry_conseq, onset_doy_dict, cess_doy_dict):
    DAILY_RAIN_THRESHOLD = 1.0 # <1 mm is a dry day
    lat_slice = slice(lat_c-1, lat_c+1)
    lon_slice = slice(lon_c-2.5, lon_c+2.5)
    box_pr = da.sel(lat=lat_slice, lon=lon_slice).mean(['lat','lon'])
    box_pr = box_pr.compute()
    
    monsoon_mask_np = build_monsoon_mask(
        box_pr.time.values,
        onset_doy_dict, cess_doy_dict
        )
    monsoon_mask = xr.DataArray(monsoon_mask_np, coords={'time': box_pr.time}, dims=['time'])

    pr_roll = box_pr.rolling(time=dry_conseq, center=False).sum()
    dry_window_end = (pr_roll < dry_thresh) & monsoon_mask & ~pr_roll.isnull()
    dry_binary = dry_window_end.copy()
    for i in range(1, dry_conseq):
        dry_binary = dry_binary | dry_window_end.shift(time=-i, fill_value=False)
    dry_binary = dry_binary & monsoon_mask
    
    dry_start = dry_binary & ~dry_binary.shift(time=1, fill_value=False)
    spell_id = dry_start.cumsum(dim='time').where(dry_binary)
    
    spells_list = []
    for spell_num in np.unique(spell_id):
        if np.isnan(spell_num): continue
        spell_mask = spell_id == spell_num
        duration = spell_mask.sum().item()
        
        start_time = pd.Timestamp(box_pr.time.where(spell_mask, drop=True).isel(time=0).item())
        end_time = pd.Timestamp(box_pr.time.where(spell_mask, drop=True).isel(time=-1).item())
        pre_start = start_time - pd.Timedelta(days=3)
        post_end = end_time + pd.Timedelta(days=3)
        pre_rain = box_pr.sel(time=slice(pre_start, start_time-pd.Timedelta(days=1))).max() > DAILY_RAIN_THRESHOLD
        post_rain = box_pr.sel(time=slice(end_time+pd.Timedelta(days=1), post_end)).max() > DAILY_RAIN_THRESHOLD

        if pre_rain and post_rain:
            spell_total_rain = float(box_pr.sel(time=slice(start_time, end_time)).sum())
            spells_list.append({
                'lat': lat_c, 'lon': lon_c, 'region': region,
                'start_time': start_time,
                'start_doy': start_time.dayofyear,
                'end_time': end_time,
                'duration': duration,
                'total_rain': spell_total_rain,
                'year': start_time.year,
                'gridbox': f"{lat_c:.1f}N_{lon_c:+5.1f}E"
            })
    
    return pd.DataFrame(spells_list)

dry_spells_db = []
years = ds_im.time.dt.year.values
years = np.unique(years)

for region in regions:
    print(region)
    dry_thresh, dry_conseq = DRY_THRESHOLDS[region]
    
    reg_spells = []
    for lat_cells in GRIDBOX_CENTERS[region]:
        for lat_c, lon_c in lat_cells:
            box_spells = detect_daily_dry_spells(pr_land, region, lat_c, lon_c, dry_thresh, dry_conseq,
                                                 onset_doy_dict=actual_onset_doy[region],
                                                 cess_doy_dict=actual_cess_doy[region]
                                                 )
            box_spells['gridbox'] = f"{lat_c:.1f}N_{lon_c:+5.1f}E"
            reg_spells.append(box_spells)
            
    dry_spells_db.append(pd.concat(reg_spells, ignore_index=True))


#%% -- merge spells within region --
 
def merge_overlapping_spells(df_spells):
    if df_spells.empty:
        return df_spells
    df = df_spells.sort_values('start_time').copy()
    
    merged = []
    current = df.iloc[0].copy()
    current['gridboxes'] = [current['gridbox']]
    current['lats'] = [current['lat']]
    current['lons'] = [current['lon']]
    current['durations'] = [current['duration']]
    current['start_doys'] = [current['start_doy']]
    
    def finalise(c):
        c['lat'] = np.mean(c['lats'])
        c['lon'] = np.mean(c['lons'])
        c['gridbox'] = ', '.join(set(c['gridboxes']))
        c['n_gridboxes'] = len(set(c['gridboxes']))
        c['year'] = c['start_time'].year
        c['duration_span'] = (c['end_time'] - c['start_time']).days + 1
        c['duration_min'] = min(c['durations'])
        c['start_doy'] = min(c['start_doys'])
        return c
    
    for _, spell in df.iloc[1:].iterrows():
        if spell['start_time'] <= current['end_time']:
            current['end_time'] = max(current['end_time'], spell['end_time'])
            current['gridboxes'].append(spell['gridbox'])
            current['lats'].append(spell['lat'])
            current['lons'].append(spell['lon'])
            current['durations'].append(spell['duration'])
            current['start_doys'].append(spell['start_doy'])
        else:
            current = finalise(current)
            merged.append(current)
            current = spell.copy()
            current['gridboxes'] = [spell['gridbox']]
            current['lats'] = [spell['lat']]
            current['lons'] = [spell['lon']]
            current['durations'] = [spell['duration']]
            current['start_doys'] = [spell['start_doy']]
    
    merged.append(finalise(current)) 
    return pd.DataFrame(merged).drop(columns=['gridboxes', 'lats', 'lons', 'durations', 'start_doys'])


yearly_stats_all = []
df_merged_all = {}
all_years = pd.DataFrame({'year': np.arange(2003, 2025)})

for i, region in enumerate(regions):
    df_merged = merge_overlapping_spells(dry_spells_db[i])  
    if df_merged.empty:
        continue
    df_merged['is_long_spell'] = df_merged['duration_span'] >= 10
    df_merged['is_early_long_spell'] = (
        (df_merged['duration_span'] >= 10) &
        (df_merged['start_doy'] < SEASON_SPLIT_DOY[region])
    )
    df_merged['is_late_long_spell'] = (
        (df_merged['duration_span'] >= 10) &
        (df_merged['start_doy'] >= SEASON_SPLIT_DOY[region])
    )
    df_merged_all[region] = df_merged
         
    yearly = all_years.merge(df_merged.groupby('year').size().reset_index(name='n_spells'), on='year', how='left')
    yearly['n_spells'] = yearly['n_spells'].fillna(0)
    mean_spells = yearly['n_spells'].mean()
    yearly['n_spells_mean'] = mean_spells
    yearly['n_spells_anom'] = yearly['n_spells'] - mean_spells
    yearly['n_spells_anom_pct'] = (yearly['n_spells_anom'] / mean_spells) * 100
    yearly['region'] = region
    yearly['region_name'] = region_names[i]
    
    yearly_extra = all_years.merge(
        df_merged.groupby('year').agg(
            n_long_spells        = ('is_long_spell',       'sum'),
            n_early_long_spells  = ('is_early_long_spell',  'sum'),
            n_late_long_spells   = ('is_late_long_spell',   'sum'),
            mean_duration        = ('duration_min',         'mean'),
            mean_start_doy       = ('start_doy',            'mean'),
        ).reset_index(), on='year', how='left')
    yearly_extra['above_normal_spells'] = (yearly['n_spells'].fillna(0) > 
                                        yearly['n_spells_mean'].iloc[0]).astype(int)

    yearly = yearly.merge(yearly_extra, on='year', how='left')
    yearly_stats_all.append(yearly)

yearly_stats_all = pd.concat(yearly_stats_all, ignore_index=True)
yearly_stats_all = yearly_stats_all.set_index('year')


#%% -- pivot & export --

def pivot_region_cols(df, col):
    pivoted = df.pivot(index='year', columns='region', values=col)
    pivoted = pivoted[regions]
    return pivoted.rename(columns={reg: f'{reg}-{col}' for reg in regions})

cols_to_pivot = [
    'n_spells', 'n_spells_anom', 'n_spells_anom_pct',
    'mean_duration', 'mean_start_doy', 'n_long_spells',
    'n_early_long_spells', 'n_late_long_spells', 'above_normal_spells'
]
yearly_stats_reset = yearly_stats_all.reset_index()
yearly_stats_reset['region'] = pd.Categorical(
    yearly_stats_reset['region'], 
    categories=regions, 
    ordered=True
)
wide = pd.concat([pivot_region_cols(yearly_stats_reset, col) 
                  for col in cols_to_pivot], axis=1)
wide.index.name = 'year'
wide = wide.sort_index()
wide.to_excel("dry_spells_2003-2024.xlsx")


#%% -- correlation heatmap --
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

def pivot_reg(df, col):
    pivoted = round(df.pivot(index='year', columns='region', values=col),0)
    pivoted = pivoted[regions]
    return pivoted.rename(columns={reg: f'{reg}' for reg in regions})

nlong = False 
if nlong:
    df_spell_anom = pd.concat([pivot_reg(yearly_stats_reset, 'n_long_spells')], axis=1)
    name = 'nlongspells'
    title = 'long dry spells'
else:
    df_spell_anom = pd.concat([pivot_reg(yearly_stats_reset, 'n_spells_anom')], axis=1)
    name = 'spellsanoms'
    title = 'dry spell anomaly'

filepath_anoms = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024_v2.csv'
df_all = pd.read_csv(filepath_anoms, index_col="year")
climate_indices = ["ENSO_NDJ", "ENSO_DJF", "ENSO_JFM", "ENSO_FMA", "ENSO_MAM", 
                   "AMM_NDJ", "AMM_DJF", "AMM_JFM", "AMM_FMA", "AMM_MAM",
                   "DUST_NDJ", "DUST_DJF", "DUST_JFM", "DUST_FMA", "DUST_MAM",
                   "OM_N_NDJ", "OM_N_DJF", "OM_N_JFM", "OM_N_FMA", "OM_N_MAM",
                   "OM_S_NDJ", "OM_S_DJF", "OM_S_JFM", "OM_S_FMA", "OM_S_MAM",
                   "IOD_NDJ", "IOD_DJF"	, "IOD_JFM", "IOD_FMA", "IOD_MAM",
                   "NAO_NDJ", "NAO_DJF"	, "NAO_JFM", "NAO_FMA", "NAO_MAM"]

df_indices = df_all.loc[2003:2024][climate_indices]

region_seasons = {
    "sah-w":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sah-e":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-sah-w": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-sah-e": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-w":     ["NDJ", "DJF", "JFM"],
    "sud-e":     ["NDJ", "DJF", "JFM"],
    "guin-w":    ["NDJ", "DJF", "JFM"],
    "guin-e":    ["NDJ", "DJF", "JFM"]
}
north_regs = {"sah-w", "sah-e", "sud-sah-w", "sud-sah-e"}

def valid_indices_for_region(region, all_index_cols):
    allowed_seasons = region_seasons.get(region, [])
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

def spell_index_association(df_spell, df_indices):
    df_spell = df_spell.copy()
    df_indices = df_indices.copy()

    df_spell.index = pd.Index(df_spell.index.astype(int), name="year")
    df_spell = df_spell.apply(pd.to_numeric, errors="coerce")
    df_indices = df_indices.apply(pd.to_numeric, errors="coerce")

    region_names = list(df_spell.columns)
    index_names = list(df_indices.columns)

    r_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)

    for reg in region_names:
        valid_seasons = set(valid_indices_for_region(reg, index_names))
        for idx in index_names:
            if idx not in valid_seasons:
                continue
            tmp = pd.concat(
                [df_indices[idx].rename("x"), df_spell[reg].rename("y")],
                axis=1
            ).dropna()

            if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                continue

            r, p = stats.pearsonr(tmp["x"], tmp["y"])
            r_matrix.loc[idx, reg] = r
            p_matrix.loc[idx, reg] = p

    r_matrix.to_csv(f"x.correlations/2003-2024/corr_matrix_{name}.csv")
    return r_matrix, p_matrix

r_matrix, p_matrix = spell_index_association(df_spell_anom, df_indices)


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
    plt.title(f"Correlation {title} with climate indices", fontsize=24, y=1.02)
    plt.grid(False)
    fig.tight_layout()
    plt.savefig(f"x.correlations/2003-2024/corr_heatmap_{name}.png")
    plt.close()

plot_correlation_heatmap(r_matrix, p_matrix)
