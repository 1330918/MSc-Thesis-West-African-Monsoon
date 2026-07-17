#%% -- imports & data --
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import glob
plt.style.use('ggplot')

data_im_loc = 'C:/Users/ikeva/z.datasets/IMERG_PR_daily_africa/daily_precip_imerg_v06_*.nc'
data_im = sorted(glob.glob(data_im_loc))
target_years = np.arange(2003, 2025, 1)
data_im_targets = [f for f in data_im if any(f'_{year}.nc' in f for year in target_years)]
ds_pr =  xr.open_mfdataset(data_im_targets, combine='nested', concat_dim='time', join='override')
ds_enso = xr.open_dataset('C:/Users/ikeva/z.datasets/oni.nc', chunks=None)
ds_amm = xr.open_dataset('C:/Users/ikeva/z.datasets/amm.nc', chunks=None)
data_ae = 'C:/Users/ikeva/z.datasets/AE_daily_2003-2024.nc'
ds_ae = xr.open_dataset(data_ae, chunks=None)
var = 'value'

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
mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-20,40), lat=slice(0,35))
ls_aligned = ls.reindex_like(ds_pr['pr'], method='nearest')
land_mask = (ls_aligned <= 50)
pr_land = ds_pr['pr'].where(land_mask, other=np.nan)
yrs = np.arange(1981, 2025, 1)
enso = ds_enso[var].sel(time=ds_enso.time.dt.year.isin(yrs))
amm = ds_amm[var].sel(time=ds_amm.time.dt.year.isin(yrs))

vars_ae = ['duaod550', 'omaod550']
vars_ae_names = ['dust', 'organic matter']

region_names = ['Sahelian-W','Sahelian-E', 'Sudano-Sahelian-W', 'Sudano-Sahelian-E', 'Sudanian-W', 'Sudanian-E', 'Guinean-W', 'Guinean-E']
regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']
region_bnds = {'sah-w': [14, 18, -18, 0],
               'sah-e': [14, 18, 0, 15],
               'sud-sah-w': [12, 14, -18, 0],
               'sud-sah-e': [12, 14, 0, 15],
               'sud-w': [8, 12, -18, 0],
               'sud-e': [8, 12, 0, 15],
               'guin-w': [4, 8, -18, 0],
               'guin-e': [4, 8, 0, 15]
               }

seasons = {
    'NDJ': [11, 12, 1], 'DJF': [12, 1, 2],
    'JFM': [1, 2, 3],   'FMA': [2, 3, 4],
    'MAM': [3, 4, 5],   'AMJ': [4, 5, 6],
    'MJJ': [5, 6, 7],   'JJA': [6, 7, 8],
    'JAS': [7, 8, 9],   'ASO': [8, 9, 10],
    }

#%% -- compute index years -- 
def add_seasonal_year(da, season, time_coord):
    if season == 'NDJ':
        return da.assign_coords(
            seasonal_year=da[time_coord].dt.year + (da[time_coord].dt.month == 12) + (da[time_coord].dt.month == 11)
            )
    elif season == 'DJF':
        return da.assign_coords(
            seasonal_year=da[time_coord].dt.year + (da[time_coord].dt.month == 12)
            )

def sel_idx(da_sst, months, pos_threshold, neg_threshold, get_diags=False, get_vals=False):
    season_mask = da_sst.time.dt.month.isin(months)
    da_sst_season = da_sst.sel(time=season_mask)
    if months == [12, 1, 2] or months == [11, 12, 1]:
        da_sst_season = add_seasonal_year(da_sst_season, 'DJF', 'time') if months == [12, 1, 2] \
            else add_seasonal_year(da_sst_season, 'NDJ', 'time')
        if 2025 in da_sst_season.seasonal_year:
            da_sst_season = da_sst_season.where(da_sst_season.seasonal_year != 2025, drop=True)

        da_sst_mean = da_sst_season.groupby('seasonal_year').mean('time')
        pos_yrs = np.unique(da_sst_mean.seasonal_year.where(da_sst_mean >= pos_threshold, drop=True))
        neg_yrs = np.unique(da_sst_mean.seasonal_year.where(da_sst_mean <= neg_threshold, drop=True))
    else:
        da_sst_mean = da_sst_season.groupby('time.year').mean('time')
        pos_yrs = np.unique(da_sst_mean.year.where(da_sst_mean >= pos_threshold, drop=True))
        neg_yrs = np.unique(da_sst_mean.year.where(da_sst_mean <= neg_threshold, drop=True))
    if get_diags:
        if months == [12, 1, 2] or months == [11, 12, 1]:
            seasonal_mean = da_sst_mean.mean('seasonal_year').values
            seasonal_std = da_sst_mean.std('seasonal_year').values
        else:
            seasonal_mean = da_sst_mean.mean('year').values
            seasonal_std = da_sst_mean.std('year').values
        return seasonal_mean, seasonal_std, pos_threshold, neg_threshold
    elif get_vals:
        if months == [12, 1, 2] or months == [11, 12, 1]:
            neutral_yrs = np.setdiff1d(np.unique(da_sst_mean.seasonal_year.values), np.concatenate([pos_yrs, neg_yrs]))
        else:
            neutral_yrs = np.setdiff1d(np.unique(da_sst_mean.year.values), np.concatenate([pos_yrs, neg_yrs]))
        return da_sst_mean.values, pos_yrs, neg_yrs, neutral_yrs
    else:
        return pos_yrs, neg_yrs


#%% -- oceanic seasonal --
amm_mean = amm.mean("time")
amm_std = amm.std("time")
amm_stdidx = (amm - amm_mean) / amm_std # use standardized AMM with ±0.5σ threshold

seasonal_diags = {}
raw_data = {}
category_data = {}

composite_yrs = {}
for season, season_months in seasons.items():
    ENSO_diags = sel_idx(enso, season_months, 0.5, -0.5, get_diags=True, get_vals=False)    
    AMM_diags = sel_idx(amm_stdidx, season_months, 0.5, -0.5, get_diags=True, get_vals=False)
    ENSO_vals, ENSO_pos, ENSO_neg, ENSO_neutral = sel_idx(enso, season_months, 0.5, -0.5, get_diags=False, get_vals=True)
    AMM_vals, AMM_pos, AMM_neg, AMM_neutral = sel_idx(amm_stdidx, season_months, 0.5, -0.5, get_diags=False, get_vals=True)
    nino_yrs, nina_yrs = sel_idx(enso, season_months, 0.5, -0.5, get_diags=False, get_vals=False)
    amm_posyrs, amm_negyrs = sel_idx(amm_stdidx, season_months, 0.5, -0.5, get_diags=False, get_vals=False)
    
    composite_yrs[f'nino_{season}'] = nino_yrs
    composite_yrs[f'nina_{season}'] = nina_yrs
    composite_yrs[f'amm_pos_{season}'] = amm_posyrs
    composite_yrs[f'amm_neg_{season}'] = amm_negyrs
    
    seasonal_diags[f'ENSO_{season}'] = ENSO_diags
    seasonal_diags[f'AMM_{season}'] = AMM_diags

    raw_data[f'ENSO_{season}'] = ENSO_vals
    raw_data[f'AMM_{season}'] = AMM_vals
    
    def categorize_years(all_years, high_yrs, low_yrs, neutral_yrs):
        categories = np.zeros(len(all_years))
        for i, year in enumerate(all_years):
            if year in high_yrs:
                categories[i] = 1
            elif year in low_yrs:
                categories[i] = -1
            else:
                categories[i] = 0
        return categories
    
    category_data[f'ENSO_{season}'] = categorize_years(yrs, ENSO_pos, ENSO_neg, ENSO_neutral)
    category_data[f'AMM_{season}'] = categorize_years(yrs, AMM_pos, AMM_neg, AMM_neutral)  
    
season_order = seasons.keys()
enso_cols = [f'ENSO_{s}' for s in season_order]
amm_cols = [f'AMM_{s}' for s in season_order]

# column_order1 = enso_cols + amm_cols
# df_seasonal_diags = pd.DataFrame(seasonal_diags)
# df_seasonal_diags = df_seasonal_diags[column_order1]
# df_seasonal_diags.to_excel('seasonal_oceanic_diags.xlsx', index=False)

column_order = ['year'] + enso_cols + amm_cols
df_raw = pd.DataFrame(raw_data)
df_raw.insert(0, 'year', yrs)
df_raw = df_raw[column_order]
df_category = pd.DataFrame(category_data)
df_category.insert(0, 'year', yrs)
df_category = df_category[column_order]

for col in df_category.columns:
    if col != 'year':
        df_category[col] = df_category[col].astype(int)

with pd.ExcelWriter('index_oceanic.xlsx') as writer:
    df_raw.to_excel(writer, sheet_name='oc_anom', index=False)
    df_category.to_excel(writer, sheet_name='oc_idx', index=False)


#%% -- aerosol seasonal --
def clean_dirty_years(ds, var, domain, months, get_diags=False, get_vals=False, clim_file=None, index=None, season=None):
    minlat, maxlat, minlon, maxlon = domain
    ae_domain = ds[var].sel(
        latitude=slice(maxlat, minlat),
        longitude=slice(minlon, maxlon)
        )

    weights = np.cos(np.deg2rad(ae_domain["latitude"]))
    weights = weights / weights.mean()
    ae_mean = (ae_domain * weights).mean(dim=("latitude", "longitude"))
    season_mask = ae_mean.valid_time.dt.month.isin(months)
    ae_season = ae_mean.sel(valid_time=season_mask)
    
    if months == [11, 12, 1]:
        ae_season = add_seasonal_year(ae_season, 'NDJ', 'valid_time')
        if clim_file is None and 2025 in ae_season.seasonal_year:
            ae_season = ae_season.where(ae_season.seasonal_year != 2025, drop=True)
        ae_seasonal_mean = ae_season.groupby('seasonal_year').mean('valid_time')
        year_coord = 'seasonal_year'

    elif months == [12, 1, 2]:
        ae_season = add_seasonal_year(ae_season, 'DJF', 'valid_time')
        if clim_file is None and 2025 in ae_season.seasonal_year:
            ae_season = ae_season.where(ae_season.seasonal_year != 2025, drop=True)
        ae_seasonal_mean = ae_season.groupby('seasonal_year').mean('valid_time')
        year_coord = 'seasonal_year'

    else:
        ae_seasonal_mean = ae_season.groupby('valid_time.year').mean('valid_time')
        year_coord = 'year'
    
    if clim_file is not None:
        col_idx = f'{index}_{season}'
        seas_mean = clim_file[col_idx].loc['seasonal_mean']
        seas_std  = clim_file[col_idx].loc['seasonal_std']
        ae_idx = (ae_seasonal_mean - seas_mean) / seas_std
    else:
        ae_idx = (ae_seasonal_mean - ae_seasonal_mean.mean(year_coord)) / ae_seasonal_mean.std(year_coord)
    ae_idx = ae_idx.rename("ae_idx")

    if clim_file is None:
        q75 = ae_idx.quantile(0.75, dim=year_coord).values.item()
        q25 = ae_idx.quantile(0.25, dim=year_coord).values.item()
        high_yrs = ae_idx[year_coord].values[ae_idx.values > q75]
        low_yrs  = ae_idx[year_coord].values[ae_idx.values < q25]
    else:
        high_yrs, low_yrs, neutral_yrs = None, None, None
    
    if get_diags:
        seasonal_mean = ae_seasonal_mean.mean(year_coord).values
        seasonal_std = ae_seasonal_mean.std(year_coord).values
        return seasonal_mean, seasonal_std, q25, q75
    elif get_vals:
        neutral_yrs = np.setdiff1d(np.unique(ae_idx[year_coord].values), np.concatenate([high_yrs, low_yrs])) if high_yrs is not None else None
        return ae_idx.values, high_yrs, low_yrs, neutral_yrs
    else:
        return high_yrs, low_yrs


def get_dust_domain(season):
    if season in ['NDJ', 'DJF', 'JFM']:
        return [5, 20, -20, 20]
    elif season in ['FMA', 'MAM', 'AMJ', 'JAS', 'ASO']:
        return [10, 20, -20, 20]
    elif season in ['MJJ', 'JJA']:
        return [15, 20, -20, 20]

def get_om_n_domain(season):
    if season in ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'JAS', 'ASO']:
        return [10, 20, -20, 15]
    elif season in ['MJJ', 'JJA']:
        return [12, 20, -20, 15] 
    
def get_om_s_domain(season):
    if season in ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'JAS', 'ASO']:
        return [4, 10, -20, 15]
    elif season in ['MJJ', 'JJA']:
        return [4, 12, -20, 15] 

seasonal_diags = {}
raw_data = {}
category_data = {}
clim_file = pd.read_csv('C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_ae-clim.csv', index_col='type')

for season, season_months in seasons.items():
    DU_domain = get_dust_domain(season)
    OMN_domain = get_om_n_domain(season)
    OMS_domain = get_om_s_domain(season)
    
    DU_diags = clean_dirty_years(ds_ae, 'duaod550', DU_domain, season_months, get_diags=True, get_vals=False)
    OMN_diags = clean_dirty_years(ds_ae, 'omaod550', OMN_domain, season_months, get_diags=True, get_vals=False)
    OMS_diags = clean_dirty_years(ds_ae, 'omaod550', OMS_domain, season_months, get_diags=True, get_vals=False)
    
    DU_vals, DU_high, DU_low, DU_neutral = clean_dirty_years(ds_ae, 'duaod550', DU_domain, season_months, get_diags=False,
                                                             get_vals=True, clim_file=clim_file, index='DUST', season=season)
    OMN_vals, OMN_high, OMN_low, OMN_neutral = clean_dirty_years(ds_ae, 'omaod550', OMN_domain, season_months, get_diags=False,
                                                                 get_vals=True, clim_file=clim_file, index='OM_N', season=season)
    OMS_vals, OMS_high, OMS_low, OMS_neutral = clean_dirty_years(ds_ae, 'omaod550', OMS_domain, season_months, get_diags=False,
                                                                 get_vals=True, clim_file=clim_file, index='OM_S', season=season)

    seasonal_diags[f'DUST_{season}'] = DU_diags
    seasonal_diags[f'OM_{season}_N'] = OMN_diags
    seasonal_diags[f'OM_{season}_S'] = OMS_diags

    raw_data[f'DUST_{season}'] = DU_vals
    raw_data[f'OM_N_{season}'] = OMN_vals
    raw_data[f'OM_S_{season}'] = OMS_vals
    
    def categorize_years(all_years, high_yrs, low_yrs, neutral_yrs):
        categories = np.zeros(len(all_years))
        for i, year in enumerate(all_years):
            if year in high_yrs:
                categories[i] = 1
            elif year in low_yrs:
                categories[i] = -1
            else:
                categories[i] = 0
        return categories
    
    category_data[f'DUST_{season}'] = categorize_years(target_years, DU_high, DU_low, DU_neutral)
    category_data[f'OM_N_{season}'] = categorize_years(target_years, OMN_high, OMN_low, OMN_neutral)
    category_data[f'OM_S_{season}'] = categorize_years(target_years, OMS_high, OMS_low, OMS_neutral)
    

df_seasonal_diags = pd.DataFrame(seasonal_diags)
season_order = seasons.keys()
dust_cols = [f'DUST_{s}' for s in season_order]
om_n_cols = [f'OM_{s}_N' for s in season_order]
om_s_cols = [f'OM_{s}_S' for s in season_order]

# column_order1 = dust_cols + om_n_cols + om_s_cols
# df_seasonal_diags = df_seasonal_diags[column_order1]
# df_seasonal_diags.to_excel('seasonal_aerosol_diags.xlsx', index=False)

column_order = ['year'] + dust_cols + om_n_cols + om_s_cols
df_raw = pd.DataFrame(raw_data)
df_raw.insert(0, 'year', target_years)
df_category = pd.DataFrame(category_data)
df_category.insert(0, 'year', target_years)
df_raw = df_raw[column_order]
df_category = df_category[column_order]

for col in df_category.columns:
    if col != 'year':
        df_category[col] = df_category[col].astype(int)

with pd.ExcelWriter('index_aerosol.xlsx') as writer:
    df_raw.to_excel(writer, sheet_name='ae_raw', index=False)
    df_category.to_excel(writer, sheet_name='ae_idx', index=False)


#%%
# def get_valid_seasons(diagnostic):
#     if diagnostic in ['doy_os', '1stmonth_anom', 'dry_spell_mean_stdoy', 'false_onset']:
#         return {'northern1': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
#                 'northern2': ['NDJ', 'DJF', 'JFM', 'FMA'],
#                 'southern1': ['NDJ', 'DJF', 'JFM'],
#                 'southern2': ['NDJ', 'DJF'],
#                 }
#     elif diagnostic in ['dry_spell_anom', 'dry_spell_mean_dur', 'dry_spell_nlong']:
#         return {'northern1': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
#                 'northern2': ['NDJ', 'DJF', 'JFM', 'FMA'],
#                 'southern1': ['NDJ', 'DJF', 'JFM'],
#                 'southern2': ['NDJ', 'DJF', 'JFM'],
#                 }
#     elif diagnostic == 'doy_cess':
#         return {'northern1': ["MAM", "AMJ", "MJJ", "JJA"],
#                 'northern2': ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
#                 'southern1': ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
#                 'southern2': ["MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO"],
#                 }
#     elif diagnostic == 'lds_dur':
#         return {'southern2': ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"]}


# idx_file = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/index_all.csv'
# df_idx = pd.read_csv(idx_file, index_col='year')
# indices = ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
# dfs = {}
    
# for idx in indices:
#     col_names = [f"{idx}_{s}" for s in seasons]
#     df_sel = df_idx.loc[:, col_names]
#     if idx in ['OM_N', 'OM_S']:
#         df_sel.columns = df_sel.columns.str.rsplit('_', n=1).str[1]
#     else:
#         df_sel.columns = df_sel.columns.str.rsplit('_').str[1]
#     dfs[idx] = df_sel

# northern_regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e']
# northern1_regions = ['sah-w', 'sah-e']
# northern2_regions = ['sud-sah-w', 'sud-sah-e']
# southern1_regions = ['sud-w', 'sud-e']
# southern2_regions = ['guin-w', 'guin-e']

# region = 'sah-w'
# diag = 'doy_os'
# loc = 'northern1' if region in northern1_regions else 'northern2' if region in northern2_regions \
#     else 'southern1' if region in southern1_regions else 'southern2' if region in southern2_regions else None
# seasons = get_valid_seasons(diag)[loc]


#%%  -- composite plotting (JJAS PRa) --

def get_idx_years(dfs, index, season):
    df_sel = dfs[index][season]
    neg_yrs  = df_sel.index[df_sel==-1].to_list()
    pos_yrs  = df_sel.index[df_sel==1].to_list()
    neut_yrs = df_sel.index[df_sel==0].to_list()
    
    return neg_yrs, neut_yrs, pos_yrs
    

def composite_diff(da_pr, dfs, index, season):
    neg_yrs, neut_yrs, pos_yrs = get_idx_years(dfs, index, season)
    pr_neg  = da_pr.sel(year=neg_yrs).mean('year')
    pr_pos  = da_pr.sel(year=pos_yrs).mean('year')
    pr_neut = da_pr.sel(year=neut_yrs).mean('year')
    pr_diff = pr_pos - pr_neg 
    
    return pr_pos, pr_neut, pr_neg, pr_diff, len(neg_yrs), len(pos_yrs)


def pr_period(da_pr, months):
    pr_sel = da_pr.sel(time=da_pr.time.dt.month.isin(months), lon=slice(-18,15), lat=slice(4,18))
    pr_anom = pr_sel.groupby('time.month') - pr_sel.groupby('time.month').mean('time')
    pr_yearly = pr_anom.groupby('time.year').mean('time').compute()
    
    return pr_yearly

import cmocean
def composite_plot(comp_pos, comp_neg, comp_diff, n_neg, n_pos, index, season, pr_period):
    max_pos, min_pos = comp_pos.max().values, comp_pos.min().values
    max_neg, min_neg = comp_neg.max().values, comp_neg.min().values
    max_diff, min_diff = comp_diff.max().values, comp_diff.min().values
    
    fig, axs = plt.subplots(ncols=3, figsize=(15,3.2), subplot_kw=dict(projection=ccrs.PlateCarree()))
    comp_pos.plot.pcolormesh(ax=axs[0], x='lon', cmap=cmocean.cm.tarn, vmin=min_pos-5, vmax=max_pos+5,  cbar_kwargs={"label": "Precipitation anomaly (mm)", "orientation": "horizontal"})
    comp_neg.plot.pcolormesh(ax=axs[1], x='lon', cmap=cmocean.cm.tarn, vmin=min_neg-5, vmax=max_neg+5,  cbar_kwargs={"label": "Precipitation anomaly (mm)", "orientation": "horizontal"})
    comp_diff.plot(ax=axs[2], x='lon', cmap='RdBu', vmin=min_diff-5, vmax=max_diff+5, cbar_kwargs={"label": "Precipitation anomaly (mm)", "orientation": "horizontal"})

    for ax in axs:
        ax.coastlines(resolution='110m')
        ax.add_feature(cfeature.BORDERS, edgecolor='gray')
        gl = ax.gridlines(color='black', linestyle=':', alpha=0.2, draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        if ax == axs[1] or ax == axs[2]:
            gl.left_labels = False
        
    axs[0].set_title(f"Positive (n={n_pos})")
    axs[1].set_title(f"Negative (n={n_neg})")
    axs[2].set_title("Difference (pos - neg)")
    plt.suptitle(f"{pr_period} precipitation composites for {index} in {season}", fontsize=16)
    plt.savefig(f'x.composites/1/{pr_period}_composite_{index}_{season}', bbox_inches='tight')
    plt.close()

idx_file = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/index_all.csv'
df_idx = pd.read_csv(idx_file, index_col='year')
indices = ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
dfs = {}

for idx in indices:
    col_names = [f"{idx}_{s}" for s in seasons]
    df_sel = df_idx.loc[:, col_names]
    if idx in ['OM_N', 'OM_S']:
        df_sel.columns = df_sel.columns.str.rsplit('_', n=1).str[1]
    else:
        df_sel.columns = df_sel.columns.str.rsplit('_').str[1]
    dfs[idx] = df_sel
    
pr_monthly = pr_land.resample(time='MS').sum('time').where(land_mask, other=np.nan)
pr_jjas = pr_period(pr_monthly, [6,7,8,9])
valid_seasons = list(seasons.keys())[:5]

for idx in indices:
    for season in valid_seasons:
        pr_pos_jjas, pr_neut_jjas, pr_neg_jjas,pr_diff_jjas, n_neg, n_pos = composite_diff(pr_jjas, dfs, idx, season)
        composite_plot(pr_pos_jjas, pr_neg_jjas, pr_diff_jjas, n_neg, n_pos, idx, season, 'JJAS')
        

#%% -- onset × index composites --

filepath_anoms = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv'
df_raw = pd.read_csv(filepath_anoms)
df_raw.columns = df_raw.columns.str.strip()
df_raw['year'] = df_raw['year'].astype(int)
df_raw = df_raw.set_index('year').loc[2003:2024]
for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
diagnostics = ['doy_os', 'doy_cess', '1stmonth_anom',
               'dry_spell_anom', 'dry_spell_nlong',
               'false_onset', 'lds_dur']
df_all = pd.read_csv(filepath_anoms, index_col='year')
dfs = {}
for diag in diagnostics:
    col_names = [f"{reg}-{diag}" for reg in regions]
    cols = [c for c in df_all.columns if c in col_names]
    df_sel = df_all.loc[:, cols]
    df_sel.columns = df_sel.columns.str.rsplit('-', n=1).str[0]
    dfs[diag] = df_sel.loc[2003:2024]

means_os, p30_os, p70_os = {}, {}, {}
means_cess, p30_cess, p70_cess = {}, {}, {}
p30_1stmnth, p70_1stmnth = {}, {}
neg_spellanom, pos_spellanom = {}, {}
p30_nlongspell, p70_nlongspell = {}, {}
med_nlongspell = {}
p30_ldsdur, p70_ldsdur = {}, {}

for region in regions:
    p30_os[region] = round(np.percentile(dfs['doy_os'][region], 30))
    p70_os[region] = round(np.percentile(dfs['doy_os'][region], 70))

    p30_cess[region] = round(np.percentile(dfs['doy_cess'][region], 30))
    p70_cess[region] = round(np.percentile(dfs['doy_cess'][region], 70))

    p30_1stmnth[region] = np.percentile(dfs['1stmonth_anom'][region], 30)
    p70_1stmnth[region] = np.percentile(dfs['1stmonth_anom'][region], 70)
    
    neg_spellanom[region] = -1
    pos_spellanom[region] = 1
    med_nlongspell[region] = np.floor(np.median(dfs['dry_spell_nlong'][region]))
    
    if region in ['guin-w', 'guin-e']:
        p30_ldsdur[region] = round(np.percentile(dfs['lds_dur'][region], 30))
        p70_ldsdur[region] = round(np.percentile(dfs['lds_dur'][region], 70))
        
DIAG_THRESHOLDS = {
    'doy_os': {'low': p30_os, 'high': p70_os},
    'doy_cess': {'low': p30_cess, 'high': p70_cess},
    '1stmonth_anom': {'low': p30_1stmnth, 'high': p70_1stmnth},
    'dry_spell_anom': {'low': neg_spellanom, 'high': pos_spellanom},
    'dry_spell_nlong': {'median': med_nlongspell},
    'lds_dur': {'low': p30_ldsdur, 'high': p70_ldsdur},
    'false_onset': {'yes': 1, 'no': 0}
}

regions      = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e',
                'sud-w', 'sud-e', 'guin-w', 'guin-e']
region_names = ['Sahelian-W', 'Sahelian-E', 'Sudano-Sahelian-W', 'Sudano-Sahelian-E',
                'Sudanian-W', 'Sudanian-E', 'Guinean-W', 'Guinean-E']
indices      = ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
seasons      = ['DJF', 'FMA', 'MAM', 'MJJ', 'JAS']

REGION_BOXES = {
    'sah-w':     (-18,  2, 14, 18),
    'sah-e':     (  2, 15, 14, 18),
    'sud-sah-w': (-18,  2, 11, 14),
    'sud-sah-e': (  2, 15, 11, 14),
    'sud-w':     (-18,  2,  8, 11),
    'sud-e':     (  2, 15,  8, 11),
    'guin-w':    (-18,  2,  4,  8),
    'guin-e':    (  2, 15,  4,  8),
}
REGION_CENTROIDS = {
    'sah-w':     (-8.0, 16.0), 'sah-e':     ( 8.5, 16.0),
    'sud-sah-w': (-8.0, 12.5), 'sud-sah-e': ( 8.5, 12.5),
    'sud-w':     (-8.0,  9.5), 'sud-e':     ( 8.5,  9.5),
    'guin-w':    (-8.0,  6.0), 'guin-e':    ( 8.5,  6.0),
}

northern1_regions = ['sah-w', 'sah-e']
northern2_regions = ['sud-sah-w', 'sud-sah-e']
southern1_regions = ['sud-w', 'sud-e']
southern2_regions = ['guin-w', 'guin-e']

DIAG_LABELS = {
    'doy_os': ['early', 'clim', 'late'],
    'doy_cess': ['early', 'clim', 'late'],
    '1stmonth_anom': ['below-normal', 'near-normal', 'above-normal'],
    'dry_spell_anom': ['below-normal', 'near-normal', 'above-normal'],
    'dry_spell_nlong': ['low risk', 'elevated risk'],
    'dry_spell_late_long': ['low risk', 'elevated risk'],
    'false_onset': ['yes', 'no'],
    'lds_dur': ['below-normal', 'near-normal', 'above-normal'],
}
diag_names = {
    'doy_os':               'Onset date',
    'doy_cess':             'Cessation date',
    '1stmonth_anom':        'RS1 anomaly',
    'dry_spell_anom':       'Dry spell anomaly',
    'dry_spell_nlong':      'No. long dry spells',
    'false_onset':          'False onset',
    'lds_dur':              'LDS duration',
}

def get_labels(diag):
    return DIAG_LABELS[diag]

def get_region_band(region):
    if region in northern1_regions: return 'northern1'
    if region in northern2_regions: return 'northern2'
    if region in southern1_regions: return 'southern1'
    if region in southern2_regions: return 'southern2'
    return None

def index_region_allowed(index, region):
    band = get_region_band(region)

    if index == 'OM_N':
        return band in ['northern1', 'northern2']
    elif index == 'OM_S':
        return band in ['southern1', 'southern2']
    else:
        return True
    
def get_valid_seasons(diagnostic):
    if diagnostic in ['doy_os', '1stmonth_anom', 'dry_spell_mean_stdoy', 'false_onset']:
        return {'northern1': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
                'northern2': ['NDJ', 'DJF', 'JFM', 'FMA'],
                'southern1': ['NDJ', 'DJF', 'JFM'],
                'southern2': ['NDJ', 'DJF'],
                }
    elif diagnostic in ['dry_spell_anom', 'dry_spell_mean_dur', 'dry_spell_nlong']:
        return {'northern1': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
                'northern2': ['NDJ', 'DJF', 'JFM', 'FMA'],
                'southern1': ['NDJ', 'DJF', 'JFM'],
                'southern2': ['NDJ', 'DJF', 'JFM'],
                }
    elif diagnostic == 'doy_cess':
        return {'northern1': ["MAM", "AMJ", "MJJ", "JJA"],
                'northern2': ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
                'southern1': ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
                'southern2': ["MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO"],
                }
    elif diagnostic == 'lds_dur':
        return {'southern2': ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"]}
    

EXTENT = [-18, 15, 4, 18]
land_mask_da = (pr_land.isel(time=0).notnull()
                .sel(lon=slice(-18, 15), lat=slice(4, 18)))

def _setup_ax(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.4)
    ax.add_feature(cfeature.LAND,  facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#dce9f5', zorder=0)
    gl = ax.gridlines(draw_labels=True, color='black',
                      linestyle=':', alpha=0.25, linewidth=0.4)
    gl.top_labels = False; gl.right_labels = False
    return gl
            
def _draw_regions_masked(ax, values_dict, cmap, norm, land_mask_da, annotate=True):
    lats = land_mask_da['lat'].values
    lons = land_mask_da['lon'].values
    land = land_mask_da.values
    if land.shape == (330, 140):
        land = land.transpose(1, 0)
    rgba = np.zeros((*land.shape, 4), dtype=float)

    for region in regions:
        if region not in values_dict or np.isnan(values_dict[region]):
            continue
        x0, x1, y0, y1 = REGION_BOXES[region]
        color = cmap(norm(values_dict[region]))
        lon_mask = (lons >= x0) & (lons <= x1)
        lat_mask = (lats >= y0) & (lats <= y1)
        box_mask = np.outer(lat_mask, lon_mask)
        fill     = box_mask & land
        rgba[fill] = color

    ax.imshow(rgba,
              extent=[lons.min(), lons.max(), lats.min(), lats.max()],
              origin='lower', transform=ccrs.PlateCarree(), alpha=0.7,
              interpolation='nearest', aspect='auto', zorder=3)

    if annotate:
        for region in regions:
            if region not in values_dict or np.isnan(values_dict[region]):
                continue
            cx, cy = REGION_CENTROIDS[region]
            if REGION_BOXES[region][0] <= cx <= REGION_BOXES[region][1]:
                ax.text(cx, cy, f'{values_dict[region]:.1f}',
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', color='black',
                        transform=ccrs.PlateCarree(), zorder=5)

def _add_colorbar(fig, ax, cmap, norm, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.1, shrink=0.85)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

def classify_tercile(series, labels, low, high):
    out = pd.Series(index=series.index, dtype='object')
    out[series <= low] = labels[0]
    out[(series > low) & (series < high)] = labels[1]
    out[series >= high] = labels[2]
    return pd.Categorical(out, categories=labels, ordered=True)

def classify_binary(series, labels, threshold, mode):
    out = pd.Series(index=series.index, dtype='object')
    if mode == 'gt':
        out[series <= threshold] = labels[0]
        out[series > threshold] = labels[1]
    elif mode == 'bool':
        out[series == 0] = labels[0]
        out[series == 1] = labels[1]
    return pd.Categorical(out, categories=labels, ordered=True)

def classify_index_phase(series, plow=0.33, phigh=0.67):
    sub = series.dropna()
    return pd.cut(series,
                  bins=[-np.inf, sub.quantile(plow), sub.quantile(phigh), np.inf],
                  labels=['negative', 'neutral', 'positive'])
            
# Early/late onset years: mean index value composite
def plot_index_by_onset_class(df_raw, indices, seasons, land_mask_da, diag,
                              plow=0.30, phigh=0.70, out_dir='x.composites'):
    """
    For each index: one multi-panel figure with
      rows = diagnostic classes
      cols = only the seasons that are valid for this diagnostic
    Each panel shows the mean standardised index value for the years
    belonging to that diagnostic class in each region.
    """
    cmap = plt.cm.RdBu_r
    classes = get_labels(diag)
    regions_use = ['guin-w', 'guin-e'] if diag == 'lds_dur' else regions

    valid_season_dict = get_valid_seasons(diag)
    seasons_use = []
    for s in seasons:
        keep = False
        for region in regions_use:
            band = get_region_band(region)
            if band is not None and s in valid_season_dict.get(band, []):
                keep = True
                break
        if keep:
            seasons_use.append(s)

    for index in indices:
        season_class_data = {}
        all_vals = []

        for season in seasons_use:
            col_idx = f'{index}_{season}'
            if col_idx not in df_raw.columns:
                continue
            data = {cls: {} for cls in classes}

            for region in regions_use:
                col_diag = f'{region}-{diag}'
                if col_diag not in df_raw.columns:
                    continue
                if not index_region_allowed(index, region):
                    continue
                band = get_region_band(region)
                if band is None:
                    continue
                if season not in valid_season_dict.get(band, []):
                    continue

                series = df_raw[col_diag]
                
                if diag in ['doy_os', 'doy_cess', '1stmonth_anom', 'dry_spell_anom', 'lds_dur']:
                    low = DIAG_THRESHOLDS[diag]['low'][region]
                    high = DIAG_THRESHOLDS[diag]['high'][region]
                    classif = pd.Series(
                        classify_tercile(series, classes, low=low, high=high),
                        index=series.index)
                
                elif diag == 'dry_spell_nlong':
                    threshold = DIAG_THRESHOLDS[diag]['median'][region]
                    classif = pd.Series(
                        classify_binary(series, classes, threshold=threshold, mode='gt'),
                        index=series.index)
                
                elif diag == 'false_onset':
                    classif = pd.Series(
                        classify_binary(series, classes, threshold=None, mode='bool'),
                        index=series.index)

                for cls in classes:
                    yrs = classif[classif == cls].index
                    common = df_raw.index.intersection(yrs)
                    if len(common) >= 2:
                        val = df_raw.loc[common, col_idx].mean()
                        data[cls][region] = val
                        if not np.isnan(val):
                            all_vals.append(val)

            season_class_data[season] = data
        if not all_vals:
            continue

        vmax = 1
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        nrows = len(classes)
        ncols = len(seasons_use)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3.8 * ncols, 2.2 * nrows),
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            constrained_layout=True
        )

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for i, cls in enumerate(classes):
            for j, season in enumerate(seasons_use):
                ax = axes[i,j]
                gl = _setup_ax(ax)
                if j > 0:
                    gl.left_labels = False
                if i < nrows - 1:
                    gl.bottom_labels = False

                _draw_regions_masked(
                    ax,
                    season_class_data.get(season, {}).get(cls, {}),
                    cmap, norm, land_mask_da
                )
                if i == 0:
                    ax.set_title(season, fontsize=10)
                if j == 0:
                    ax.text(
                        -0.2, 0.5, cls,
                        transform=ax.transAxes,
                        rotation=90,
                        va='center', ha='center',
                        fontsize=10
                    )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            orientation='horizontal',
            shrink=0.9,
            pad=0.04
        )
        cb.set_label('Mean index value', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        fig.suptitle(f'Mean {index} by {diag_names[diag]}', fontsize=13)
        fname = f'{out_dir}/diag_vs_idx/idx_by_{diag}_{index}.png'
        fig.savefig(fname, bbox_inches='tight')
        plt.close()

for diag in diag_names.keys():
    plot_index_by_onset_class(df_raw, indices, seasons, land_mask_da,
                               diag=diag, out_dir='x.composites')
