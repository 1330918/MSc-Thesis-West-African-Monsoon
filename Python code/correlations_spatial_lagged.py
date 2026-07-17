 #%% -- imports --
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#%% -- variables --

LAT_MIN, LAT_MAX = 4, 18
LON_MIN, LON_MAX = -18, 15

REGIONS = ['sah-w','sah-e','sud-sah-w','sud-sah-e','sud-w','sud-e','guin-w','guin-e']
MONTHS_TO_USE = list(range(3,11))

indices = ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
seasons = ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM',
           'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO']
ae_idxs = [f'{ae}_{s}' for ae in indices[2:5] for s in seasons]
oc_idxs = [f'{oc}_{s}' for oc in indices[:2] for s in seasons]
iodnao_idxs = [f'{oc}_{s}' for oc in indices[5:] for s in seasons]
all_idxs = ae_idxs + oc_idxs + iodnao_idxs


 #%% -- functions --

def p_to_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def corr_pair(x, y, method='pearson'):
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan, len(df)
    if method == 'pearson':
        r, p = pearsonr(df['x'], df['y'])
    else:
        r, p = spearmanr(df['x'], df['y'])
    return r, p, len(df)


def load_index_csv(path):
    raw = pd.read_csv(path, header=None)
    header = raw.iloc[0].astype(str).tolist()
    df = raw.iloc[1:].copy()
    df.columns = header
    return df.reset_index(drop=True)


def clean_numeric(df):
    out = df.copy()
    for c in out.columns:
        if c != 'year':
            out[c] = pd.to_numeric(out[c], errors='ignore')
    out['year'] = pd.to_numeric(out['year'], errors='coerce')
    out = out.dropna(subset=['year'])
    out['year'] = out['year'].astype(int)
    return out


def load_monthly_region_anoms_from_csv(path):
    df = pd.read_csv(path)
    colmap = {c.lower(): c for c in df.columns}
    year_col = colmap.get('year', 'year')
    region_col = colmap.get('region', 'region')
    month_col = colmap.get('month', 'month')
    anomaly_col = colmap.get('anomaly', 'anomaly')

    out = df[[year_col, region_col, month_col, anomaly_col]].copy()
    out.columns = ['year', 'region', 'month', 'precip_anom']
    out['year'] = pd.to_numeric(out['year'], errors='coerce')
    out['month'] = pd.to_numeric(out['month'], errors='coerce')
    out['precip_anom'] = pd.to_numeric(out['precip_anom'], errors='coerce')
    out['region'] = out['region'].astype(str).str.strip()
    out = out.dropna(subset=['year', 'month', 'precip_anom'])
    out['year'] = out['year'].astype(int)
    out['month'] = out['month'].astype(int)
    out = out[out['region'].isin(REGIONS)]
    out = out[out['month'].isin(MONTHS_TO_USE)]
    return out.sort_values(['year', 'month', 'region']).reset_index(drop=True)

seasonal_mapping_to_monthly_pr = {
    'NDJ': list(range(3,11)),
    'DJF': list(range(3,11)),
    'JFM': list(range(4,11)),
    'FMA': list(range(5,11)),
    'MAM': list(range(6,11)),
    'AMJ': [7, 8, 9, 10],
    'MJJ': [8, 9, 10],
    'JJA': [9, 10],
    'JAS': [10],
    'ASO': [11],    
}

def build_corr_table(index_df, monthly_long, index_cols):
    rows = []
    for idx in index_cols:
        if idx not in ['ENSO_ASO', 'AMM_ASO', 'DUST_ASO',
                       'OM_N_ASO','OM_S_ASO', 'IOD_ASO', 'NAO_ASO']:
            tmp_idx = index_df[['year', idx]].copy().rename(columns={idx: 'index_value'})
            MONTHS_TO_USE = seasonal_mapping_to_monthly_pr[idx[-3:]]
            for m in MONTHS_TO_USE:
                mdf = monthly_long[monthly_long['month'] == m]
                merged = mdf.merge(tmp_idx, on='year', how='inner')
                for reg in REGIONS:
                    sdf = merged[merged['region'] == reg]
                    r, p, n = corr_pair(sdf['index_value'], sdf['precip_anom'], method='pearson')
                    rows.append({
                        'index_name': idx,
                        'method': 'pearson',
                        'month': m,
                        'region': reg,
                        'r': r,
                        'p': p,
                        'n': n,
                        'sig': p_to_stars(p)
                    })
    return pd.DataFrame(rows)


def season_to_month(idx, used_seasons):
    mapping = {}
    for i, s in enumerate(used_seasons):
        mapping[s] = i+3
    suffix = idx[-3:]
    return mapping[suffix]


def correlation_map(index_series, precip_da, target_month, method='pearson'):
    months_all = pd.DatetimeIndex(precip_da['time'].values).month
    sel = np.where(months_all == target_month)[0]
    da = precip_da.isel(time=sel)
    years = pd.DatetimeIndex(da['time'].values).year
    idx = pd.Series(index_series).dropna()
    idx.index = idx.index.astype(int)
    common = np.intersect1d(years, idx.index.values)
    da = da.isel(time=np.isin(years, common))
    years = pd.DatetimeIndex(da['time'].values).year
    idx_vals = np.array([idx.loc[y] for y in years])
    arr = da.values
    nx, ny = arr.shape[1], arr.shape[2]
    rmap = np.full((nx, ny), np.nan)
    pmap = np.full((nx, ny), np.nan)
    plist = []
    for i in range(nx):
        for j in range(ny):
            y = arr[:, i, j]
            valid = np.isfinite(y) & np.isfinite(idx_vals)
            if valid.sum() >= 3:
                if method == 'pearson':
                    r, p = pearsonr(idx_vals[valid], y[valid])
                else:
                    r, p = spearmanr(idx_vals[valid], y[valid])
                rmap[i, j] = r
                pmap[i, j] = p
                plist.append(p)
    return (
        xr.DataArray(rmap, coords={'lon': da['lon'], 'lat': da['lat']}, dims=('lon','lat')),
        xr.DataArray(pmap, coords={'lon': da['lon'], 'lat': da['lat']}, dims=('lon','lat')),
    )


def spatial_corr(index_df, precip_da, index_cols, method, output_dir):
    month_names = ['March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']
    used_seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO']
    fig, axs = plt.subplots(3, 3, figsize=(16,8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    axs = axs.flatten()
    c=0
        
    for idx in index_cols:
        if idx[-3:] not in used_seasons:
            continue
        target_month = season_to_month(idx, used_seasons)
        print(idx, target_month)
        ser = pd.Series(pd.to_numeric(index_df[idx], errors='coerce').values, index=index_df['year'].values).dropna()
        rmap, pmap = correlation_map(ser, precip_da, target_month, method=method)
    
        for ax, m in zip(axs, np.arange(0, len(month_names), 1)):
            ax.coastlines(resolution='110m')
            ax.add_feature(cfeature.BORDERS, edgecolor='gray')
            ax.set_title(f"{month_names[m]}", fontsize=22)
            gl = ax.gridlines(color='gray', linestyle=':', draw_labels=True, alpha=0.2)
            gl.top_labels = False
            gl.right_labels = False
            if m < 6:
                gl.bottom_labels = False
            if m != 0 and m != 3 and m != 6:
                gl.left_labels = False
            gl.xlabel_style = {'size': 20, 'color': 'gray'}
            gl.ylabel_style = {'size': 20, 'color': 'gray'}
            
            if m == 0 or m == 8:
                lon1, lon2, lat1, lat2 = [-18, 15, 4, 8]
            elif m == 1 or m == 7:
                lon1, lon2, lat1, lat2 = [-18, 15, 4, 12]
            elif m == 2:
                lon1, lon2, lat1, lat2 = [-18, 15, 4, 14]
            else:
                lon1 = None
            if lon1 is not None:
                rect = mpatches.Rectangle(
                    (lon1, lat1),
                    lon2 - lon1,
                    lat2 - lat1,
                    edgecolor='black',
                    facecolor='none',
                    linestyle='--',
                    linewidth=1,
                    zorder=5,
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(rect)
                
        pcm = axs[c].pcolormesh(rmap['lon'], rmap['lat'], rmap.T, cmap='RdBu_r', vmin=-1, vmax=1)
        axs[c].contourf(rmap['lon'], rmap['lat'], xr.where(pmap.T < 0.05, 1, np.nan), levels=[0.5, 1.5], colors='none', hatches=['....'])
        axs[c].set_xlim(LON_MIN-0.15, LON_MAX+0.15)
        axs[c].set_ylim(LAT_MIN-0.15, LAT_MAX+0.15)
        c+=1
        
    plt.suptitle(f"{idx[:-4]} post-season spatial correlation (2003-2024)", fontsize=24, y=1.01)
    plt.tight_layout()
    cax = fig.add_axes([1.02, 0.05, 0.02, 0.85])
    cb = fig.colorbar(pcm, cax=cax)
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label="Pearson r", fontsize=20, labelpad=4)
    plt.savefig(f'{output_dir}/spatial_corr_{idx[:-4]}.png', bbox_inches='tight')
    plt.close()
        

def corr_heatmap(corr_df, out_prefix, indexes, region_order):
    mdf = corr_df[corr_df['method'] == 'pearson'].copy()
    for reg in region_order:
        sub = mdf[mdf['region'] == reg].copy()
        if indexes == ae_idxs:
            if reg in ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e']:
                index_order = indexes[:20]
            else:
                index_order = indexes[:10] + indexes[20:]
        else:
            index_order = indexes

        mat = sub.pivot(index='index_name', columns='month', values='r')
        ann = sub.assign(label=sub.apply(lambda r: '' if pd.isna(r['r']) else f"{r['r']:.2f}{r['sig']}", axis=1))\
                 .pivot(index='index_name', columns='month', values='label')
        mat = mat.reindex([i for i in index_order if i in mat.index])
        ann = ann.reindex(mat.index)
        
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6*len(mat.index))), constrained_layout=True)
        ax.grid(False)

        im = sns.heatmap(
            mat,
            ax=ax,
            mask=mat.isna(),
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=ann,
            fmt='',
            linewidths=0.5,
            annot_kws={"fontsize": 12},
            cbar=True,
            cbar_kws={'label': 'Pearson correlation coefficient'}
        )
        ax.set_title(f'{reg}', fontsize=24, y=1.01)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=16)
        cax = im.figure.axes[-1]
        cax.tick_params(labelsize=16)
        cax.yaxis.label.set_size(16)
        fig.savefig(f'{out_prefix}_{reg}.png', bbox_inches='tight')
        plt.close(fig)
   

WINDOW_LABELS = ['NDJ','DJF','JFM','FMA','MAM']

def compute_jjas_region_anoms(monthly_long):
    jjas = monthly_long[monthly_long['month'].isin([6,7,8,9])].copy()
    out = jjas.groupby(['year','region'], as_index=False)['precip_anom'].mean()
    out = out.rename(columns={'precip_anom': 'jjas_precip_anom'})
    return out
    
def build_jjas_leadlag_table(index_df, monthly_long, index, method):
    jjas = compute_jjas_region_anoms(monthly_long)
    rows = []
    label_order = {lab: i for i, lab in enumerate(WINDOW_LABELS)}

    if index != 'OM':
        IDX_WINDOWS = [f'{index}_{w}' for w in WINDOW_LABELS]
        IDX_WINDOWS = sorted(IDX_WINDOWS, key=lambda x: label_order[x.split('_')[-1]])

    for reg in REGIONS:
        if index == 'OM':
            if reg in ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e']:
                OM_N_WINDOWS = [f'OM_N_{w}' for w in WINDOW_LABELS]
                OM_N_WINDOWS = sorted(OM_N_WINDOWS, key=lambda x: label_order[x.split('_')[-1]])
                IDX_WINDOWS = OM_N_WINDOWS
            else:
                OM_S_WINDOWS = [f'OM_S_{w}' for w in WINDOW_LABELS]
                OM_S_WINDOWS = sorted(OM_S_WINDOWS, key=lambda x: label_order[x.split('_')[-1]])
                IDX_WINDOWS = OM_S_WINDOWS
                
        reg_y = jjas[jjas['region'] == reg][['year', 'jjas_precip_anom']].copy()
        for col, lab in zip(IDX_WINDOWS, WINDOW_LABELS):
            if col not in index_df.columns:
                continue
            tmp = index_df[['year', col]].copy().rename(columns={col: 'value'})
            merged = reg_y.merge(tmp, on='year', how='inner')
            r, p, n = corr_pair(merged['value'], merged['jjas_precip_anom'], method=method)
            rows.append({
                'region': reg,
                'window': col,
                'window_label': lab,
                'r': r,
                'p': p,
                'n': n
            })
    if index == 'OM':
        return pd.DataFrame(rows), OM_N_WINDOWS, OM_S_WINDOWS
    else:
        return pd.DataFrame(rows), IDX_WINDOWS

def plot_jjas_leadlag_by_region(index_df, monthly_long, method, output_png, oceanic=False, aerosols=False):
    if aerosols == False and oceanic == True:
        table1, windows1 = build_jjas_leadlag_table(index_df, monthly_long, 'ENSO', method=method)
        table2, windows2 = build_jjas_leadlag_table(index_df, monthly_long, 'AMM', method=method)
    elif aerosols == True and oceanic == False:
        table1, windows1 = build_jjas_leadlag_table(index_df, monthly_long, 'DUST', method=method)
        table2, windows2, windows3 = build_jjas_leadlag_table(index_df, monthly_long, 'OM', method=method)
    else:
        table1, windows1 = build_jjas_leadlag_table(index_df, monthly_long, 'IOD', method=method)
        table2, windows2 = build_jjas_leadlag_table(index_df, monthly_long, 'NAO', method=method)
    
    fig, axs = plt.subplots(1, 2, figsize=(15,5), sharey=True)
    colors = plt.get_cmap('tab10').colors
    
    for ax in axs:
        ns = []
        windows_idx = windows1 if ax==axs[0] else windows2
        table_idx = table1 if ax==axs[0] else table2
        if aerosols == False and oceanic == True:
            title_sub = 'ENSO' if ax==axs[0] else 'AMM'
        elif aerosols == True and oceanic == False:
            title_sub = 'DUST' if ax==axs[0] else 'OM'
        else:
            title_sub = 'IOD' if ax==axs[0] else 'NAO'
        
        for i, reg in enumerate(REGIONS):
            sub = table_idx[table_idx['region'] == reg].copy()
            if aerosols == True and ax==axs[1]:
                if reg in ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e']:
                    sub = sub.set_index('window').reindex(windows2).reset_index()
                else:
                    sub = sub.set_index('window').reindex(windows3).reset_index()
            else:
                sub = sub.set_index('window').reindex(windows_idx).reset_index()
            x = np.arange(len(sub))
            y = sub['r'].values.astype(float)
            ax.plot(x, y, marker='o', markersize=5.5, linewidth=2.0, color=colors[i % len(colors)], label=reg)
            if sub['n'].notna().any():
                ns.append(int(sub['n'].dropna().iloc[0]))
        ax.axhline(0, color='k', linestyle=':', zorder=0)
        ax.set_xlim(-0.2, len(windows_idx)-0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xticks(np.arange(len(windows_idx)))
        ax.set_xticklabels(WINDOW_LABELS)
        ax.set_xlabel('Seasonal window')
        ax.set_title(f'{title_sub}')
    axs[0].set_ylabel('R')
    axs[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(1, -0.1), frameon=False)
    fig.subplots_adjust(wspace=0.05)
    fig.suptitle(f'Lagged correlation of seasonal index with JJAS precipitation anomalies ({method})', fontsize=16)
    fig.savefig(output_png, bbox_inches='tight')
    plt.close(fig)
    

#%% -- files --
plt.style.use('ggplot')
pr_anoms_1998 = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/Coding/y.code_output/monthly_rs_anoms_1998-2024.csv'
pr_anoms_2003 = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/Coding/y.code_output/monthly_rs_anoms_2003-2024.csv'
idx_diags = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv'

index_df = clean_numeric(load_index_csv(idx_diags))
index_df2003 = index_df[index_df['year']>=2003].reset_index()
monthly_long = load_monthly_region_anoms_from_csv(pr_anoms_1998)
monthly_long2003 = load_monthly_region_anoms_from_csv(pr_anoms_2003)


#%% -- heatmap seasonal idx correlation with monthly PR anomalies --
corr_aerosol = build_corr_table(index_df2003, monthly_long2003, ae_idxs)
corr_heatmap(
    corr_aerosol,
    out_prefix='x.correlations/2003-2024/heatmap_corr_ae_monthly',
    indexes=ae_idxs,
    region_order=REGIONS,
)

corr_enso_amm = build_corr_table(index_df2003, monthly_long2003, oc_idxs)
corr_heatmap(
    corr_enso_amm,
    out_prefix='x.correlations/2003-2024/heatmap_corr_oc_monthly',
    indexes=oc_idxs,
    region_order=REGIONS,
)

corr_iod_nao = build_corr_table(index_df2003, monthly_long2003, iodnao_idxs)
corr_heatmap(
    corr_iod_nao,
    out_prefix='x.correlations/2003-2024/heatmap_corr_iodnao_monthly',
    indexes=iodnao_idxs,
    region_order=REGIONS,
)


#%% -- lagged index / JJAS precipitation correlation --
index_df, monthly_long = index_df2003, monthly_long2003
tab_p0 = plot_jjas_leadlag_by_region(
    index_df,
    monthly_long,
    method='pearson',
    output_png='x.correlations/2003-2024/jjas_leadlag_oc_pearson.png',
    oceanic=True,
    aerosols=False
)

tab_p1 = plot_jjas_leadlag_by_region(
    index_df,
    monthly_long,
    method='pearson',
    output_png='x.correlations/2003-2024/jjas_leadlag_ae_pearson.png',
    oceanic=False,
    aerosols=True
)

tab_p2 = plot_jjas_leadlag_by_region(
    index_df,
    monthly_long,
    method='pearson',
    output_png='x.correlations/2003-2024/jjas_leadlag_iodnao_pearson.png',
    oceanic=False,
    aerosols=False
)


#%% -- spatial correlation --
## index season end --> precipitation in following month

import glob
data_im_loc = 'C:/Users/ikeva/z.datasets/IMERG_PR_daily_africa/daily_precip_imerg_v06_*.nc'
data_im = sorted(glob.glob(data_im_loc))
target_years = np.arange(2003, 2025, 1)
data_im_targets = [f for f in data_im if any(f'_{year}.nc' in f for year in target_years)]
ds_pr =  xr.open_mfdataset(data_im_targets, combine='nested', concat_dim='time', join='override')

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
monthly = ds_pr.resample(time="1MS").sum(dim="time") # (time, lon, lat)

mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-18,15), lat=slice(4,18))
pr = monthly['pr'].sel(lon=slice(-18,15), lat=slice(4,18))
ls_aligned = ls.reindex_like(pr, method='nearest')
land_mask = (ls_aligned <= 50)
pr_land = pr.where(land_mask, other=np.nan)
pr_land_coarse = pr_land.coarsen(lat=5, lon=5, boundary='trim').mean()

clim = pr_land_coarse.groupby("time.month").mean("time")
anoms = pr_land_coarse.groupby("time.month") - clim
precip_da = anoms

for i in range(0, len(all_idxs), len(seasons)):
    indices = all_idxs[i:i+len(seasons)]
    spatial_corr(index_df, precip_da, indices, method='pearson', output_dir='x.correlations/2003-2024/')


#%% -- pixelwise correlation --
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def compute_pixelwise_index_corr(pr_jjas, dfs, indices, seasons):
    """
    For each pixel, compute Pearson r between JJAS precip anomaly and
    each index-season combination. Take the max |r| across all combinations
    and record which index and season achieved it.
    """
    years   = pr_jjas['year'].values
    lats    = pr_jjas['lat'].values
    lons    = pr_jjas['lon'].values
    pr_vals = pr_jjas.values
    if pr_jjas.dims == ('year', 'lon', 'lat'):
        pr_vals = pr_vals.transpose(0, 2, 1)

    max_r_arr   = np.full((len(lats), len(lons)), np.nan)
    max_idx_arr = np.full((len(lats), len(lons)), -1, dtype=int)
    feat_list   = []

    for index in indices:
        for season in seasons:
            if index not in dfs or season not in dfs[index]:
                continue
            idx_series = dfs[index][season].dropna()
            common_years = sorted(set(years) & set(idx_series.index))
            if len(common_years) < 5:
                continue

            feat_name = f'{index}_{season}'
            fi        = len(feat_list)
            feat_list.append(feat_name)

            yr_idx   = np.isin(years, common_years)
            idx_ts   = idx_series.loc[common_years].values
            pr_sub = pr_vals[yr_idx, :, :]
            if index == 'OM_S':
                pr_sub[yr_idx, 80:, :] = None
            elif index == 'OM_N':
                pr_sub[yr_idx, :80, :] = None

            # Vectorised Pearson r
            pr_anom   = pr_sub  - pr_sub.mean(axis=0)
            idx_anom  = idx_ts  - idx_ts.mean()
            cov       = (pr_anom * idx_anom[:, None, None]).mean(axis=0)
            pr_std    = pr_anom.std(axis=0)
            idx_std   = idx_anom.std()
            with np.errstate(invalid='ignore', divide='ignore'):
                r_map = np.where(pr_std > 0, cov / (pr_std * idx_std), np.nan)

            abs_better  = np.abs(r_map) > np.abs(np.nan_to_num(max_r_arr))
            max_r_arr   = np.where(abs_better, r_map, max_r_arr)
            max_idx_arr = np.where(abs_better, fi, max_idx_arr)

    max_r_da   = xr.DataArray(max_r_arr,   coords={'lat': lats, 'lon': lons},
                               dims=['lat', 'lon'])
    max_idx_da = xr.DataArray(max_idx_arr, coords={'lat': lats, 'lon': lons},
                               dims=['lat', 'lon'])
    return max_r_da, max_idx_da, feat_list


def plot_pixelwise_index_corr(max_r_da, max_idx_da, feat_list, period, out_dir='x.composites'):
    """
    Top    : max signed r per pixel
    Bottom : which index-season achieved max |r| per pixel
    """
    INDEX_COLORS = {
        'ENSO': '#D32F2F',
        'AMM':  '#1565C0',
        'DUST': '#F9A825',
        'OM_N': '#2E7D32',
        'OM_S': '#66BB6A',
        'IOD':  '#00B2C2',
        'NAO':  '#6A1B9A',
    }

    def feat_color(fname):
        for key, col in INDEX_COLORS.items():
            if fname.startswith(key):
                return col
        return '#999999'

    extent = [-18, 15, 4, 18]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                              subplot_kw=dict(projection=ccrs.PlateCarree()))
    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, color='black',
                          linestyle=':', alpha=0.25, linewidth=0.4)
        gl.top_labels = False; gl.right_labels = False

    # Top panel: max r
    vmax = float(np.nanmax(np.abs(max_r_da.values)))
    vmax = min(vmax, 0.7)
    max_r_da.plot.pcolormesh(ax=axes[0], x='lon', y='lat',
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              cbar_kwargs={'label': 'Correlation coefficient',
                                           'orientation': 'vertical',
                                           'shrink': 0.8})
    axes[0].set_title(f'Max r between {period} precipitation and climate indices',
                       fontsize=11)

    # Bottom panel: which index
    color_arr = np.full((*max_idx_da.shape, 4), np.nan)
    for fi, fname in enumerate(feat_list):
        mask = max_idx_da.values == fi
        color_arr[mask] = mcolors.to_rgba(feat_color(fname))
    color_arr[np.isnan(max_r_da.values)] = [1, 1, 1, 0]

    lons2d, lats2d = np.meshgrid(max_r_da['lon'].values,
                                  max_r_da['lat'].values)
    axes[1].imshow(color_arr,
                   extent=[float(max_r_da['lon'].min()), float(max_r_da['lon'].max()),
                           float(max_r_da['lat'].min()), float(max_r_da['lat'].max())],
                   origin='lower',
                   transform=ccrs.PlateCarree(),
                   interpolation='nearest',
                   alpha=0.7
                   )
    c1 = ax.contourf(
        max_r_da['lon'], max_r_da['lat'],
        xr.where(max_r_da >= 0.5, 1, np.nan),
        levels=[0.5, 1.5], colors='none', hatches=['////']
    )
    c2 = ax.contourf(
        max_r_da['lon'], max_r_da['lat'],
        xr.where(max_r_da <= -0.5, 1, np.nan),
        levels=[0.5, 1.5], colors='none', hatches=['////']
    )
    
    c1.set_edgecolor('darkred')
    c1.set_linewidth(0.)
    c2.set_edgecolor('darkblue')
    c2.set_linewidth(0.)
    
    seen = set()
    legend_handles = []
    for fname in feat_list:
        idx_name = fname.rsplit('_', 1)[0]
        if idx_name not in seen:
            seen.add(idx_name)
            legend_handles.append(
                mpatches.Patch(color=INDEX_COLORS.get(idx_name, '#999'),
                               label=idx_name))
    axes[1].legend(handles=legend_handles, loc='lower right',
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.17, 0.58, 1, 1),
                   title='Index with max |r|', title_fontsize=8)
    axes[1].set_title('Index with maximum correlation', fontsize=11)

    fig.tight_layout()
    fname_out = f'{out_dir}/pixelwise_max_corr_indices_{period}.png'
    fig.savefig(fname_out, bbox_inches='tight')
    plt.show()

pr_jjas = pr_period(pr_land, [6,7,8,9])
valid_seasons = ['NDJ','DJF','JFM','FMA','MAM']
max_r_da, max_idx_da, feat_list = compute_pixelwise_index_corr(pr_jjas, dfs, indices, valid_seasons)
plot_pixelwise_index_corr(max_r_da, max_idx_da, feat_list, 'JJAS', out_dir='x.correlations')

