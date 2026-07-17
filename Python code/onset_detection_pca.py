#%% -- imports --
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import xeofs as xe
import datetime
import time


#%% -- data --
file_loc_chirpsv3 = 'C:/Users/ikeva/z.datasets/CHIRPS_pentads/chirps_africa_pentad_*.nc'
file_ob_chirpsv3 = sorted(glob.glob(file_loc_chirpsv3))
ds_chirps = xr.open_mfdataset(file_ob_chirpsv3, combine='nested', concat_dim='time', join='override')
ds_chirps = ds_chirps.sel(lon=slice(-20,30),lat=slice(0,35))
chirps_downscaled = ds_chirps.coarsen(lat=10, lon=10).mean()
chirps_clim = chirps_downscaled.groupby('time.dayofyear').mean(dim='time')
chirps_anom = chirps_downscaled.groupby('time.dayofyear') - chirps_clim

region_names = ['Sahelian-W','Sahelian-E', 'Sudano-Sahelian-W', 'Sudano-Sahelian-E', 'Sudanian-W', 'Sudanian-E', 'Guinean-W', 'Guinean-E']
regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']
region_bnds = {'sah-w': [14, 18, -18, 0],
               'sah-e': [14, 18, 0, 15],
               'sud-sah-w': [12, 14, -18, 0],
               'sud-sah-e': [12, 14, 0, 15],
               'sud-w': [8, 12, -18, 0],
               'sud-e': [8, 12, 0, 15],
               'guin-w': [4, 8, -18, 0],
               'guin-e': [4, 8, 0, 15]}

period = [1981, 2003, 2003, 2025]
y1, y2, y3, y4 = period[0], period[1], period[2], period[3]

# for saving
EOF_regions, EOF_regions2 = {}, {}
PC_regions, PC_regions2 = {}, {}
ds_regions, ds_regions2 = {}, {}
os_regions = {}
os_regions2 = {}
end_regions = {}
end_regions2 = {}


#%% -- EOF analysis --
choice = 'slice2'
if choice == 'slice1':
    yr1 = y1
    yr2 = y2
elif choice == 'slice2':
    yr1 = y3
    yr2 = y4
    
for region, region_name in zip(regions, region_names):
    print(f"{region}")
    minlat, maxlat, minlon, maxlon = region_bnds[region]    # regional
    # minlat, maxlat, minlon, maxlon = 4, 18, -18, 15       # entire domain
    chirps_regional = chirps_downscaled.pr.sel(
        time=slice(f'{str(yr1)}-03-01', f'{str(yr1)}-10-31'),
        lat=slice(minlat, maxlat),
        lon=slice(minlon, maxlon)
        )
    chirps_anom_regional = chirps_anom.pr.sel(
        time=slice(f'{str(yr1)}-03-01', f'{str(yr1)}-10-31'),
        lat=slice(minlat, maxlat),
        lon=slice(minlon, maxlon)
        )
    chirps_sqrt = np.sqrt(chirps_anom_regional.clip(min=0))
    chirps_sqrt = chirps_sqrt.fillna(0)

    # EOF analysis
    model = xe.single.EOF(n_modes=2, use_coslat=True, check_nans=False, compute=False)
    model.fit(chirps_sqrt, dim='time')

    comps = model.components() # EOFs (spatial patterns)
    scores = model.scores(normalized=False) # Principal components (time series)

    PC1 = scores.sel(mode=1) # first principal component as time series
    EOF1 = comps.sel(mode=1) # first EOF spatial pattern

    if choice == 'slice1':
        EOF_regions[region] = EOF1
        PC_regions[region] = PC1
        ds_regions[region] = chirps_regional
    elif choice == 'slice2':
        EOF_regions2[region] = EOF1
        PC_regions2[region] = PC1
        ds_regions2[region] = chirps_regional
    
    fig = plt.figure(figsize=(5,8))
    ax0 = fig.add_subplot(211, projection=ccrs.PlateCarree())
    ax0.coastlines(resolution='110m')
    gl = ax0.gridlines(color='gray', linestyle='-:', draw_labels=True, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    EOF1.plot.contourf(ax=ax0, levels=25, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal'})
    ax1 = fig.add_subplot(212)
    PC1.plot(ax=ax1)
    ax0.set_title(f"EOF1 ({model.explained_variance_ratio().values[0]*100:.1f}%)")
    ax1.set_title("PC1")
    ax1.set_xlabel("")
    plt.show()
    
    fig = plt.figure(figsize=(5,8))
    ax0 = fig.add_subplot(211, projection=ccrs.PlateCarree())
    ax0.coastlines(resolution='110m')
    gl = ax0.gridlines(color='gray', linestyle='-:', draw_labels=True, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    comps.sel(mode=2).plot.contourf(ax=ax0, levels=25, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal'})
    ax1 = fig.add_subplot(212)
    scores.sel(mode=2).plot(ax=ax1)
    ax0.set_title(f"EOF2 ({model.explained_variance_ratio().values[1]*100:.1f}%)")
    ax1.set_title("PC2")
    ax1.set_xlabel("")
    plt.show()

    
#%% -- detect onset & cessation --
start = time.time()
calc_peaks = False
make_graphs = True
plt.style.use('default')

for region, region_name in zip(regions[:2], region_names[:2]):
    if choice == 'slice1':
        EOF1, PC1, chirps_regional = EOF_regions[region], PC_regions[region], ds_regions[region]
    elif choice == 'slice2':
        EOF1, PC1, chirps_regional = EOF_regions2[region], PC_regions2[region], ds_regions2[region]
    onset_date = []
    end_date = []

    for year in range(yr1, yr2+1):
        print(f'Computing dates for {region_name} - Year: {year}')
        startdate = datetime.datetime(year=year, month=3, day=1)
        enddate = datetime.datetime(year=year, month=12, day=1)
        
        if calc_peaks == True:
            ## Threshold-based peak period
            window = 2 # rolling window size
            rolling_precip = chirps_regional.sel(time=slice(startdate, enddate)).mean(('lat', 'lon')).rolling(time=window, center=True).mean()
            threshold = rolling_precip.quantile(0.9).compute()  # Top 10% rolling mean
            peak_period = rolling_precip.where(rolling_precip > threshold).dropna(dim='time')
            peak_start = peak_period['time'].min().values
            peak_end = peak_period['time'].max().values

        ## Select PC1 time series for the rainy season
        PC1_y = PC1.sel(time=slice(startdate, enddate))
        PC1_cumsum = np.cumsum(PC1_y)

        ## Detect onset date from cumulative PC1
        onset_date_y = PC1_cumsum.idxmin(dim='time').values
        end_date_y = PC1_cumsum.idxmax(dim='time').values
        print("onset date:", onset_date_y)
        print("end date:", end_date_y)
        print(time.time()-start)

        if make_graphs == True:
            ## Plot PC1 cumulative sums and precipitation time series
            chirps_yr = chirps_regional.sel(time=PC1_cumsum.time).mean(('lat', 'lon'))
            vmin = 0
            vmax = 200
    
            fig, ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(PC1_cumsum.time, chirps_yr, color='tab:blue')
            ax2 = ax1.twinx()
            ax2.plot(PC1_cumsum.time, PC1_cumsum, color='black', label='PC1')
            ax1.fill_between(PC1_cumsum.time, chirps_yr, color='lightblue')
            ax1.set_ylabel('Precipitation (mm/pentad)', color='tab:blue', fontsize=12)
            ax1.set_ylim(vmin, vmax)
            ax1.axvline(onset_date_y, color='tab:red', linestyle='dashed', linewidth=1.5,
                        label='Onset date: ' + pd.to_datetime(onset_date_y).strftime('%d %b %Y'))
            ax1.axvline(end_date_y, color='tab:orange', linestyle='dashed', linewidth=1.5,
                        label='End date: ' + pd.to_datetime(end_date_y).strftime('%d %b %Y'))
            ax2.set_ylabel('Cumulative PC1 score', fontsize=12)
            ax2.legend(loc='upper right')
            ax1.legend(loc='upper left')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax1.set_title(f'{region_name} - PCA Rainy Season Detection ({year})', fontsize=15)
            plt.savefig(f"{region_name}_PC1_{year}_os")
            plt.close()

        onset_date.append(onset_date_y)
        end_date.append(end_date_y)
    os_regions[region] = onset_date
    end_regions[region] = end_date

df_onsets = pd.DataFrame.from_dict(os_regions)
df_ends = pd.DataFrame.from_dict(end_regions)
df_onsets.to_csv("onsets_pca.csv")
df_ends.to_csv("cess_pca.csv")

