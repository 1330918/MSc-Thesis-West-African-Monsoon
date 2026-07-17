#%% -- imports --
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colormaps as cmaps
import glob


#%% -- plot study area --

regions_map = {'Sahelian-W':        [[14, 18, -18, 0], -6,   17],
               'Sahelian-E':        [[14, 18, 0, 15],  9.5,  17],
               'Sudano-Sahelian-W': [[12, 14, -18, 0], -9.9, 13],
               'Sudano-Sahelian-E': [[12, 14, 0, 15],  5.5,  13],
               'Sudanian-W':        [[8, 12, -18, 0],  -6.2, 11],
               'Sudanian-E':        [[8, 12, 0, 15],   9,    11],
               'Guinean-W':         [[4, 8, -18, 0],   -6,   7],
               'Guinean-E':         [[4, 8, 0, 15],    9.5,  7]}

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree(), extent=[340,20,0,30])
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.4, alpha=0.4)
gl = ax.gridlines(draw_labels=True, alpha=0.1, color='k')
gl.top_labels = False
gl.right_labels = False
ax.stock_img()
ax.hlines(y=4, xmin=-18, xmax=15, color='darkred', linewidth=1, linestyle='--')
ax.hlines(y=8, xmin=-18, xmax=15, color='darkred', linewidth=1, linestyle='--')
ax.hlines(y=12, xmin=-18, xmax=15, color='darkred', linewidth=1, linestyle='--')
ax.hlines(y=14, xmin=-18, xmax=15, color='darkred', linewidth=1, linestyle='--')
ax.hlines(y=18, xmin=-18, xmax=15, color='darkred', linewidth=1, linestyle='--')
ax.vlines(x=-18, ymin=4, ymax=18, color='darkred', linewidth=1, linestyle='--')
ax.vlines(x=15, ymin=4, ymax=18, color='darkred', linewidth=1, linestyle='--')
ax.vlines(x=0, ymin=4, ymax=18, color='darkred', linewidth=1, linestyle='--')

for region, (bbox, xpos, ypos) in regions_map.items():
    ax.text(
        x=xpos,
        y=ypos,
        s=region,
        va='center',
        color='darkred',
        fontsize=10,
        transform=ccrs.PlateCarree(),
    )
plt.show()


#%% -- plot onset / cessation date --

plt.style.use('ggplot')
mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-20,40), lat=slice(0,35))
land_mask = (ls <= 70)

regions = { 'sah-w':     [[14, 18, -18, 0], '18-6-2026', '8-10-2026'],
            'sah-e':     [[14, 18, 0, 15],  '19-6-2026', '24-9-2026'],
            'sud-sah-w': [[12, 14, -18, 0], '18-5-2026', '19-10-2026'],
            'sud-sah-e': [[12, 14, 0, 15],  '27-5-2026', '3-10-2026'],
            'sud-w':     [[8, 12, -18, 0],  '12-4-2026', '2-11-2026'],
            'sud-e':     [[8, 12, 0, 15],   '23-4-2026', '18-10-2026'],
            'guin-w':    [[4, 8, -18, 0],   '13-3-2026', '18-11-2026'],
            'guin-e':    [[4, 8, 0, 15],    '23-3-2026', '10-11-2026']}

region_os, region_cess = {}, {}
region_os_doy, region_cess_doy = {}, {}
for region, (bbox, os_str, cess_str) in regions.items():
    os_date = pd.to_datetime(os_str, dayfirst=True)
    region_os[region] = os_date
    region_os_doy[region] = os_date.dayofyear
    
    cess_date = pd.to_datetime(cess_str, dayfirst=True)
    region_cess[region] = cess_date
    region_cess_doy[region] = cess_date.dayofyear
    
 
def plot_dates(regions, region_dates, region_doy, title, land_mask):
    doys = np.array(list(region_doy.values()))
    norm = mcolors.Normalize(vmin=min(doys), vmax=max(doys))
    cmap = plt.cm.YlGnBu
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    lats = land_mask['lat'].values
    lons = land_mask['lon'].values
    land = land_mask.values
    rgba = np.zeros((len(lats), len(lons), 4), dtype=float)
    
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.2)
    ax.set_extent([-20, 18, 2, 20], crs=ccrs.PlateCarree())
    gl = ax.gridlines(color='gray', linestyle='-:', draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    for region, (bbox, os_str, cess_str) in regions.items():
        lat_min, lat_max, lon_min, lon_max = bbox
        doy = region_doy[region]
        
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        box_mask = np.outer(lat_mask, lon_mask)
        fill_mask = box_mask & land
        rgba[fill_mask] = cmap(norm(doy))
        
        ax.imshow(
            rgba,
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            origin='lower',
            transform=ccrs.PlateCarree(),
            interpolation='nearest',
            aspect='auto',
            )
        
        ax.plot(
            [lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color='black', linewidth=0.8,
            transform=ccrs.PlateCarree()
        )
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    tick_doys = np.linspace(min(doys), max(doys), 6)
    tick_dates = [pd.to_datetime(doy, format='%j').strftime('%d %b') for doy in tick_doys]
    cbar.set_ticks(tick_doys)
    cbar.set_ticklabels(tick_dates)
    plt.title(f"Mean {title} date")
    return plt.show()

plot_dates(regions, region_os, region_os_doy, 'onset', land_mask)
plot_dates(regions, region_cess, region_cess_doy, 'cessation', land_mask)


#%% -- plot annual average precipitation --

plt.style.use('ggplot')
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

mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-20,30), lat=slice(2,20))
monthly = ds_pr.resample(time="1MS").sum(dim="time")
pr_monthly_domain = monthly['pr'].sel(lon=slice(-20,30), lat=slice(2,20))
ls_aligned = ls.reindex_like(pr_monthly_domain, method='nearest')
land_mask = (ls_aligned <= 50)
pr_monthly_land = pr_monthly_domain.where(land_mask, other=np.nan)

months = list(range(1,13))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
               'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sel = monthly.pr.where(monthly.time.dt.month.isin(months), drop=True)
monthly_sel = monthly_sel.groupby('time.year').sum('time')
yearly_mean = monthly_sel.mean('year')
yearly_mean = yearly_mean.compute()

regions_map = {'Sahelian-W':        [[14, 18, -18, 0], -6,   17],
               'Sahelian-E':        [[14, 18, 0, 15],  9.5,  17],
               'Sudano-Sahelian-W': [[12, 14, -18, 0], -9.5, 13],
               'Sudano-Sahelian-E': [[12, 14, 0, 15],  5.8,  13],
               'Sudanian-W':        [[8, 12, -18, 0],  -6,   11],
               'Sudanian-E':        [[8, 12, 0, 15],   9.5,  11],
               'Guinean-W':         [[4, 8, -18, 0],   -6,   7],
               'Guinean-E':         [[4, 8, 0, 15],    9.5,  7]}

plt.figure(figsize=(8,5))
ax = plt.axes(projection=ccrs.PlateCarree(), extent=[340,20,0,22])
ax.coastlines(resolution='110m', linewidth=1.2)
gl = ax.gridlines(draw_labels=True, alpha=0.1, color='k')
gl.top_labels = False
gl.right_labels = False
level = np.array([250, 400, 800, 1200, 2200])

yearly_mean.plot(x='lon', vmax=4000, cmap=cmaps.GMT_drywet,
                 cbar_kwargs={'label':'Average annual precipitation [mm]', 'orientation': 'horizontal', 'shrink': 0.7, 'pad': 0.08})
contours = yearly_mean.plot.contour(x='lon', colors='white', levels=level, linewidths=0.7, transform=ccrs.PlateCarree())
contours.clabel(inline=True, inline_spacing=2, fontsize=9, fmt='%d', colors='white')

for region, (bbox, xpos, ypos) in regions_map.items():
    lat_min, lat_max, lon_min, lon_max = bbox
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color='darkred', linewidth=1, linestyle='--',
        transform=ccrs.PlateCarree()
    )
    
    ax.text(
        x=xpos,
        y=ypos,
        s=region,
        va='center',
        color='darkred',
        fontsize=10,
        transform=ccrs.PlateCarree(),
    )
plt.tight_layout()
plt.show()

