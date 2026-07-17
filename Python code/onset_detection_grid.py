#%% -- imports --
import numpy as np
import pandas as pd
import xarray as xr
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#%% -- variables & functions for onset detection --

region_bnds = {'sah-w': [14, 18, -18, 0],   # [lat_min, lat_max, lon_min, lon_max]
               'sah-e': [14, 18, 0, 15],
               'sud-sah-w': [12, 14, -18, 0],
               'sud-sah-e': [12, 14, 0, 15],
               'sud-w': [8, 12, -18, 0],
               'sud-e': [8, 12, 0, 15],
               'guin-w': [4, 8, -18, 0],
               'guin-e': [4, 8, 0, 15]}


ONSET_SEARCH_START_MONTH = 3
ONSET_SEARCH_END_MONTH   = 8
DRY_SPELL_MAX_DAYS       = 10
DRY_SPELL_WINDOW_DAYS    = 30
DRY_DAY_THRESHOLD_MM     = 1.0

RAIN_THRESHOLD_MM = {
    "sah-w":      5.0,
    "sah-e":      5.0,
    "sud-sah-w":  10.0,
    "sud-sah-e":  10.0,
    "sud-w":      15.0,
    "sud-e":      15.0,
    "guin-w":     20.0,
    "guin-e":     20.0,
}


def _has_long_dry_spell(precip_window, max_allowed, dry_threshold, window_days=20,
                        spell_length=7, spell_total_max=5.0, false_detection=False):
    """
    Real-onset check:
    True if any consecutive run of days each below dry_threshold
    exceeds max_allowed days in length.
    Used only for the real onset 30-day validation.

    False-onset check:
    True if ANY consecutive spell_length-day window within precip_window has a
    total rainfall below spell_total_max mm.
    """
    if not false_detection:
        consecutive = 0
        for p in precip_window:
            if p < dry_threshold:
                consecutive += 1
                if consecutive > max_allowed:
                    return True
            else:
                consecutive = 0
        return False
    
    elif false_detection:
        n = len(precip_window)
        for start in range(n - spell_length + 1):
            if np.sum(precip_window[start : start + spell_length]) < spell_total_max:
                return True
        return False


def _doy_of(doy_array, month, day):
    days_per_month = [31, 29 if len(doy_array) == 366 else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(days_per_month[:month - 1]) + day
    return doy


def regional_onset(onset_dict, region_bnds):
    years   = next(iter(onset_dict.values())).year.values
    records = {"year": years}

    for region, (lat_min, lat_max, lon_min, lon_max) in region_bnds.items():
        da     = onset_dict[region]
        subset = da.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )
        weights       = np.cos(np.deg2rad(subset.lat))
        weighted_mean = subset.weighted(weights).mean(dim=["lon", "lat"])
        records[region] = weighted_mean.values

    df = pd.DataFrame(records).set_index("year")
    return df


def find_onset_1d(daily_precip, doy_array, rain_threshold, false_onset_params):
    """
    Find the onset day-of-year for a single pixel and a single year.

    Real onset  : first day meeting rainfall criterion with no consecutive dry
                  spell >DRY_SPELL_MAX_DAYS days in the following 30 days

    False onset : first day meeting rainfall criterion where any 7-day window
                  within the following 20 days has total rainfall below spell_total_max mm.
                  Only recorded if a real onset is also found,
                  and discarded if it IS the real onset day.
    """
    n = len(daily_precip)
    FALSE_WINDOW_DAYS = 20
    spell_length, spell_total_max = false_onset_params

    start_idx = np.where(doy_array == _doy_of(doy_array, month=ONSET_SEARCH_START_MONTH, day=1))[0]
    end_idx   = np.where(doy_array == _doy_of(doy_array, month=ONSET_SEARCH_END_MONTH,   day=31))[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        return -1, -1

    start_idx = int(start_idx[0])
    end_idx   = int(end_idx[0])
    false_onset_doy = -1   # first rainfall candidate that fails the dry spell check

    for i in range(start_idx, end_idx + 1):
        one_day = daily_precip[i]
        two_day = daily_precip[i] + (daily_precip[i+1] if i+1 < n else 0.0)

        if one_day >= rain_threshold or two_day >= rain_threshold:
            # Check dry spell condition
            look_start = i+1
            look_end30 = min(look_start + DRY_SPELL_WINDOW_DAYS, n)
            look_end20 = min(look_start + FALSE_WINDOW_DAYS, n)
            window_30  = daily_precip[look_start:look_end30]
            window_20  = daily_precip[look_start:look_end20]

            if not _has_long_dry_spell(window_30, DRY_SPELL_MAX_DAYS, DRY_DAY_THRESHOLD_MM, false_detection=False):
                # Real onset: passes both criteria
                real_doy = int(doy_array[i])
                if false_onset_doy != -1 and (real_doy - false_onset_doy) > 35:
                    false_onset_doy = -1   # too far before real onset, discard
                return real_doy, false_onset_doy
            else:
                # Failed check — now test false onset criterion                
                if false_onset_doy == -1 and _has_long_dry_spell(
                        window_20, DRY_SPELL_MAX_DAYS, DRY_DAY_THRESHOLD_MM, FALSE_WINDOW_DAYS, spell_length, spell_total_max, false_detection=True):
                    false_onset_doy = int(doy_array[i])
    return -1, -1


def compute_onset_grid(da, lat_min, rain_threshold, false_onset_params):
    years = np.unique(da.time.dt.year.values)
    real_results  = []
    false_results = []
    
    for year in years:
        print(f"  Processing year {year} …")
        da_yr = da.sel(time=da.time.dt.year == year)
        doy   = da_yr.time.dt.dayofyear.values
        arr   = da_yr.values

        n_days, n_lon, n_lat = arr.shape
        real_grid  = np.full((n_lon, n_lat), np.nan)
        false_grid = np.full((n_lon, n_lat), np.nan)

        for i in range(n_lon):
            for j in range(n_lat):
                pixel = arr[:,i,j]
                if np.all(np.isnan(pixel)):
                    continue
                pixel_clean = np.where(np.isnan(pixel), 0.0, pixel)
                real_doy, false_doy = find_onset_1d(pixel_clean, doy, rain_threshold, false_onset_params)
                real_grid[i,j]  = real_doy  if real_doy  != -1 else np.nan
                false_grid[i,j] = false_doy if false_doy != -1 else np.nan

        real_results.append(real_grid)
        false_results.append(false_grid)

    coords = {"year": years, "lon": da.lon.values, "lat": da.lat.values}
    dims   = ["year", "lon", "lat"]
    attrs_base = {"units": "day of year (1 = Jan 1)"}
    
    real_da = xr.DataArray(
        np.stack(real_results, axis=0), dims=dims, coords=coords,
        name="onset_doy",
        attrs={**attrs_base, "long_name": "Julian day of real rainy season onset",
               "dry_spell":  f"No consecutive dry spell >{DRY_SPELL_MAX_DAYS} days "
                             f"in next {DRY_SPELL_WINDOW_DAYS} days"},
    )
    false_da = xr.DataArray(
        np.stack(false_results, axis=0), dims=dims, coords=coords,
        name="false_onset_doy",
        attrs={**attrs_base,
               "long_name": "Julian day of false onset",
               "dry_spell":  f"Any {false_onset_params[0]}-day window with total "
                             f"rainfall <{false_onset_params[1]} mm in 20 days "},
    )
    
    return real_da, false_da


def compute_all_regions(da, false_onset_params):
    """Detect onset for all regions"""
    zones = defaultdict(list)
    for region, bnds in region_bnds.items():
        zones[bnds[0]].append(region)

    real_dict  = {}
    false_dict = {}
    years = np.unique(da.time.dt.year.values)
    false_df_binary = pd.DataFrame(index=years)

    for lat_min, regions_in_zone in zones.items():
        lat_max   = region_bnds[regions_in_zone[0]][1]
        lon_min   = min(region_bnds[r][2] for r in regions_in_zone)
        lon_max   = max(region_bnds[r][3] for r in regions_in_zone)
        threshold = RAIN_THRESHOLD_MM[regions_in_zone[0]]
        params    = false_onset_params[regions_in_zone[0]]

        print(f"\nZone {lat_min}–{lat_max}°N | lon {lon_min}–{lon_max}°E "
              f"| regions: {regions_in_zone} | false_onset_params: {params}")

        da_zone = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        zone_real, zone_false = compute_onset_grid(da_zone, lat_min, threshold, params)
        
        for region in regions_in_zone:
            r_lat_min, r_lat_max, r_lon_min, r_lon_max = region_bnds[region]
            sel = dict(lon=slice(r_lon_min, r_lon_max), lat=slice(r_lat_min, r_lat_max))
            real_dict[region]  = zone_real.sel(**sel)
            false_dict[region] = zone_false.sel(**sel)

            # binary flag for false onsets if >20% of the grid has a false onset
            flags = []
            for year in years:
                real_grid  = real_dict[region].sel(year=year).values
                false_grid = false_dict[region].sel(year=year).values
                has_onset       = ~np.isnan(real_grid)
                has_false_onset = ~np.isnan(false_grid)

                n_onset = has_onset.sum()
                frac = has_false_onset.sum() / n_onset
                print(region, year, frac)
                flags.append(1 if frac >= 0.2 else 0)

            false_df_binary[region] = flags
            
    return real_dict, false_dict, false_df_binary

#%% -- datasets --

data_sorted = sorted(glob.glob('C:/Users/ikeva/z.datasets/IMERG_PR_daily_africa/daily_precip_imerg_v06_*.nc'))
ds = xr.open_mfdataset(data_sorted, combine='nested', concat_dim='time', join='override')

def convert_time(time_array):
    converted_time = []
    for t in time_array:
        if isinstance(t, np.datetime64) or isinstance(t, pd.Timestamp):
            converted_time.append(pd.Timestamp(t))
        else:
            converted_time.append(t)
    return np.array(converted_time, dtype="datetime64[ns]")

normalized_time = convert_time(ds['time'].values)
ds = ds.assign_coords(time=normalized_time)

all_lat_min = min(b[0] for b in region_bnds.values())
all_lat_max = max(b[1] for b in region_bnds.values())
all_lon_min = min(b[2] for b in region_bnds.values())
all_lon_max = max(b[3] for b in region_bnds.values())

ds = ds.sel(
    lat=slice(all_lat_min, all_lat_max),
    lon=slice(all_lon_min, all_lon_max),
)

ds = ds.sel(time=slice("2003-01-01", "2024-12-31"))
mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-18,15), lat=slice(4,18))
pr = ds['pr']
ls_aligned = ls.reindex_like(pr, method='nearest')
land_mask = (ls_aligned <= 50)
pr_land = pr.where(land_mask, other=np.nan)
pr_land_coarse = pr_land.coarsen(lat=5, lon=5, boundary='trim').mean()


#%% -- detect onset --
da = pr_land_coarse

# Zone-specific false onset parameters: [spell_length, spell_total_max_mm]
FALSE_ONSET_PARAMS = {
    'sah-w':     [7, 3.0],
    'sah-e':     [7, 3.0],
    'sud-sah-w': [7, 4.0],
    'sud-sah-e': [7, 4.0],
    'sud-w':     [7, 5.0],
    'sud-e':     [7, 5.0],
    'guin-w':    [7, 5.0],
    'guin-e':    [7, 5.0],
}

real_dict, false_dict, false_df_binary = compute_all_regions(da, FALSE_ONSET_PARAMS)
df_onset = regional_onset(real_dict, region_bnds)
df_false = regional_onset(false_dict, region_bnds)
df_onset.to_csv("onsets_agro.csv")

da10 = pr_land.coarsen(lat=10, lon=10, boundary='trim').mean()
real_dict10, false_dict10, false_df_binary10 = compute_all_regions(da10, FALSE_ONSET_PARAMS)
false_df_binary10.to_csv("onsets_false_binary_coarse10.csv")


#%% -- plotting --
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_onset_map(region, year, onset_dict, region_bnds, title,
                   vmin, vmax, cmap="RdYlGn_r", multiple=False):
    """
    Plot per-pixel onset DOY as a colour map for a given region and year.
    """
    if not multiple:
        da = onset_dict[region].sel(year=year)
    
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='110m')
        ax.add_feature(cfeature.BORDERS, edgecolor='gray')
        gl = ax.gridlines(color='black', linestyle=':', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        
        da.plot(
            ax=ax,
            x="lon", y="lat",
            # vmin=vmin, vmax=vmax,
            cmap=cmap,
            cbar_kwargs={"label": "Onset DOY", "orientation": "horizontal"},
        )
        ax.set_title(f"{region}  |  {year} — {title} onset day of year")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.show()
    
    elif multiple:
        years = np.arange(2003, 2025, 1)
        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(25,15), subplot_kw=dict(projection=ccrs.PlateCarree()))
        axs = axs.flatten()
        fig.delaxes(axs[-1])
        fig.delaxes(axs[-2])
        fig.delaxes(axs[-3])
        for ax in axs:
            ax.coastlines(resolution='110m')
            gl = ax.gridlines(color='black', linestyle=':', draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
                    
        for y, yr in enumerate(years):
            da = onset_dict[region].sel(year=yr)
            da.plot(
                ax=axs[y],
                x="lon", y="lat",
                # vmin=vmin, vmax=vmax,
                cmap=cmap,
                cbar_kwargs={"label": "Onset DOY", "orientation": "horizontal"},
            )
            axs[y].set_title(f"{region}  |  {yr} — {title} onset")

        plt.tight_layout()
        plt.show()
        
# plot_onset_map("sah-w", 2003, real_dict, region_bnds, "real", 60, 240, multiple=True)
# plot_onset_map("sah-w", 2003, false_dict10, region_bnds, "false", 60, 240, multiple=True)

def plot_pixel_precip_selection(region, year, pixels, onset_dict,
                                false_dict, region_bnds, da):
    """
    Plot precipitation time series for a selection of pixels.
    """
    lat_min, lat_max, lon_min, lon_max = region_bnds[region]
    onset_grid = onset_dict[region].sel(year=year)   # (lon, lat)
    false_grid = false_dict[region].sel(year=year)

    da_region = da.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    da_yr = da_region.sel(time=da_region.time.dt.year == year)
    time_dates = pd.to_datetime(da_yr.time.values)
    ref = pd.Timestamp(f"{year}-01-01")

    fig, axes = plt.subplots(len(pixels), 1,
                             figsize=(12, 4 * len(pixels)),
                             sharex=True)
    if len(pixels) == 1:
        axes = [axes]

    for ax, (plon, plat) in zip(axes, pixels):
        precip = da_yr.sel(lon=plon, lat=plat, method="nearest").values
        doy    = float(onset_grid.sel(lon=plon, lat=plat, method="nearest").values)
        doy_f  = float(false_grid.sel(lon=plon, lat=plat, method="nearest").values)

        actual_lon = float(da_region.lon.sel(lon=plon, method="nearest"))
        actual_lat = float(da_region.lat.sel(lat=plat, method="nearest"))

        ax.bar(time_dates, precip, width=1, color="steelblue", alpha=0.7)
        ax.axhline(5.0, color="red", lw=0.8, linestyle=":", label="5 mm threshold")
        ax.axhline(10.0, color="blue", lw=0.8, linestyle=":", label="10 mm threshold")
        ax.axhline(20.0, color="green", lw=0.8, linestyle=":", label="20 mm threshold")

        if not np.isnan(doy):
            onset_date = ref + pd.Timedelta(days=int(doy) - 1)
            ax.axvline(onset_date, color="black", lw=1.2, linestyle="--",
                       label=f"Onset DOY {int(doy)} ({onset_date.strftime('%d-%b')})")
        if not np.isnan(doy_f):
            false_date = ref + pd.Timedelta(days=int(doy_f) - 1)
            ax.axvline(false_date, color="gray", lw=1.2, linestyle="--",
                       label=f"False onset DOY {int(doy_f)} ({false_date.strftime('%d-%b')})")

        ax.set_title(f"lon={actual_lon:.2f}°E, lat={actual_lat:.2f}°N")
        ax.set_ylabel("Precipitation [mm/day]")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
        for label in ax.get_xticklabels(which='minor'):
            label.set(fontsize=6)
        ax.get_xaxis().set_tick_params(which='major', pad=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0,20)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{region}  |  {year}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
# plot_pixel_precip_selection(
#     region     = "guin-w",
#     year       = 2012,
#     pixels     = [(-5, 7.5), (-5, 5.5)],
#     onset_dict = real_dict,
#     false_dict = false_dict,
#     region_bnds= region_bnds,
#     da         = pr,
# )


#%% -- cessation detection -- 

CESS_SEARCH_START_MONTH = 9
CESS_SEARCH_END_MONTH   = 11

def find_cessation_1d_agronomic(daily_precip, doy_array, rain_threshold):
    """
    Agronomic cessation date, adopting the method for onset and
    extend it symmetrically to cessation.

    Definition:
        "The agronomic cessation date (offset) of the rainy season is
        determined as the last day (+1) of a wet sequence of 4 consecutive
        days receiving at least 20 mm, without being preceded by a dry spell
        of more than 10 consecutive days in the 20 days preceding it."
    """
    WET_WINDOW_DAYS  = 4    # consecutive days for wet sequence
    PRE_CHECK_DAYS   = 20   # days preceding the wet sequence to check
    DRY_SPELL_MAX    = 10   # max allowed consecutive dry days in pre-window

    n = len(daily_precip)

    start_idx = np.where(doy_array == _doy_of(doy_array,
                         month=CESS_SEARCH_START_MONTH, day=1))[0]
    end_idx   = np.where(doy_array == _doy_of(doy_array,
                         month=CESS_SEARCH_END_MONTH,   day=30))[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        return -1

    start_idx = int(start_idx[0])
    end_idx   = int(end_idx[0])

    # Scan backward: find the last qualifying wet sequence
    for i in range(end_idx - WET_WINDOW_DAYS + 1, start_idx - 1, -1):
        # Check: 4-day window sum >= threshold
        wet_window = daily_precip[i : i + WET_WINDOW_DAYS]
        if np.sum(wet_window) < rain_threshold:
            continue

        # Check: no dry spell >10 consecutive days in the prior 20 days
        pre_start = max(0, i - PRE_CHECK_DAYS)
        pre_window = daily_precip[pre_start : i]

        if _has_long_dry_spell(pre_window,
                               max_allowed=DRY_SPELL_MAX,
                               dry_threshold=DRY_DAY_THRESHOLD_MM):
            continue   # preceded by long dry spell — not a valid wet sequence

        # Valid: return D+1 after the last day of the wet sequence
        last_wet_day_idx = i + WET_WINDOW_DAYS - 1
        return int(doy_array[min(last_wet_day_idx + 1, n - 1)])
    return -1


def compute_cessation_grid(da, lat_min, rain_threshold):
    """
    Compute per-pixel cessation DOY for every year
    using agronomic (zone-specific mm) threshold.
    """
    years        = np.unique(da.time.dt.year.values)
    agro_results  = []

    for year in years:
        print(f"  Cessation year {year} …")
        da_yr = da.sel(time=da.time.dt.year == year)
        doy   = da_yr.time.dt.dayofyear.values
        arr   = da_yr.values

        n_days, n_lon, n_lat = arr.shape
        agro_grid  = np.full((n_lon, n_lat), np.nan)

        for i in range(n_lon):
            for j in range(n_lat):
                pixel = arr[:, i, j]
                if np.all(np.isnan(pixel)):
                    continue
                pixel_clean = np.where(np.isnan(pixel), 0.0, pixel)
                agro_doy  = find_cessation_1d_agronomic(
                    pixel_clean, doy, rain_threshold)
                agro_grid[i, j]  = agro_doy  if agro_doy  != -1 else np.nan

        agro_results.append(agro_grid)

    coords     = {"year": years, "lon": da.lon.values, "lat": da.lat.values}
    dims       = ["year", "lon", "lat"]
    attrs_base = {"units": "day of year (1 = Jan 1)"}
    agro_da = xr.DataArray(
        np.stack(agro_results, axis=0), dims=dims, coords=coords,
        name="cessation_doy_agronomic",
        attrs={**attrs_base,
               "long_name": "Agronomic cessation: last day+1 of 4-day wet "
                            f"sequence >={rain_threshold} mm not preceded by "
                            f">10 consecutive dry days in 20 days"},
    )
    return agro_da


def compute_all_regions_cessation(da):
    """Run compute_cessation_grid() for each zone"""
    zones = defaultdict(list)
    for region, bnds in region_bnds.items():
        zones[bnds[0]].append(region)
    agro_dict  = {}

    for lat_min, regions_in_zone in zones.items():
        lat_max        = region_bnds[regions_in_zone[0]][1]
        lon_min        = min(region_bnds[r][2] for r in regions_in_zone)
        lon_max        = max(region_bnds[r][3] for r in regions_in_zone)
        rain_threshold = RAIN_THRESHOLD_MM[regions_in_zone[0]]

        print(f"\nCessation zone {lat_min}–{lat_max}°N | "
              f"lon {lon_min}–{lon_max}°E | regions: {regions_in_zone}")

        da_zone = da.sel(lat=slice(lat_min, lat_max),
                         lon=slice(lon_min, lon_max))
        zone_agro = compute_cessation_grid(da_zone, lat_min, rain_threshold)

        for region in regions_in_zone:
            r_lat_min, r_lat_max, r_lon_min, r_lon_max = region_bnds[region]
            sel = dict(lon=slice(r_lon_min, r_lon_max),
                       lat=slice(r_lat_min, r_lat_max))
            agro_dict[region]  = zone_agro.sel(**sel)

    return agro_dict

agro_dict = compute_all_regions_cessation(pr_land_coarse)
df_cess_agro  = regional_onset(agro_dict,  region_bnds)
df_cess_agro.to_csv("cess_agro.csv")


#%% -- lds detection --

def find_lds_agronomic(daily_precip, doy_array, rain_threshold):
    """
    Detect the Little Dry Season using the agronomic method of
    onset & cessation detection for the bimodal Guinean rainfall regime.

    LDS start: cessation of the first wet season — last day (+1) of a
               4-day wet sequence >= rain_threshold not preceded by a
               dry spell >10 consecutive days in the 20 days before it.
               Scanned backward from August 15 to June 1.

    LDS end:   onset of the second wet season — first day of a 1-or-2-day
               event >= rain_threshold not followed by a dry spell >10
               consecutive days in the next 30 days.
               Scanned forward from August 1 to October 31.
    """
    WET_WINDOW_DAYS   = 4    # consecutive days for wet sequence (Faye 2024)
    PRE_CHECK_DAYS    = 20   # preceding window for dry spell check
    DRY_SPELL_MAX     = 10   # max consecutive dry days allowed

    LDS_START_SEARCH_START_MONTH, LDS_START_SEARCH_START_DAY = 6, 1
    LDS_START_SEARCH_END_MONTH,   LDS_START_SEARCH_END_DAY   = 8, 15
    LDS_END_SEARCH_START_MONTH, LDS_END_SEARCH_START_DAY = 8, 1
    LDS_END_SEARCH_END_MONTH,   LDS_END_SEARCH_END_DAY   = 9, 30
    n = len(daily_precip)

    def get_idx(month, day):
        target = _doy_of(doy_array, month=month, day=day)
        idx    = np.where(doy_array == target)[0]
        return int(idx[0]) if len(idx) > 0 else -1

    s1 = get_idx(LDS_START_SEARCH_START_MONTH, LDS_START_SEARCH_START_DAY)
    e1 = get_idx(LDS_START_SEARCH_END_MONTH,   LDS_START_SEARCH_END_DAY)
    s2 = get_idx(LDS_END_SEARCH_START_MONTH,   LDS_END_SEARCH_START_DAY)
    e2 = get_idx(LDS_END_SEARCH_END_MONTH,     LDS_END_SEARCH_END_DAY)

    if any(x == -1 for x in [s1, e1, s2, e2]):
        return -1, -1, -1

    # LDS start: backward scan for last valid wet sequence
    lds_start_doy = -1
    for i in range(e1 - WET_WINDOW_DAYS + 1, s1 - 1, -1):
        wet_window = daily_precip[i : i + WET_WINDOW_DAYS]
        if np.sum(wet_window) < rain_threshold:
            continue
        pre_start  = max(0, i - PRE_CHECK_DAYS)
        pre_window = daily_precip[pre_start : i]
        if _has_long_dry_spell(pre_window, DRY_SPELL_MAX,
                                    DRY_DAY_THRESHOLD_MM):
            continue
        # D+1 after last day of wet sequence
        last_wet_idx  = i + WET_WINDOW_DAYS - 1
        lds_start_doy = int(doy_array[min(last_wet_idx + 1, n - 1)])
        break

    if lds_start_doy == -1:
        return -1, -1, -1

    # LDS end: forward scan for first valid onset of second wet season
    lds_end_doy = -1
    for i in range(s2, e2 + 1):
        one_day = daily_precip[i]
        two_day = daily_precip[i] + (daily_precip[i + 1] if i + 1 < n else 0.0)
        if one_day >= rain_threshold or two_day >= rain_threshold:
            look_start = i + 1
            window_30  = daily_precip[look_start : min(look_start + DRY_SPELL_WINDOW_DAYS, n)]
            if not _has_long_dry_spell(window_30, DRY_SPELL_MAX,
                                            DRY_DAY_THRESHOLD_MM):
                lds_end_doy = int(doy_array[i])
                break

    if lds_end_doy == -1:
        return -1, -1, -1

    duration = lds_end_doy - lds_start_doy
    if duration <= 0:
        return -1, -1, -1

    return lds_start_doy, lds_end_doy, duration


def compute_lds_grid(da, rain_threshold):
    years         = np.unique(da.time.dt.year.values)
    start_results = []
    end_results   = []
    dur_results   = []

    for year in years:
        print(f"  LDS year {year} …")
        da_yr = da.sel(time=da.time.dt.year == year)
        doy   = da_yr.time.dt.dayofyear.values
        arr   = da_yr.values

        n_days, n_lon, n_lat = arr.shape
        start_grid = np.full((n_lon, n_lat), np.nan)
        end_grid   = np.full((n_lon, n_lat), np.nan)
        dur_grid   = np.full((n_lon, n_lat), np.nan)

        for i in range(n_lon):
            for j in range(n_lat):
                pixel = arr[:, i, j]
                if np.all(np.isnan(pixel)):
                    continue
                pixel_clean = np.where(np.isnan(pixel), 0.0, pixel)
                s, e, d = find_lds_agronomic(pixel_clean, doy, rain_threshold)
                if s != -1:
                    start_grid[i, j] = s
                    end_grid[i, j]   = e
                    dur_grid[i, j]   = d

        start_results.append(start_grid)
        end_results.append(end_grid)
        dur_results.append(dur_grid)

    coords     = {"year": years, "lon": da.lon.values, "lat": da.lat.values}
    dims       = ["year", "lon", "lat"]
    attrs_base = {"units":  "day of year (1 = Jan 1)"}
    start_da = xr.DataArray(np.stack(start_results, axis=0),
                             dims=dims, coords=coords,
                             name="lds_start_doy", attrs=attrs_base)
    end_da   = xr.DataArray(np.stack(end_results,   axis=0),
                             dims=dims, coords=coords,
                             name="lds_end_doy",   attrs=attrs_base)
    dur_da   = xr.DataArray(np.stack(dur_results,   axis=0),
                             dims=dims, coords=coords,
                             name="lds_duration",
                             attrs={**attrs_base, "units": "days"})
    return start_da, end_da, dur_da


def compute_lds_regions(da):
    records = []

    for region in ["guin-w", "guin-e"]:
        lat_min, lat_max, lon_min, lon_max = region_bnds[region]
        rain_threshold = RAIN_THRESHOLD_MM[region]
        da_region = da.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )

        start_da, end_da, dur_da = compute_lds_grid(da_region, rain_threshold)
        weights = np.cos(np.deg2rad(da_region.lat))
        da_mean = da_region.weighted(weights).mean(dim=["lon", "lat"])
        years   = np.unique(da_region.time.dt.year.values)

        for year in years:
            s_grid = start_da.sel(year=year).values
            e_grid = end_da.sel(year=year).values
            d_grid = dur_da.sel(year=year).values

            lds_start = int(np.nanmedian(s_grid)) if not np.all(np.isnan(s_grid)) else np.nan
            lds_end   = int(np.nanmedian(e_grid)) if not np.all(np.isnan(e_grid)) else np.nan
            lds_dur   = float(np.nanmean(d_grid)) if not np.all(np.isnan(d_grid)) else np.nan

            # Count dry days in the regional mean within the LDS window
            da_yr  = da_mean.sel(time=da_mean.time.dt.year == year)
            doy_yr = da_yr.time.dt.dayofyear.values
            pr_yr  = da_yr.values

            if not np.isnan(lds_start) and not np.isnan(lds_end):
                mask   = (doy_yr >= lds_start) & (doy_yr <= lds_end)
                n_dry  = int(np.sum(pr_yr[mask] < 1.0))
            else:
                n_dry  = np.nan

            records.append({
                "region":        region,
                "year":          year,
                "lds_start_doy": lds_start,
                "lds_end_doy":   lds_end,
                "lds_duration":  lds_dur,
                "n_dry_days":    n_dry,
            })

    return (pd.DataFrame(records)
              .set_index(["year"])
              .sort_index())

lds = compute_lds_regions(pr_land)


#%% -- correlation climate indices & (false) onset, cessation, LDS duration --
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
plt.style.use('ggplot')

filepath_anoms = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024_v2.csv'
df_all = pd.read_csv(filepath_anoms, index_col="year")

north_regs = {"sah-w", "sah-e", "sud-sah-w", "sud-sah-e"}

def valid_indices_for_region(region, all_index_cols, region_seasons):
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

def index_association(df_target, df_indices, region_seasons, out_csv):
    df_target = df_target.copy()
    df_indices = df_indices.copy()

    df_target.index = pd.Index(df_target.index.astype(int), name="year")
    df_target = df_target.apply(pd.to_numeric, errors="coerce")
    df_indices = df_indices.apply(pd.to_numeric, errors="coerce")

    region_names = list(df_target.columns)
    index_names = list(df_indices.columns)

    r_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_names, columns=region_names, dtype=float)

    for reg in region_names:
        valid_cols = set(valid_indices_for_region(reg, index_names, region_seasons))
        for idx in index_names:
            if idx not in valid_cols:
                continue
            tmp = pd.concat(
                [df_indices[idx].rename("x"), df_target[reg].rename("y")],
                axis=1
            ).dropna()

            if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                continue

            r, p = stats.pearsonr(tmp["x"], tmp["y"])
            r_matrix.loc[idx, reg] = r
            p_matrix.loc[idx, reg] = p

    r_matrix.to_csv(out_csv)
    return r_matrix, p_matrix

def plot_correlation_heatmap(r_matrix, p_matrix, title, out_png, save=True):
    fig = plt.figure(figsize=(20, 5))
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

    plt.title(title, fontsize=24, y=1.02)
    plt.grid(False)
    fig.tight_layout()
    if save:
        plt.savefig(out_png)
        plt.close()
    else:
        plt.show()

# -- onset --
climate_indices_onset = ["ENSO_NDJ", "ENSO_DJF", "ENSO_JFM", "ENSO_FMA", "ENSO_MAM",
                          "AMM_NDJ", "AMM_DJF", "AMM_JFM", "AMM_FMA", "AMM_MAM",
                          "DUST_NDJ", "DUST_DJF", "DUST_JFM", "DUST_FMA", "DUST_MAM",
                          "OM_N_NDJ", "OM_N_DJF", "OM_N_JFM", "OM_N_FMA", "OM_N_MAM",
                          "OM_S_NDJ", "OM_S_DJF", "OM_S_JFM", "OM_S_FMA", "OM_S_MAM",
                          "IOD_NDJ", "IOD_DJF", "IOD_JFM", "IOD_FMA", "IOD_MAM",
                          "NAO_NDJ", "NAO_DJF", "NAO_JFM", "NAO_FMA", "NAO_MAM"]
region_seasons_onset = {
    "sah-w":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sah-e":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-sah-w": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-sah-e": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-w":     ["NDJ", "DJF", "JFM"],
    "sud-e":     ["NDJ", "DJF", "JFM"],
    "guin-w":    ["NDJ", "DJF"],
    "guin-e":    ["NDJ", "DJF"]
}
df_indices_onset = df_all.loc[2003:2024][climate_indices_onset]
r_matrix_onset, p_matrix_onset = index_association(
    df_onset.loc[2003:2024].fillna(0), df_indices_onset, region_seasons_onset,
    out_csv="x.correlations/2003-2024/corr_matrix_onset.csv"
)
plot_correlation_heatmap(
    r_matrix_onset, p_matrix_onset,
    title='Correlation onset with climate indices',
    out_png="x.correlations/2003-2024/corr_heatmap_onset_vs_idx.png"
)

# -- false onset --
r_matrix_falseos, p_matrix_falseos = index_association(
    false_df_binary10[false_df_binary10 == 1].loc[2003:2024].fillna(0),
    df_indices_onset, region_seasons_onset,
    out_csv="x.correlations/2003-2024/corr_matrix_falseos_coarsegrid.csv"
)
plot_correlation_heatmap(
    r_matrix_falseos, p_matrix_falseos,
    title='Correlation false onset with climate indices',
    out_png="x.correlations/2003-2024/corr_heatmap_falseos_vs_idx_coarsegrid.png"
)

# cessation
climate_indices_cess = ["ENSO_MAM", "ENSO_AMJ", "ENSO_MJJ", "ENSO_JJA", "ENSO_JAS", "ENSO_ASO",
                         "AMM_MAM", "AMM_AMJ", "AMM_MJJ", "AMM_JJA", "AMM_JAS", "AMM_ASO",
                         "DUST_MAM", "DUST_AMJ", "DUST_MJJ", "DUST_JJA", "DUST_JAS", "DUST_ASO",
                         "OM_N_MAM", "OM_N_AMJ", "OM_N_MJJ", "OM_N_JJA", "OM_N_JAS", "OM_N_ASO",
                         "OM_S_MAM", "OM_S_AMJ", "OM_S_MJJ", "OM_S_JJA", "OM_S_JAS", "OM_S_ASO",
                         "IOD_MAM", "IOD_AMJ", "IOD_MJJ", "IOD_JJA", "IOD_JAS", "IOD_ASO",
                         "NAO_MAM", "NAO_AMJ", "NAO_MJJ", "NAO_JJA", "NAO_JAS", "NAO_ASO"]
region_seasons_cess = {
    "sah-w":     ["MAM", "AMJ", "MJJ", "JJA"],
    "sah-e":     ["MAM", "AMJ", "MJJ", "JJA"],
    "sud-sah-w": ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
    "sud-sah-e": ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
    "sud-w":     ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
    "sud-e":     ["MAM", "AMJ", "MJJ", "JJA", "JAS"],
    "guin-w":    ["MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO"],
    "guin-e":    ["MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO"]
}
df_indices_cess = df_all.loc[2003:2024][climate_indices_cess]
r_matrix_cess, p_matrix_cess = index_association(
    df_cess_agro.loc[2003:2024], df_indices_cess, region_seasons_cess,
    out_csv="x.correlations/2003-2024/corr_matrix_cess.csv"
)
plot_correlation_heatmap(
    r_matrix_cess, p_matrix_cess,
    title='Correlation cessation with climate indices',
    out_png="x.correlations/2003-2024/corr_heatmap_cess_vs_idx.png",
)

# LDS duration
climate_indices_lds = ["ENSO_NDJ", "ENSO_DJF", "ENSO_JFM", "ENSO_FMA", "ENSO_MAM", "ENSO_AMJ",
                        "AMM_NDJ", "AMM_DJF", "AMM_JFM", "AMM_FMA", "AMM_MAM", "AMM_AMJ",
                        "DUST_NDJ", "DUST_DJF", "DUST_JFM", "DUST_FMA", "DUST_MAM", "DUST_AMJ",
                        "OM_S_NDJ", "OM_S_DJF", "OM_S_JFM", "OM_S_FMA", "OM_S_MAM", "OM_S_AMJ",
                        "IOD_NDJ", "IOD_DJF", "IOD_JFM", "IOD_FMA", "IOD_MAM", "IOD_AMJ",
                        "NAO_NDJ", "NAO_DJF", "NAO_JFM", "NAO_FMA", "NAO_MAM", "NAO_AMJ"]
region_seasons_lds = {
    "guin-w": ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"],
    "guin-e": ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ"]
}
df_indices_lds = df_all.loc[2003:2024][climate_indices_lds]
df_lds = (
    lds.loc[lds.index >= 2003, ['region', 'lds_duration']]
       .reset_index()
       .pivot(index='year', columns='region', values='lds_duration')
)
r_matrix_lds, p_matrix_lds = index_association(
    df_lds, df_indices_lds,
    region_seasons_lds,
    out_csv="x.correlations/2003-2024/corr_matrix_ldsdur.csv"
)
plot_correlation_heatmap(
    r_matrix_lds, p_matrix_lds,
    title='Correlation LDS duration with climate indices',
    out_png="x.correlations/2003-2024/corr_heatmap_ldsdur_vs_idx.png"
)
