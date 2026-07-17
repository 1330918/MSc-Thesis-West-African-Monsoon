#%% -- scatter plots: diagnostics x index anomalies --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

df1998 = pd.read_csv('C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv')
df2003 = df1998[df1998['year']>=2003].reset_index()

north_regs   = {"sah-w", "sah-e", "sud-sah-w", "sud-sah-e"}
base_indices = ["ENSO", "AMM", "DUST", "OM", "IOD", "NAO"]
DOT_SIZE = 55

def get_index_col(idx, region, season):
    if idx == "OM":
        suffix = "N" if region in north_regs else "S"
        return f"OM_{suffix}_{season}"
    return f"{idx}_{season}"


def scatter_grid(df, region_seasons, y_col_fn, xlims, ylim_fn, shade_fn, color_fn, title_fn, save_fn):
    for region, seasons in region_seasons.items():
        nrows, ncols = len(base_indices), len(seasons)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.0*nrows), sharey=True)
        if ncols == 1:
            axes = np.array(axes).reshape(nrows, 1)
        doy_col = y_col_fn(region)
        ylo, yhi = ylim_fn(region)
        for j, season in enumerate(seasons):
            for i, idx in enumerate(base_indices):
                ax = axes[i, j]
                idx_col = get_index_col(idx, region, season)
                sub = df[["year", idx_col, doy_col]].dropna() if idx in ['DUST', 'OM'] else df[["year", idx_col, doy_col]]
                x = sub[idx_col].values
                y = sub[doy_col].values

                shade_fn(ax, region, ylo, yhi)

                colors = color_fn(y) if color_fn else 'steelblue'
                ax.scatter(x, y, s=DOT_SIZE, color=colors, edgecolors='white', linewidths=0.5, alpha=0.85, zorder=4)

                slope, intercept, r, p_val, _ = stats.linregress(x, y)
                xfit = np.linspace(x.min(), x.max(), 200)
                yfit = slope*xfit + intercept
                n = len(x)
                se = np.sqrt(np.sum((y-(slope*x+intercept))**2)/(n-2))
                sx = np.sqrt(np.sum((x-x.mean())**2))
                t95 = stats.t.ppf(0.975, df=n-2)
                ci = t95*se*np.sqrt(1/n + (xfit-x.mean())**2/sx**2)
                lc = '#c0392b' if p_val < 0.05 else '#555555'
                ax.plot(xfit, yfit, color=lc, lw=1.8, zorder=3)
                ax.fill_between(xfit, yfit-ci, yfit+ci, color=lc, alpha=0.12, zorder=2)
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                ax.text(0.97, 0.04, f"r={r:.2f}{sig}", transform=ax.transAxes, fontsize=12,
                        ha='right', va='bottom', color=lc,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

                ax.set_ylim(ylo, yhi)
                ax.set_xlim(xlims[idx])
                ax.axvline(0, color='grey', lw=0.7, linestyle=':', zorder=1)
                ax.tick_params(labelsize=16)
                ax.grid(False)

                if i == 0:
                    ax.set_title(season, fontsize=18, fontweight='bold', pad=6)
                ax.set_ylabel(idx, fontsize=18) if j == 0 else ax.set_ylabel("")
                ax.set_xlabel("Index (standardised)", fontsize=18) if i == nrows-1 else ax.set_xlabel("")

        fig.suptitle(title_fn(region), fontsize=24, y=1.0)
        plt.tight_layout()
        plt.savefig(save_fn(region), bbox_inches='tight')
        plt.close()

xlims_std = {'ENSO': (-1.6, 2.6), 'AMM': (-1.9, 2.2), 'DUST': (-2.6, 2.6),
             'OM': (-1.9, 2.6), 'IOD': (-0.6, 1), 'NAO': (-1.7, 2)}

# onset
region_seasons_onset = {
    "sah-w": ["NDJ","DJF","JFM","FMA","MAM"], "sah-e": ["NDJ","DJF","JFM","FMA","MAM"],
    "sud-sah-w": ["NDJ","DJF","JFM","FMA"], "sud-sah-e": ["NDJ","DJF","JFM","FMA"],
    "sud-w": ["NDJ","DJF","JFM"], "sud-e": ["NDJ","DJF","JFM"],
    "guin-w": ["NDJ","DJF"], "guin-e": ["NDJ","DJF"]
}
maxmin_os = {"sah-w":[19,-18],"sah-e":[20,-16],"sud-sah-w":[17,-16],"sud-sah-e":[23,-14],
             "sud-w":[12,-16],"sud-e":[20,-16],"guin-w":[12,-10],"guin-e":[19,-12]}
means_os_2003 = {'sah-w':168,'sah-e':169,'sud-sah-w':138,'sud-sah-e':145,'sud-w':101,'sud-e':112,'guin-w':71,'guin-e':81}
p20_os_2003 = {'sah-w':159,'sah-e':164,'sud-sah-w':133,'sud-sah-e':141,'sud-w':96,'sud-e':108,'guin-w':66,'guin-e':75}
p80_os_2003 = {'sah-w':176,'sah-e':174,'sud-sah-w':144,'sud-sah-e':149,'sud-w':108,'sud-e':117,'guin-w':74,'guin-e':86}

def shade_onset(ax, region, ylo, yhi):
    ax.axhspan(ylo, p20_os_2003[region], color="#fffacd", alpha=0.55, zorder=0)
    ax.axhspan(p20_os_2003[region], p80_os_2003[region], color="white", alpha=0.7, zorder=0)
    ax.axhspan(p80_os_2003[region], yhi, color="#e8d5f5", alpha=0.55, zorder=0)
    ax.axhline(means_os_2003[region], color='royalblue', linestyle='--', linewidth=1.4, zorder=2)

scatter_grid(df2003, region_seasons_onset, lambda r: f"{r}-doy_os", xlims_std,
             lambda r: (p20_os_2003[r]+maxmin_os[r][1], p80_os_2003[r]+maxmin_os[r][0]),
             shade_onset, None,
             lambda r: f"Onset DOY vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/onset_vs_idx_scatter_{r}.png')

# first/second month anomaly
region_seasons1 = region_seasons_onset
region_seasons2 = {
    "sah-w": ["NDJ","DJF","JFM","FMA","MAM","AMJ"], "sah-e": ["NDJ","DJF","JFM","FMA","MAM","AMJ"],
    "sud-sah-w": ["NDJ","DJF","JFM","FMA","MAM"], "sud-sah-e": ["NDJ","DJF","JFM","FMA","MAM"],
    "sud-w": ["NDJ","DJF","JFM","FMA"], "sud-e": ["NDJ","DJF","JFM","FMA"],
    "guin-w": ["NDJ","DJF","JFM"], "guin-e": ["NDJ","DJF","JFM"]
}
mnth1 = False
region_seasons_mnth = region_seasons1 if mnth1 else region_seasons2
anom_col = lambda r: f"{r}-1stmonth_anom" if mnth1 else f"{r}-2ndmonth_anom"
title_mnth = 'onset month' if mnth1 else 'second RS month'
save_mnth = '1stmnth' if mnth1 else '2ndmnth'

def shade_zero(ax, region, ylo, yhi):
    ax.axhline(0, color='k', linestyle='--', linewidth=1.4, zorder=2)

scatter_grid(df2003, region_seasons_mnth, anom_col, xlims_std, lambda r: (-2, 2.3),
             shade_zero, None,
             lambda r: f"PR anomaly in {title_mnth} vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/{save_mnth}_vs_idx_scatter_{r}.png')

# cessation
region_seasons_cess = {
    "sah-w": ["MAM","AMJ","MJJ","JJA"], "sah-e": ["MAM","AMJ","MJJ","JJA"],
    "sud-sah-w": ["MAM","AMJ","MJJ","JJA","JAS"], "sud-sah-e": ["MAM","AMJ","MJJ","JJA","JAS"],
    "sud-w": ["MAM","AMJ","MJJ","JJA","JAS"], "sud-e": ["MAM","AMJ","MJJ","JJA","JAS"],
    "guin-w": ["MAM","AMJ","MJJ","JJA","JAS","ASO"], "guin-e": ["MAM","AMJ","MJJ","JJA","JAS","ASO"]
}
means_cess = {'sah-w':281,'sah-e':267,'sud-sah-w':292,'sud-sah-e':276,'sud-w':306,'sud-e':292,'guin-w':322,'guin-e':314}
p20_cess = {'sah-w':276,'sah-e':264,'sud-sah-w':289,'sud-sah-e':270,'sud-w':300,'sud-e':289,'guin-w':317,'guin-e':308}
p80_cess = {'sah-w':286,'sah-e':271,'sud-sah-w':298,'sud-sah-e':281,'sud-w':310,'sud-e':297,'guin-w':326,'guin-e':318}
maxmin_cess = {"sah-w":[12,-10],"sah-e":[9,-13],"sud-sah-w":[10,-14],"sud-sah-e":[12,-15],
               "sud-w":[11,-11],"sud-e":[14,-12],"guin-w":[11,-11],"guin-e":[15,-11]}
xlims_cess = {'ENSO': (-1.6, 2.2), 'AMM': (-2.1, 1.9), 'DUST': (-2.6, 2.8),
              'OM': (-2.5, 3.3), 'IOD': (-1.1, 1.5), 'NAO': (-1.9, 1.7)}

def shade_cess(ax, region, ylo, yhi):
    ax.axhspan(ylo, p20_cess[region], color="#fffacd", alpha=0.55, zorder=0)
    ax.axhspan(p20_cess[region], p80_cess[region], color="white", alpha=0.7, zorder=0)
    ax.axhspan(p80_cess[region], yhi, color="#e8d5f5", alpha=0.55, zorder=0)
    ax.axhline(means_cess[region], color='royalblue', linestyle='--', linewidth=1.4, zorder=2)

scatter_grid(df2003, region_seasons_cess, lambda r: f"{r}-doy_cess", xlims_cess,
             lambda r: (p20_cess[r]+maxmin_cess[r][1], p80_cess[r]+maxmin_cess[r][0]),
             shade_cess, None,
             lambda r: f"Cessation DOY vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/cess_vs_idx_scatter_{r}.png')

# dry spells
region_seasons_ds = {
    "sah-w": ["NDJ","DJF","JFM","FMA","MAM"], "sah-e": ["NDJ","DJF","JFM","FMA","MAM"],
    "sud-sah-w": ["NDJ","DJF","JFM","FMA"], "sud-sah-e": ["NDJ","DJF","JFM","FMA"],
    "sud-w": ["NDJ","DJF","JFM"], "sud-e": ["NDJ","DJF","JFM"],
    "guin-w": ["NDJ","DJF","JFM"], "guin-e": ["NDJ","DJF","JFM"]
}
nlong = True
ds_col = lambda r: f"{r}-dry_spell_nlong" if nlong else f"{r}-dry_spell_anom"

def shade_ds(ax, region, ylo, yhi):
    if nlong:
        ax.axhspan(-1, 2, color='green', alpha=0.15, zorder=0)
        ax.axhspan(2, 4, color='yellow', alpha=0.15, zorder=0)
        ax.axhspan(4, 6, color='orange', alpha=0.15, zorder=0)
    else:
        ax.axhline(0, color='k', linestyle='--', linewidth=1.4, zorder=2)

suptitle_ds, save_ds = ('# long dry spells', 'spellsnlong') if nlong else ('Dry spell anomaly', 'spellsanom')

scatter_grid(df2003, region_seasons_ds, ds_col, xlims_std, lambda r: (-1, 6) if nlong else (-4.5, 4.5),
             shade_ds, None,
             lambda r: f"{suptitle_ds} vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/{save_ds}_vs_idx_scatter_{r}.png')

# false onset
region_seasons_fos = region_seasons_onset

def shade_fos(ax, region, ylo, yhi):
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.7, zorder=1)

def color_fos(y):
    return ['red' if v > 0 else 'green' for v in (y > 0)]

scatter_grid(df2003, region_seasons_fos, lambda r: f"{r}-false_onset", xlims_std, lambda r: (-1, 2),
             shade_fos, color_fos,
             lambda r: f"False onset vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/falseos_vs_idx_scatter_{r}.png')

# LDS duration
region_seasons_lds = {
    "guin-w": ["NDJ","DJF","JFM","FMA","MAM","AMJ"], "guin-e": ["NDJ","DJF","JFM","FMA","MAM","AMJ"]
}
y_lims_lds = {'guin-w': [(30, 70), 40, 63, 52], 'guin-e': [(10, 50), 25, 36, 31]}

def shade_lds(ax, region, ylo, yhi):
    ax.axhspan(y_lims_lds[region][0][0], y_lims_lds[region][1], color="#fffacd", alpha=0.55, zorder=0)
    ax.axhspan(y_lims_lds[region][1], y_lims_lds[region][2], color="white", alpha=0.7, zorder=0)
    ax.axhspan(y_lims_lds[region][2], y_lims_lds[region][0][1], color="#e8d5f5", alpha=0.55, zorder=0)
    ax.axhline(y_lims_lds[region][3], color='royalblue', linestyle='--', linewidth=1.4, zorder=2)

scatter_grid(df2003, region_seasons_lds, lambda r: f"{r}-lds_dur", xlims_std,
             lambda r: y_lims_lds[r][0],
             shade_lds, None,
             lambda r: f"LDS duration vs climate index [{r}]",
             lambda r: f'x.correlations/2003-2024/ldsdur_vs_idx_scatter_{r}.png')
    
    
    
#%% -- percentile of the seasonal correlations --
df = df2003
aer_thresh = pd.read_csv('C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_ae-clim.csv')
q25_row = aer_thresh.loc[aer_thresh["type"] == "q25"].iloc[0]
q75_row = aer_thresh.loc[aer_thresh["type"] == "q75"].iloc[0]
results = []

# (different for each diagnostic)
region_seasons = {
    "sah-w":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sah-e":     ["NDJ", "DJF", "JFM", "FMA", "MAM"],
    "sud-sah-w": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-sah-e": ["NDJ", "DJF", "JFM", "FMA"],
    "sud-w":     ["NDJ", "DJF", "JFM"],
    "sud-e":     ["NDJ", "DJF", "JFM"],
    "guin-w":    ["NDJ", "DJF"],
    "guin-e":    ["NDJ", "DJF"]
}

for region, seasons in region_seasons.items():
    # per diagnostic:
    doy_col = ...
    p30 = ...
    p70 = ...

    for idx in base_indices:
        n_high = 0
        n_low = 0
        late_high = 0
        early_high = 0
        late_low = 0
        early_low = 0

        for season in seasons:
            idx_col = get_index_col(idx, region, season)
            sub = df[[idx_col, doy_col]].dropna()
            if len(sub) < 5:
                continue
            if idx in ["ENSO", "AMM"]:
                high_thr = 0.5
                low_thr  = -0.5
            elif idx == "IOD":
                high_thr = 0.4
                low_thr  = -0.4
            elif idx == "NAO":
                high_thr = 0.001
                low_thr  = -0.001
            elif idx in ["DUST", "OM"]:
                high_thr = q75_row[idx_col]
                low_thr  = q25_row[idx_col]
            
            late = sub[doy_col] >= p70
            early = sub[doy_col] <= p30
            high_idx = sub[idx_col] >= high_thr
            low_idx = sub[idx_col] <= low_thr
            n_high += high_idx.sum()
            n_low += low_idx.sum()
            late_high += (late & high_idx).sum()
            early_high += (early & high_idx).sum()
            late_low += (late & low_idx).sum()
            early_low += (early & low_idx).sum()

        if n_high > 0:
            pct_late_high = 100 * late_high / n_high
            pct_early_high = 100 * early_high / n_high
        else:
            pct_late_high = np.nan
            pct_early_high = np.nan

        if n_low > 0:
            pct_late_low = 100 * late_low / n_low
            pct_early_low = 100 * early_low / n_low
        else:
            pct_late_low = np.nan
            pct_early_low = np.nan

        results.append({
            "region": region,
            "index": idx,
            "n_high": n_high,
            "n_low": n_low,
            "late_high_count": late_high,
            "early_high_count": early_high,
            "late_low_count": late_low,
            "early_low_count": early_low,
            "late_given_high (%)": round(pct_late_high, 1),
            "early_given_high (%)": round(pct_early_high, 1),
            "late_given_low (%)": round(pct_late_low, 1),
            "early_given_low (%)": round(pct_early_low, 1),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["region", "index"])
results_df.to_csv("x.correlations/perc_mapping/ldsdur_index_pooled_probabilities.csv", index=False)
    

#%% -- percentile heatmap --
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import colormaps as cmaps
plt.style.use('ggplot')

FILES = {
    "RS1 anomaly":          "x.correlations/perc_mapping/1stmnth_index_pooled_probabilities.csv",
    "Onset":                "x.correlations/perc_mapping/onset_index_pooled_probabilities.csv",
    "Cessation":            "x.correlations/perc_mapping/cess_index_pooled_probabilities.csv",
    "Dry spell anomaly":    "x.correlations/perc_mapping/spellanom_index_pooled_probabilities.csv",
    "Long dry spells":      "x.correlations/perc_mapping/spellnlong_index_pooled_probabilities.csv",
    "False onset":          "x.correlations/perc_mapping/falseos_index_pooled_probabilities.csv",
    "LDS duration":         "x.correlations/perc_mapping/ldsdur_index_pooled_probabilities.csv",
}
 
COL_MAP1 = {
    "RS1 anomaly":          ("above-normal_given_high (%)",  "below-normal_given_high (%)",
                              "above-normal_given_low (%)",   "below-normal_given_low (%)"),
    "Onset":                ("late_given_high (%)",           "early_given_high (%)",
                              "late_given_low (%)",            "early_given_low (%)"),
    "Cessation":            ("late_given_high (%)",           "early_given_high (%)",
                              "late_given_low (%)",            "early_given_low (%)"),
    "Dry spell anomaly":    ("above-normal_given_high (%)",  "below-normal_given_high (%)",
                              "above-normal_given_low (%)",   "below-normal_given_low (%)"),
    "Long dry spells":      ("elevated-risk_given_high (%)", "low-risk_given_high (%)",
                              "elevated-risk_given_low (%)",  "low-risk_given_low (%)"),
    "False onset":          ("FO_given_high (%)",             "NOFO_given_high (%)",
                              "FO_given_low (%)",              "NOFO_given_low (%)"),
    "LDS duration":         ("above-normal_given_high (%)",  "below-normal_given_high (%)",
                              "above-normal_given_low (%)",   "below-normal_given_low (%)"),
}
COL_MAP2 = {
    "RS1 anomaly":          ("above-normal_high_count",  "below-normal_high_count",
                              "above-normal_low_count",   "below-normal_low_count"),
    "Onset":                ("late_high_count",           "early_high_count",
                              "late_low_count",            "early_low_count"),
    "Cessation":            ("late_high_count",           "early_high_count",
                              "late_low_count",            "early_low_count"),
    "Dry spell anomaly":    ("above-normal_high_count",  "below-normal_high_count",
                              "above-normal_low_count",   "below-normal_low_count"),
    "Long dry spells":      ("elevated-risk_high_count", "low-risk_high_count",
                              "elevated-risk_low_count",  "low-risk_low_count"),
    "False onset":          ("FO_high_count",             "NOFO_high_count",
                              "FO_low_count",              "NOFO_low_count"),
    "LDS duration":         ("above-normal_high_count",  "below-normal_high_count",
                              "above-normal_low_count",   "below-normal_low_count"),
}

CLASS_LABELS = {
    "RS1 anomaly":          ("above-normal",  "below-normal"),
    "Onset":                ("late",          "early"),
    "Cessation":            ("late",          "early"),
    "Dry spell anomaly":    ("above-normal",  "below-normal"),
    "Long dry spells":      ("elevated risk", "low risk"),
    "False onset":          ("false onset",   "no false onset"),
    "LDS duration":         ("above-normal",  "below-normal"),
}
 
REGIONS = ["sah-w", "sah-e", "sud-sah-w", "sud-sah-e", "sud-w", "sud-e", "guin-w", "guin-e"]
INDICES = ["ENSO", "AMM", "DUST", "OM", "IOD", "NAO"]
 
VMIN, VCEN, VMAX = 0, 25, 100
cmap = cmaps.BlueWhiteOrangeRed
norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=VCEN, vmax=VMAX)
 
def cell_color(val):
    if np.isnan(val):
        return (0.88, 0.88, 0.88, 1.0)
    return cmap(norm(np.clip(val, VMIN, VMAX)))
 
def text_color(val):
    if np.isnan(val):
        return (0.6, 0.6, 0.6, 1.0)
    rgba = cell_color(val)
    lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
    return 'white' if lum < 0.45 else 'black'
 
def draw_figure(dname, df):
    c2h, c1h, c2l, c1l = COL_MAP1[dname]
    c2h_, c1h_, c2l_, c1l_ = COL_MAP2[dname]
    cl2_lbl, cl1_lbl   = CLASS_LABELS[dname]
    NR, NI = len(REGIONS), len(INDICES)
 
    CELL_W = 1.20 
    CELL_H = 1.10
    PAD_L  = 1.40   # index labels
    PAD_T  = 0.80   # region labels + n-labels
    PAD_B  = 0.80   # legend
    PAD_R  = 1.10   # colorbar
 
    FIG_W = PAD_L + NR * CELL_W + PAD_R
    FIG_H = PAD_T + NI * CELL_H + PAD_B
 
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(NI, NR,
                  left   = PAD_L / FIG_W,
                  right  = (FIG_W - PAD_R) / FIG_W,
                  top    = (FIG_H - PAD_T) / FIG_H,
                  bottom = PAD_B / FIG_H,
                  hspace = 0.08,
                  wspace = 0.08)
 
    for ii, idx in enumerate(INDICES):
        for ri, reg in enumerate(REGIONS):
            ax = fig.add_subplot(gs[ii, ri])
            def lkp(col):
                row = df[(df['region'] == reg) & (df['index'] == idx)]
                if row.empty or col not in df.columns:
                    return np.nan
                return float(row.iloc[0][col])
 
            vals1 = {
                (0, 0): lkp(c1l),
                (1, 0): lkp(c1h),
                (0, 1): lkp(c2l),
                (1, 1): lkp(c2h),
            }
            vals2 = {
                (0, 0): lkp(c1l_),
                (1, 0): lkp(c1h_),
                (0, 1): lkp(c2l_),
                (1, 1): lkp(c2h_),
            }
 
            for (row_q, col_q), val in vals1.items():
                fc = cell_color(val)
                
                rect = mpatches.FancyBboxPatch(
                    (col_q + 0.04, row_q + 0.04), 0.92, 0.92,
                    boxstyle="square,pad=0",
                    facecolor=fc, edgecolor='none', zorder=2)
                ax.add_patch(rect)
            for (row_q, col_q), val in vals2.items():
                if not np.isnan(val):
                    bg_val = vals1[(row_q, col_q)]
                    tc = text_color(bg_val)
                    ax.text(col_q + 0.5, row_q + 0.5, f"{val:.0f}",
                            ha='center', va='center',
                            fontsize=12, color=tc, fontweight='normal', zorder=3)
 
            ax.axvline(1.0, color='#999', lw=0.7, zorder=4)
            ax.axhline(1.0, color='#999', lw=0.7, zorder=4)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color('#777')
 
            if ii == 0:
                ax.set_title(reg, fontsize=14, pad=10, color='#111')
            if ii == NI - 1:
                ax.text(0.5, -0.22, '1', ha='center', va='top', fontsize=12,
                        color='orange', fontweight='bold', transform=ax.transData)
                ax.text(1.5, -0.22, '2', ha='center', va='top', fontsize=12,
                        color='purple', fontweight='bold', transform=ax.transData)
            if ri == 0:
                ax.text(-0.20, 0.5, idx, transform=ax.transAxes, ha='right', va='center',
                        fontsize=14, fontweight='bold', color='#111')
            if ri == 0:
                ax.text(-0.05, 1.5 / 2, 'H', transform=ax.transAxes,
                        ha='right', va='center', fontsize=12,
                        color='r', fontweight='bold')
                ax.text(-0.05, 0.5 / 2, 'L', transform=ax.transAxes,
                        ha='right', va='center', fontsize=12,
                        color='b', fontweight='bold')
 
    cbar_ax = fig.add_axes([(FIG_W - 0.72) / FIG_W, 0.18, 0.018, 0.60])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 10, 25, 50, 75, 100])
    cb.set_ticklabels(['0%', '10%', '25%', '50%', '75%', '100%'], fontsize=14)
    cb.ax.tick_params(labelsize=14, length=2)
    cb.set_label('% of seasons', fontsize=14, labelpad=6)
 
    leg = (f"Class 2: {cl2_lbl}    "
           f"Class 1: {cl1_lbl}    |    "
           f"H = Positive index phase    L = Negative index phase")
    fig.text(0.5, 0.015, leg, ha='center', va='bottom', fontsize=14,
             bbox=dict(boxstyle='round,pad=0.4', linewidth=0.7, facecolor='white'))
    fig.suptitle(dname, fontsize=16, fontweight='bold', y=0.98, color='#111')
 
    return fig
 
for dname, fname in FILES.items():
    df   = pd.read_csv(fname)
    fig  = draw_figure(dname, df)
    fig.savefig(f'x.correlations/perc_mapping/perc_heatmap_{fname[2:-31]}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    