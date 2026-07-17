#%% -- imports --
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
import shap
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
mpl.rcParams['font.size'] = 16
plt.style.use('ggplot')


#%% -- data preparation --

filepath_anoms = 'C:/Users/ikeva/OneDrive - HydroLogic/Documents/Thesis/idxvsRS_anoms_1998-2024.csv'
region_names = ['Sahelian-W', 'Sahelian-E', 'Sudano-Sahelian-W', 'Sudano-Sahelian-E',
                'Sudanian-W', 'Sudanian-E', 'Guinean-W', 'Guinean-E']
regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']
northern_regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e']
northern1_regions = ['sah-w', 'sah-e']
northern2_regions = ['sud-sah-w', 'sud-sah-e']
southern1_regions = ['sud-w', 'sud-e']
southern2_regions = ['guin-w', 'guin-e']

# Region position matrix
REGION_POSITION = {
    'sah-w':     [1, 1],
    'sah-e':     [1, 2],
    'sud-sah-w': [2, 1],
    'sud-sah-e': [2, 2],
    'sud-w':     [3, 1],
    'sud-e':     [3, 2],
    'guin-w':    [4, 1],
    'guin-e':    [4, 2],
}

indices = ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
seasons = ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM',
           'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO']

diagnostics = ['doy_os', 'doy_cess', '1stmonth_anom',
               'dry_spell_anom', 'dry_spell_nlong',
               'lds_dur', 'false_onset']
diag_labels = {
    'doy_os':               'Onset date',
    'doy_cess':             'Cessation date',
    '1stmonth_anom':        'RS1 anomaly',
    'dry_spell_anom':       'Dry spell anomaly',
    'dry_spell_nlong':      'Long dry spells',
    'false_onset':          'False onset',
    'lds_dur':              'LDS duration',
}
diag_class = {
    'binary_diags': ['false_onset', 'dry_spell_nlong'],
    'doy_diags': ['doy_os', 'doy_cess'],
    'anom_diags': ['1stmonth_anom', 'dry_spell_anom', 'lds_dur']
}

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
neg_spellanom, pos_spellanom, med_nlongspell = {}, {}, {}
means_ldsdur, p30_ldsdur, p70_ldsdur = {}, {}, {}

for region in regions:
    means_os[region] = round(dfs['doy_os'][region].mean())
    p30_os[region]   = round(np.percentile(dfs['doy_os'][region], 30))
    p70_os[region]   = round(np.percentile(dfs['doy_os'][region], 70))

    means_cess[region] = round(dfs['doy_cess'][region].mean())
    p30_cess[region]   = round(np.percentile(dfs['doy_cess'][region], 30))
    p70_cess[region]   = round(np.percentile(dfs['doy_cess'][region], 70))

    p30_1stmnth[region] = np.percentile(dfs['1stmonth_anom'][region], 30)
    p70_1stmnth[region] = np.percentile(dfs['1stmonth_anom'][region], 70)
    
    neg_spellanom[region]  = -1
    pos_spellanom[region]  = 1
    med_nlongspell[region] = np.floor(np.median(dfs['dry_spell_nlong'][region]))
    
    if region in ['guin-w', 'guin-e']:
        means_ldsdur[region] = round(dfs['lds_dur'][region].mean())
        p30_ldsdur[region]   = round(np.percentile(dfs['lds_dur'][region], 30))
        p70_ldsdur[region]   = round(np.percentile(dfs['lds_dur'][region], 70))


#%% -- helper functions --

def allowed_indices_for_diag(diag, season):
    if diag in ['doy_os', '1stmonth_anom', 'dry_spell_anom', 'dry_spell_nlong', 'false_onset']:
        if season in ['NDJ', 'DJF', 'JFM']:
            return ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
        elif season in ['FMA', 'MAM']:
            return ['ENSO', 'AMM', 'DUST', 'OM_N', 'IOD', 'NAO']
        else:
            return []
    elif diag == 'lds_dur':
        return ['ENSO', 'AMM', 'DUST', 'OM_S', 'IOD', 'NAO']
    elif diag == 'doy_cess':
        if season in ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS']:
            return ['ENSO', 'AMM', 'DUST', 'OM_N', 'OM_S', 'IOD', 'NAO']
        elif season == 'ASO':
            return ['ENSO', 'AMM', 'DUST', 'OM_S', 'IOD', 'NAO']
        else:
            return []


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
        return {'northern1': ['MAM', 'AMJ', 'MJJ', 'JJA'],
                'northern2': ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS'],
                'southern1': ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS'],
                'southern2': ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO'],
                }
    elif diagnostic == 'lds_dur':
        return {'southern2': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ']}
    
    
def build_feature_columns(indices, seasons, diag, df_columns): 
    cols = []
    valid_map = get_valid_seasons(diag)
    valid_seasons = set(sum(valid_map.values(), [])) if valid_map is not None else set()
    
    for idx in indices:
        for s in seasons:
            if s not in valid_seasons:
                continue
    
            allowed = set(allowed_indices_for_diag(diag, s))
            if idx not in allowed:
                continue
    
            col = f'{idx}_{s}'
            if col in df_columns:
                cols.append(col)
    return cols
    

def classify_value(value, region, diagnostic, clim_src, doy_clim, diag_class):
    if pd.isna(value):
        return np.nan
    if diagnostic in diag_class['binary_diags']:
        if diagnostic == 'dry_spell_nlong':
            phigh = doy_clim['phigh']
            if value > phigh[region]:
                return 1 # elevated risk
            return 0     # low risk
        else:
            return int(value)  # 0 --> no, 1 --> yes
    if diagnostic in diag_class['doy_diags'][:2]:
        plow = doy_clim['plow']
        phigh = doy_clim['phigh']
        if value <= plow[region]:
            return 1   # early
        elif value >= phigh[region]:
            return 2   # late
        return 0       # climatology
    if diagnostic in diag_class['anom_diags']:
        plow = doy_clim['plow']
        phigh = doy_clim['phigh']
        if value <= plow[region]:
            return 1   # below-normal
        elif value >= phigh[region]:
            return 2   # above-normal
        return 0       # near-normal


def get_class_labels(diagnostic, diag_class):
    if diagnostic in diag_class['doy_diags']:
        return {0: 'clim', 1: 'early', 2: 'late'}
    elif diagnostic in diag_class['binary_diags']:
        if diagnostic == 'dry_spell_nlong':
            return {0: 'low risk', 1: 'elevated risk'}
        else:
            return {0: 'no', 1: 'yes'}
    else:
        return {0: 'near-normal', 1: 'below-normal', 2: 'above-normal'}


def load_data(filepath_anoms):
    raw = pd.read_csv(filepath_anoms)
    raw.columns = raw.columns.str.strip()
    df = raw.reset_index(drop=True).copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    for col in df.columns:
        if col != 'year':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_short  = df[df['year'] >= 2003]
    diag_clim = {}
    for region in regions:
        diag_clim[region] = {}
        for diag in diagnostics:
            col = f'{region}-{diag}'
            if col not in df_short.columns:
                continue
            vals = df_short[col].dropna()
            if len(vals) < 5:
                continue
            diag_clim[region][diag] = {
                'mean': vals.mean(),
                'std':  vals.std() if vals.std() > 0 else 1.0,
                'p20':  vals.quantile(0.20),
                'p80':  vals.quantile(0.80),
                'q25':  vals.quantile(0.25),
                'q75':  vals.quantile(0.75),
            }
    return df, {'diag_clim': diag_clim}


#%% -- model configuration --

def fit_conditional_regressors(X, y, class_labels, min_samples=3):
    regressors = {}
    for label in np.unique(class_labels):
        mask = class_labels == label
        if mask.sum() >= min_samples:
            rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, min_samples_split=4, max_features='sqrt', random_state=13)
            rf.fit(X[mask], y[mask])
            regressors[label] = rf
    return regressors


def predict_value_uncertainty(regressors, pred_class, X_row, y_train_median):
    if pred_class in regressors:
        reg = regressors[pred_class]
        leaf_preds = np.array([t.predict(X_row)[0] for t in reg.estimators_])
        return np.mean(leaf_preds), np.percentile(leaf_preds, 25), np.percentile(leaf_preds, 75)
    return y_train_median, y_train_median, y_train_median


def hp_tuning():
    param_grid = {
        'n_estimators':      [100, 200],
        'min_samples_leaf':  [2, 3, 5],
        'max_features':      [0.5, 0.7, 'sqrt'],
        'max_depth':         [3, 5, 8],
        'sampling_strategy': ['not minority', 'all']
    }

    hp_cols = {
        'min_samples_leaf':  'Min samples per leaf',
        'min_samples_split': 'Min samples per split',
        'max_features':      'Max features',
        'max_depth':         'Max depth',
        'n_estimators':      'No. estimators',
        'class_weight':      'Class weight',
        'sampling_strategy': 'Sampling'
    }
    
    base_clf = BalancedRandomForestClassifier(random_state=13)
    sc = 'balanced_accuracy'
    inner_cv = TimeSeriesSplit(n_splits=2, test_size=2)
    gs = GridSearchCV(base_clf, param_grid, cv=inner_cv, scoring=sc, n_jobs=-1)
    return gs, sc, hp_cols

configs = [
    {'label': 'onset', 'means': means_os, 'plow': p30_os, 'phigh': p70_os},
    {'label': 'cess', 'means': means_cess, 'plow': p30_cess, 'phigh': p70_cess},
    {'label': 'RS1_anom', 'plow': p30_1stmnth, 'phigh': p70_1stmnth},
    {'label': 'spell_anoms', 'plow': neg_spellanom, 'phigh': pos_spellanom},
    {'label': 'spell_nlong', 'phigh': med_nlongspell},
    {'label': 'lds_dur', 'plow': p30_ldsdur, 'phigh': p70_ldsdur},
    {'label': 'false_os'},
]


def build_unified_dataset(feat_cols, diag, df, clim_src, doy_clim, min_year=2003):
    """
    Pool all 8 regions into a single dataset.
    For each region:
      - climate-index features (feat_cols)              shape: (n_years, n_feat)
      - position features: lat_band, lon_band           shape: (n_years, 2)
      - target: region-specific diagnostic value        shape: (n_years,)
      - target class: classified using regional p       shape: (n_years,)
    """
    X_parts, y_parts, yc_parts, yr_parts, reg_parts = [], [], [], [], []
    valid_seasons = get_valid_seasons(diag)

    for region in regions:
        target_col = f'{region}-{diag}'
        if target_col not in df.columns:
            continue
        
        loc = 'northern1' if region in northern1_regions else 'northern2' if region in northern2_regions \
            else 'southern1' if region in southern1_regions else 'southern2' if region in southern2_regions else None
        allowed_seasons = valid_seasons[loc]
        region_valid_cols = [c for c in feat_cols if any(c.endswith('_' + s) for s in allowed_seasons)]

        df_use = df[df['year'] >= min_year].copy()
        df_use = df_use[['year'] + feat_cols + [target_col]].dropna()
        df_use = df_use.sort_values('year').reset_index(drop=True)

        if len(df_use) < 4:
            continue

        X_idx = df_use[feat_cols].values 
        for fi, col in enumerate(feat_cols):
            if col not in region_valid_cols:
                X_idx[:,fi] = 0.0

        lat_band = REGION_POSITION[region][0]
        lon_band = REGION_POSITION[region][1]
        pos      = np.full((len(df_use), 2), [lat_band, lon_band])
        X_region = np.hstack([X_idx, pos])

        y_region = df_use[target_col].values
        yc_region = np.array([
            classify_value(v, region, diag, clim_src, doy_clim, diag_class)
            for v in y_region
        ])

        X_parts.append(X_region)
        y_parts.append(y_region)
        yc_parts.append(yc_region)
        yr_parts.append(df_use['year'].values)
        reg_parts.extend([region] * len(df_use))

    if not X_parts:
        return None

    X_all       = np.vstack(X_parts)
    y_all       = np.concatenate(y_parts)
    y_class_all = np.concatenate(yc_parts)
    years_all   = np.concatenate(yr_parts)
    regions_all = reg_parts

    clim_means = {}
    for region in regions:
        clim_means[region] = clim_src.get(region, {}).get(diag, {}).get('mean', np.nan)

    return X_all, y_all, y_class_all, years_all, regions_all, clim_means


def get_unified_feature_names(feat_cols):
    return feat_cols + ['lat_band', 'lon_band']


#%% -- run LOOCV and validate --

from sklearn.metrics import precision_recall_fscore_support
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    
all_clf_rows  = []   # classification performance rows
all_acc_rows  = []   # value accuracy rows
all_df_loo    = {}   # store full df_loo per model label for downstream use

diagnostics_used = ['doy_os', 'doy_cess', '1stmonth_anom',
                    'dry_spell_anom', 'dry_spell_nlong', 'lds_dur', 'false_onset']
gs_records = []

for c, config in enumerate(configs):
    label = config['label']
    print(f'Model {c+1}/{len(configs)}: {label}')
    df, clim  = load_data(filepath_anoms)
    clim_src  = clim['diag_clim']
    diag      = diagnostics_used[c]
    label_map = get_class_labels(diag, diag_class)
    
    if diag in diag_class['doy_diags'][:2] or diag in diag_class['anom_diags']:
        doy_clim = {'plow': config['plow'], 'phigh': config['phigh']}
    elif diag == 'dry_spell_nlong':
        doy_clim = {'phigh': config['phigh']}
    else:
        doy_clim = None
    
    feat_cols = build_feature_columns(indices, seasons, diag, df.columns)
    X_all, y_all, y_class_all, years_all, regions_all, clim_means = build_unified_dataset(feat_cols, diag, df, clim_src, doy_clim)
    feature_names = get_unified_feature_names(feat_cols)        
    n = len(X_all)
    
    print(np.sum(y_class_all==1), "class1")
    print(np.sum(y_class_all==0), "class0")
    print(np.sum(y_class_all==2), "class2")
    
    # Tune hyperparameters once on the full pooled dataset
    gs_global, scoring_method, hp_cols = hp_tuning()
    gs_global.fit(X_all, y_class_all)
    global_best_params = gs_global.best_params_
    print(f'\nGlobal best params: {global_best_params}')
    
    res = gs_global.cv_results_
    n_combos = len(res['mean_test_score'])
    for i in range(n_combos):
        row = {'diagnostic': diag, 'mean_test_score': res['mean_test_score'][i],
               'std_test_score':  res['std_test_score'][i]}
        params = res['params'][i]
        row.update(params)
        gs_records.append(row)
        
    # Main modelling
    print(f'Running LOO-CV: {diag} | n={n}')
    loo_records = []    
    loo = LeaveOneOut()
    
    # Precompute combined mask (valid seasons + correlation)
    feat_col_idx = {c: i for i, c in enumerate(feat_cols)}
    combined_col_mask = {}
    for region in np.unique(regions_all):
        loc = 'northern1' if region in northern1_regions else 'northern2' if region in northern2_regions \
            else 'southern1' if region in southern1_regions else 'southern2' if region in southern2_regions else None
        allowed_seasons = get_valid_seasons(diag)[loc]
        valid_cols = np.array([
            any(c.endswith('_' + s) for s in allowed_seasons) for c in feat_cols])
        combined_col_mask[region] = valid_cols
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(X_all)):       
        i         = test_idx[0]
        X_train   = X_all[train_idx].copy()
        y_train   = y_all[train_idx]
        yc_train  = y_class_all[train_idx]
        X_test    = X_all[[i]].copy()
        yc_test   = y_class_all[i]
        y_test    = y_all[i]
        year_test = int(years_all[i])
        region    = regions_all[i]
        
        best_params = global_best_params

        col_mask_fold = combined_col_mask[region]
        n_feat        = len(feat_cols)
        X_train[:, :n_feat] = X_train[:, :n_feat] * col_mask_fold
        X_test[:, :n_feat]  = X_test[:, :n_feat]  * col_mask_fold
    
        # Fit classifier on n-1 samples
        clf_fold = BalancedRandomForestClassifier(**best_params, min_samples_split=4, class_weight='balanced', random_state=13)
        # if fold == 0:
        #     print(clf_fold)
        clf_fold.fit(X_train, yc_train)
    
        # Predict the left-out sample
        pred_class  = int(clf_fold.predict(X_test)[0])
        pred_probas = clf_fold.predict_proba(X_test)[0]
        correct    = label_map.get(pred_class, '') == label_map.get(int(yc_test), '')

        p_arr = np.full(3, np.nan)
        for ci, cls in enumerate(clf_fold.classes_):
            if cls in (0, 1, 2):
                p_arr[int(cls)] = pred_probas[ci]
                
        # Fit conditional regressors for value prediction
        if diag in diag_class['doy_diags'] or diag == 'lds_dur':
            region_mask = np.array(regions_all)[train_idx] == region
            X_train_reg = X_train[region_mask]
            y_train_reg  = y_train[region_mask]
            yc_train_reg = yc_train[region_mask]
            
            regs_fold = fit_conditional_regressors(X_train_reg, y_train_reg, yc_train_reg)
            pred_val, q25, q75 = predict_value_uncertainty(
                regs_fold, pred_class, X_test, np.median(y_train_reg))
            clim_mean  = clim_means.get(region, np.nan)
        else:
            pred_val, q25, q75 = None, None, None
    
        loo_records.append({
            'model':            label,
            'fold':             fold,
            'region':           region,
            'diagnostic':       diag,
            'year':             year_test,
            'true_value':       float(y_test),
            'pred_value':       pred_val,
            'pred_q25':         q25,
            'pred_q75':         q75,
            'days_offset':      pred_val - clim_mean if pred_val else np.nan,
            'true_class':       label_map.get(int(yc_test), ''),
            'pred_class':       label_map.get(pred_class, ''),
            'p_class0':         round(float(p_arr[0]) * 100, 1) if not np.isnan(p_arr[0]) else np.nan,
            'p_class1':         round(float(p_arr[1]) * 100, 1) if not np.isnan(p_arr[1]) else np.nan,
            'p_class2':         round(float(p_arr[2]) * 100, 1) if not np.isnan(p_arr[2]) else np.nan,
            'class0_label':     label_map.get(0, ''),
            'class1_label':     label_map.get(1, ''),
            'class2_label':     label_map.get(2, ''),
            'correct':          correct,
            'n_train_fold':     len(train_idx),
            'feature_cols':     '|'.join(feature_names),
            'calibrated':       False,
            'n_estimators':     best_params.get('n_estimators'),
            'min_samples_leaf': best_params.get('min_samples_leaf'),
            'max_features':     str(best_params.get('max_features')),
            'max_depth':        str(best_params.get('max_depth')),
            'class_weight':     str(best_params.get('class_weight')),
            'X_test_values':    X_test[0],
            'year_test':        year_test,
        })
    
        # print(f'  Fold {fold+1:>3}/{n} | year={year_test} | region={region:<12} | '
        #       f'true={label_map.get(int(yc_test), "?"):>12} | '
        #       f'pred={label_map.get(pred_class, "?"):>12} | '
        #       f'{"✓" if correct else "✗"}')


# Aggregate LOO metrics
    df_loo = pd.DataFrame(loo_records)
    df_loo['clim_mean']   = df_loo['region'].map(clim_means)
    df_loo['days_offset'] = df_loo['pred_value'] - df_loo['clim_mean']
    df_loo['abs_error']   = (df_loo['pred_value'] - df_loo['true_value']).abs()
    df_loo['true_deviation'] = df_loo['true_value'] - df_loo['clim_mean']
    df_loo['p_pred'] = df_loo[['p_class0', 'p_class1', 'p_class2']].max(axis=1)
 
    all_df_loo[label] = df_loo
 
    # Classification performance
    le = LabelEncoder().fit(list(label_map.values()))
    all_true = df_loo['true_class'].values
    all_pred = df_loo['pred_class'].values
 
    for region_filter in [None] + regions:
        sub = df_loo if region_filter is None else df_loo[df_loo['region'] == region_filter]
        if sub.empty:
            continue
        reg_label = 'all' if region_filter is None else regions[regions.index(region_filter)]
        t = sub['true_class'].values
        p = sub['pred_class'].values
        labels = ['early', 'clim', 'late'] if diag in diag_class['doy_diags'] \
            else ['below-normal', 'near-normal', 'above-normal'] if diag in diag_class['anom_diags'] \
                else ['low risk', 'elevated risk'] if diag=='dry_spell_nlong' else ['yes', 'no']
 
        acc     = (t == p).mean()
        bal_acc = balanced_accuracy_score(le.transform(t), le.transform(p))
        prec, rec, f1, _ = precision_recall_fscore_support(
            t, p, labels=labels,
            average=None, zero_division=0)
 
        for ci, cls in enumerate(labels):
            all_clf_rows.append({
                'model':        label,
                'region':       reg_label,
                'class':        cls,
                'n':            int((t == cls).sum()),
                'accuracy':     round(acc, 3),
                'bal_accuracy': round(bal_acc, 3),
                'precision':    round(prec[ci], 3),
                'recall':       round(rec[ci], 3),
                'f1':           round(f1[ci], 3),
            })
        all_clf_rows.append({
            'model':        label,
            'region':       reg_label,
            'class':        'overall',
            'n':            len(sub),
            'accuracy':     round(acc, 3),
            'bal_accuracy': round(bal_acc, 3),
            'precision':    round(prec.mean(), 3),
            'recall':       round(rec.mean(), 3),
            'f1':           round(f1.mean(), 3),
        })
 
    #  Value accuracy
    if diag in diag_class['doy_diags'] or diag == 'lds_dur':
        for region_filter in [None] + regions:
            sub_reg = df_loo if region_filter is None else df_loo[df_loo['region'] == region_filter]
            if sub_reg.empty:
                continue
            reg_label = 'all' if region_filter is None else regions[regions.index(region_filter)]
     
            if diag in diag_class['doy_diags']:
                for cls in ['early', 'clim', 'late', 'overall']:
                    sub = sub_reg if cls == 'overall' else sub_reg[sub_reg['true_class'] == cls]
                    all_acc_rows.append({
                        'model':      label,
                        'region':     reg_label,
                        'true_class': cls,
                        'n':          len(sub),
                        'MAE':        round(sub['abs_error'].mean(), 2),
                        'MedAE':      round(sub['abs_error'].median(), 2),
                        'within_5d':  round((sub['abs_error'] <= 5).mean() * 100, 1),
                        'within_10d': round((sub['abs_error'] <= 10).mean() * 100, 1),
                    })
                    
            else:
                for cls in ['below-normal', 'near-normal', 'above-normal', 'overall']:
                    sub = sub_reg if cls == 'overall' else sub_reg[sub_reg['true_class'] == cls]
                
                    all_acc_rows.append({
                        'model':      label,
                        'region':     reg_label,
                        'true_class': cls,
                        'n':          len(sub),
                        'MAE':        round(sub['abs_error'].mean(), 2),
                        'MedAE':      round(sub['abs_error'].median(), 2),
                        'within_5d':  round((sub['abs_error'] <= 5).mean() * 100, 1),
                        'within_10d': round((sub['abs_error'] <= 10).mean() * 100, 1),
                    })
 
    print(f'  Done. n={n}, acc={df_loo["correct"].mean():.3f}, '
          f'bal_acc={balanced_accuracy_score(le.transform(all_true), le.transform(all_pred)):.3f}')

df_clf_results = pd.DataFrame(all_clf_rows)
print(df_clf_results[df_clf_results['region'] == 'all'].to_string(index=False))
df_acc_results = pd.DataFrame(all_acc_rows)
print(df_acc_results[df_acc_results['region'] == 'all'].to_string(index=False))
    
df_clf_results.to_csv(f'y.code_output/model_comparison_classification.csv', index=False)
df_acc_results.to_csv(f'y.code_output/model_comparison_accuracy.csv', index=False)
                

#%% -- final model retrained on all pooled data --

all_clf_final = {}
for c, config in enumerate(configs):
    df, clim  = load_data(filepath_anoms)
    clim_src  = clim['diag_clim']
    label = config['label']
    print(f'\n  Final model: {label}')

    diag          = diagnostics_used[c]
    label_map     = get_class_labels(diag, diag_class)

    if diag in diag_class['doy_diags'][:2] or diag in diag_class['anom_diags']:
        doy_clim = {'plow': config['plow'], 'phigh': config['phigh']}
    elif diag == 'dry_spell_nlong':
        doy_clim = {'phigh': config['phigh']}
    else:
        doy_clim = None

    feat_cols = build_feature_columns(indices, seasons, diag, df.columns)
    X_all, y_all, y_class_all, years_all, regions_all, clim_means = \
        build_unified_dataset(feat_cols, diag, df, clim_src, doy_clim)
    feature_names = get_unified_feature_names(feat_cols)
    n = len(X_all)
    best_params_final = global_best_params

    # Precompute combined mask (identical to LOO loop)
    active_regions    = list(dict.fromkeys(regions_all))
    feat_col_idx      = {col: i for i, col in enumerate(feat_cols)}
    combined_col_mask = {}
    for region in active_regions:
        loc = ('northern1' if region in northern1_regions else
               'northern2' if region in northern2_regions else
               'southern1' if region in southern1_regions else
               'southern2' if region in southern2_regions else None)
        if loc is None:
            continue
        valid_seasons_map = get_valid_seasons(diag)
        if loc not in valid_seasons_map:
            continue
        allowed_seasons = valid_seasons_map[loc]
        combined_col_mask[region] = np.array([
            any(col.endswith('_' + s) for s in allowed_seasons)
            for col in feat_cols
        ])

    X_all_final = X_all.copy()
    n_feat      = len(feat_cols)
    for i in range(n):
        region = regions_all[i]
        if region in combined_col_mask:
            X_all_final[i, :n_feat] = X_all[i, :n_feat] * combined_col_mask[region]

    # Fit final classifier & regressor on fully masked pooled data
    clf_final = BalancedRandomForestClassifier(**best_params_final, min_samples_split=4, class_weight='balanced', random_state=13)
    clf_final.fit(X_all_final, y_class_all)
    
    regs_final = {}
    if diag in diag_class['doy_diags'] or diag == 'lds_dur':
        for region in active_regions:
            region_mask  = np.array(regions_all) == region
            X_reg        = X_all_final[region_mask]
            y_reg        = y_all[region_mask]
            yc_reg       = y_class_all[region_mask]
            regs_final[region] = fit_conditional_regressors(X_reg, y_reg, yc_reg)

    train_preds_final = clf_final.predict(X_all_final)
    train_acc_final   = (train_preds_final == y_class_all).mean()
    print(f'  Train accuracy: {train_acc_final:.3f}')

    cv_records = []
    for i in range(n):
        X_row  = X_all_final[[i]]
        region = regions_all[i]
        p_tr   = clf_final.predict_proba(X_row)[0]
        p_arr  = np.full(3, np.nan)
        for ci, cls in enumerate(clf_final.classes_):
            if cls in (0, 1, 2):
                p_arr[int(cls)] = p_tr[ci]

        if diag in diag_class['doy_diags'] or diag == 'lds_dur':
            pred_class = int(clf_final.predict(X_row)[0])
            pred_val, q25, q75 = predict_value_uncertainty(
                regs_final[region], pred_class, X_row,
                np.median(y_all[np.array(regions_all) == region]))
            clim_mean = clim_means.get(region, np.nan)
        else:
            pred_val, q25, q75, clim_mean = None, None, None, None

        cv_records.append({
            'region':       region,
            'diagnostic':   diag,
            'year':         int(years_all[i]),
            'true_class':   label_map.get(int(y_class_all[i]), ''),
            'pred_class':   label_map.get(int(clf_final.predict(X_row)[0]), ''),
            'pred_value':   pred_val,
            'days_offset':  pred_val - clim_mean if pred_val is not None else np.nan,
            'p_class0':     round(float(p_arr[0]) * 100, 1) if not np.isnan(p_arr[0]) else np.nan,
            'p_class1':     round(float(p_arr[1]) * 100, 1) if not np.isnan(p_arr[1]) else np.nan,
            'p_class2':     round(float(p_arr[2]) * 100, 1) if not np.isnan(p_arr[2]) else np.nan,
            'class0_label': label_map.get(0, ''),
            'class1_label': label_map.get(1, ''),
            'class2_label': label_map.get(2, ''),
            'train_accuracy': round(train_acc_final, 3),
            'feature_cols': '|'.join(feature_names),
        })

    all_clf_final[label] = {
        'clf':            clf_final,
        'X_all':          X_all_final,
        'feat_cols':      feat_cols,
        'feature_names':  feature_names,
        'cv_records':     cv_records,
        'regs':           regs_final,
        'clim_means':     clim_means,
        'regions_all':    regions_all,
        'active_regions': active_regions,
        'diagnostic':     diag
    }
    print(f'  Stored clf_final for {label}')


#%% -- plotting functions --

def conf_matrix(df_loo, diag, diag_labels, output_dir, region=None):
    """
    Confusion matrix.
    If region is None: plot across all regions combined.
    If region is given: filter to that region only.
    """
    sub = df_loo[df_loo['diagnostic'] == diag].dropna(
        subset=['true_class', 'pred_class'])
    if region is not None:
        if diag == 'lds_dur':
            if region not in ['guin-w', 'guin-e']: return None
        sub = sub[sub['region'] == region]

    if diag in diag_class['doy_diags']:
        sort_order = {'early': 0, 'clim': 1, 'late': 2}
    elif diag in diag_class['anom_diags']:
        sort_order = {'below-normal': 0, 'near-normal': 1, 'above-normal': 2}
        lbl = ['below- \n normal', 'near- \n normal', 'above- \n normal']
    elif diag in diag_class['binary_diags']:
        if diag == 'dry_spell_nlong':
            sort_order = {'low risk': 0, 'elevated risk': 1}
        else:
            sort_order = {'no': 0, 'yes': 1}
                
    all_labels = list(dict.fromkeys(sorted(
        list(sub['true_class'].unique().tolist() + sub['pred_class'].unique().tolist()
             ), key=lambda d: sort_order[d])))
    if diag not in diag_class['anom_diags']:
        lbl = all_labels
    
    if region is not None:
        bal_acc = df_clf_results[
            (df_clf_results['model'] == label) &
            (df_clf_results['region'] == region) &
            (df_clf_results['class'] == 'overall')
        ]['bal_accuracy'].values[0]
    else:
        bal_acc = df_clf_results[
            (df_clf_results['model'] == label) &
            (df_clf_results['region'] == 'all') &
            (df_clf_results['class'] == 'overall')
        ]['bal_accuracy'].values[0]

    with mpl.rc_context({'font.size': 14}):
        # title_region = region if region else 'All regions'
        fig, ax = plt.subplots(figsize=(6,5))
        cm_arr = confusion_matrix(sub['true_class'], sub['pred_class'],
                                   labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr,
                                       display_labels=all_labels)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        plt.gca().set_xticklabels(lbl)
        plt.gca().set_yticklabels(lbl)
        ax.set_title(f'{diag_labels.get(diag, diag)}\n'
                     f'bal_acc={bal_acc:.3f}', fontsize=14)
        ax.grid(False)
        fig.tight_layout()
        if region is not None:
            fig.savefig(f'{output_dir}/conf matrix/regional/{diag}_confusion_matrix_{region}', bbox_inches='tight')
        else:
            fig.savefig(f'{output_dir}/conf matrix/{diag}_confusion_matrix', bbox_inches='tight')
        # plt.show()
        plt.close()


def plot_loo_probabilities(df_loo, diag, diag_labels, output_dir):
    """
    Stacked bar chart of LOO predicted probabilities per year.
    """
    sub_all = df_loo[df_loo['diagnostic'] == diag].copy()
    label_map = get_class_labels(diag, diag_class)

    if diag not in diag_class['binary_diags']:
        colours = {'class0': 'whitesmoke', 'class1': 'moccasin', 'class2': 'thistle'}
        true_class_colors = {
            label_map.get(0): 'darkgrey',
            label_map.get(1): 'orange',
            label_map.get(2): 'mediumpurple',
        }
    else:
        colours = {'class0': 'whitesmoke', 'class1': 'lightsalmon'}
        true_class_colors = {
            label_map.get(0): 'darkgrey',
            label_map.get(1): 'orangered',
        }
    
    grid = [['sah-w',     'sah-e'],
            ['sud-sah-w', 'sud-sah-e'],
            ['sud-w',     'sud-e'],
            ['guin-w',    'guin-e']]
    col_titles = ['w', 'e']
    row_titles = ['guin'] if diag == 'lds_dur' else ['sah', 'sud-sah', 'sud', 'guin']
    nrows  = 1 if diag == 'lds_dur' else 4
    height = 3 if diag == 'lds_dur' else 12

    fig, axes = plt.subplots(nrows, 2, figsize=(10, height))
    fig.suptitle(
        f'LOO-CV predicted probabilities | {diag_labels.get(diag, diag)}',
        fontsize=13, y=1.01,
    )
    
    for ri in range(nrows):
        for ci in range(2):
            ax     = axes[ri,ci]  if nrows == 4 else axes[ci]
            region = grid[ri][ci] if nrows == 4 else grid[ri+3][ci]
            sub    = sub_all[sub_all['region'] == region].sort_values('year').copy()
            x = np.arange(len(sub))

            b1 = ax.bar(x, sub['p_class0'], color=colours['class0'],
                        label=label_map.get(0, 'class0'))
            b2 = ax.bar(x, sub['p_class1'], bottom=sub['p_class0'],
                        color=colours['class1'], label=label_map.get(1, 'class1'))
            if diag not in diag_class['binary_diags']:
                b3 = ax.bar(x, sub['p_class2'],
                            bottom=sub['p_class0'].values + sub['p_class1'].values,
                            color=colours['class2'], label=label_map.get(2, 'class2'))
                for b in [b1, b2, b3]:
                    ax.bar_label(b, label_type='center', fontsize=6.5,
                                 fmt=lambda v: f'{v:.0f}' if v >= 5 else '')
            else:
                for b in [b1, b2]:
                    ax.bar_label(b, label_type='center', fontsize=6.5,
                                 fmt=lambda v: f'{v:.0f}' if v >= 5 else '')

            for j, (_, row) in enumerate(sub.iterrows()):
                ax.scatter(j, 105, marker='s', s=50, zorder=5,
                           color=true_class_colors.get(row['true_class'], 'black'))
                ax.scatter(j, 114, marker='o', s=50, zorder=5,
                           color=true_class_colors.get(row['pred_class'], 'black'))

            if ci == 0:
                ax.text(-1.5, 104, 'real', fontstyle='italic', fontsize=7)
                ax.text(-1.5, 113, 'pred', fontstyle='italic', fontsize=7)
                ax.set_ylabel('Probability (%)', fontsize=8)
            
            if ri == nrows-1:
                ax.set_xticks(x)
                ax.set_xticklabels(sub['year'].astype(int).astype(str),
                                   rotation=45, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.set_ylim(0, 125)
            ax.set_yticks(np.linspace(0,100,6))
            ax.set_title(f'{row_titles[ri]}-{col_titles[ci]}', fontsize=9, pad=3)
            ax.tick_params(axis='y', labelsize=7)

    first_pos = axes[0,0] if nrows == 4 else axes[0]
    handles, labels = first_pos.get_legend_handles_labels()
    legend_tab = 0.96 if diag == 'lds_dur' else 0.99
    order = [labels.index(lbl) for lbl in [label_map.get(1, 'class1'),label_map.get(0, 'class0'), label_map.get(2, 'class2')] \
             if lbl in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center',
               ncols=3, fontsize=9, bbox_to_anchor=(0.5, legend_tab), frameon=False)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/probs timeseries/{diag}_probs_ts.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    

def feat_importance(clf, feat_cols, X_data, label, diag, output_dir):
    """
    Feature importances: Gini and SHAP
    """
    importances    = clf.feature_importances_
    feature_imp_df = pd.DataFrame(
        {'Feature': feat_cols, 'Gini Importance': importances}
    ).sort_values('Gini Importance', ascending=False).set_index('Feature')[:15]

    explainer   = shap.TreeExplainer(clf)
    shap_values = np.array(explainer.shap_values(X_data))
    shap_summary = np.abs(shap_values).mean(axis=(0,2))
    shap_summary = shap_summary.flatten()[:len(feat_cols)]

    shap_summary_df = pd.DataFrame(
        {'Feature': feat_cols, 'Mean |SHAP Value|': shap_summary}
    ).sort_values('Mean |SHAP Value|', ascending=False)[:15]

    def feature_color(f):
        if 'ENSO' in f: return 'red'
        if 'AMM'  in f: return 'blue'
        if 'DUST' in f: return 'gold'
        if 'OM'   in f: return 'green'
        if 'IOD'  in f: return 'skyblue'
        if 'NAO'  in f: return 'indigo'
        return '#666666'

    feature_imp_df['Color']  = feature_imp_df.index.map(feature_color)
    shap_summary_df['Color'] = shap_summary_df['Feature'].map(feature_color)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].barh(feature_imp_df.index[::-1], feature_imp_df['Gini Importance'][::-1],
                color=feature_imp_df['Color'][::-1], height=0.8)
    axs[1].barh(shap_summary_df['Feature'], shap_summary_df['Mean |SHAP Value|'],
                color=shap_summary_df['Color'], height=0.8)
    axs[1].invert_yaxis()
    axs[0].set_xlabel('Gini importance')
    axs[1].set_xlabel('Mean abs. SHAP value')
    axs[0].set_title('Gini importance', fontsize=14)
    axs[1].set_title('SHAP Values', fontsize=14)
    plt.suptitle(f'Feature importances | {diag_labels.get(diag, diag)}', fontsize=16)
    handles = [
        mpatches.Patch(color='red', label='ENSO'),
        mpatches.Patch(color='blue', label='AMM'),
        mpatches.Patch(color='gold', label='DUST'),
        mpatches.Patch(color='green', label='OM'),
        mpatches.Patch(color='skyblue', label='IOD'),
        mpatches.Patch(color='indigo', label='NAO'),
    ]
    axs[1].legend(handles=handles, fontsize=10, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/feat importances/{diag}_importances.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_value_preds(df_loo, diag, diag_labels, clim, regions, output_dir):
    """
    Predicted values for onset, cessation, LDS duration from the regressor.
    """
    sub_all = df_loo[df_loo['diagnostic'] == diag].copy()
    if diag == 'lds_dur':
        grid = [['guin-w', 'guin-e']]
        nrows, ncols = 1, 2
        title_tab  = 1.13
        legend_tab = -0.3
    else:
        grid = [['sah-w',     'sah-e'],
                ['sud-sah-w', 'sud-sah-e'],
                ['sud-w',     'sud-e'],
                ['guin-w',    'guin-e']]
        nrows, ncols = 4, 2
        title_tab  = 0.94
        legend_tab = 0.00

    fig = plt.figure(figsize=(14, 2.4 * nrows))
    fig.suptitle(
        f'Predicted anomaly risk  |  {diag_labels.get(diag, diag)}\n'
        f'Bars: deviation from climatology | Markers: true value',
        fontsize=11, y=title_tab,
    )
    gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.12)

    for ri in range(nrows):
        for ci in range(ncols):
            region   = grid[ri][ci]
            reg_name = region_names[regions.index(region)]
            sub      = sub_all[sub_all['region'] == region].sort_values('year')
            ax_dev   = fig.add_subplot(gs[ri, ci]) if nrows == 4 else fig.add_subplot(gs[ci])
            
            years    = sub['year'].values
            y_pos    = np.arange(len(years))
            true_dev = sub['true_value'].values - clim[region]

            # ── Deviation bars ────────────────────────────────────────────────
            ax_dev.axvline(0, color='#666', lw=0.8, zorder=2)
            for bound in RISK_BINS[1:-1]:
                ax_dev.axvline( bound, color='gray', lw=1, ls=':', zorder=1)
                ax_dev.axvline(-bound, color='gray', lw=1, ls=':', zorder=1)

            for j, (dev_val, tdev) in enumerate(zip(sub['days_offset'].values,
                                                      true_dev)):
                sign   = np.sign(dev_val) if dev_val != 0 else 1
                abs_dv = abs(dev_val)
                colors = C_LATE if sign > 0 else C_EARLY

                # Stack segments bin-by-bin up to abs_dv
                remaining = abs_dv
                cursor    = 0.0
                for bi in range(len(RISK_LABELS)):
                    lo = RISK_BINS[bi];  hi = RISK_BINS[bi + 1]
                    seg = min(remaining, hi - lo)
                    if seg <= 0:
                        break
                    ax_dev.barh(j, sign * seg, left=sign * cursor,
                                color=colors[bi], edgecolor='none',
                                height=0.65, zorder=3)
                    remaining -= seg;  cursor += seg

                # True deviation marker
                ax_dev.scatter(tdev, j, marker='|', color='black',
                               s=55, linewidths=1.8, zorder=5)

            ax_dev.set_yticks(y_pos[::2])
            ax_dev.set_yticklabels(years[::2], fontsize=9)
            ax_dev.set_xlabel('Anomaly', fontsize=9)
            xlim = max(12, np.nanmax(np.abs(sub['days_offset'].values)) + 2)
            ax_dev.set_xlim(-xlim, xlim)
            ax_dev.tick_params(axis='x', labelsize=9)
            ax_dev.set_title(reg_name, fontsize=10, loc='left', pad=2)

    legend_handles = (
        [mpatches.Patch(color=C_LATE[i],  label=f'Late {RISK_LABELS[i]}')
         for i in range(len(RISK_LABELS))]
        + [mpatches.Patch(color=C_EARLY[i], label=f'Early {RISK_LABELS[i]}')
           for i in range(len(RISK_LABELS))]
        + [plt.Line2D([0], [0], marker='|', color='black', lw=0,
                      markersize=9, markeredgewidth=2, label='True value')]
    )
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, legend_tab),
               frameon=False)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/probs timeseries/{diag}_pred_ts.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    
    
def plot_gridsearch_results_pooled(df_gs, output_dir):
    """
    For each diagnostic, plot mean_test_score vs each hyperparameter,
    marginalised over all other parameters.
    """
    param_cols = [c for c in
                  ['n_estimators', 'min_samples_leaf', 'max_features',
                   'max_depth', 'sampling_strategy']
                  if c in df_gs.columns]

    DIAG_LABELS = {
        'doy_os':        'Onset',
        'doy_cess':      'Cessation',
        '1stmonth_anom': 'RS1 precip anomaly',
        'dry_spell_anom':'Dry spell anomaly',
        'dry_spell_nlong':'Long dry spells',
        'false_onset':   'False onset',
        'lds_dur':       'LDS duration',
    }

    for diag in df_gs['diagnostic'].unique():
        sub = df_gs[df_gs['diagnostic'] == diag].copy()
        for p in param_cols:
            if p in sub.columns:
                sub[p] = sub[p].astype(str)

        active_params = [p for p in param_cols if p in sub.columns
                         and sub[p].nunique() > 1]
        n_params = len(active_params)
        if n_params == 0:
            continue

        fig, axes = plt.subplots(1, n_params,
                                 figsize=(3.5 * n_params, 3.5),
                                 sharey=False)
        if n_params == 1:
            axes = [axes]
        for ax, param in zip(axes, active_params):
            grouped = (sub.groupby(param)['mean_test_score']
                       .agg(['mean', 'std'])
                       .reset_index()
                       .sort_values(param))
            means  = grouped['mean'].values.astype(float)
            stds   = grouped['std'].values.astype(float)
            labels = grouped[param].values
            x      = range(len(labels))

            best_val = labels[np.argmax(means)]
            colors   = ['#E53935' if str(v) == str(best_val) else '#1565C0'
                        for v in labels]

            ax.bar(x, means, yerr=stds, color=colors, alpha=0.78,
                   error_kw={'elinewidth': 1.5, 'capsize': 4, 'capthick': 1.5})
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=14, rotation=20)
            score_range = means.max() - means.min()
            ax.set_title(f'{param}\nrange={score_range:.3f}', fontsize=14)
            ax.set_ylabel('Balanced accuracy', fontsize=14)
            ax.set_ylim(max(0, means.min() - stds.max() - 0.05),
                        min(1, means.max() + stds.max() + 0.05))
            ax.tick_params(labelsize=14)

        fig.suptitle(f'Hyperparameter sensitivity — {DIAG_LABELS.get(diag, diag)}',
                     fontsize=18)
        fig.tight_layout()
        plt.show()


output_dir = 'x.model_plots'
labels = [config['label'] for config in configs]

RISK_BINS   = [0, 2, 5, 10, np.inf]
RISK_LABELS = ['≤2 d', '2–5 d', '5–10 d', '>10 d']
C_LATE  = ['thistle', 'mediumpurple', 'rebeccapurple', 'indigo']
C_EARLY = ['moccasin', 'orange', 'darkorange', 'brown']
C_CLIM  = 'whitesmoke'

for n, label in enumerate(labels):    
    conf_matrix(all_df_loo[label], diagnostics[n], diag_labels, output_dir)
    for reg in [r for r in regions if r in set(regions_all)]:
        conf_matrix(all_df_loo[label], diagnostics[n], diag_labels, output_dir, region=reg)
    plot_loo_probabilities(all_df_loo[label], diagnostics[n], diag_labels, output_dir)

    if diagnostics[n] in ['doy_os', 'doy_cess', 'lds_dur']:
        clims = [means_os, means_cess, None, None, None, means_ldsdur, None]
        plot_value_preds(all_df_loo[label], diagnostics[n], diag_labels, clims[n], regions, output_dir)

for label, bundle in all_clf_final.items():
    feat_importance(bundle['clf'], bundle['feature_names'], bundle['X_all'], label, bundle['diagnostic'], output_dir)
df_gs = pd.DataFrame(gs_records)
plot_gridsearch_results_pooled(df_gs, output_dir=None)    
    

#%% -- issue dates models --

ISSUE_DATES = {
    'NDJ':     ['NDJ'],
    'NDJ-DJF': ['NDJ', 'DJF'],
    'NDJ-JFM': ['NDJ', 'DJF', 'JFM'],
    'NDJ-FMA': ['NDJ', 'DJF', 'JFM', 'FMA'],
    'NDJ-MAM': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
    'NDJ-AMJ': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ'],
    'MAM':     ['MAM'],
    'MAM-AMJ': ['MAM', 'AMJ'],
    'MAM-MJJ': ['MAM', 'AMJ', 'MJJ'],
    'MAM-JJA': ['MAM', 'AMJ', 'MJJ', 'JJA'],
    'MAM-JAS': ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS',],
    'MAM-ASO': ['MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO']
}


def get_issue_date_range(diag, get_valid_seasons, ISSUE_DATES, seasons):
    """
    Keep only issue dates whose available_seasons are all within the window
    [earliest_valid_season, latest_valid_season], as ordered in 'seasons'.
    """
    all_valid = set(
        s for loc_seasons in get_valid_seasons(diag).values() for s in loc_seasons
    )
    valid_positions = [i for i, s in enumerate(seasons) if s in all_valid]
    if not valid_positions:
        return []
    earliest_pos = min(valid_positions)
    latest_pos   = max(valid_positions)
    start_season = 'MAM' if diag == 'doy_cess' else 'NDJ'

    valid_keys = [
        label for label, avail in ISSUE_DATES.items()
        if avail[0] == start_season
        and all(earliest_pos <= seasons.index(s) <= latest_pos for s in avail)
    ]
    return valid_keys


all_clf_issue   = {}
issue_skill_rows = []
all_loo_issue_records = []

configs = [
    {'label': 'onset', 'means': means_os, 'plow': p30_os, 'phigh': p70_os},
    {'label': 'cess', 'means': means_cess, 'plow': p30_cess, 'phigh': p70_cess},
    {'label': 'RS1_anom', 'plow': p30_1stmnth, 'phigh': p70_1stmnth},
    {'label': 'spell_anoms', 'plow': neg_spellanom, 'phigh': pos_spellanom},
    {'label': 'spell_nlong', 'phigh': med_nlongspell},
    {'label': 'lds_dur', 'plow': p30_ldsdur, 'phigh': p70_ldsdur},
    {'label': 'false_os'},
]
diagnostics_used = ['doy_os', 'doy_cess', '1stmonth_anom',
                    'dry_spell_anom', 'dry_spell_nlong', 'lds_dur', 'false_onset']

for c, config in enumerate(configs):
    label = config['label']
    diag  = diagnostics_used[c]
    print(f'\nIssue-date models: {label} ({diag})')

    df, clim = load_data(filepath_anoms)
    clim_src  = clim['diag_clim']
    label_map = get_class_labels(diag, diag_class)

    if diag in diag_class['doy_diags'][:2] or diag in diag_class['anom_diags']:
        doy_clim = {'plow': config['plow'], 'phigh': config['phigh']}
    elif diag == 'dry_spell_nlong':
        doy_clim = {'phigh': config['phigh']}
    else:
        doy_clim = None

    feat_cols_full = build_feature_columns(indices, seasons, diag, df.columns)
    valid_issue_labels = get_issue_date_range(diag, get_valid_seasons, ISSUE_DATES, seasons)

    for issue_label, available_seasons in ISSUE_DATES.items():
        if issue_label not in valid_issue_labels:
            continue

        feat_cols_issue = [
            c for c in feat_cols_full
            if any(c.endswith('_' + s) for s in available_seasons)
        ]
        result = build_unified_dataset(
            feat_cols_issue, diag, df, clim_src, doy_clim)
        if result is None:
            print(f'    {issue_label}: build_unified_dataset returned None, skipping.')
            continue
        X_all_issue, y_all_issue, y_class_all_issue, \
            years_all_issue, regions_all_issue, clim_means_issue = result
        feature_names_issue = get_unified_feature_names(feat_cols_issue)
        n_issue = len(X_all_issue)
        best_params_issue = global_best_params
        
        le_issue = LabelEncoder().fit(list(label_map.values()))
        feat_col_idx_issue = {col: i for i, col in enumerate(feat_cols_issue)}
        combined_col_mask_issue = {}
        for region in np.unique(regions_all_issue):
            loc = ('northern1' if region in northern1_regions else
                   'northern2' if region in northern2_regions else
                   'southern1' if region in southern1_regions else
                   'southern2' if region in southern2_regions else None)
            allowed_seasons_reg = get_valid_seasons(diag).get(loc, [])
            effective_seasons = [s for s in allowed_seasons_reg
                                 if s in available_seasons]
            combined_col_mask_issue[region] = np.array([
                any(col.endswith('_' + s) for s in effective_seasons)
                for col in feat_cols_issue
            ])

        loo_issue   = LeaveOneOut()
        loo_records_issue = []

        for fold, (train_idx, test_idx) in enumerate(
                loo_issue.split(X_all_issue)):
            i          = test_idx[0]
            X_train_i  = X_all_issue[train_idx].copy()
            yc_train_i = y_class_all_issue[train_idx]
            X_test_i   = X_all_issue[[i]].copy()
            yc_test_i  = y_class_all_issue[i]
            year_test  = int(years_all_issue[i])
            region     = regions_all_issue[i]

            col_mask = combined_col_mask_issue.get(region)
            if col_mask is not None:
                n_feat_i = len(feat_cols_issue)
                X_train_i[:, :n_feat_i] = X_train_i[:, :n_feat_i] * col_mask
                X_test_i[:, :n_feat_i]  = X_test_i[:, :n_feat_i]  * col_mask

            clf_fold_i = BalancedRandomForestClassifier(
                **best_params_issue, min_samples_split=4,
                class_weight='balanced', random_state=13)
            clf_fold_i.fit(X_train_i, yc_train_i)
            
            proba_i      = clf_fold_i.predict_proba(X_test_i)[0]
            p_arr        = np.full(3, np.nan)
            for ci, cls in enumerate(clf_fold_i.classes_):
                p_arr[int(cls)] = proba_i[ci]
            
            pred_class_i = int(clf_fold_i.classes_[np.argmax(proba_i)])
            correct_i    = (label_map.get(pred_class_i, '') ==
                            label_map.get(int(yc_test_i), ''))
            
            loo_records_issue.append({
                'model':         label,
                'diagnostic':    diag,
                'issue_label':   issue_label,
                'region':        region,
                'year':          year_test,
                'true_class':    label_map.get(int(yc_test_i), ''),
                'pred_class':    label_map.get(pred_class_i, ''),
                'p_class0':      round(float(p_arr[0]) * 100, 1) if not np.isnan(p_arr[0]) else np.nan,
                'p_class1':      round(float(p_arr[1]) * 100, 1) if not np.isnan(p_arr[1]) else np.nan,
                'p_class2':      round(float(p_arr[2]) * 100, 1) if not np.isnan(p_arr[2]) else np.nan,
                'class0_label':  label_map.get(0, ''),
                'class1_label':  label_map.get(1, ''),
                'class2_label':  label_map.get(2, ''),
                'correct':       correct_i,
            })

        df_loo_issue = pd.DataFrame(loo_records_issue)
        all_true_i   = df_loo_issue['true_class'].values
        all_pred_i   = df_loo_issue['pred_class'].values
        bal_acc_issue = balanced_accuracy_score(
            le_issue.transform(all_true_i),
            le_issue.transform(all_pred_i))        
        df_loo_issue['bal_acc_loo'] = bal_acc_issue
        all_loo_issue_records.append(df_loo_issue)

        # Retrain final model
        X_all_masked = X_all_issue.copy()
        for i in range(n_issue):
            reg = regions_all_issue[i]
            msk = combined_col_mask_issue.get(reg)
            if msk is not None:
                X_all_masked[i, :len(feat_cols_issue)] = \
                    X_all_issue[i, :len(feat_cols_issue)] * msk

        clf_issue = BalancedRandomForestClassifier(
            **best_params_issue, min_samples_split=4,
            class_weight='balanced', random_state=13)
        clf_issue.fit(X_all_masked, y_class_all_issue)

        regs_issue = {}
        if diag in diag_class['doy_diags'] or diag == 'lds_dur':
            active_regions    = list(dict.fromkeys(regions_all_issue))
            for region in active_regions:
                region_mask  = np.array(regions_all_issue) == region
                X_reg        = X_all_masked[region_mask]
                y_reg        = y_all_issue[region_mask]
                yc_reg       = y_class_all_issue[region_mask]
                regs_issue[region] = fit_conditional_regressors(X_reg, y_reg, yc_reg)

        key = f'{label}_{issue_label}'
        all_clf_issue[key] = {
            'clf':               clf_issue,
            'regs':              regs_issue,
            'feat_cols':         feat_cols_issue,
            'feature_names':     feature_names_issue,
            'available_seasons': available_seasons,
            'X_all':             X_all_masked,
            'y_class_all_issue': y_class_all_issue,    
            'years_all_issue':   years_all_issue,
            'regions_all':       list(regions_all_issue),
            'bal_acc_loo':       round(bal_acc_issue, 3),
            'diagnostic':        diag,
            'clim_means':        clim_means_issue
        }
        issue_skill_rows.append({
            'model':             label,
            'diagnostic':        diag,
            'issue_label':       issue_label,
            'n_seasons':         len(available_seasons),
            'n_features':        len(feat_cols_issue),
            'bal_acc_loo':       round(bal_acc_issue, 3),
        })
        print(f'    {issue_label}: {len(feat_cols_issue):>3} features | '
              f'LOO bal_acc={bal_acc_issue:.3f}')

df_issue_skill = pd.DataFrame(issue_skill_rows)
df_loo_issue_all = pd.concat(all_loo_issue_records, ignore_index=True)
df_loo_issue_all['p_pred'] = df_loo_issue_all[['p_class0', 'p_class1', 'p_class2']].max(axis=1)


#%% -- save the models locally --
save_models = True
if save_models:
    import joblib, os
    os.makedirs('z.models', exist_ok=True)
    
    for c, config in enumerate(configs):
        label = config['label']
        diag  = diagnostics_used[c]
        
        diag_bundle = {}
        for issue_label in ISSUE_DATES:
            key = f'{label}_{issue_label}'
            if key not in all_clf_issue:
                continue
            b = all_clf_issue[key]
            diag_bundle[issue_label] = {k: v for k, v in b.items()
                                        if k not in ('X_all',)}
        
        payload = {
            'label':      label,
            'diagnostic': diag,
            'bundles':    diag_bundle,
            'label_map':  get_class_labels(diag, diag_class),
        }
        path = f'z.models/clf_issue_{label}.pkl'
        joblib.dump(payload, path, compress=3)
        print(f'Saved {path}  ({len(diag_bundle)} issue-date models)')
    
    
#%% -- issue-date plots --

issue_date_order = [k for k in ISSUE_DATES if k.startswith('NDJ')] + \
                   [k for k in ISSUE_DATES if k.startswith('MAM')]
df_issue_skill['issue_label'] = pd.Categorical(
    df_issue_skill['issue_label'],
    categories=issue_date_order,
    ordered=True
)
df_issue_skill = df_issue_skill.sort_values('issue_label')
                   
fig, ax = plt.subplots(figsize=(9,4))
for lbl, grp in df_issue_skill.groupby('model', sort=False):
    ax.plot(grp['issue_label'], grp['bal_acc_loo'], marker='o', label=lbl)
ax.set_xlabel('Issue date (last available season)')
ax.set_ylabel('LOO balanced accuracy')
ax.set_title('Forecast skill vs. issue date')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('z.models/forecast_skill.png')
plt.show()

def conf_matrix_issue(df_loo_issue_all, diag, diag_labels, diag_class, issue_date_order):
    sub = df_loo_issue_all[
        df_loo_issue_all['diagnostic'] == diag
    ].dropna(subset=['true_class', 'pred_class'])

    if diag in diag_class['doy_diags']:
        sort_order = {'early': 0, 'clim': 1, 'late': 2}
    elif diag in diag_class['anom_diags']:
        sort_order = {'below-normal': 0, 'near-normal': 1, 'above-normal': 2}
    elif diag in diag_class['binary_diags']:
        if diag == 'dry_spell_nlong':
            sort_order = {'low risk': 0, 'elevated risk': 1}
        else:
            sort_order = {'no': 0, 'yes': 1}

    all_labels = list(dict.fromkeys(sorted(
        sub['true_class'].unique().tolist() +
        sub['pred_class'].unique().tolist(),
        key=lambda d: sort_order.get(d, 99)
    )))

    present_issues = sub['issue_label'].unique()
    issue_labels   = [k for k in issue_date_order if k in present_issues]
    n_issues       = len(issue_labels)

    fig, axes = plt.subplots(1, n_issues, figsize=(4*n_issues, 4.5), sharey=True)
    if n_issues == 1:
        axes = [axes]
    fig.suptitle(f'{diag_labels.get(diag, diag)}', fontsize=16, y=1.05)

    for ax, issue_label in zip(axes, issue_labels):
        df_i       = sub[sub['issue_label'] == issue_label]
        bal_acc    = df_i['bal_acc_loo'].iloc[0]
        n_samples  = len(df_i)

        cm_arr = confusion_matrix(df_i['true_class'], df_i['pred_class'], labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=all_labels)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')

        ax.set_title(f'{issue_label}\nbal_acc={bal_acc:.3f}  (n={n_samples})', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True' if ax == axes[0] else '', fontsize=11)
        ax.tick_params(axis='x', labelrotation=30)
        ax.grid(False)
    
    fig.tight_layout()
    os.makedirs('z.models/conf_matrix_issue', exist_ok=True)
    fig.savefig(f'z.models/conf_matrix_issue/{diag}_confusion_matrices', bbox_inches='tight')
    plt.close()

for diag in diagnostics_used:
    conf_matrix_issue(df_loo_issue_all, diag, diag_labels, diag_class, issue_date_order)


#%% -- run 2025 test forecast --

from forecast_2025 import run_forecast_2025, load_results
import joblib, glob

def load_issue_models(model_dir='z.models'):
    all_clf_issue = {}
    for path in sorted(glob.glob(f'{model_dir}/clf_issue_*.pkl')):
        payload = joblib.load(path)
        label   = payload['label']
        for issue_label, bundle in payload['bundles'].items():
            all_clf_issue[f"{label}_{issue_label}"] = bundle
    return all_clf_issue

all_clf_issue = load_issue_models()

df_fc = run_forecast_2025(
    all_clf_issue        = all_clf_issue,
    diagnostics_used     = diagnostics_used,
    configs              = configs,
    forecast_year        = 2025,
    output_dir           = 'z.models/forecast_2025_output',
    diag_class           = diag_class,
    regions              = regions,
    region_names         = region_names,
    northern1_regions    = northern1_regions,
    northern2_regions    = northern2_regions,
    southern1_regions    = southern1_regions,
    southern2_regions    = southern2_regions,
    REGION_POSITION      = REGION_POSITION,
    get_valid_seasons    = get_valid_seasons,
    get_class_labels     = get_class_labels,
    )

df_fc, by_diag, by_region = load_results(output_dir='z.models/forecast_2025_output/')


#%% -- 2025 forecast output plots --
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr
    
def extract_preds(df, diag):
    valid = {'sah-w':     'NDJ-MAM',
             'sah-e':     'NDJ-MAM',
             'sud-sah-w': 'NDJ-FMA',
             'sud-sah-e': 'NDJ-FMA',
             'sud-w':     'NDJ-JFM',
             'sud-e':     'NDJ-JFM',
             'guin-w':    'NDJ-DJF',
             'guin-e':    'NDJ-DJF',
        }
    classes = []
    probs = []
    probs_all = {}
    vals = {}
    
    for region in regions:
        if diag == 'lds_dur':
            if region not in ['guin-w', 'guin-e']:
                continue
            else:
                valid = {'guin-w': 'NDJ-AMJ',
                         'guin-e': 'NDJ-AMJ'}
        sub = df[diag][
            (df[diag]['issue_label']==valid[region]) & (df[diag]['region']==region)
            ]
        classes.append(sub['pred_class'].item())
        probs.append(max([sub['p_class0'].item(), sub['p_class1'].item(), sub['p_class2'].item()]))
        probs_all[region] = [sub['p_class0'].item(), sub['p_class1'].item(), sub['p_class2'].item()]
        
        if diag in ['doy_os', 'doy_cess', 'lds_dur']:
            vals[region] = {
                'val': sub['pred_value'].item(),
                'offset': sub['days_offset'].item(),
                'q25': sub['pred_q25'].item(), 'q75': sub['pred_q75'].item()
                }
        
    return classes, probs, probs_all, vals


def plot_preds(result_dict, diag, land_mask):   
    if diag in ['doy_os', 'doy_cess']:
        labels = ['Early', 'Normal', 'Late']
        colors = {0: 'whitesmoke', 1: 'moccasin', 2: 'thistle'}
    elif diag in ['1stmonth_anom', 'dry_spell_anom', 'lds_dur']:
        labels = ['Below-normal', 'Near-normal', 'Above-normal']
        colors = {0: 'whitesmoke', 1: 'moccasin', 2: 'thistle'}
    elif diag in ['dry_spell_nlong', 'false_onset']:
        colors = {0: 'whitesmoke', 1: 'lightsalmon'}
        if diag == 'dry_spell_nlong':
            labels = ['Low risk', 'Elevated risk']
        else:
            labels = ['No', 'Yes']
            
    regions_map = {'sah-w':     [[14, 18, -18, 0], -5, 16],
                   'sah-e':     [[14, 18, 0, 15],  10, 16],
                   'sud-sah-w': [[12, 14, -18, 0], -5, 13],
                   'sud-sah-e': [[12, 14, 0, 15],  10, 13],
                   'sud-w':     [[8, 12, -18, 0],  -5, 10],
                   'sud-e':     [[8, 12, 0, 15],   10, 10],
                   'guin-w':    [[4, 8, -18, 0],   -5, 6.5],
                   'guin-e':    [[4, 8, 0, 15],    10, 6.5]}

    lats = land_mask['lat'].values
    lons = land_mask['lon'].values
    land = land_mask.values
    rgba = np.zeros((len(lats), len(lons), 4), dtype=float)
    
    cmap = mcolors.ListedColormap([colors[1], colors[0], colors[2]]) if len(labels)==3 else mcolors.ListedColormap([colors[0], colors[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5] if len(labels) == 3 else [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.2)
    ax.set_extent([-20, 18, 2, 20], crs=ccrs.PlateCarree())
    gl = ax.gridlines(color='gray', linestyle='-:', draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    for region, (bbox, xpos, ypos) in regions_map.items():
        if diag == 'lds_dur' and region not in ['guin-w', 'guin-e']:
            continue
        lat_min, lat_max, lon_min, lon_max = bbox
        pred_class, pred_prob = result_dict[region][0], result_dict[region][1]
        color = mcolors.to_rgba(colors[pred_class])
        
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        box_mask = np.outer(lat_mask, lon_mask)
        fill_mask = box_mask & land
        rgba[fill_mask] = color
        
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
        
        ax.text(
            x=xpos,
            y=ypos,
            s=f'{np.round(pred_prob)}%',
            va='center',
            color='darkred',
            backgroundcolor='white',
            fontsize=10,
            transform=ccrs.PlateCarree(),
        )
    
    cb = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[-1, 0, 1] if len(labels)==3 else [0, 1])
    cb.minorticks_off()
    cb.ax.set_yticklabels(labels)
    plt.title(f"{diag_labels[diag]} class probability per region")
    plt.savefig(f'z.models/forecast_2025_output/prob_map_{diag}.png', bbox_inches='tight')
    plt.close()


def plot_probs(probs_dict, diag, regions, vals):
    if diag in ['doy_os', 'doy_cess']:
        labels = ['Early', 'Normal', 'Late']
        colors = ['orange', 'darkgrey', 'mediumpurple']
    elif diag in ['1stmonth_anom', 'dry_spell_anom', 'lds_dur']:
        labels = ['Below-normal', 'Near-normal', 'Above-normal']
        colors = ['orange', 'darkgrey', 'mediumpurple']
    elif diag in ['dry_spell_nlong', 'false_onset']:
        colors = ['darkgrey', 'orangered']
        if diag == 'dry_spell_nlong':
            labels = ['Low risk', 'Elevated risk']
        else:
            labels = ['No', 'Yes']
    
    if diag == 'lds_dur':
        nrows = 1
        regions = regions[-2:]
        legend_tab = 0.96
    else:
        nrows = 4
        legend_tab = 0.99
        
    fig, axs = plt.subplots(nrows, 2, sharex=True, sharey=True, figsize=(8, 3*nrows))
    axs = axs.flatten()
    for r, region in enumerate(regions):
        ax = axs[r]
        probs_reg = [prob for prob in probs_dict[region] if str(prob) != 'nan']
        x = np.arange(1, len(labels)+1, 1)
        bars = ax.bar(x, probs_reg, color=colors, label=labels)
        ax.bar_label(bars, fmt='{:,.0f}%', padding=5, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if r % 2 == 0:
            ax.set_ylabel("Probability")
        ax.set_title(f'{region}')
        ax.grid(axis='x')
        ax.set_ylim([0,100])        
    
    handles, lbls = axs[0].get_legend_handles_labels()
    order = [lbls.index(lbl) for lbl in lbls if lbl in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center',
               ncols=3, fontsize=11, bbox_to_anchor=(0.54, legend_tab), frameon=False)
    plt.suptitle(f"{diag_labels[diag]} all class probabilities", y=1.01, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'z.models/forecast_2025_output/prob_bar_{diag}.png', bbox_inches='tight')
    plt.close()
    plt.show()
    
    if diag in ['doy_os', 'doy_cess', 'lds_dur']:
        RISK_BINS   = [0, 2, 5, 10, np.inf]
        RISK_LABELS = ['≤2 d', '2–5 d', '5–10 d', '>10 d']
        C_LATE  = ['thistle', 'mediumpurple', 'rebeccapurple', 'indigo']
        C_EARLY = ['moccasin', 'orange', 'darkorange', 'brown']
        
        if diag == 'lds_dur':
            fig, axs = plt.subplots(nrows, 2, sharex=True, sharey=True, figsize=(8, 1.8*nrows))
            n = -0.15
            m = 20
        else:
            fig, axs = plt.subplots(nrows, 2, sharex=True, sharey=True, figsize=(8, 1.2*nrows))
            n = -0.0
            m = 15
        axs = axs.flatten()

        for r, region in enumerate(regions):
            ax = axs[r]
            pred_val = vals[region]['val']
            days_offset = vals[region]['offset']
            day_clim = pred_val + days_offset
            q25, q75 = day_clim - vals[region]['q25'], day_clim - vals[region]['q75']
            
            ax.axvline(0, color='k', lw=1.5, zorder=2)
            ax.hlines(0, q25, q75, color='grey', lw=3, alpha=0.7, zorder=4)
            ax.vlines([q25, q75], ymin=-0.10, ymax=0.10, color='grey', lw=2, alpha=0.7, zorder=4)
            
            for bound in RISK_BINS[1:-1]:
                ax.axvline( bound, color='gray', lw=2, ls=':', zorder=1)
                ax.axvline(-bound, color='gray', lw=2, ls=':', zorder=1)
            sign   = np.sign(days_offset) if days_offset != 0 else 1
            colors = C_LATE if sign > 0 else C_EARLY

            remaining = abs(days_offset)
            cursor    = 0.0
            for bi in range(len(RISK_LABELS)):
                lo = RISK_BINS[bi];  hi = RISK_BINS[bi + 1]
                seg = min(remaining, hi - lo)
                if seg <= 0:
                    break
                ax.barh(0, sign * seg, left=sign * cursor,
                            color=colors[bi], edgecolor='none',
                            height=0.65, zorder=3)
                remaining -= seg;  cursor += seg
            
            if r in [len(axs)-2, len(axs)-1]:
                ax.set_xlabel('Anomaly', fontsize=14)
            xlim = max(m, np.nanmax(np.abs(days_offset)) + 2)
            ax.set_xlim(-xlim, xlim)
            ax.tick_params(axis='x', labelsize=14)
            ax.set_title(region, fontsize=16, loc='left', pad=2)
            ax.set_yticks([])

        plt.suptitle(f'{diag_labels[diag]} deviation from climatology', fontsize=20, y=0.97)
        legend_handles = (
            [mpatches.Patch(color=C_LATE[i],  label=f'Late {RISK_LABELS[i]}')
             for i in range(len(RISK_LABELS))]
            + [mpatches.Patch(color=C_EARLY[i], label=f'Early {RISK_LABELS[i]}')
               for i in range(len(RISK_LABELS))]
            + [plt.Line2D([0], [0], marker='_', color='grey', lw=0,
                          markersize=10, markeredgewidth=2, label='Uncertainty')]
        )
        fig.legend(handles=legend_handles, loc='lower center',
                   ncol=5, fontsize=12, bbox_to_anchor=(0.5, -0.15+n),
                   frameon=False)
        fig.tight_layout()
        plt.savefig(f'z.models/forecast_2025_output/dates_range_{diag}.png', bbox_inches='tight')
        plt.show()
        # plt.close()
       
# plot using land mask        
mask_ds = xr.open_dataset('C:/Users/ikeva/z.datasets/IMERG_land_sea_mask.nc')
mask_ds = mask_ds.assign_coords(lon=((mask_ds.lon + 180) % 360) - 180).sortby('lon')
ls = mask_ds['landseamask'].sel(lon=slice(-20,40), lat=slice(0,35))
land_mask = (ls <= 70)

for diag in diagnostics:
    if diag == 'doy_cess':
        continue
    pred_class, pred_probs, probs_all, vals = extract_preds(by_diag, diag)
    if diag == 'lds_dur':
        result = {reg: [pred_class[r], pred_probs[r]] for r, reg in enumerate(regions[-2:])}
    else:
        result = {reg: [pred_class[r], pred_probs[r]] for r, reg in enumerate(regions)}
            
    plot_preds(result, diag, land_mask)
    plot_probs(probs_all, diag, regions, vals)
    

#%% -- feature space --

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_fc_sah = [[-0.42, -0.45, -0.24, -0.06,  0.02,  1.74,  1.38,  0.76, -0.06,
             -0.53, -0.04, -0.02, -1.08, -1.03, -0.13, -1.35, -0.93, -0.9 ,
             -1.04, -1.07, -1.21, -0.61, -0.65, -0.46, -0.3 , -0.05,  0.15,
             0.22,  0.15,  0.76,  0.46,  0.69,  0.32,  1.  ,  1.  ]]

X_fc_sudsah = [[-0.42, -0.45, -0.24, -0.06,  1.74,  1.38,  0.76, -0.06,
                -0.04, -0.02, -1.08, -1.03,  -1.35, -0.93, -0.9 ,
                -1.04, -1.21, -0.61, -0.65, -0.46, -0.3 , -0.05,  0.15,
                0.15,  0.76,  0.46,  0.69,  2.  ,  1.  ]]

X_fc_sud = [[-0.42, -0.45, -0.24,  1.74,  1.38,  0.76,
             -0.04, -0.02, -1.08, -1.35, -0.93, -0.9 ,
             -1.21, -0.61, -0.65, -0.46, -0.3 , -0.05,
             0.15,  0.76,  0.46,  3.  ,  1.  ]]

X_fc_guin = [[-0.42, -0.45,  1.74,  1.38,
              -0.04, -0.02, -1.35, -0.93,
              -1.21, -0.61, -0.46, -0.3,
              0.15,   0.76,    4.,   1. ]]

X_fc_guin_lds = [[-0.42, -0.45, -0.24, -0.06,  0.02, -0.02,  1.74,  1.38,  0.76,
                  -0.06, -0.53, -0.77, -0.04, -0.02, -1.08, -1.03, -0.13, -0.42,
                  -1.21, -0.61, -0.65, -0.58, -0.98, -1.28, -0.46, -0.3 , -0.05,
                  0.15,  0.22,  0.12,  0.15,  0.76,  0.46,  0.69,  0.32,  0.46,
                  4.  ,  1.  ]]

    
def feature_space_clustering(clf, diag, regions_to_check, lds=False):
    if 'sah-w' in regions_to_check:
        X_fc = X_fc_sah
        issue = 'NDJ-MAM'
    elif 'sud-sah-w' in regions_to_check:
        X_fc = X_fc_sudsah
        issue = 'NDJ-FMA'
    elif 'sud-w' in regions_to_check:
        X_fc = X_fc_sud
        issue = 'NDJ-JFM'
    elif 'guin-w' in regions_to_check:
        if not lds:
            X_fc = X_fc_guin
            issue = 'NDJ-DJF'
        else:
            X_fc = X_fc_guin_lds
            issue = 'NDJ-AMJ'
            
    diag_map = {'onset': 'doy_os', 'RS1_anom': '1stmonth_anom',
                'spell_anoms': 'dry_spell_anom', 'spell_nlong': 'dry_spell_nlong',
                'lds_dur': 'lds_dur', 'false_os': 'false_onset'}

    bundle = clf[f'{diag}_{issue}']
    regions_arr = np.array(bundle['regions_all'])
    X_masked    = bundle['X_all']
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    for ax, region in zip(axes, regions_to_check):
        mask      = regions_arr == region   
        X_reg     = X_masked[mask][:, :-2]   # drop position features
        yc_reg    = bundle['y_class_all_issue'][mask]
        yrs_reg   = np.array(bundle['years_all_issue'])[mask]
        # Scale and PCA
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X_reg)
        pca       = PCA(n_components=2)
        X_pca     = pca.fit_transform(X_scaled)
        # Project X_fc onto same space (drop position features)
        X_fc_arr  = np.array(X_fc)
        X_fc_proj = pca.transform(scaler.transform(X_fc_arr[0, :-2].reshape(1, -1)))
        
        if len(np.unique(yc_reg)) < 2:
            ax.set_title(f'{region} — {issue}', fontsize=16, pad=10)
            ax.text(0.5, 0.5, 'insufficient class variation', transform=ax.transAxes, ha='center', fontsize=14)
            ax.set_facecolor('#E5E5E5')
            continue
    
        if diag == 'onset':
            labels_cl = {0: 'clim', 1: 'early', 2: 'late'}
            colors    = {0: 'grey', 1: 'yellow', 2: 'purple'}
            clss      = [0, 1, 2]
        elif diag in ['RS1_anom', 'spell_anoms', 'lds_dur']:
            labels_cl = {0: 'near-normal', 1: 'below-normal', 2: 'above-normal'}
            colors    = {0: 'grey', 1: 'yellow', 2: 'purple'}
            clss      = [0, 1, 2]
        elif diag in ['spell_nlong', 'false_os']:
            colors    = {0: 'grey', 1: 'orange'}
            clss      = [0, 1]
            if diag == 'spell_nlong':
                labels_cl = {0: 'low risk', 1: 'elevated risk'}
            else:
                labels_cl = {0: 'no', 1: 'yes'}
    
        for cls in clss:
            idx = yc_reg == cls
            if idx.sum() == 0:
                continue
            ax.scatter(X_pca[idx,0], X_pca[idx,1],
                       c=colors[cls], label=labels_cl[cls], s=70,
                       alpha=0.7, edgecolors='k', linewidths=0.4)
            for xi, yi, yr in zip(X_pca[idx,0], X_pca[idx,1], yrs_reg[idx]):
                ax.annotate(str(yr), (xi, yi), xytext=(xi, yi+0.2), fontsize=12, ha='center', va='bottom')
    
        ax.scatter(X_fc_proj[0,0], X_fc_proj[0,1],
                   c='r', s=120, marker='*', edgecolors='k',
                   linewidths=0.8, label='2025', zorder=5)
        ax.annotate(2025, (X_fc_proj[0,0], X_fc_proj[0,1]), xytext=(X_fc_proj[0,0], X_fc_proj[0,1]+0.2),
                    color='darkred', fontsize=12, ha='center', va='bottom')
        ax.set_title(f'{region}  —  {issue}\n'
                     f'(var explained: {pca.explained_variance_ratio_.sum():.2f})', fontsize=16, pad=10)
        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 1)
        ax.legend(loc='upper right', fontsize=10)
    plt.suptitle(f"PCA feature space | {diag_labels[diag_map[diag]]}", fontsize=20, y=0.98)
    plt.tight_layout()
    os.makedirs('z.models/feature_space_figs', exist_ok=True)
    plt.savefig(f'z.models/feature_space_figs/{diag}_pca_feature_space_{regions_to_check[0][:-2]}.png')
    plt.close()
    

for diag in ['onset', 'RS1_anom', 'spell_anoms', 'spell_nlong', 'lds_dur', 'false_os']:
    for regions_to_check in [['sah-w', 'sah-e'], ['sud-sah-w', 'sud-sah-e'], ['sud-w', 'sud-e'], ['guin-w', 'guin-e']]:
        if diag == 'lds_dur':
            if regions_to_check != ['guin-w', 'guin-e']:
                continue
            feature_space_clustering(all_clf_issue, diag, regions_to_check, lds=True)
        else:
            feature_space_clustering(all_clf_issue, diag, regions_to_check, lds=False)


#%% -- 2025 summary matrix --

def plot_prediction_matrix(data_dict, regions, diag_order=None, diag_labels=None,
                           cmap_name='Blues', figsize=(15,7)):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    if diag_order is None:
        diag_order = list(data_dict.keys())

    if diag_labels is None:
        diag_labels = {k: k for k in diag_order}

    nrows = len(diag_order)
    ncols = len(regions)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_facecolor('white')

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=10, vmax=110)

    correct_edge = '#7FB069'
    incorrect_edge = '#D14141'
    missing_face = '#F2F2F2'
    mapping = {'early': 1, 'clim': 0, 'late': 2,
               'below-normal': 1, 'near-normal': 0, 'above-normal': 2,
               'low risk': 0, 'elevated risk': 1, 'no': 0, 'yes': 1}

    for i, diag in enumerate(diag_order):
        for j, region in enumerate(regions):
            cell = data_dict.get(diag, {}).get(region, None)

            x, y = j, i

            if cell is None or cell.get('true') in [None, ''] or cell.get('pred') in [None, '']:
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor=missing_face,
                    edgecolor='lightgray',
                    linewidth=1.0
                )
                ax.add_patch(rect)
                continue

            true_label = str(cell['true'])
            pred_label = str(cell['pred'])
            prob = float(cell['prob'])
            is_correct = true_label == pred_label

            facecolor = cmap(norm(prob))
            edgecolor = correct_edge if is_correct else incorrect_edge
            pad = 0.02
            rect = patches.Rectangle(
                (x + pad, y + pad),
                1 - 2*pad,
                1 - 2*pad,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2.5
            )
            ax.add_patch(rect)
            inner = patches.Rectangle(
                (x + 0.015, y + 0.015), 0.97, 0.97,
                facecolor='none',
                edgecolor=(1, 1, 1, 0.35),
                linewidth=0.8
            )
            ax.add_patch(inner)
            txt_color = 'black'
            ax.text(
                x + 0.5, y + 0.24,
                mapping[true_label],
                ha='center', va='center',
                fontsize=14, color=txt_color
            )
            ax.text(
                x + 0.5, y + 0.50,
                mapping[pred_label],
                ha='center', va='center',
                fontsize=14, color=txt_color, fontweight='semibold'
            )
            ax.text(
                x + 0.5, y + 0.80,
                f'{prob:.1f}%',
                ha='center', va='center',
                fontsize=14, color=txt_color
            )
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(regions, fontsize=16)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels([diag_labels.get(d, d) for d in diag_order], fontsize=16)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in range(ncols + 1):
        ax.axvline(x, color='white', lw=1.2, zorder=0)
    for y in range(nrows + 1):
        ax.axhline(y, color='white', lw=1.2, zorder=0)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Prediction probability (%)', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    edge_handles = [
        patches.Patch(facecolor='white', edgecolor=correct_edge, linewidth=3, label='Correct'),
        patches.Patch(facecolor='white', edgecolor=incorrect_edge, linewidth=3, label='Incorrect'),
        plt.Line2D([0], [0], marker='_', color='k', lw=3, markersize=3, markeredgewidth=3, label='Prediction')
    ]
    fig.legend(handles=edge_handles, loc='lower center', ncol=3,
               frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.05))
    plt.title('Prediction summary 2025', fontsize=20, pad=35)
    plt.tight_layout()
    plt.show()
    
    
regions = ['sah-w', 'sah-e', 'sud-sah-w', 'sud-sah-e', 'sud-w', 'sud-e', 'guin-w', 'guin-e']
data_dict = {
    'Onset': {
        'sah-w':   {'true': 'early', 'pred': 'clim', 'prob': 43},
        'sah-e':   {'true': 'early', 'pred': 'clim', 'prob': 43},
        'sud-sah-w': {'true': 'early', 'pred': 'early', 'prob': 42},
        'sud-sah-e': {'true': 'late', 'pred': 'early', 'prob': 43},
        'sud-w':   {'true': 'late', 'pred': 'clim', 'prob': 43},
        'sud-e':   {'true': 'late', 'pred': 'clim', 'prob': 43},
        'guin-w':  {'true': 'late', 'pred': 'early', 'prob': 47},
        'guin-e':  {'true': 'late', 'pred': 'early', 'prob': 47},
    },
    'False onset': {
        'sah-w':   {'true': 'yes', 'pred': 'yes', 'prob': 82},
        'sah-e':   {'true': 'yes', 'pred': 'yes', 'prob': 82},
        'sud-sah-w': {'true': 'no', 'pred': 'yes', 'prob': 63},
        'sud-sah-e': {'true': 'yes', 'pred': 'yes', 'prob': 61},
        'sud-w':   {'true': 'no', 'pred': 'no', 'prob': 54},
        'sud-e':   {'true': 'no', 'pred': 'no', 'prob': 54},
        'guin-w':  {'true': 'no', 'pred': 'no', 'prob': 64},
        'guin-e':  {'true': 'no', 'pred': 'no', 'prob': 66},
    },
    'RS1 anomaly': {
        'sah-w':   {'true': 'near-normal', 'pred': 'above-normal', 'prob': 48},
        'sah-e':   {'true': 'near-normal', 'pred': 'above-normal', 'prob': 48},
        'sud-sah-w': {'true': 'near-normal', 'pred': 'above-normal', 'prob': 50},
        'sud-sah-e': {'true': 'below-normal', 'pred': 'above-normal', 'prob': 50},
        'sud-w':   {'true': 'below-normal', 'pred': 'above-normal', 'prob': 52},
        'sud-e':   {'true': 'below-normal', 'pred': 'above-normal', 'prob': 52},
        'guin-w':  {'true': 'below-normal', 'pred': 'above-normal', 'prob': 56},
        'guin-e':  {'true': 'below-normal', 'pred': 'above-normal', 'prob': 56},
    },
    'Dry spell anomaly': {
        'sah-w':   {'true': 'near-normal', 'pred': 'near-normal', 'prob': 42},
        'sah-e':   {'true': 'below-normal', 'pred': 'near-normal', 'prob': 43},
        'sud-sah-w': {'true': 'below-normal', 'pred': 'near-normal', 'prob': 40},
        'sud-sah-e': {'true': 'near-normal', 'pred': 'near-normal', 'prob': 41},
        'sud-w':   {'true': 'below-normal', 'pred': 'above-normal', 'prob': 37},
        'sud-e':   {'true': 'near-normal', 'pred': 'above-normal', 'prob': 39},
        'guin-w':  {'true': 'below-normal', 'pred': 'below-normal', 'prob': 40},
        'guin-e':  {'true': 'below-normal', 'pred': 'above-normal', 'prob': 37},
    },
    'Long dry spells': {
        'sah-w':   {'true': 'low risk', 'pred': 'elevated risk', 'prob': 68},
        'sah-e':   {'true': 'low risk', 'pred': 'elevated risk', 'prob': 65},
        'sud-sah-w': {'true': 'low risk', 'pred': 'elevated risk', 'prob': 71},
        'sud-sah-e': {'true': 'low risk', 'pred': 'elevated risk', 'prob': 68},
        'sud-w':   {'true': 'low risk', 'pred': 'elevated risk', 'prob': 71},
        'sud-e':   {'true': 'elevated risk', 'pred': 'elevated risk', 'prob': 65},
        'guin-w':  {'true': 'low risk', 'pred': 'elevated risk', 'prob': 71},
        'guin-e':  {'true': 'low risk', 'pred': 'elevated risk', 'prob': 64},
    },
    'LDS duration': {
        'guin-w':  {'true': 'above-normal', 'pred': 'below-normal', 'prob': 54},
        'guin-e':  {'true': 'above-normal', 'pred': 'below-normal', 'prob': 54},
    }
}

plot_prediction_matrix(data_dict, regions)


#%% -- issue-date prediction progression --

from matplotlib.lines import Line2D
df = pd.read_csv('z.models/forecast_2025_output/forecast_2025_results.csv')
REGION_ORDER = ['sah-w','sah-e','sud-sah-w','sud-sah-e',
                'sud-w','sud-e','guin-w','guin-e']
ISSUE_ORDER = ['NDJ','NDJ-DJF','NDJ-JFM','NDJ-FMA','NDJ-MAM','NDJ-AMJ']
ISSUE_SHORT  = ['NDJ','NDJ-DJF','NDJ-JFM','NDJ-FMA','NDJ-MAM','NDJ-AMJ']
DIAG_ORDER  = ['doy_os', 'false_onset', '1stmonth_anom','dry_spell_anom',
                'dry_spell_nlong','lds_dur']
DIAG_LABELS = {
    'doy_os':         'Onset',
    'false_onset':    'False onset',
    '1stmonth_anom':  'RS1 anomaly',
    'dry_spell_anom': 'Dry spell anomaly',
    'dry_spell_nlong':'Long dry spells',
    'lds_dur':        'LDS duration',
}

LABEL_COLOR = {
    'early':        'orange',
    'clim':         'darkgrey',
    'late':         'mediumpurple',
    'above-normal': 'mediumpurple',
    'near-normal':  'darkgrey',
    'below-normal': 'orange',
    'low risk':     'darkgrey',
    'elevated risk':'orangered',
    'no':           'darkgrey',
    'yes':          'orangered',
}

def get_pred_prob(row):
    """Return probability of the predicted class."""
    label = row['pred_class_label']
    mapping = {
        row['class0_label']: row['p_class0'],
        row['class1_label']: row['p_class1'],
        row['class2_label']: row['p_class2'],
    }
    return mapping.get(label, np.nan)

df['pred_prob'] = df.apply(get_pred_prob, axis=1)
n_diag   = len(DIAG_ORDER)
n_region = len(REGION_ORDER)
n_issue  = len(ISSUE_ORDER)
fig, axes = plt.subplots(n_region, n_diag, figsize=(15,7))
plt.style.use('ggplot')
for c, diag in enumerate(DIAG_ORDER):
    sub_diag = df[df['diagnostic'] == diag]

    for r, region in enumerate(REGION_ORDER):
        ax = axes[r, c]
        sub = sub_diag[sub_diag['region'] == region]
        ax.set_xlim(-0.5, n_issue - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])

        for xi in range(n_issue):
            ax.axvline(xi, color='#E8E8E8', lw=0.5, zorder=0)
        if sub.empty:
            ax.set_facecolor('#F5F5F5')
            if r == 0:
                ax.set_title(DIAG_LABELS[diag], fontsize=16, pad=4)
            continue

        ax.set_facecolor('white')
        for xi, issue in enumerate(ISSUE_ORDER):
            row = sub[sub['issue_label'] == issue]
            if row.empty:
                continue
            row = row.iloc[0]
            label = row['pred_class_label']
            prob  = row['pred_prob']
            color = LABEL_COLOR.get(label, '#AAAAAA')

            # Circle size scales with forecast probability
            ms = 60 + (prob - 33) / (100 - 33) * 240
            ms = max(40, min(300, ms))
            ax.scatter(xi, 0.5, s=ms, color=color, zorder=3, linewidths=0)

        if r == n_region - 1:
            ax.set_xticks(range(n_issue))
            ax.set_xticklabels(
                [ISSUE_SHORT[i] for i in range(n_issue)],
                fontsize=12, rotation=90)
            ax.tick_params(axis='x', length=0, pad=2)
        if r == 0:
            ax.set_title(DIAG_LABELS[diag], fontsize=16, pad=4)
        if c == 0:
            ax.set_ylabel(REGION_ORDER[r], fontsize=16,
                          rotation=0, labelpad=45, va='center')
        ax.spines[['top','right','bottom','left']].set_visible(False)

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange',
           markersize=8,  label='early / below-normal'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='darkgrey',
           markersize=8,  label='clim / near-normal / low risk / no FO'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='mediumpurple',
           markersize=8,  label='late / above-normal'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orangered',
           markersize=8,  label='elevated risk / FO')
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=2, fontsize=14, frameon=False,
           bbox_to_anchor=(0.5, -0.15),
           handletextpad=0.4, columnspacing=1.2)
fig.suptitle('2025 forecast by issue date', fontsize=20, y=0.98)
plt.show()
