#%% -- Run the issue-date models against 2025 index data --
import os
import numpy as np
import pandas as pd

OUTPUT_DIR = 'z.models/forecast_2025_output'
FORECAST_CSV_MAP = {
    'NDJ':     'z.data_2025/forecast_NDJ.csv',
    'NDJ-DJF': 'z.data_2025/forecast_NDJ-DJF.csv',
    'NDJ-JFM': 'z.data_2025/forecast_NDJ-JFM.csv',
    'NDJ-FMA': 'z.data_2025/forecast_NDJ-FMA.csv',
    'NDJ-MAM': 'z.data_2025/forecast_NDJ-MAM.csv',
    'NDJ-AMJ': 'z.data_2025/forecast_NDJ-AMJ.csv',
}
FORECAST_YEAR = 2025

# helper functions
def load_forecast_csv(path, year=FORECAST_YEAR):
    """
    Load one issue-date CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Forecast CSV not found: {path}"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    row = df[df['year'] == year]
    if row.empty:
        raise ValueError(f"No row for year={year} found in {path}")
    return row.reset_index(drop=True)


def build_X_forecast(df_row, feat_cols_issue, region,
                      northern1_regions, northern2_regions,
                      southern1_regions, southern2_regions,
                      get_valid_seasons_fn, diag,
                      available_seasons, REGION_POSITION):
    """
    Build a feature array for a single forecast row.
    """
    n_feat = len(feat_cols_issue)
    x = np.zeros(n_feat)
    for fi, col in enumerate(feat_cols_issue):
        if col in df_row.columns:
            val = df_row[col].values[0]
            x[fi] = 0.0 if pd.isna(val) else float(val)

    # Apply the same region × season mask used during training
    loc = ('northern1' if region in northern1_regions else
           'northern2' if region in northern2_regions else
           'southern1' if region in southern1_regions else
           'southern2' if region in southern2_regions else None)
    if loc is not None:
        allowed_reg = get_valid_seasons_fn(diag).get(loc, [])
        effective   = [s for s in allowed_reg if s in available_seasons]
        mask = np.array([any(c.endswith('_' + s) for s in effective)
                         for c in feat_cols_issue])
        x = x * mask

    # Append position features
    lat_band = REGION_POSITION[region][0]
    lon_band = REGION_POSITION[region][1]
    X = np.hstack([x, [lat_band, lon_band]])[np.newaxis, :]  # (1, n_feat+2)
    return X


# Main forecasting loop
def run_forecast_2025(all_clf_issue,
                      diagnostics_used,
                      configs,
                      diag_class,
                      regions,
                      region_names,
                      northern1_regions, northern2_regions,
                      southern1_regions, southern2_regions,
                      REGION_POSITION,
                      get_valid_seasons,
                      get_class_labels,
                      clim_means_per_diag=None,
                      output_dir=OUTPUT_DIR,
                      csv_map=FORECAST_CSV_MAP,
                      forecast_year=FORECAST_YEAR):
    """
    Run the forecast using the issue-dates models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def get_issue_date_range(diag, get_valid_seasons, ISSUE_DATES, seasons):
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

    ISSUE_DATES = {
        'NDJ':     ['NDJ'],
        'NDJ-DJF': ['NDJ', 'DJF'],
        'NDJ-JFM': ['NDJ', 'DJF', 'JFM'],
        'NDJ-FMA': ['NDJ', 'DJF', 'JFM', 'FMA'],
        'NDJ-MAM': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM'],
        'NDJ-AMJ': ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ'],
        }
    seasons = ['NDJ', 'DJF', 'JFM', 'FMA', 'MAM',
               'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO']

    records = []

    for c, config in enumerate(configs):
        label = config['label']
        diag  = diagnostics_used[c]
        label_map = get_class_labels(diag, diag_class)
        
        valid_issue_labels = get_issue_date_range(diag, get_valid_seasons, ISSUE_DATES, seasons)
        
        for issue_label, available_seasons in ISSUE_DATES.items():
            if issue_label not in valid_issue_labels:
                continue
            print(f'\n── Forecasting: {issue_label} ({diag}) ──')
            key = f'{label}_{issue_label}'
            csv_path = csv_map.get(issue_label)
            if csv_path is None:
                print(f'  [{issue_label}] no CSV path configured, skipping.')
                continue
            try:
                df_row = load_forecast_csv(csv_path, year=forecast_year)
            except (FileNotFoundError, ValueError) as e:
                print(f'  [{issue_label}] {e}')
                continue

            bundle          = all_clf_issue[key]
            clf             = bundle['clf']
            feat_cols_issue = bundle['feat_cols']
            bal_acc_loo     = bundle.get('bal_acc_loo', np.nan)
            clim_means      = bundle.get('clim_means', {})
            regs            = bundle.get('regs', {})
            print(f'  [{issue_label}]  {len(feat_cols_issue)} features  '
                  f'LOO bal_acc={bal_acc_loo:.3f}')

            for region in regions:
                if region not in bundle.get('regions_all', regions):
                    continue
                
                loc = ('northern1' if region in northern1_regions else
                       'northern2' if region in northern2_regions else
                       'southern1' if region in southern1_regions else
                       'southern2' if region in southern2_regions else None)
                if loc is not None:
                    allowed_reg = get_valid_seasons(diag).get(loc, [])
                    if not all(s in allowed_reg for s in available_seasons):
                        print(f'    Skipping {region}: {available_seasons} exceeds '
                              f'allowed seasons {allowed_reg}')
                        continue

                X_fc = build_X_forecast(
                    df_row, feat_cols_issue, region,
                    northern1_regions, northern2_regions,
                    southern1_regions, southern2_regions,
                    get_valid_seasons, diag,
                    available_seasons, REGION_POSITION
                )
                if region == 'guin-w' and issue_label == 'NDJ-AMJ':
                    print(list(X_fc))

                #  Classification
                pred_class_idx  = int(clf.predict(X_fc)[0])
                pred_class_lbl  = label_map.get(pred_class_idx, '')
                pred_probas     = clf.predict_proba(X_fc)[0]

                p_arr = np.full(3, np.nan)
                for ci, cls in enumerate(clf.classes_):
                    if cls in (0, 1, 2):
                        p_arr[int(cls)] = pred_probas[ci]

                # Value prediction
                pred_value  = np.nan
                q25, q75    = np.nan, np.nan
                days_offset = np.nan
                clim_mean   = clim_means.get(region, np.nan)

                if diag in diag_class.get('doy_diags', []) or diag == 'lds_dur':
                    region_regs = regs.get(region, {})
                    if region_regs:
                        if pred_class_idx in region_regs:
                            reg = region_regs[pred_class_idx]
                            leaf_preds = np.array([t.predict(X_fc)[0] for t in reg.estimators_])
                            pred_value = np.mean(leaf_preds)
                            q25, q75 = np.percentile(leaf_preds, 25), np.percentile(leaf_preds, 75)

                        days_offset = pred_value - clim_mean \
                            if not np.isnan(clim_mean) else np.nan

                records.append({
                    'year':             forecast_year,
                    'diagnostic':       diag,
                    'model_label':      label,
                    'region':           region,
                    'region_name':      region_names[regions.index(region)],
                    'issue_label':      issue_label,
                    'available_seasons': '|'.join(available_seasons),
                    'n_features':       len(feat_cols_issue),
                    'bal_acc_loo':      round(bal_acc_loo, 3),
                    'pred_class':       pred_class_idx,
                    'pred_class_label': pred_class_lbl,
                    'p_class0':         round(float(p_arr[0]) * 100, 1) if not np.isnan(p_arr[0]) else np.nan,
                    'p_class1':         round(float(p_arr[1]) * 100, 1) if not np.isnan(p_arr[1]) else np.nan,
                    'p_class2':         round(float(p_arr[2]) * 100, 1) if not np.isnan(p_arr[2]) else np.nan,
                    'class0_label':     label_map.get(0, ''),
                    'class1_label':     label_map.get(1, ''),
                    'class2_label':     label_map.get(2, ''),
                    'pred_value':       round(pred_value, 1)  if not np.isnan(pred_value) else np.nan,
                    'pred_q25':         q25 if not np.isnan(q25) else np.nan,
                    'pred_q75':         q75 if not np.isnan(q75) else np.nan,
                    'days_offset':      round(days_offset, 1) if not np.isnan(days_offset) else np.nan,
                    'clim_mean':        round(clim_mean, 1)   if not np.isnan(clim_mean) else np.nan,
                })

    df_results = pd.DataFrame(records)

    # Save forecast
    csv_out = os.path.join(output_dir, 'forecast_2025_results1.csv')
    df_results.to_csv(csv_out, index=False)
    
    summary_rows = []
    for (diag_s, region_s, issue_s), grp in df_results.groupby(
            ['diagnostic', 'region', 'issue_label']):
        row = grp.iloc[0]
        summary_rows.append({
            'diagnostic':       diag_s,
            'region':           region_s,
            'issue_label':      issue_s,
            'pred_class_label': row['pred_class_label'],
            'p_dominant':       max(
                row['p_class0'] if not pd.isna(row['p_class0']) else 0,
                row['p_class1'] if not pd.isna(row['p_class1']) else 0,
                row['p_class2'] if not pd.isna(row['p_class2']) else 0,
            ),
            'bal_acc_loo':  row['bal_acc_loo'],
            'days_offset':  row.get('days_offset', np.nan),
        })
    df_summary = pd.DataFrame(summary_rows)
    sum_out = os.path.join(output_dir, 'forecast_2025_summary.csv')
    df_summary.to_csv(sum_out, index=False)

    return df_results


def load_results(output_dir=OUTPUT_DIR):
    """
    Load saved forecast results for figure making.
    """
    path = os.path.join(output_dir, 'forecast_2025_results.csv')
    df = pd.read_csv(path)

    by_diag   = {d: g.reset_index(drop=True)
                 for d, g in df.groupby('diagnostic')}
    by_region = {r: g.reset_index(drop=True)
                 for r, g in df.groupby('region')}

    print(f'Loaded {len(df)} forecast records from {path}')
    print(f'  Diagnostics : {sorted(by_diag.keys())}')
    print(f'  Regions     : {sorted(by_region.keys())}')
    return df, by_diag, by_region


