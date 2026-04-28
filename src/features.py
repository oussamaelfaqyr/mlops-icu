import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import timedelta

def compute_risk_features(series, vital_name=None):
    """
    Computes clinical risk features for a vital sign series.
    """
    # Normal centers for deviation calculation
    normals = {
        'heart_rate': 80,
        'spo2': 98,
        'resp_rate': 16,
        'temp': 37,
        'mean_bp': 90
    }
    
    if series.empty:
        return {
            'mean': 0, 'std': 0, 'trend': 0, 'dev_from_normal': 0
        }
    
    mean_val = series.mean()
    res = {
        'mean': mean_val,
        'std': series.std() if len(series) > 1 else 0,
        'trend': np.polyfit(np.arange(len(series)), series.values, 1)[0] if len(series) > 1 else 0
    }
    
    if vital_name in normals:
        res['dev_from_normal'] = mean_val - normals[vital_name]
    else:
        res['dev_from_normal'] = 0
        
    return res

def generate_sliding_window_features(vitals_path, stays_path, obs_hours=12, gap_hours=6, pred_hours=6):
    print(f"Generating features with horizon: {gap_hours}-{gap_hours+pred_hours}h...")
    df_vitals = pd.read_csv(vitals_path)
    df_stays = pd.read_csv(stays_path)
    
    df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime'])
    df_stays['intime'] = pd.to_datetime(df_stays['intime'])
    
    vital_names = ['heart_rate', 'spo2', 'resp_rate', 'temp', 'mean_bp']
    
    dataset = []
    
    for stay_id, group in df_vitals.groupby('stay_id'):
        group = group.sort_values('charttime')
        start_time = group['charttime'].min()
        end_time = group['charttime'].max()
        
        # Sliding window logic
        current_time = start_time + timedelta(hours=obs_hours)
        
        while current_time + timedelta(hours=gap_hours + pred_hours) <= end_time:
            # Observation Window
            obs_window = group[(group['charttime'] >= current_time - timedelta(hours=obs_hours)) & 
                               (group['charttime'] < current_time)]
            
            # Prediction Window (Gap + Prediction)
            pred_start = current_time + timedelta(hours=gap_hours)
            pred_end = pred_start + timedelta(hours=pred_hours)
            pred_window = group[(group['charttime'] >= pred_start) & 
                                (group['charttime'] < pred_end)]
            
            if obs_window.empty or pred_window.empty:
                current_time += timedelta(hours=4) # Step 4 hours
                continue
                
            features = {'stay_id': stay_id, 'timestamp': current_time}
            
            # Feature extraction for each vital
            for v in vital_names:
                v_data = obs_window[obs_window['vital_name'] == v]['valuenum']
                stats = compute_risk_features(v_data, vital_name=v)
                for stat_name, val in stats.items():
                    features[f'{v}_{stat_name}'] = val
            
            # Label: Deterioration in prediction window
            # Defined as: SpO2 < 90 or HR > 120 in the FUTURE window
            is_deteriorated = (pred_window[(pred_window['vital_name'] == 'spo2') & (pred_window['valuenum'] < 90)].shape[0] > 0) or \
                              (pred_window[(pred_window['vital_name'] == 'heart_rate') & (pred_window['valuenum'] > 110)].shape[0] > 0)
            
            features['target'] = int(is_deteriorated)
            dataset.append(features)
            
            current_time += timedelta(hours=4) # Slide by 4 hours
            
    df_final = pd.DataFrame(dataset)
    
    # Merge with static stays for age/gender
    df_final = df_final.merge(df_stays[['stay_id', 'subject_id', 'gender', 'anchor_age']], on='stay_id', how='left')
    df_final['gender'] = df_final['gender'].map({'M': 1, 'F': 0})
    
    return df_final

if __name__ == "__main__":
    DATA_DIR = Path("data")
    df_features = generate_sliding_window_features(
        DATA_DIR / "vitals_cleaned.csv", 
        DATA_DIR / "stays_cleaned.csv"
    )
    df_features.to_csv(DATA_DIR / "features_final.csv", index=False)
    print(f"Generated {len(df_features)} samples. Target distribution:\n{df_features['target'].value_counts()}")
