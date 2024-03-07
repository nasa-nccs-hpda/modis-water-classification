#!/usr/bin/env python
# coding: utf-8

# #  MODIS Water Cluster Training
# 
# Version: 0.1.0
# 
# Date modified: 05.01.2023
# 
# Modified by: Amanda Burke

# In[1]:


import csv
import datetime
import glob
import time
import joblib
import numpy as np
import os
import math 
import pandas as pd
from pathlib import Path   
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split 

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')


import optuna
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold

# GPU-based frameworks
import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRFC


# In[2]:


GPU = False


# In[3]:

outlier_threshold = 10000
MODEL = 'rf'
TEST_RATIO = 0.2
RANDOM_STATE = 42
LABEL_NAME = 'water'
if GPU is False:
    DATA_TYPE = np.int16
else: 
    DATA_TYPE = cp.float32
FRAC_LAND=0.5


# In[4]:


##############################
# VERSION 4.2.1 (targeted 500k points)
#TILE_IN = 'Golden'#v4.2.1
#DATA_VERSION='v4.2.1'
#offsets_indexes = ['x_offset', 'y_offset', 'year', 'julian_day','tileID']
#############################

##############################
#VERSION 2.0.1 (5 million points)
TILE_IN = 'GLOBAL'#v2.0.1
DATA_VERSION='v2.0.1'
offsets_indexes = ['x_offset', 'y_offset', 'year', 'julian_day']
##############################

# # #############################
# #VERSION 0.0.0 (2billion data points)
# TILE_IN = 'cleaned'#v2.0.1
# DATA_VERSION='AGU'
# offsets_indexes = []
# # ##############################

training_data_basepath = f'/explore/nobackup/projects/ilab/data/MODIS/MODIS_WATER_ML/training_data/{DATA_VERSION}'
glob_string = os.path.join(training_data_basepath,'MOD*{}*.parquet.gzip'.format(TILE_IN))
data_paths = sorted([fv for fv in glob.glob(glob_string)])

#Only want the one with 4.2.0 because the other file doesnt work. 
print(data_paths)
data_path = data_paths[0]
print(data_path)

colsToDrop = [
    # 'sur_refl_b01_1',
    # 'sur_refl_b02_1',
    'sur_refl_b03_1',
    'sur_refl_b04_1','sur_refl_b05_1','sur_refl_b06_1',
    # 'sur_refl_b07_1',
    # 'ndvi',
    'ndwi1','ndwi2'
        ]

colsToDropTraining = colsToDrop.copy()
colsToDropTraining.extend(offsets_indexes)
v_names = ['sur_refl_b01_1','sur_refl_b02_1','sur_refl_b03_1',
           'sur_refl_b04_1','sur_refl_b05_1','sur_refl_b06_1',
           'sur_refl_b07_1','ndvi','ndwi1','ndwi2']


# In[5]:


def load_cpu_data(fpath, colsToDrop, yCol='water', testSize=0.2, randomState=42, 
            dataType=np.int16, cpu=True, splitXY=False, trainTestSplit=False,
            applyLog=False, imbalance=False, frac=0.1, land=False, multi=False, 
            multisample=1000000):
    """
    Simple helper function for loading data to be used by models
    :param fpath: Path to the data to be ingested.
    :param dataType: Data type to convert ingested data to.
    :param colsToDrop: Columns which are not necessary, from which to drop.
    :param testSize: Ration to
    """
    if multi:
        all_dfs = [pd.read_csv(path_) for path_ in fpath]
        df = pd.concat(all_dfs).sample(n=multisample, random_state=randomState)
        print('DF length: {}'.format(len(df.index)))
    else:   
        df = pd.read_parquet(fpath) if '.parquet' in fpath else pd.read_csv(fpath)
    df = df[df['sur_refl_b01_1'] + df['sur_refl_b02_1'] != 0]
    df = df[df['sur_refl_b07_1'] + df['sur_refl_b02_1'] != 0]
    df = df[df['sur_refl_b06_1'] + df['sur_refl_b02_1'] != 0]

    df = df.drop(columns=colsToDrop)
    cleanedDF = df[~df.isin([np.NaN, np.inf, -np.inf]).any(1)].dropna(axis=0).astype(dataType)
    if applyLog:
        for col in cleanedDF.drop([yCol], axis=1).columns:
            print('Applying log1p func to {}'.format(col))
            cleanedDF[col] = np.log1p(cleanedDF[col])
        cleanedDF = cleanedDF[~cleanedDF.isin([np.NaN, np.inf, -np.inf]).any(1)].dropna(axis=0)
    df = None
    if imbalance:
        if land:
            print('Imbalancing data, sampling {} from water'.format(frac))
        else:
            print(f'Imbalancing data, sampling {frac} from land, {1-frac} from water')
        groupedDF = cleanedDF.groupby('water')
        dfs = [groupedDF.get_group(y) for y in groupedDF.groups]
        sampledDF = dfs[1].sample(frac=frac)if land else dfs[0].sample(frac=frac)
        concatDF = sampledDF.append(dfs[0]) if land else sampledDF.append(dfs[1])
        concatDF = concatDF.sample(frac=1)
        concatDF = concatDF.reset_index()
        cleanedDF = concatDF.drop(columns=['index'])
    if not splitXY:
        return cleanedDF
    X = cleanedDF.drop([yCol], axis=1).astype(dataType)
    y = cleanedDF[yCol].astype(dataType)
    if trainTestSplit:
        return train_test_split(X, y, test_size=TEST_RATIO)
    else:
        return X, y


# In[6]:


def load_gpu_data(fpath, colsToDrop, yCol='water', testSize=0.2, randomState=42, 
            dataType=cp.float32, cpu=False, splitXY=True, trainTestSplit=True,
            applyLog=False, imbalance=False, frac=0.1, land=False, multi=False, 
            multisample=1000000):
    """
    Simple helper function for loading data to be used by models
    :param fpath: Path to the data to be ingested.
    :param dataType: Data type to convert ingested data to.
    :param colsToDrop: Columns which are not necessary, from which to drop.
    :param testSize: Ration to
    """
    if multi:
        all_dfs = [pd.read_csv(path_) for path_ in fpath]
        df = pd.concat(all_dfs).sample(n=multisample, random_state=randomState)
        print('DF length: {}'.format(len(df.index)))
    else:   
        df = pd.read_parquet(fpath) if '.parquet' in fpath else pd.read_csv(fpath)
    df = df[df['sur_refl_b01_1'] + df['sur_refl_b02_1'] != 0]
    df = df[df['sur_refl_b07_1'] + df['sur_refl_b02_1'] != 0]
    df = df[df['sur_refl_b06_1'] + df['sur_refl_b02_1'] != 0]
    df = df.drop(columns=colsToDrop)
    cleanedDF = df[~df.isin([np.NaN, np.inf, -np.inf]).any(1)].dropna(axis=0).astype(dataType)
    cleanedDF = cudf.from_pandas(cleanedDF) if not cpu else cleanedDF
    if applyLog:
        for col in cleanedDF.drop([yCol], axis=1).columns:
            print('Applying log1p func to {}'.format(col))
            cleanedDF[col] = np.log1p(cleanedDF[col])
        cleanedDF = cleanedDF[~cleanedDF.isin([np.NaN, np.inf, -np.inf]).any(1)].dropna(axis=0)
    df = None
    if imbalance:
        if land:
            print('Imbalancing data, sampling {} from water'.format(frac))
        else:
            print('Imbalancing data, sampling {} from land'.format(frac))
        groupedDF = cleanedDF.groupby('water')
        dfs = [groupedDF.get_group(y) for y in groupedDF.groups]
        sampledDF = dfs[1].sample(frac=frac)if land else dfs[0].sample(frac=frac)
        concatDF = sampledDF.append(dfs[0]) if land else sampledDF.append(dfs[1])
        concatDF = concatDF.sample(frac=1)
        concatDF = concatDF.reset_index()
        cleanedDF = concatDF.drop(columns=['index'])
    if not splitXY:
        return cleanedDF
    X = cleanedDF.drop([yCol], axis=1).astype(dataType)
    y = cleanedDF[yCol].astype(dataType)
    cleanedX = cleanedDF.drop([yCol], axis=1).astype(dataType)
    cleanedy = cleanedDF[yCol].astype(dataType)
    if trainTestSplit:
        return train_test_split(cleanedX, cleanedy, test_size=TEST_RATIO)
    else:
        return cleanedX, cleanedy


# In[7]:





# ### Input data

# In[8]:


colsToDrop


# In[9]:

start_time = time.time()
load_data_params = {'fpath':data_path,'colsToDrop':colsToDropTraining,'splitXY':True,'imbalance':False,'trainTestSplit':True}
if GPU is False: 
    X, X_test, y, y_test = load_cpu_data(**load_data_params)
else:
    X, X_test, y, y_test = load_gpu_data(**load_data_params)
    X_cpu,y_cpu,y_cpu_test,X_cpu_test = load_cpu_data(**load_data_params)
    
    ###############
    # Will need to change this part when GPU is true, so 
    #the final skRF works with X_cpu NOT X
    X_cpu_no_outlier = X_cpu.copy()
    y_cpu_no_outlier = y_cpu.copy()
    cpu_outlier_inds = X_cpu[X_cpu['sur_refl_b01_1'] > outlier_threshold].index
    X_cpu_no_outlier.drop(cpu_outlier_inds, inplace=True)
    y_cpu_no_outlier.drop(cpu_outlier_inds, inplace=True)
    ###############

print(f'data shape: {X.shape}, {y.shape}')
elapsed_time = time.time() - start_time
print(f'Load in Data Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')


# In[10]:


print(X)
X_no_outlier = X.copy()
y_no_outlier = y.copy()
outlier_inds = X[X['sur_refl_b01_1'] > outlier_threshold].index
X_no_outlier.drop(outlier_inds, inplace=True)
y_no_outlier.drop(outlier_inds, inplace=True)


print(f' Removing {len(X) - len(X_no_outlier)} outliers')

_ = [print(column) for column in X.columns]

# # Random forest

# In[12]:


def cpu_rf_objective(trial):
    list_trees = [75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
    max_depth = [5, 10, 30, 50, 80, 90, 100, 110]
    min_samples_leaf = [1, 2, 3, 4, 5]
    min_samples_split = [2, 4, 8, 10]
    bootstrap = [True, False]
    max_features = ['auto', 'sqrt', 'log2']
    
    param = {'n_estimators': trial.suggest_categorical('n_estimators', list_trees), 
       'max_depth':trial.suggest_categorical('max_depth', max_depth), 
       'min_samples_split':trial.suggest_categorical('min_samples_split', min_samples_split), 
       'min_samples_leaf':trial.suggest_categorical('min_samples_leaf', min_samples_leaf), 
       'bootstrap': trial.suggest_categorical('bootstrap', bootstrap),
       'criterion':'gini', 
       #'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 1e-8, 1.0, log=True), 
       'max_features':trial.suggest_categorical('max_features', max_features), 
       'max_leaf_nodes':None, 
       'min_impurity_decrease':0.0, 
       'oob_score':False, 
       'n_jobs':-1, 
       # 'random_state':42, 
       'verbose':0, 
       'warm_start':False, 
       'class_weight':None, 
       'ccp_alpha':0.0, 
       'max_samples':None
        }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X_no_outlier,y_no_outlier)):
        X_train, X_val = X_no_outlier.iloc[train_idx], X_no_outlier.iloc[val_idx]
        y_train, y_val = y_no_outlier.iloc[train_idx],y_no_outlier.iloc[val_idx]

        model = skRF(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        cv_scores[idx] = f1_score(y_val, preds)
        if cv_scores[idx] == 0.0:
            print('Pruning because of 0.0 score.')
            return 0.0
        print('Fold {}: {}'.format(idx, cv_scores[idx]))
    return np.mean(cv_scores)

search_space={
    "n_estimators": [75, 100, 125, 150, 175, 200, 250, 300, 400, 500],
    "max_depth" : [5,10, 30, 50, 80, 90, 100, 110],
    "min_samples_leaf" : [1, 2, 3, 4, 5],
    "min_samples_split" : [2, 4, 8, 10],
    "bootstrap" : [True, False],
    "max_features" : ['auto', 'sqrt', 'log2'],
    
}
TREES_AND_DEPTH_ONLY = False
GRID_SEARCH = True


# In[13]:


def gpu_rf_objective(trial):
    list_trees = [75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
    max_depth = [5, 10, 30, 50, 80, 90, 100, 110]
    min_samples_leaf = [1, 2, 3, 4, 5]
    min_samples_split = [2, 4, 8, 10]
    bootstrap = [True, False]
    max_features = ['auto', 'sqrt', 'log2']
    
    param = {'n_estimators': trial.suggest_categorical('n_estimators', list_trees), 
        'max_depth':trial.suggest_categorical('max_depth', max_depth), 
        'min_samples_split':trial.suggest_categorical('min_samples_split', min_samples_split), 
        'min_samples_leaf':trial.suggest_categorical('min_samples_leaf', min_samples_leaf), 
        'max_features':trial.suggest_categorical('max_features', max_features), 
            }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #######################
    # HERE IS WHERE TO CHANGE THE X,Y DATASET USED FOR TRAINING
    #######################
   
    cv_scores = np.empty(5)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X_no_outlier.to_pandas(),y_no_outlier.to_pandas())):
        X_train, X_val = X_no_outlier.iloc[train_idx], X_no_outlier.iloc[val_idx]
        y_train, y_val = y_no_outlier.iloc[train_idx], y_no_outlier.iloc[val_idx]
        
        model = cuRFC(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        cv_scores[idx] = f1_score(y_val.to_numpy(), preds.to_numpy())
        del model, preds
        if cv_scores[idx] == 0.0:
            print('Pruning because of 0.0 score.')
            return 0.0
        print('Fold {}: {}'.format(idx, cv_scores[idx]))
    return np.mean(cv_scores)
    
search_space={
    "n_estimators": [75, 100, 125, 150, 175, 200, 250, 300, 400, 500],
    "max_depth" : [5,10, 30, 50, 80, 90, 100, 110],
    "min_samples_leaf" : [1, 2, 3, 4, 5],
    "min_samples_split" : [2, 4, 8, 10],
    "bootstrap" : [True, False],
    "max_features" : ['auto', 'sqrt', 'log2'],
    
}
TREES_AND_DEPTH_ONLY = False
GRID_SEARCH = True


# In[14]:
start_time = time.time()
optuna.logging.set_verbosity(optuna.logging.INFO)
if GRID_SEARCH:
    study = optuna.create_study(study_name='RF Tuning Grid Search', 
                                direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space))
else:
    study = optuna.create_study(study_name='RF Tuning',direction='maximize')
if GPU is False:
    study.optimize(cpu_rf_objective, n_trials=25, timeout=30*600)
else: 
    study.optimize(gpu_rf_objective, n_trials=25, timeout=30*600)
elapsed_time = time.time() - start_time
print(f'Tuning time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')

# #### Training and output best model

# In[ ]:


trials = study.best_trials            
max_trial_score = max([trial.values[0] for trial in trials])
max_trial_params = [trial.params for trial in trials 
                        if trial.values[0] == max_trial_score][0]
max_trial_params['n_jobs'] = -1
score_print = int(np.round(max_trial_score,4)*1000)
print(max_trial_score)
print(score_print)


# In[ ]:

hyperparameters = max_trial_params
hyperparameters['n_jobs'] = -1
print('Using these params:')
print(hyperparameters)
tuned_classifier = skRF(**hyperparameters)

# In[ ]:

start_time = time.time()
tuned_classifier.fit(X,y)
elapsed_time = time.time() - start_time
print(f'Full Train time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')

# # save the model to disk
import pickle
filename = f'rfa_models/MODIS_RFA_v201_Total_no-outlier-pts_MaxScore{score_print}.pkl'
print(filename)
pickle.dump(tuned_classifier, open(filename, 'wb'))

