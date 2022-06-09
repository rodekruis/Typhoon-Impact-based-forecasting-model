#!/usr/bin/env python
# coding: utf-8

# ## Modelling
# 
# This note book explains the different steps in the machine learning model for the binary classfication model. First the model is trained on the full dataset to obtain the optimal features followed by hyper parameter tunning and model performance estimate using Nested Cross Validation.
# 
# * Nested Cross Validation for
#     * Feature selection 
#     * hyper parameter tunning 
# * Performance metrics
# * Baseline Models
# 
# ### Binary Classification
# At the end of this section we will obtain  the optimal Binary Classification models and the performance estimates, 
# for a 10% threshold. Two models are implemented: Random Forest Classifier, XGBoost Classifier. 
# First, the model is trained on the full dataset to obtain the optimal features followed by a model 
# that obtains the performance estimate using Nested Cross Validation.
# 

# In[41]:



import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import os
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import importlib
import os
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    mutual_info_regression,
    f_regression,
    mutual_info_classif,
)
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import xgboost as xgb
import random
import pickle
import openpyxl
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import pickle
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import random
import importlib


# In[42]:



def binary_damage_class(x):
    damage = x[0]   
    if damage > 10:
        value = 1
    else:
        value = 0
    return value


def splitting_train_test(df):

    # To save the train and test sets
    df_train_list = []
    df_test_list = []

    # List of typhoons that are to be used as a test set 
 
    typhoons_with_impact_data=list(np.unique(df.typhoon))

    for typhoon in typhoons_with_impact_data:
        if len(df[df["typhoon"] == typhoon]) >1:
            df_train_list.append(df[df["typhoon"] != typhoon])
            df_test_list.append(df[df["typhoon"] == typhoon])

    return df_train_list, df_test_list

def unweighted_random(y_train, y_test):
    options = y_train.value_counts(normalize=True)
    y_pred = random.choices(population=list(options.index), k=len(y_test))
    return y_pred

def weighted_random(y_train, y_test):
    options = y_train.value_counts()
    y_pred = random.choices(
        population=list(options.index), weights=list(options.values), k=len(y_test)
    )
    return y_pred


# In[43]:


# Setting directory

wor_dir="/home/fbf/"
wor_dir='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/IBF-Typhoon-model/'

os.chdir(wor_dir)

cdir = os.getcwd()


# In[44]:


# Import functions
from models.binary_classification.rf_binary import (rf_binary_features,rf_binary_performance,)
from models.binary_classification.xgb_binary import (xgb_binary_features,xgb_binary_performance,)
from models.regression.rf_regression import (rf_regression_features,rf_regression_performance,)
from models.regression.xgb_regression import (xgb_regression_features,xgb_regression_performance,)



# In[45]:




combined_input_data=pd.read_csv("data/model_input/combined_input_data.csv")

combined_input_data["DAM_binary_dmg"] = combined_input_data[["DAM_perc_dmg"]].apply(binary_damage_class, axis="columns")


combined_input_data =combined_input_data.filter(['typhoon','HAZ_rainfall_Total', 
        'HAZ_rainfall_max_6h',
        'HAZ_rainfall_max_24h',
        'HAZ_v_max',
        'HAZ_dis_track_min',
        'GEN_landslide_per',
        'GEN_stormsurge_per',
        'GEN_Bu_p_inSSA', 
        'GEN_Bu_p_LS', 
        'GEN_Red_per_LSbldg',
        'GEN_Or_per_LSblg', 
        'GEN_Yel_per_LSSAb', 
        'GEN_RED_per_SSAbldg',
        'GEN_OR_per_SSAbldg',
        'GEN_Yellow_per_LSbl',
        'TOP_mean_slope',
        'TOP_mean_elevation_m', 
        'TOP_ruggedness_stdev', 
        'TOP_mean_ruggedness',
        'TOP_slope_stdev', 
        'VUL_poverty_perc',
        'GEN_with_coast',
        'GEN_coast_length', 
        'VUL_Housing_Units',
        'VUL_StrongRoof_StrongWall', 
        'VUL_StrongRoof_LightWall',
        'VUL_StrongRoof_SalvageWall', 
        'VUL_LightRoof_StrongWall',
        'VUL_LightRoof_LightWall', 
        'VUL_LightRoof_SalvageWall',
        'VUL_SalvagedRoof_StrongWall',
        'VUL_SalvagedRoof_LightWall',
        'VUL_SalvagedRoof_SalvageWall', 
        'VUL_vulnerable_groups',
        'VUL_pantawid_pamilya_beneficiary',
        'DAM_binary_dmg'])


# In[46]:


features =['HAZ_rainfall_Total', 
        'HAZ_rainfall_max_6h',
        'HAZ_rainfall_max_24h',
        'HAZ_v_max',
        'HAZ_dis_track_min',
        'GEN_landslide_per',
        'GEN_stormsurge_per',
        'GEN_Bu_p_inSSA', 
        'GEN_Bu_p_LS', 
        'GEN_Red_per_LSbldg',
        'GEN_Or_per_LSblg', 
        'GEN_Yel_per_LSSAb', 
        'GEN_RED_per_SSAbldg',
        'GEN_OR_per_SSAbldg',
        'GEN_Yellow_per_LSbl',
        'TOP_mean_slope',
        'TOP_mean_elevation_m', 
        'TOP_ruggedness_stdev', 
        'TOP_mean_ruggedness',
        'TOP_slope_stdev', 
        'VUL_poverty_perc',
        'GEN_with_coast',
        'GEN_coast_length', 
        'VUL_Housing_Units',
        'VUL_StrongRoof_StrongWall', 
        'VUL_StrongRoof_LightWall',
        'VUL_StrongRoof_SalvageWall', 
        'VUL_LightRoof_StrongWall',
        'VUL_LightRoof_LightWall', 
        'VUL_LightRoof_SalvageWall',
        'VUL_SalvagedRoof_StrongWall',
        'VUL_SalvagedRoof_LightWall',
        'VUL_SalvagedRoof_SalvageWall', 
        'VUL_vulnerable_groups',
        'VUL_pantawid_pamilya_beneficiary']


# ####  Random forest 

# In[19]:


combined_input_data.query("DAM_binary_dmg>0")


# In[30]:



df=combined_input_data.dropna()
 
#combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
X = df[features]
y = df["DAM_binary_dmg"]

# Setting the train and the test sets for obtaining performance estimate
df_train_list, df_test_list = splitting_train_test(df)


# In[31]:


# Setting the random forest search grid
rf_search_space = [
    {
        "estimator__n_estimators": [100, 250],
        "estimator__max_depth": [20, None],
        "estimator__min_samples_split": [2, 8, 10, 15],
        "estimator__min_samples_leaf": [1, 3, 5],
    }
]

# Obtaining the selected features based on the full dataset
selected_features_rf_binary, selected_params_rf_binary_full = rf_binary_features(
    X=X,
    y=y,
    features=features,
    search_space=rf_search_space,
    cv_splits=5,
    class_weight="balanced",
    min_features_to_select=1,
    GS_score="f1",
    GS_randomized=False,
    GS_n_iter=10,
    verbose=10,
)

print(f"Number of selected features RF Binary: {len(selected_features_rf_binary)}")
print(f"Selected features RF Binary: {selected_features_rf_binary}")
print(f"Selected Parameters RF Binary {selected_params_rf_binary_full}")


# In[33]:


selected_features_rf_binary


# In[35]:


selected_features_rf_binary=[
    'HAZ_rainfall_Total',
     'HAZ_rainfall_max_6h',
     'HAZ_rainfall_max_24h',
     'HAZ_v_max',
     'HAZ_dis_track_min',
     'GEN_landslide_per',
     'GEN_stormsurge_per',
     'TOP_mean_slope',
     'TOP_mean_elevation_m',
     'TOP_ruggedness_stdev',
     'TOP_mean_ruggedness',
     'TOP_slope_stdev',
     'VUL_poverty_perc',
     'GEN_coast_length',
     'VUL_Housing_Units',
     'VUL_StrongRoof_StrongWall',
     'VUL_StrongRoof_SalvageWall',
     'VUL_LightRoof_StrongWall',
     'VUL_LightRoof_SalvageWall',
     'VUL_SalvagedRoof_StrongWall',
     'VUL_vulnerable_groups',
     'VUL_pantawid_pamilya_beneficiary'
]


# #### Training the optimal model

# In[23]:



file_name = "models/output/v1/selected_params_rf_binary2.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_rf_binary, open(path, "wb"))

file_name = "models/output/v1/df_predicted_rf_binary2.csv"
path = os.path.join(cdir, file_name)
df_predicted_rf_binary.to_csv(path, index=False)


# In[39]:


# Setting the random forest search grid

 
rf_search_space = [
    {
        "rf__n_estimators": [500],
        "rf__max_depth": [22],
        "rf__min_samples_split": [2],
        "rf__min_samples_leaf": [3]
        
    }
]
# Obtaining the performance estimate
df_predicted_rf_binary, selected_params_rf_binary = rf_binary_performance(
    df_train_list=df_train_list,
    df_test_list=df_test_list,
    y_var='DAM_binary_dmg',
    features=selected_features_rf_binary,
    search_space=rf_search_space,
    stratK=True,
    cv_splits=5,
    class_weight="balanced",
    GS_score="f1",
    GS_randomized=False,
    GS_n_iter=50,
    verbose=10,
)

#n_samples / (n_classes * np.bincount(y))


# #### XG Boost

# In[47]:



combined_input_data = combined_input_data[combined_input_data['DAM_binary_dmg'].notnull()]
X = combined_input_data[features]
y = combined_input_data["DAM_binary_dmg"]

# Setting the train and the test sets for obtaining performance estimate
df_train_list, df_test_list = splitting_train_test(combined_input_data)


# In[25]:


# Setting the XGBoost search grid for full dataset
xgb_search_space = [
    {
        "estimator__learning_rate": [0.1, 0.5, 1],
        "estimator__gamma": [0.1, 0.5, 2],#0
        "estimator__max_depth": [6, 8],
        "estimator__reg_lambda": [0.001, 0.1, 1],
        "estimator__n_estimators": [100, 200],
        "estimator__colsample_bytree": [0.5, 0.7],
    }
]

# Obtaining the selected features based on the full dataset
selected_features_xgb_binary, selected_params_xgb_binary_full = xgb_binary_features(
    X=X,
    y=y,
    features=features,
    search_space=xgb_search_space,
    objective="binary:hinge",
    cv_splits=5,
    min_features_to_select=1,
    GS_score="f1",
    GS_n_iter=50,
    GS_randomized=True,
    verbose=10,
)


# In[ ]:





# #### Training the optimal model

# In[48]:


selected_features_xgb_regr =[
    'HAZ_v_max',
    'HAZ_dis_track_min',
    'VUL_StrongRoof_StrongWall',
    'TOP_mean_elevation_m',
    'HAZ_rainfall_max_6h',
    'HAZ_rainfall_max_24h',
    'HAZ_rainfall_Total',
    'VUL_vulnerable_groups',
    'VUL_pantawid_pamilya_beneficiary',
    'VUL_StrongRoof_LightWall',
    'VUL_poverty_perc',
    'TOP_ruggedness_stdev',
    'TOP_slope_stdev',
    'TOP_mean_slope',
    'GEN_coast_length',
    'VUL_Housing_Units',
    'GEN_stormsurge_per',
    'GEN_landslide_per',
    'TOP_mean_ruggedness',
    'GEN_Yel_per_LSSAb',
    'GEN_Yellow_per_LSbl',
    'GEN_Red_per_LSbldg',
    'GEN_with_coast',
    'GEN_OR_per_SSAbldg',
    'GEN_RED_per_SSAbldg',
    'GEN_Or_per_LSblg']


# In[49]:



df_train_list, df_test_list = splitting_train_test(combined_input_data)

selected_features_xgb_binary=selected_features_xgb_regr


xgb_search_space = [
    {
        "xgb__learning_rate": [0.3], #0.03
        "xgb__gamma": [0.1], #0
        "xgb__max_depth": [6], #6
        "xgb__reg_lambda": [0.001],
        "xgb__n_estimators": [50],
        "xgb__colsample_bytree": [0.7],#1
    }
]
# Obtaining the performance estimate
df_predicted_xgb_binary, selected_params_xgb_binary = xgb_binary_performance(
    df_train_list=df_train_list,
    df_test_list=df_test_list,
    y_var='DAM_binary_dmg',
    features=selected_features_xgb_binary,
    search_space=xgb_search_space,
    stratK=True,
    cv_splits=5,
    objective="binary:hinge",
    GS_score="f1",
    GS_randomized=True,
    GS_n_iter=100,
    verbose=10,
)

file_name = "models/output/v1/selected_params_xgb_binary.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_xgb_binary, open(path, "wb"))

file_name = "models/output/v1/df_predicted_xgb_binary.csv"
path = os.path.join(cdir, file_name)
df_predicted_xgb_binary.to_csv(path, index=False)


# #### Base line

# In[29]:



def unweighted_random(y_train, y_test):
    options = y_train.value_counts(normalize=True)
    y_pred = random.choices(population=list(options.index), k=len(y_test))
    return y_pred

def weighted_random(y_train, y_test):
    options = y_train.value_counts()
    y_pred = random.choices(
        population=list(options.index), weights=list(options.values), k=len(y_test)
    )
    return y_pred

df_predicted_random = pd.DataFrame(columns=["typhoon", "actual", "predicted"])

for i in range(len(df_train_list)):

    train = df_train_list[i]
    test = df_test_list[i]

    y_train = train["DAM_binary_dmg"]
    y_test = test["DAM_binary_dmg"]

    y_pred_test = unweighted_random(y_train, y_test)
    df_predicted_temp = pd.DataFrame(
        {"typhoon": test["typhoon"], "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted_random = pd.concat([df_predicted_random, df_predicted_temp])


file_name = "models/output/v1/df_predicted_random.csv"
path = os.path.join(cdir, file_name)
df_predicted_random.to_csv(path, index=False)
    
df_predicted_random_weighted = pd.DataFrame(columns=["typhoon", "actual", "predicted"])

for i in range(len(df_train_list)):

    train = df_train_list[i]
    test = df_test_list[i]

    y_train = train["DAM_binary_dmg"]
    y_test = test["DAM_binary_dmg"]

    y_pred_test = weighted_random(y_train, y_test)
    df_predicted_temp = pd.DataFrame(
        {"typhoon": test["typhoon"], "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted_random_weighted = pd.concat(
        [df_predicted_random_weighted, df_predicted_temp]
    )

    
file_name = "models/output/v1/df_predicted_random_weighted.csv"
path = os.path.join(cdir, file_name)
df_predicted_random_weighted.to_csv(path, index=False)
 


# In[ ]:




