#!/usr/bin/env python
# coding: utf-8

# ## Modelling
# 
# This note book explains the different steps in the machine learning model.For the trigger model we used a Regression model. First the model is trained on the full dataset to obtain the optimal features followed by hyper parameter tunning and model performance estimate using Nested Cross Validation.
# 
# * Nested Cross Validation for
#     * Feature selection 
#     * hyper parameter tunning 
# * Performance metrics
# * Baseline Models
#  
# 
# 
# 
# 
# 
# 
# 

# In[1]:

 
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


# ### Define functions 

# In[2]:


def splitting_train_test(df):

    # To save the train and test sets
    df_train_list = []
    df_test_list = []

    # List of typhoons that are to be used as a test set 
 
    typhoons_with_impact_data=list(np.unique(df.typhoon))

    for typhoon in typhoons_with_impact_data:
        if len(df[df["typhoon"] == typhoon]) >150:
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


# In[3]:


# Setting directory

wor_dir="C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\IBF-typhoon-model"

os.chdir(wor_dir)

cdir = os.getcwd()


# In[4]:


# Import functions
from models.binary_classification.rf_binary import (rf_binary_features,rf_binary_performance,)
from models.binary_classification.xgb_binary import (xgb_binary_features,xgb_binary_performance,)
from models.regression.rf_regression import (rf_regression_features,rf_regression_performance,)
from models.regression.xgb_regression import (xgb_regression_features,xgb_regression_performance,)


# ## Loading the data

# In[15]:
    

combined_input_data=pd.read_csv("data\\model_input\\combined_input_data.csv")

typhoons_with_impact_data=['bopha2012', 'conson2010', 'durian2006', 'fengshen2008',
       'fung-wong2014', 'goni2015', 'goni2020', 'hagupit2014',
       'haima2016', 'haiyan2013', 'jangmi2014', 'kalmaegi2014',
       'kammuri2019', 'ketsana2009', 'koppu2015', 'krosa2013',
       'linfa2015', 'lingling2014', 'mangkhut2018', 'mekkhala2015',
       'melor2015', 'meranti2016', 'molave2020', 'mujigae2015',
       'nakri2019', 'nari2013', 'nesat2011', 'nock-ten2016', 'noul2015',
       'phanfone2019', 'rammasun2014', 'sarika2016', 'saudel2020',
       'tokage2016', 'trami2013', 'usagi2013', 'utor2013', 'vamco2020',
       'vongfong2020', 'yutu2018']


        
#combined_input_data=combined_input_data[combined_input_data.typhoon.isin(typhoons_with_impact_data)]





# In[17]:


#combined_input_data['year']=combined_input_data['typhoon'].apply(lambda x: int(x[-4:]))



# In[19]:



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
        'DAM_perc_dmg'])


# In[20]:


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


# In[21]:



# Full dataset for feature selection

df=combined_input_data.dropna()
 
#combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
X = df[features]
y = df["DAM_perc_dmg"]


 
    
 
# Setting the train and the test sets for obtaining performance estimate
df_train_list, df_test_list = splitting_train_test(df)

#%%
# #Random Forest
# #Training the optimal model
# Number of selected features RF Regression: 12
# 
# Selected features RF Regression:


#Selected Parameters RF Regression: {'estimator__max_depth': None,'estimator__min_samples_leaf': 3, 'estimator__min_samples_split': 5,                                     'estimator__n_estimators': 150}
 # 'HAZ_rainfall_Total',
 # 'HAZ_rainfall_max_6h',
 # 'HAZ_rainfall_max_24h',
 # 'HAZ_v_max',
 # 'HAZ_dis_track_min',
 # 'GEN_landslide_per',
 # 'GEN_Bu_p_inSSA',
 # 'GEN_Red_per_LSbldg',
 # 'GEN_Yel_per_LSSAb',
 # 'TOP_mean_slope',
 # 'TOP_mean_elevation_m',
 # 'TOP_ruggedness_stdev',
 # 'VUL_poverty_perc',
 # 'GEN_coast_length',
 # 'VUL_Housing_Units',
 # 'VUL_StrongRoof_StrongWall',
 # 'VUL_StrongRoof_SalvageWall',
 # 'VUL_vulnerable_groups',
 # 'VUL_pantawid_pamilya_beneficiary'
# 
# Selected Parameters RF Regression: 
# - max_depth = 20
# - min_samples_leaf = 2
# - min_samples_split = 5
# - n_estimators = 100
# 
# 
# 
# 
# 
# 

# ### Feature selection
# 
# Feature Selection is an important step in devloping a machine learning model.Data features used to train a machine learning model will influence model performance,less important features can have a negative impact on model performance.
# Feature Selection aims to solve the problem of identifying relevant features from a dataset by removing the less important features, which have little/no contribution to our target variable. Feature selection helps to achieve better model accuracy.
# 
# There are different techniques for feature selection. For this research we used Recursive feature elimination (RFE),which is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Features are ranked by the model’s coef_ or feature_importances_ attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.
# To find the optimal number of features we applied cross-validation with RFE on the entire data set. 
# 
#  
# 

# In[23]:


#%% Setting input varialbes
rf_search_space = [
    {
        "estimator__n_estimators": [100, 150],
        "estimator__max_depth": [20, None],
        "estimator__min_samples_split": [4, 5, 8],
        "estimator__min_samples_leaf":[1, 3, 5],
    }
]

(
    selected_features_rf_regr,
    selected_params_rf_regr_full,
) = rf_regression_features(
    X=X,
    y=y,
    features=features,
    search_space=rf_search_space,
    min_features_to_select=1,
    cv_splits=3,
    GS_score="neg_root_mean_squared_error",
    GS_randomized=False,
    GS_n_iter=10,
    verbose=10,
)

print(
    f"Number of selected features RF Regression {len(selected_features_rf_regr)}"
)
print(f"Selected features RF Regression: {selected_features_rf_regr}")
print(f"Selected Parameters RF Regression: {selected_params_rf_regr_full}")


# In[105]:


# Based on output previous cell



# 
# ### Hyper Parameter optimization  
# 
# Machine learning models have hyperparameters that you must set in order to customize the model to your dataset. Often the general effects of hyperparameters on a model are known, but how to best set a hyperparameter and combinations of interacting hyperparameters for a given dataset is challenging. There are often general rules of thumb for configuring hyperparameters. A better approach is to objectively search different values for model hyperparameters and choose a subset that results in a model that achieves the best performance on a given dataset. This is called hyperparameter optimization or hyperparameter tuning and is available in the scikit-learn Python machine learning library. [Source](https://machinelearningmastery.com/) 
# 
# 
# 
# 
# Hyperparameters are essentila components for machine learning algorithms, they control behaviour and performance of a machine learning model. For a learning algorithm optimal hyperparameter selection, hyperparameter tuning is esstil first step as it helps to achive best model performance on the data set with a reasonable amount of time.[source](https://www.sciencedirect.com/science/article/pii/S1674862X19300047)
# 
# To reduce the bias in performance evaluation, model selection should be treated as an integral part of the model fitting procedure, and should be conducted independently in each trial in order to prevent selection bias.[source](https://www.jmlr.org/papers/v11/cawley10a.html)
# 
# There are different techniques for Hyperparameters, for this research we used neasted K-fold cross validation technique. 
# Nested cross-validation uses inner and outer loops when optimizing the hyperparameters of a model on a dataset, and when comparing and selecting a model for the dataset. This reduced biased evaluation of model performance as different dataset are used to for hyperparameter tunning and model selection.
# 
# In our implementation of nested CV the outer loop iterates over typhoon events in our datasets, holiding data for one typhoon for test set and assigning the remaining data as training set. In the inner loop a k-fold CV is applied on the training dataset
# 
# 

# In[ ]:


#%% Setting input varialbes

selected_features_rf_regr=[
    'HAZ_rainfall_Total', 
    'HAZ_rainfall_max_6h',
    'HAZ_rainfall_max_24h',
    'HAZ_v_max', 
    'HAZ_dis_track_min',
    'GEN_landslide_per', 
    'GEN_stormsurge_per',
    'GEN_Bu_p_inSSA', 
    'GEN_Bu_p_LS',
    'GEN_Red_per_LSbldg',
    'GEN_Yel_per_LSSAb',
    'GEN_RED_per_SSAbldg', 
    'GEN_OR_per_SSAbldg',
    'TOP_mean_slope', 
    'TOP_mean_elevation_m', 
    'TOP_ruggedness_stdev', 
    'TOP_mean_ruggedness', 
    'TOP_slope_stdev', 
    'VUL_poverty_perc', 
    'GEN_coast_length',
    'VUL_Housing_Units', 
    'VUL_StrongRoof_StrongWall', 
    'VUL_StrongRoof_LightWall',
    'VUL_StrongRoof_SalvageWall',
    'VUL_LightRoof_LightWall', 
    'VUL_LightRoof_SalvageWall', 
    'VUL_SalvagedRoof_StrongWall',
    'VUL_SalvagedRoof_LightWall', 
    'VUL_SalvagedRoof_SalvageWall',
    'VUL_vulnerable_groups', 
    'VUL_pantawid_pamilya_beneficiary'
]

#%%



rf_search_space = [
    {
        "rf__n_estimators": [100, 250],
        "rf__max_depth": [18, 22],
        "rf__min_samples_split": [2, 8, 10],
        "rf__min_samples_leaf": [1, 3, 5],
    }
]

df_predicted_rf_regr, selected_params_rf_regr = rf_regression_performance(
    df_train_list=df_train_list,
    df_test_list=df_test_list,
    y_var='DAM_perc_dmg',
    features=selected_features_rf_regr,
    search_space=rf_search_space,
    cv_splits=5,
    GS_score="neg_root_mean_squared_error",
    GS_randomized=False,
    GS_n_iter=10,
    verbose=10,
)

#%%

Selected_Parameters={'rf__max_depth': 22, 
 'rf__min_samples_leaf': 3,
 'rf__min_samples_split': 2,
 'rf__n_estimators': 100}

#Train score: 0.006772373372310738
#Test score: 0.005727847779853535


# In[ ]:


file_name = "models\\output\\02\\selected_params_rf_regr.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_rf_regr, open(path, "wb"))

file_name = "models\\output\\02\\df_predicted_rf_regr.csv"
path = os.path.join(cdir, file_name)
df_predicted_rf_regr.to_csv(path)

#df_predicted_rf_regr=pd.read_csv(path)

#%%




#%%
# ### XGBoost Regression 
# Obtaining the optimal model

# In[24]:



# Full dataset for feature selection

combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
X = combined_input_data[features]
y = combined_input_data["DAM_perc_dmg"]

# Setting the train and the test sets for obtaining performance estimate
df_train_list, df_test_list = splitting_train_test(combined_input_data)


# In[ ]:


xgb_search_space = [
    {
        "estimator__learning_rate": [0.1, 0.5, 1],
        "estimator__gamma": [0.1, 0.5, 2],
        "estimator__max_depth": [6, 8],
        "estimator__reg_lambda": [0.001, 0.1, 1],
        "estimator__n_estimators": [100, 200],
        "estimator__colsample_bytree": [0.5, 0.7],
    }
]

selected_features_xgb_regr, selected_params_xgb_regr_full = xgb_regression_features(
    X=X,
    y=y,
    features=features,
    search_space=xgb_search_space,
    min_features_to_select=1,
    cv_splits=5,
    GS_score="neg_root_mean_squared_error",
    objective="reg:squarederror",
    GS_randomized=True,
    GS_n_iter=50,
    verbose=10,
)


print(f"Number of selected features XGBoost Regression {len(selected_features_xgb_regr)}")
print(f"Selected features XGBoost Regression: {selected_features_xgb_regr}")
print(f"Selected Parameters XGBoost Regression: {selected_params_xgb_regr_full}")


# ### Obtaining performance estimate¶

# In[ ]:


# Setting the selected features for XGB
selected_features_xgb_regr = ['HAZ_rainfall_Total',
 'HAZ_v_max',
 'HAZ_dis_track_min',
 'GEN_landslide_per',
 'TOP_mean_elevation_m',
 'TOP_mean_ruggedness',
 'VUL_Housing_Units',
 'VUL_StrongRoof_StrongWall',
 'VUL_StrongRoof_LightWall',
 'VUL_LightRoof_StrongWall',
 'VUL_vulnerable_groups',
 'VUL_pantawid_pamilya_beneficiary']

#%%

selected_params_xgb_regr_full={'estimator__reg_lambda': 0.001,
 'estimator__n_estimators': 200,
 'estimator__max_depth': 6,
 'estimator__learning_rate': 0.1,
 'estimator__gamma': 0.1,
 'estimator__colsample_bytree': 0.5}


# ### parameter optimization first based on selected model features 

# In[ ]:


xgb_search_space = [
    {
        "xgb__learning_rate": [0.1, 0.5, 1],
        "xgb__gamma": [0.1, 0.5, 2],
        "xgb__max_depth": [6, 8],
        "xgb__reg_lambda": [0.001, 0.1, 1],
        "xgb__n_estimators": [100, 200],
        "xgb__colsample_bytree": [0.5, 0.7],
    }
]

df_predicted_xgb_regr, selected_params_xgb_regr = xgb_regression_performance(
    df_train_list=df_train_list,
    df_test_list=df_test_list,
    y_var='DAM_perc_dmg',
    features=selected_features_xgb_regr,
    search_space=xgb_search_space,
    cv_splits=5,
    objective="reg:squarederror",
    GS_score="neg_root_mean_squared_error",
    GS_randomized=True,
    GS_n_iter=50,
    verbose=10,
)


# In[ ]:
    
#Selected Parameters {'xgb__reg_lambda': 0.1, 'xgb__n_estimators': 200, 'xgb__max_depth': 8, 'xgb__learning_rate': 0.1, 'xgb__gamma': 0.1, 'xgb__colsample_bytree': 0.5}
#Train score: 0.007331493507685711
#Test score: 0.0061379163528594355


file_name = "models\\output\\v1\\selected_params_xgb_regr.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_xgb_regr, open(path, "wb"))

file_name = "models\\output\\v1\\df_predicted_xgb_regr.csv"
path = os.path.join(cdir, file_name)
df_predicted_xgb_regr.to_csv(path)


# In[ ]:

file_name = "models\\output\\02\\df_predicted_xgb_regr.csv"
path = os.path.join(cdir, file_name)
df_predicted_xgb_regr=pd.read_csv(path)
### Benchmark


# In[ ]:



# Predict the average
df_predicted_mean = pd.DataFrame(columns=["typhoon", "actual", "predicted"])

for i in range(len(df_train_list)):

    train = df_train_list[i]
    test = df_test_list[i]

    y_train = train["DAM_perc_dmg"]
    y_test = test["DAM_perc_dmg"]

    y_test_pred = [np.mean(y_train)] * len(y_test)

    df_predicted_temp = pd.DataFrame(
        {"typhoon": test["typhoon"], "actual": y_test, "predicted": y_test_pred}
    )

    df_predicted_mean = pd.concat([df_predicted_mean, df_predicted_temp])


# In[ ]:


# Simle Linear Regression with Wind Speed
input_variable = "HAZ_v_max"
df_predicted_lr = pd.DataFrame(columns=["typhoon", "actual", "predicted"])

for i in range(len(df_train_list)):

    train = df_train_list[i]
    test = df_test_list[i]

    x_train = train[input_variable].values.reshape(-1, 1)
    y_train = train["DAM_perc_dmg"].values.reshape(-1, 1)

    x_test = test[input_variable].values.reshape(-1, 1)
    y_test = test["DAM_perc_dmg"]

    model = LinearRegression()
    lr_fitted = model.fit(x_train, y_train)

    y_pred_train = lr_fitted.predict(x_train)
    y_pred_test = lr_fitted.predict(x_test)
    y_pred_test = y_pred_test.tolist()
    y_pred_test = [val for sublist in y_pred_test for val in sublist]

    df_predicted_temp = pd.DataFrame(
        {"typhoon": test["typhoon"], "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted_lr = pd.concat([df_predicted_lr, df_predicted_temp])




# In[ ]:


### Results 

models = {
    #"Random Forest": df_predicted_rf_regr,
    "XGBoost": df_predicted_xgb_regr,
    "Average": df_predicted_mean,
    "Simple Linear Regression": df_predicted_lr,
}

mae = []
rmse = []

# add 'list' if error
for df_temp in models.values():
    mae.append(mean_absolute_error(df_temp["actual"], df_temp["predicted"]))
    rmse.append(mean_squared_error(df_temp["actual"], df_temp["predicted"], squared=False))

df_results_regr = pd.DataFrame({"Models": list(models.keys()), "MAE": mae, "RMSE": rmse})
#%%
display(df_results_regr)


#Binary Classification
#This section obtain the optimal Binary Classification models and the performance estimates, 
#for a 10% threshold. Two models are implemented: Random Forest Classifier, XGBoost Classifier. 
#First, the model is trained on the full dataset to obtain the optimal features followed by a model 
#that obtains the performance estimate using Nested Cross Validation.


# In[ ]:


def binary_damage_class(x):
    damage = x[0]   
    if damage > 0.1:
        value = 1
    else:
        value = 0
    return value

combined_input_data=pd.read_csv("data\\combined_input_data.csv")
combined_input_data["binary_dmg"] = combined_input_data[["perc_dmg"]].apply(binary_damage_class, axis="columns")

typhoons_with_impact_data=['bopha2012', 'conson2010', 'durian2006', 'fengshen2008',
       'fung-wong2014', 'goni2015', 'goni2020', 'hagupit2014','haima2016', 'haiyan2013', 'kalmaegi2014', 'kammuri2019',
       'ketsana2009', 'koppu2015', 'krosa2013', 'lingling2014','mangkhut2018', 'mekkhala2015', 'melor2015', 'mujigae2015',
       'nari2013', 'nesat2011', 'nock-ten2016', 'noul2015','rammasun2014', 'sarika2016', 'trami2013', 'usagi2013', 'utor2013',
       'vamco2020']

combined_input_data=combined_input_data[combined_input_data.typhoon.isin(typhoons_with_impact_data)]


combined_input_data.rename(columns ={"rainfall_Total":"HAZ_rainfall_Total",
                                     'rainfall_max_6h':'HAZ_rainfall_max_6h',
                                     'rainfall_max_24h':'HAZ_rainfall_max_24h',
                                     'v_max':'HAZ_v_max',
                                     'dis_track_min':'HAZ_dis_track_min',
                                     'binary_dmg':'DAM_binary_dmg',
                                     'perc_dmg':'DAM_perc_dmg',
                                    'landslide_per':'GEN_landslide_per',
                                    'stormsurge_per':'GEN_stormsurge_per',
                                    'Bu_p_inSSA':'GEN_Bu_p_inSSA',
                                    'Bu_p_LS':'GEN_Bu_p_LS',
                                     'Red_per_LSbldg':'GEN_Red_per_LSbldg',
                                    'Or_per_LSblg':'GEN_Or_per_LSblg',
                                     'Yel_per_LSSAb':'GEN_Yel_per_LSSAb',
                                    'RED_per_SSAbldg':'GEN_RED_per_SSAbldg',
                                     'OR_per_SSAbldg':'GEN_OR_per_SSAbldg',
                                    'Yellow_per_LSbl':'GEN_Yellow_per_LSbl',
                                     'mean_slope':'TOP_mean_slope',
                                    'mean_elevation_m':'TOP_mean_elevation_m',
                                     'ruggedness_stdev':'TOP_ruggedness_stdev',
                                    'mean_ruggedness':'TOP_mean_ruggedness',
                                     'slope_stdev':'TOP_slope_stdev',
                                     'poverty_perc':'VUL_poverty_perc',
                                    'with_coast':'GEN_with_coast',
                                     'coast_length':'GEN_coast_length',
                                     'Housing Units':'VUL_Housing_Units',
                                    'Strong Roof/Strong Wall':"VUL_StrongRoof_StrongWall",
                                    'Strong Roof/Light Wall':'VUL_StrongRoof_LightWall',
                                    'Strong Roof/Salvage Wall':'VUL_StrongRoof_SalvageWall',
                                    'Light Roof/Strong Wall':'VUL_LightRoof_StrongWall',
                                    'Light Roof/Light Wall':'VUL_LightRoof_LightWall',
                                    'Light Roof/Salvage Wall':'VUL_LightRoof_SalvageWall',
                                    'Salvaged Roof/Strong Wall':'VUL_SalvagedRoof_StrongWall',
                                    'Salvaged Roof/Light Wall':'VUL_SalvagedRoof_LightWall',
                                    'Salvaged Roof/Salvage Wall':'VUL_SalvagedRoof_SalvageWall',
                                    'vulnerable_groups':'VUL_vulnerable_groups',
                                    'pantawid_pamilya_beneficiary':'VUL_pantawid_pamilya_beneficiary'},inplace=True)


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


#%%
#Random forest 
df=combined_input_data.dropna()
 
#combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
X = df[features]
y = df["DAM_binary_dmg"]

# Setting the train and the test sets for obtaining performance estimate
df_train_list, df_test_list = splitting_train_test(df)
#%%

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

#%%

selected_features_rf_binary=['HAZ_rainfall_Total', 'HAZ_rainfall_max_6h',
                             'HAZ_rainfall_max_24h', 'HAZ_v_max',
                             'HAZ_dis_track_min', 'GEN_landslide_per', 
                             'GEN_Bu_p_LS', 'TOP_mean_slope', 
                             'TOP_mean_elevation_m', 'TOP_ruggedness_stdev',
                             'TOP_mean_ruggedness', 'TOP_slope_stdev',
                             'VUL_poverty_perc', 'GEN_coast_length', 
                             'VUL_Housing_Units', 'VUL_StrongRoof_StrongWall',
                             'VUL_StrongRoof_SalvageWall', 
                             'VUL_SalvagedRoof_StrongWall', 
                             'VUL_vulnerable_groups', 
                             'VUL_pantawid_pamilya_beneficiary']

#%%

# Setting the random forest search grid


rf_search_space = [
    {
        "rf__n_estimators": [100, 250],
        "rf__max_depth": [18, 22],
        "rf__min_samples_split": [2, 8, 15],
        "rf__min_samples_leaf": [1, 3, 5],
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
    GS_n_iter=10,
    verbose=10,
)

file_name = "models\\output\\02\\selected_params_rf_binary.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_rf_binary, open(path, "wb"))

file_name = "models\\output\\02\\df_predicted_rf_binary.csv"
path = os.path.join(cdir, file_name)
df_predicted_rf_binary.to_csv(path, index=False)

### Training the optimal model
#%%

#%%
df_train_list, df_test_list = splitting_train_test(combined_input_data)
df=combined_input_data
 
#combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
X = df[features]
y = df["DAM_binary_dmg"]


# Setting the XGBoost search grid for full dataset
xgb_search_space = [
    {
        "estimator__learning_rate": [0.1, 0.5, 1],
        "estimator__gamma": [0.1, 0.5, 2],
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

print(f"Number of selected features XGBoost Binary {len(selected_features_xgb_binary)}")
print(f"Selected features XGBoost Binary: {selected_features_xgb_binary}")
print(f"Selected parameters XGBoost Binary: {selected_params_xgb_binary_full}")

#%%

selected_features_xgb_binary=['HAZ_rainfall_max_24h', 'HAZ_v_max',
                              'HAZ_dis_track_min', 'VUL_StrongRoof_StrongWall',
                              'VUL_StrongRoof_SalvageWall', 
                              'VUL_LightRoof_StrongWall',
                              'VUL_LightRoof_LightWall', 
                              'VUL_vulnerable_groups']

selected_params_xgb_binary_full={'estimator__reg_lambda': 0.001,
                                 'estimator__n_estimators': 200,
                                 'estimator__max_depth': 6,
                                 'estimator__learning_rate': 0.1,
                                 'estimator__gamma': 0.5,
                                 'estimator__colsample_bytree': 0.5}
#%%
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
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


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return "f1_err", err

def xgb_binary_performance1(
    df_train_list,
    df_test_list,
    y_var,
    features,
    search_space,
    stratK,
    cv_splits,
    objective,
    GS_score,
    GS_randomized,
    GS_n_iter,
    verbose,
):

    train_score = []
    test_score = []
    selected_params = []
    df_predicted = pd.DataFrame(columns=["typhoon", "actual", "predicted"])

    for i in range(len(df_train_list)):

        print(f"Running for {i+1} out of a total of {len(df_train_list)}")

        train = df_train_list[i]
        test = df_test_list[i]

        x_train = train[features]
        y_train = train[y_var]

        x_test = test[features]
        y_test = test[y_var]

        # negative instances / positive instances
        # weight_scale = sum(y_train == 0) / sum(y_train == 1)

        # Stratified or non-stratified CV
        if stratK == True:
            cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)
        else:
            cv_folds = KFold(n_splits=cv_splits, shuffle=True)

        steps = [
            (
                "xgb",
                XGBClassifier(use_label_encoder=False, objective=objective, n_jobs=-3,),
            )
        ]

        pipe = Pipeline(steps, verbose=0)

        # Applying GridSearch or RandomizedGridSearch
        if GS_randomized == True:
            mod = RandomizedSearchCV(
                pipe,
                search_space,
                scoring=GS_score,
                cv=cv_folds,
                verbose=verbose,
                return_train_score=True,
                refit=True,
                n_iter=GS_n_iter,
            )
        else:
            mod = GridSearchCV(
                pipe,
                search_space,
                scoring=GS_score,
                cv=cv_folds,
                verbose=verbose,
                return_train_score=True,
                refit=True,
            )

        # Fitting the model on the full dataset
        xgb_fitted = mod.fit(x_train, y_train, xgb__eval_metric=f1_eval)
        results = xgb_fitted.cv_results_

        y_pred_test = xgb_fitted.predict(x_test)
        y_pred_train = xgb_fitted.predict(x_train)

        train_score_f1 = f1_score(y_train, y_pred_train)
        test_score_f1 = f1_score(y_test, y_pred_test)

        train_score.append(train_score_f1)
        test_score.append(test_score_f1)

        df_predicted_temp = pd.DataFrame(
            {"typhoon": test["typhoon"], "actual": y_test, "predicted": y_pred_test}
        )

        df_predicted = pd.concat([df_predicted, df_predicted_temp])
        selected_params.append(xgb_fitted.best_params_)

        print(f"Selected Parameters: {xgb_fitted.best_params_}")
        print(f"Train score: {train_score_f1}")
        print(f"Test score: {test_score_f1}")

    return df_predicted, selected_params

#%%

# Setting the XGBoost search grid
xgb_search_space = [
    {
        "xgb__learning_rate": [0.1, 0.5, 1],
        "xgb__gamma": [0.1, 0.5, 2],
        "xgb__max_depth": [6, 8],
        "xgb__reg_lambda": [0.001, 0.1, 1],
        "xgb__n_estimators": [100, 200],
        "xgb__colsample_bytree": [0.5, 0.7],
    }
]

# Obtaining the performance estimate
df_predicted_xgb_binary, selected_params_xgb_binary = xgb_binary_performance1(
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
    GS_n_iter=50,
    verbose=10,
)
#%%
file_name = "models\\output\\02\\selected_params_xgb_binary.p"
path = os.path.join(cdir, file_name)
pickle.dump(selected_params_xgb_binary, open(path, "wb"))

file_name = "models\\output\\02\\df_predicted_xgb_binary.csv"
path = os.path.join(cdir, file_name)
df_predicted_xgb_binary.to_csv(path, index=False)

#%%
#Baseline

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


# In[ ]:

models = {
    "Random Fores": df_predicted_rf_binary,
    "XGBoost": df_predicted_xgb_binary,
    "Random": df_predicted_random,
    "Weighted Random": df_predicted_random_weighted,
}

f1 = []
precision = []
recall = []

# add 'list' if error
for df_temp in models.values():
    f1.append(f1_score(list(df_temp["actual"]), list(df_temp["predicted"])))
    precision.append(precision_score(list(df_temp["actual"]), list(df_temp["predicted"])))
    recall.append(recall_score(list(df_temp["actual"]), list(df_temp["predicted"])))

df_results_binary = pd.DataFrame(
    {"Models": list(models.keys()), "F1 score": f1, "Recall": recall, "Precision": precision}
)
#%%
display(df_results_binary)


