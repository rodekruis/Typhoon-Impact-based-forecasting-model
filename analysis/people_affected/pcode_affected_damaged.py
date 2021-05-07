#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nelepijnenburg
"""

###### goals of code ######
### GOAL1: add pcodes to data of affected people and damaged houses ######
### GOAL2: add population and housing units to data of affected people and damaged houses ######
### GOAL3: add thresholds (>15% people affected, >10% houses damaged) ######
### GOAL4: merge data of affected people and data of damaged houses ######
### GOAL5: data per municipality and correlation ######

### things which need to be adjusted / corrected / checked
# check if outer join works
# need to check if values are realistic
# need to check reliability of fuzzy matcher


# =============================================================================
# prepare file
# =============================================================================
###### clear variables and console ######
try:
    from IPython import get_ipython

    get_ipython().magic("clear")
    get_ipython().magic("reset -f")
except:
    pass


###### import libraries ######
import pandas as pd
import numpy as np
import fuzzymatcher
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


###### set directory ######
INPUT_DIR = "data/"
OUTPUT_DIR = "analysis/people_affected/output/"

DROMIC_REPORT_FILENAME = (
    "2014 - 20 DROMIC Reports_PeopleAffected_DamagedHouses_Consolidated.xlsx"
)


###### import data ######
### affected family data
affected = pd.read_excel(
    (INPUT_DIR + DROMIC_REPORT_FILENAME), sheet_name="People affected", header=4
)
affected = affected.iloc[:, [0, 1, 2, 4]]
affected.columns = ["typhoon_aff", "year_aff", "location_aff", "aff_fam"]


### damaged houses (totally) data
damaged = pd.read_excel(
    (INPUT_DIR + DROMIC_REPORT_FILENAME), sheet_name="Damaged houses", header=3
)
damaged = damaged.iloc[:, 0:4]
damaged.columns = ["typhoon_dam", "year_dam", "location_dam", "dam_totally"]


### pcode level 3 data
pcodes3 = pd.read_excel(
    (INPUT_DIR + "phl_adminboundaries_tabulardata.xlsx"), sheet_name="Admin3"
)
pcodes3 = pcodes3.iloc[:, 4:6]  # admin level 3 pcode
pcodes3.columns = ["admin3_name", "admin3_pcode"]


### population data
pop_data = pd.read_csv((INPUT_DIR + "population_2015.csv"))


### housing data
housing_data = pd.read_csv((INPUT_DIR + "housing_units.csv"))
housing_data.columns = ["all"]
(
    housing_data["Region"],
    housing_data["Province"],
    housing_data["Municipality_City"],
    housing_data["Mun_Code"],
    housing_data["Housing_Units"],
) = (
    housing_data["all"].str.split(";", 4).str
)
housing_data.drop(["all", "Region", "Province"], axis=1, inplace=True)
housing_data.columns = ["Municipal_City", "pcode", "Housing_Units"]


# =============================================================================
# functions
# =============================================================================
###### function: fuzzy matcher ######
def fun_fuzzymatcher(left_data, right_data, left_on, right_on, dropyn):
    match = fuzzymatcher.fuzzy_left_join(left_data, right_data, left_on, right_on)
    if dropyn == "drop_yes":
        match.drop(
            ["best_match_score", "__id_left", "__id_right"], axis=1, inplace=True
        )
    return match


###### function: combine values ######
def fun_dubble(data, var1, var2):
    data[var1].replace("", np.nan, inplace=True)
    data[var2].replace("", np.nan, inplace=True)
    for i in range(len(data)):
        if pd.isna(data[var1].iloc[i]) == True and pd.isna(data[var2].iloc[i]) == False:
            data.loc[i].replace(data.iloc[i, 0], data.iloc[i, 1], inplace=True)
    return data


###### function: adding indicators for threshold affected and damage ######
def fun_threshold_aff_dam(data, threshold_aff, threshold_dam):
    # 1. threshold affected families
    threshold_aff_str = str(threshold_aff)
    name_aff = "aff_fam_above_" + threshold_aff_str
    data[name_aff] = ""
    for i in range(len(data)):
        if data["perc_affected_fam"].iloc[i] > threshold_aff:
            data[name_aff].iloc[i] = 1
        else:
            data[name_aff].iloc[i] = 0

    # 2. threshold houses totally damaged
    threshold_dam_str = str(threshold_dam)
    name_dam = "dam_hous_above_" + threshold_dam_str
    data[name_dam] = ""
    for i in range(len(data)):
        if data["perc_damaged_hous"].iloc[i] > threshold_dam:
            data[name_dam].iloc[i] = 1
        else:
            data[name_dam].iloc[i] = 0

    # 3. threshold both
    name_affxdam = (
        "aff_fam_above_" + threshold_aff_str + "_dam_hous_above_" + threshold_dam_str
    )
    data[name_affxdam] = ""
    for i in range(len(data)):
        if data[name_aff].iloc[i] == 1 and data[name_dam].iloc[i] == 1:
            data[name_affxdam].iloc[i] = 1
        else:
            data[name_affxdam].iloc[i] = 0

    # sum
    threshold_affected_fam = sum(data[name_aff])
    threshold_damaged_hous = sum(data[name_dam])
    threshold_affected_fam_damaged_hous = sum(data[name_affxdam])

    # print
    print(
        "Times {:1.0f}% threshold families affected is met: {:d}.".format(
            threshold_aff * 100, threshold_affected_fam
        )
    )
    print(
        "Times {:1.0f}% threshold houses totally damaged is met: {:d}.".format(
            threshold_dam * 100, threshold_damaged_hous
        )
    )
    print(
        "Times {:1.0f}% threshold families affected AND threshold {:1.0f}% houses totally damaged are met: {:d}.".format(
            threshold_aff * 100,
            threshold_dam * 100,
            threshold_affected_fam_damaged_hous,
        )
    )

    return data


# =============================================================================
# assigning values to variables
# =============================================================================
false_threshold = 0.6  # assuming false data point if percentage above this value


# =============================================================================
# prepare data
# =============================================================================
######## prep data affected people ########
### add pcodes to data affected people
affectedxpcodes = fun_fuzzymatcher(
    affected, pcodes3, ["location_aff"], ["admin3_name"], "drop_yes"
)


### add population data to data affected people
affectedxpcodes = fun_fuzzymatcher(
    affectedxpcodes, pop_data, ["admin3_pcode"], ["pcode"], "drop_yes"
)


### calculate percentage families affected (family = 5 pers)
affectedxpcodes["aff_fam"] = pd.to_numeric(
    affectedxpcodes["aff_fam"].iloc[:], errors="coerce"
).fillna(0)
affectedxpcodes["perc_affected_fam"] = affectedxpcodes["aff_fam"] / (
    affectedxpcodes["total_pop"] / 5
)

# only if percentage if below threshold; not taking into account false data values
for i in range(len(affectedxpcodes)):
    if affectedxpcodes["perc_affected_fam"].iloc[i] > false_threshold:
        affectedxpcodes["perc_affected_fam"].iloc[i] = 0


### drop unnecessary columns
# affectedxpcodes.drop(['pcode', 'Municipal_City'], axis=1, inplace = True)


### export to excel
affectedxpcodes.to_excel(OUTPUT_DIR + "affected_pcoded.xlsx", index=False)


######## prep data damaged houses ########
### add pcodes to data damaged houses
damagedxpcodes = fun_fuzzymatcher(
    damaged, pcodes3, ["location_dam"], ["admin3_name"], "drop_yes"
)


### add housing data to data damaged houses
damagedxpcodes = fun_fuzzymatcher(
    damagedxpcodes, housing_data, ["admin3_pcode"], ["pcode"], "drop_yes"
)


### calculate perentage damaged houses (total)
damagedxpcodes["dam_totally"] = pd.to_numeric(
    damagedxpcodes["dam_totally"].iloc[:], errors="coerce"
).fillna(0)
damagedxpcodes["Housing_Units"] = damagedxpcodes["Housing_Units"].astype("float")
damagedxpcodes["perc_damaged_hous"] = (
    damagedxpcodes["dam_totally"] / damagedxpcodes["Housing_Units"]
)

# only if percentage if below threshold; not taking into account false data values
for i in range(len(damagedxpcodes)):
    if damagedxpcodes["perc_damaged_hous"].iloc[i] > false_threshold:
        damagedxpcodes["perc_damaged_hous"].iloc[i] = 0


### drop unnecessary columns
damagedxpcodes.drop(["pcode", "Municipal_City"], axis=1, inplace=True)


### export to excel
# damagedxpcodes.to_excel(OUTPUT_DIR + 'damaged_pcoded.xlsx', index = False)


### CONCLUSION ###
# goal 1, 2, 3 completed


# =============================================================================
# merge data affected people and damaged houses
# =============================================================================
### create format of which there can be a match (typhoon_year_pcode)
affectedxpcodes.rename(columns={"admin3_pcode": "pcode_aff"}, inplace=True)
damagedxpcodes.rename(columns={"admin3_pcode": "pcode_dam"}, inplace=True)

data = affectedxpcodes
data["typhoon_year_pcode_aff"] = ""
data["year_aff"] = data["year_aff"].astype(str)
data["pcode_aff"] = data["pcode_aff"].astype(str)
for i in range(len(data)):
    if data["pcode_aff"][i] != "nan":
        data["typhoon_year_pcode_aff"].iloc[i] = (
            data["typhoon_aff"].iloc[i]
            + "_"
            + data["year_aff"].iloc[i]
            + "_"
            + data["pcode_aff"].iloc[i]
        )

data = damagedxpcodes
data["typhoon_year_pcode_dam"] = ""
data["year_dam"] = data["year_dam"].astype(str)
data["pcode_dam"] = data["pcode_dam"].astype(str)
for i in range(len(data)):
    if data["pcode_dam"][i] != "nan":
        data["typhoon_year_pcode_dam"].iloc[i] = (
            data["typhoon_dam"].iloc[i]
            + "_"
            + data["year_dam"].iloc[i]
            + "_"
            + data["pcode_dam"].iloc[i]
        )


### merge: need outer join
# merge 1: affected people left
affectedxdamaged1 = fun_fuzzymatcher(
    affectedxpcodes,
    damagedxpcodes,
    ["typhoon_year_pcode_aff"],
    ["typhoon_year_pcode_dam"],
    "drop_yes",
)

# merge 2: damaged houses left
affectedxdamaged2 = fun_fuzzymatcher(
    damagedxpcodes,
    affectedxpcodes,
    ["typhoon_year_pcode_dam"],
    ["typhoon_year_pcode_aff"],
    "drop_yes",
)

# combine
affectedxdamaged1.reset_index(drop=True, inplace=True)
affectedxdamaged2.reset_index(drop=True, inplace=True)
affectedxdamaged = pd.concat([affectedxdamaged1, affectedxdamaged2], axis=1)
affectedxdamaged = affectedxdamaged.loc[:, ~affectedxdamaged.columns.duplicated()]


### combine columns (string, loc, typhoon, year, pcode)
# string
affectedxdamaged[["typhoon_year_pcode_aff", "typhoon_year_pcode_dam"]] = fun_dubble(
    affectedxdamaged[["typhoon_year_pcode_aff", "typhoon_year_pcode_dam"]],
    "typhoon_year_pcode_aff",
    "typhoon_year_pcode_dam",
)

# loc
affectedxdamaged[["location_aff", "location_dam"]] = fun_dubble(
    affectedxdamaged[["location_aff", "location_dam"]], "location_aff", "location_dam"
)

# typhoon
affectedxdamaged[["typhoon_aff", "typhoon_dam"]] = fun_dubble(
    affectedxdamaged[["typhoon_aff", "typhoon_dam"]], "typhoon_aff", "typhoon_dam"
)

# year
affectedxdamaged[["year_aff", "year_dam"]] = fun_dubble(
    affectedxdamaged[["year_aff", "year_dam"]], "year_aff", "year_dam"
)

# pcode
affectedxdamaged[["pcode_aff", "pcode_dam"]] = fun_dubble(
    affectedxdamaged[["pcode_aff", "pcode_dam"]], "pcode_aff", "pcode_dam"
)


### create df with most important variables
affectedxdamaged_short = affectedxdamaged[
    [
        "typhoon_year_pcode_aff",
        "location_aff",
        "typhoon_aff",
        "year_aff",
        "pcode_aff",
        "aff_fam",
        "perc_affected_fam",
        "total_pop",
        "dam_totally",
        "perc_damaged_hous",
        "Housing_Units",
    ]
]
affectedxdamaged_short.drop_duplicates(inplace=True)
affectedxdamaged_short.rename(
    columns={
        "typhoon_year_pcode_aff": "typhoon_year_pcode",
        "location_aff": "location",
        "typhoon_aff": "typhoon",
        "year_aff": "year",
        "pcode_aff": "pcode",
    },
    inplace=True,
)


### CONCLUSION ###
# goal 4 completed


# =============================================================================
# create df with data for every typhoon and municipality
# =============================================================================
### create format to view unique typhoons and years (sometimes reoccuring typhoons)
data = affectedxdamaged_short
data["typhoon_year"] = ""
data["year"] = data["year"].astype(str)
for i in range(len(data)):
    data["typhoon_year"].iloc[i] = data["typhoon"].iloc[i] + "_" + data["year"].iloc[i]

typhoon_year = affectedxdamaged_short["typhoon_year"].unique()
typhoon_count = len(typhoon_year)  # 29
mun_count = len(pcodes3)  # 1647
# 29*1647 = 47763

### create large df with values per mun and per typhoon
df1 = pd.concat([pcodes3, pcodes3], axis=0)
for i in range(typhoon_count - 2):
    df1 = pd.concat([df1, pcodes3], axis=0)

df2 = pd.DataFrame({"typhoon_year": np.repeat(typhoon_year, mun_count)})
df2.sort_values(by="typhoon_year", inplace=True)

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3 = pd.concat([df1, df2], axis=1)

data = df3
data["typhoon_year_pcode"] = ""
data["admin3_pcode"] = data["admin3_pcode"].astype(str)
for i in range(len(data)):
    data["typhoon_year_pcode"].iloc[i] = (
        data["typhoon_year"].iloc[i] + "_" + data["admin3_pcode"].iloc[i]
    )


### merge with affected people and housing damage data
df_affectedxdamaged = fun_fuzzymatcher(
    data,
    affectedxdamaged_short,
    ["typhoon_year_pcode"],
    ["typhoon_year_pcode"],
    "drop_yes",
)
df_affectedxdamaged.drop(
    ["typhoon_year_right", "typhoon_year_pcode_right"], axis=1, inplace=True
)
df_affectedxdamaged.rename(
    columns={
        "typhoon_year_pcode_left": "typhoon_year_pcode",
        "typhoon_year_left": "typhoon_year",
    },
    inplace=True,
)


### fill missing values with 0 (quantitative vars)
df_affectedxdamaged[
    [
        "aff_fam",
        "total_pop",
        "perc_affected_fam",
        "dam_totally",
        "Housing_Units",
        "perc_damaged_hous",
    ]
] = df_affectedxdamaged[
    [
        "aff_fam",
        "total_pop",
        "perc_affected_fam",
        "dam_totally",
        "Housing_Units",
        "perc_damaged_hous",
    ]
].fillna(
    0
)


# =============================================================================
# add indicator variables if thresholds are met
# =============================================================================
# data, threshold affected families, threshold damaged houses
df_affectedxdamaged = fun_threshold_aff_dam(df_affectedxdamaged, 0.15, 0.1)
df_affectedxdamaged = fun_threshold_aff_dam(df_affectedxdamaged, 0.15, 0.09)
df_affectedxdamaged = fun_threshold_aff_dam(df_affectedxdamaged, 0.15, 0.08)


### export to excel
# df_affectedxdamaged.to_excel(OUTPUT_DIR + 'dflarge_munxtyp_affected_damaged.xlsx', index = False)


# =============================================================================
# correlation analysis
# =============================================================================
######## per municipality ########
### select rows for one municipality
mun = "PH030812000"
df_mun = df_affectedxdamaged[df_affectedxdamaged["admin3_pcode"] == mun]
df_mun.sum()
df_mun.dtypes
df_mun.to_excel(OUTPUT_DIR + "test_mun1.xlsx", index=False)

x1 = "perc_affected_fam"
x2 = "perc_damaged_hous"
var1 = df_mun[x1]
var1.reset_index(drop=True, inplace=True)
var2 = df_mun[x2]
var2.reset_index(drop=True, inplace=True)
df_cor = df_mun[[x1, x2]]
labels = df_mun["typhoon_year"]

# correlation coefficient
rho = stats.pearsonr(var1, var2)

# correlation plots
plt.matshow(df_cor.corr())
sns.heatmap(df_cor.corr())

# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(var1, var2, s=100, color="red")
plt.xlabel("% affected families")
plt.ylabel("% damaged houses")
plt.title(mun, fontsize=15)
for i, label in enumerate(labels):
    plt.annotate(label, (var1[i], var2[i]))
plt.show()


### overview correlation for every municipality
df = df_affectedxdamaged
pcodes = df["admin3_pcode"].unique()
x1 = "perc_affected_fam"
x2 = "perc_damaged_hous"
rho_mun = pd.DataFrame(columns=["pcode", "cor_mun", "pvalue_mun"])


for i, pcode in enumerate(pcodes):
    df_mun = df[df["admin3_pcode"] == pcodes[i]]
    var1 = df_mun[x1]
    var1.reset_index(drop=True, inplace=True)
    var2 = df_mun[x2]
    var2.reset_index(drop=True, inplace=True)
    rho = stats.pearsonr(var1, var2)
    rho_mun = rho_mun.append(
        {"pcode": pcode, "cor_mun": rho[0], "pvalue_mun": rho[1]}, ignore_index=True
    )
# rho_mun.to_excel(OUTPUT_DIR +'cor_per_mun.xlsx', index = False)

mun = rho_mun["pcode"]
rho = rho_mun["cor_mun"]
plt.figure(figsize=(200, 160))
plt.barh(mun, rho)
plt.title("correlation between % affected families and % houses totally damaged")
# plt.savefig(OUTPUT_DIR + 'cor_per_mun.png', dpi = 150)
plt.show()


######## per typhoon ########
### select rows per typhoon
typ = "EGAY_2015"
df_typ = df_affectedxdamaged[df_affectedxdamaged["typhoon_year"] == typ]
df_typ.sum()
df_typ.dtypes
df_typ.to_excel(OUTPUT_DIR + "test_typ1.xlsx", index=False)

x1 = "perc_affected_fam"
x2 = "perc_damaged_hous"
var1 = df_typ[x1]
var1.reset_index(drop=True, inplace=True)
var2 = df_typ[x2]
var2.reset_index(drop=True, inplace=True)
df_cor = df_typ[[x1, x2]]
labels = df_typ["pcode"]

# correlation coefficient
rho = stats.pearsonr(var1, var2)

# correlation plots
plt.matshow(df_cor.corr())
sns.heatmap(df_cor.corr())

# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(var1, var2, s=100, color="red")
plt.xlabel("% affected families")
plt.ylabel("% damaged houses")
plt.title(typ, fontsize=15)
for i, label in enumerate(labels):
    plt.annotate(label, (var1[i], var2[i]))
plt.show()


### overview correlation for every typhoon
df = df_affectedxdamaged
typhoon_years = df["typhoon_year"].unique()
x1 = "perc_affected_fam"
x2 = "perc_damaged_hous"
rho_typ = pd.DataFrame(columns=["typhoon_year", "cor_typ", "pvalue_typ"])


for i, typ in enumerate(typhoon_years):
    df_typ = df[df["typhoon_year"] == typhoon_years[i]]
    var1 = df_typ[x1]
    var1.reset_index(drop=True, inplace=True)
    var2 = df_typ[x2]
    var2.reset_index(drop=True, inplace=True)
    rho = stats.pearsonr(var1, var2)
    rho_typ = rho_typ.append(
        {"typhoon_year": typ, "cor_typ": rho[0], "pvalue_typ": rho[1]},
        ignore_index=True,
    )
# rho_typ.to_excel(OUTPUT_DIR+'cor_per_typ.xlsx', index = False)

typ = rho_typ["typhoon_year"]
rho = rho_typ["cor_typ"]
plt.figure(figsize=(30, 20))
plt.barh(typ, rho)
plt.title("correlation between % affected families and % houses totally damaged")
# plt.savefig(OUTPUT_DIR + 'cor_per_typ.png', dpi = 150)
plt.show()


### CONCLUSION ###
# goal 5 completed


### df with non zero values
df_nonzero = df_affectedxdamaged[
    (df_affectedxdamaged["perc_affected_fam"] > 0)
    | (df_affectedxdamaged["perc_damaged_hous"] > 0)
]
df_nonzero.to_excel(OUTPUT_DIR + "df_nonzero.xlsx", index=False)
