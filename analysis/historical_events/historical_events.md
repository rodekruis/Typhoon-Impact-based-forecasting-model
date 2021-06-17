```python
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("../../data/")
OUTPUT_DIR = Path("output/")

DROMIC_REPORT_FILENAME = "2014 - 20 DROMIC Reports_PeopleAffected_DamagedHouses_Consolidated.xlsx"
IMPACT_DATA_FILENAME = "IMpact_data_philipines_final4.csv"
PCODE_FILENAME = "phl_adminboundaries_tabulardata.xlsx"

TYPHOON_LIST = [
    "Reming",
    "Frank",
    "Yolanda",
    "Ruby",
    "Nina",
    "Tisoy",
    "Ursula ",
    "Ambo",
    "Rolly",
]

TYPHOON_INTERNATIONAL_LIST = [
"Durian",
"Fengshen",
"Haiyan",
"Hagupit",
"Nock-Ten",
"Kammuri",
"Phanfone",
"Vongfong",
"Goni"
]
```

### Get the numbers from the DROMIC report Excel file

```python
base_cols = ['hurricane', 'year', 'location']

df_damaged = pd.read_excel(INPUT_DIR / DROMIC_REPORT_FILENAME, 
                            sheet_name="Damaged houses", 
                            header=None,
                            skiprows=4,
                            usecols=[0,1,2,3],
                            names=base_cols+['nhouses']
                          )
```

```python
df_damaged = df_damaged[(df_damaged['hurricane'].isin([t.upper() for t in TYPHOON_LIST])) & 
                        (df_damaged['location'].isin(['REGION V', 'REGION VIII']))
                       ]

df_damaged
```

### Using the impact csv

```python
df_impact = pd.read_csv(INPUT_DIR / IMPACT_DATA_FILENAME, index_col='id')
df_impact['typhoon'] = df_impact['typhoon'].str.lower()

# Use pcodes to add province
df_pcodes = pd.read_excel(
    INPUT_DIR / PCODE_FILENAME, 
     sheet_name="Admin3",
     usecols=[5, 11],
     names=['admin3_pcode', 'admin1_name']
)

df_impact = df_impact.merge(df_pcodes, left_on='pcode', right_on='admin3_pcode')

```

```python
# Take only admin regions and typhoons of interest
df_impact = df_impact[(df_impact['admin1_name'].isin(['Region V', 'Region VIII']))
                      & (df_impact['typhoon'].isin([typhoon.lower() for typhoon in TYPHOON_INTERNATIONAL_LIST]))
                     ]
```

```python
df_impact.groupby(['typhoon', 'admin1_name']).sum()['Totally']
```

# Appendix


### Checking which typhoons are in which files

```python
for typhoon in TYPHOON_INTERNATIONAL_LIST:
    x = typhoon.lower() in df_impact['typhoon'].unique()
    print(typhoon, x)
```

```python
for typhoon in TYPHOON_LIST:
    x = typhoon.lower() in df_damaged['hurricane'].str.lower().unique()
    print(typhoon, x)
```

```python

```
