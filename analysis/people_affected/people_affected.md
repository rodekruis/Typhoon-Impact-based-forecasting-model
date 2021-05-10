```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl

INPUT_DIR = Path("../../data/")
OUTPUT_DIR = Path("output/")

DROMIC_REPORT_FILENAME = "2014 - 20 DROMIC Reports_PeopleAffected_DamagedHouses_Consolidated.xlsx"

mpl.rcParams['figure.dpi'] = 200

```

```python
base_cols = ['hurricane', 'year', 'location']
df_affected = pd.read_excel(INPUT_DIR / DROMIC_REPORT_FILENAME, 
                            sheet_name="People affected",
                            header=None,
                            skiprows=5,
                           usecols=[0,1,2,4, 5],
                            names=base_cols + ['nfam', 'npeople']
                           )

df_damaged = pd.read_excel(INPUT_DIR / DROMIC_REPORT_FILENAME, 
                            sheet_name="Damaged houses", 
                            header=None,
                            skiprows=4,
                            usecols=[0,1,2,3],
                            names=base_cols+['nhouses']
                          )

```

```python
def sum_duplicates(df, base_cols):
    # Get duplicates as separate df, and sum the number columns
    duplicates = (df[df.duplicated(subset=base_cols, keep=False)]
              .groupby(base_cols).sum().reset_index())
    # Remove them from main df
    df = df.drop_duplicates(subset=base_cols, keep=False)
    # Add the summed version
    df = pd.concat([df, duplicates], ignore_index=True)
    return(df)


for df, q in zip([df_affected, df_affected, df_damaged], 
                 ['nfam', 'npeople', 'nhouses']):
    # Remove dashes and replace with 0
    df[q] = pd.to_numeric(df[q], errors='coerce').fillna(0)
    
df_affected = sum_duplicates(df_affected, base_cols)
df_damaged = sum_duplicates(df_damaged, base_cols)
```

```python
df_combined = pd.merge(df_affected, df_damaged,
        how='inner',
        left_on=base_cols,
        right_on=base_cols).dropna()
```

```python
# split up into small areas and totals
total_cnames = ['TOTAL', 'GRAND TOTAL', 
                'REGION V', 'REGION III', 'REGION VIII', 'REGION VI', 
               'REGION I', 'REGION II', 'REGION X']
df_totals = df_combined[df_combined['location'].isin(total_cnames)]

df_combined =  df_combined[~df_combined['location'].isin(total_cnames)]
```

```python
def plot_people_vs_houses(df, log=False, xlims=None, ylims=None):
    x = df['nhouses']
    y = df['npeople']

    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.5)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel('Number of severely damaged houses')
    ax.set_ylabel('Number of affected people')
        
plot_people_vs_houses(df_totals, log=True)
df_no_zeros = df_combined[['nfam', 'npeople', 'nhouses']].replace({0: 0.1})
plot_people_vs_houses(df_no_zeros, log=True)

```

```python
# Get typical household size
ratio = df_combined['npeople'] / df_combined['nfam']
ratio = ratio[ratio < 100]
plt.hist(ratio, bins=np.arange(0.0, 10.0, 0.1))
```

```python
from scipy.optimize import minimize


houses_thresh = 10
people_thresh = 10
df_large = df_combined[(df_combined['npeople'] > people_thresh) & 
                      (df_combined['nhouses'] > houses_thresh)]

x = df_large['nhouses']
y = df_large['npeople']

# define likelihood function
def mle(params):
    m, sd = params[0], params[1]
    yhat = m*x # predictions
    return -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )

guess = np.array([25, 1])
results = minimize(mle, guess, method='Nelder-Mead', 
                   options={'disp': True})
print('Best fit slope:', results.x[0])
```
