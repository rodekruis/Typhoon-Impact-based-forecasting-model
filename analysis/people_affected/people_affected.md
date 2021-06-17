```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import matplotlib as mpl

INPUT_DIR = Path("../../data/")
OUTPUT_DIR = Path("output/")

DROMIC_REPORT_FILENAME = "2014 - 20 DROMIC Reports_PeopleAffected_DamagedHouses_Consolidated.xlsx"

mpl.rcParams['figure.dpi'] = 200

```

```python
# Read in an Excel file that summarizes the number of houses severely damaged 
# and people affected per tyhpoon event

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

# Sum events have multiple entries, sum them
df_affected = sum_duplicates(df_affected, base_cols)
df_damaged = sum_duplicates(df_damaged, base_cols)
```

```python
# Combine people affected and houses damaged
df_combined = pd.merge(df_affected, df_damaged,
        how='inner',
        left_on=base_cols,
        right_on=base_cols).dropna()
```

```python
# Split up dataframes into just looking at totals or admin 1 totals, and 
# more specific municipalities
total_cnames = ['TOTAL', 'GRAND TOTAL', 
                'REGION V', 'REGION III', 'REGION VIII', 'REGION VI', 
               'REGION I', 'REGION II', 'REGION X']
df_totals = df_combined[df_combined['location'].isin(total_cnames)]

df_combined =  df_combined[~df_combined['location'].isin(total_cnames)]
```

```python
# Fit a line with 0 intercept through the data

# Limit events to where there are at least 100 damaged houess / people affected
# (in reality setting this to 0 has little effect as the fit is dominated by 
# the large numbers)
houses_thresh = 100
people_thresh = 100
df_to_fit = df_combined[(df_combined['npeople'] >= people_thresh) & 
                      (df_combined['nhouses'] >= houses_thresh)]

x = df_to_fit['nhouses']
y = df_to_fit['npeople']

# likelihood function
def mle(params):
    m, b, sd = params[0], params[1], params[2]
    yhat = m*x + b # predictions
    return -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )

guess = np.array([25, 35000, 1])
results = minimize(mle, guess, method='Nelder-Mead', 
                   options={'disp': True})
print('Best fit slope:', results.x[0])
```

The best-fit line ha a sloe of ~25 (rounding due to uncertainties), which means that
for each severly damaged house there are about 25 people affected. 
Even though the average household size is around 5, the higher number can explained because we
expect that for each severely damaged house, there will be more people affected than
just those residing in the building itself.

```python
# Plot the data and best-fit line

def plot_people_vs_houses(df, log=False, xlims=None, ylims=None):
    x = df['nhouses']
    y = df['npeople']

    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.5)
    x_line = np.linspace(100, x.max(), 100)
    ax.plot(x_line, results.x[0] * x_line + results.x[1])
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel('Number of severely damaged houses')
    ax.set_ylabel('Number of affected people')
        
#plot_people_vs_houses(df_totals, log=True)
df_no_zeros = df_combined[['nfam', 'npeople', 'nhouses']].replace({0: 0.1})
plot_people_vs_houses(df_no_zeros, log=False)
plot_people_vs_houses(df_no_zeros, log=True)

```

Some things to note:
 - a log-log plot is used because the numbers span a very large range, which makes the fit look a bit off, 
   because it obfuscates the fact that it's dominated by the higher values
 - 0 values were set to 0.1 so that they show up as 10^-1 on the log-log plot
 - the fit is performaed to points in the top-right quadrant, whhere both the number of 
   damaged houses and affected people are both > 100 (10^2)


## Appendix

```python
# Examine typical household sizelims=(10, 2000), ylims=(10, 100000)results.x[1])
ratio = df_combined['npeople'] / df_combined['nfam']
ratio = ratio[ratio < 100]
plt.hist(ratio, bins=np.arange(0.0, 10.0, 0.1))
```

```python

```
