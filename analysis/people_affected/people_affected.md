```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib as mpl
from statsmodels.stats import diagnostic


INPUT_DIR = Path("../../IBF-Typhoon-model/data")
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
def fit_line(x, y):

    def linear_function(x, m, b):
        return m * x + b

    results = curve_fit(linear_function, x, y)
    m_fit, b_fit = results[0][0], results[0][1]
    y_fit = m_fit * x + b_fit

    print(f'Best fit slope & y-int: {m_fit:.1f}, {b_fit:.1f}')

    # Report correlation results
    for stat in ['pearsonr', 'spearmanr']:
        print(f'\nResults for {stat}')
        result = getattr(stats, stat)(x, y)
        print(f'Correlation: {result[0]:.2f}, p-value: {result[1]:.2e}')
        
    # Analyze residuals 
    res = y - y_fit
    
    #fig, ax = plt.subplots()
    #ax.hist(res, bins=100)
    #ax.set_xlabel('Residuals')
    
    print('\nResult for shapiro')
    result = stats.shapiro(res)
    print(f'Correlation: {result[0]:.2f}, p-value: {result[1]:.2e}')

    
    for stat in ['het_white', 'het_breuschpagan']:
        print(f'\nResult for {stat}')
        result = getattr(diagnostic, stat)(res, np.array([np.ones(len(x)), x]).T)
        print(result)
    
    return m_fit, b_fit

```

```python
# Limit events to where there are at least 100 damaged houess / people affected
# (in reality setting this to 0 has little effect as the fit is dominated by 
# the large numbers)
houses_thresh = 100 # Don't skip any for now
people_thresh = 100
df_to_fit = df_combined[(df_combined['npeople'] >= people_thresh) & 
                      (df_combined['nhouses'] >= houses_thresh)]

x = df_to_fit['nhouses']
y = df_to_fit['npeople']

results_linear = fit_line(x, y)
```

Both Pearson's and Spearman's show a moderate correlation (>0.5) and reject the null hypothesis that there is no correlation.

With Shapiro we do reject the null hypothesis that the residuals are normally distributed.

Last two tests: Null hypothesis homoscedasticity is present is rejected, therefore we find heteroscodasticity. 

```python
# Try again but for log-log 
results_log = fit_line(np.log(x), np.log(y))
```

Pearson's correlation is slightly better, also rejecting the null hypothesis of no correlation. Spearman is unchaged as it should be since it is just the rank.

As previously with Shapiro we do reject the null hypothesis that the residuals are normally distributed.

However, we are not able to reject the null hypothesis of homoscodasticity. 

Thus an exponential of the form:
```
n_people = e ^ 5.7  x n_houses ^ 0.7
```
is probably a better fit, but maybe less practical for interpretation

```python
# Plot the data and best-fit line

x_line = np.linspace(100, x.max(), 100)
y_linear = results_linear[0] * x_line + results_linear[1]
y_log = np.exp(results_log[1]) * x_line ** results_log[0]

def plot_people_vs_houses(df, log=False, xlims=None, ylims=None):
    x = df['nhouses']
    y = df['npeople']

    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.3, c='Grey')
    x_line = np.linspace(100, x.max(), 100)
    ax.plot(x_line, y_linear, label='linear fit')
    ax.plot(x_line,  y_log, label='exponential fit')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel('Number of severely damaged houses')
    ax.set_ylabel('Number of affected people')
    ax.legend()
        
#plot_people_vs_houses(df_totals, log=True)
df_no_zeros = df_combined[['nfam', 'npeople', 'nhouses']].replace({0: 0.1})
plot_people_vs_houses(df_no_zeros, log=False)
plot_people_vs_houses(df_no_zeros, log=True)

```

Some things to note:
 - a log-log plot is used because the numbers span a very large range
 - 0 values were set to 0.1 so that they show up as 10^-1 on the log-log plot
 - the fit is performaed to points in the top-right quadrant, whhere both the number of 
   damaged houses and affected people are both > 100 (10^2)

```python
# Plot the scaling factor -- i.e. the number you need to multiply the number of 
# totally damaged houses to get the people affected

y1 = results_linear[0] * x_line + results_linear[1]
y3 = np.exp(results_log[1]) * x_line ** results_log[0]
fig, ax = plt.subplots()
ax.plot(x_line, y_linear / x_line, label='linear fit')
ax.plot(x_line, y_log / x_line, label='exponential fit')
ax.set_ylim(0, 50)
ax.legend()
ax.set_xlabel('Number of totally damaged houses')
ax.set_ylabel('Scaling factor to get people affected')
ax.minorticks_on()
ax.grid()
ax.grid(which='minor')
```

## Summary

We perform a liner fit and obtain a slope of ~25, meaning that for a single totally damaged 
house we expect about 25 affected people. This is more than a typical household size of 5, 
however this is not unexpected since by the time houses are severely damaged, 
we expected more people than just the residents of these houess in the nearby community to be affected.

Since we are paticularly interested in the high-impact regime, i.e. > 10,000 damaged houses, 
we can consider the exponential fit since it becomes flat. In this case the scaling factor is somewhat lower,
about 10-20. 


## Appendix

```python
# Examine typical household sizelims=(10, 2000), ylims=(10, 100000)results.x[1])
ratio = df_combined['npeople'] / df_combined['nfam']
ratio = ratio[ratio < 100]
plt.hist(ratio, bins=np.arange(0.0, 10.0, 0.1))
```
