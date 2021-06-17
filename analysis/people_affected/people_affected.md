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

def fit_line(x, y, yint_zero = False):

    # likelihood function
    def mle(params):
        m, b, sd = params[0], params[1], params[2]
        yhat = m*x + b # predictions
        return -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) ) 
    
    
    def mle_yint_zero(params):
        m, sd = params[0], params[1]
        yhat = m*x  # predictions
        return -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) ) 

    if yint_zero:
        guess = np.array([25, 1])
        function = mle_yint_zero

    else:
        guess = np.array([25, 35000, 1])
        function = mle
    
    results = minimize(function, guess, method='Nelder-Mead', 
                       options={'disp': True})

    # Plot histogram of residuals
    residuals = y - (results.x[0] * y + results.x[1])
    
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=100) 
    ax.set_xlabel('residuals')
    print('Best fit slope:', results.x[0])
    if not yint_zero:
        print('Best fit y-int:', results.x[1])

    return results
```

```python

# Limit events to where there are at least 100 damaged houess / people affected
# (in reality setting this to 0 has little effect as the fit is dominated by 
# the large numbers)
houses_thresh = 100
people_thresh = 100
df_to_fit = df_combined[(df_combined['npeople'] >= people_thresh) & 
                      (df_combined['nhouses'] >= houses_thresh)]

x = df_to_fit['nhouses']
y = df_to_fit['npeople']

# Try fitting with a free y intercept and one fixed at 0 (to get a scaling factor)
print('For linear fit:')
results_linear = fit_line(x, y)
print('\nFor linear fit with y int fixed to 0:')
results_yint_zero = fit_line(x, y, yint_zero=True)
```

The best-fit line ha a slope of ~25 (rounding due to uncertainties), with both a free y-intercept and one that is fixed at 0,
which means that for each severly damaged house there are about 25 people affected. 
Even though the average household size is around 5, the higher number can explained because we
expect that for each severely damaged house, there will be more people affected than
just those residing in the building itself.

However, residuals are non-normal so there is quite some heteroscedasticity, It might be worthwhile fitting an exponential

```python
# Try again but for log-log 
results_log = fit_line(np.log(x), np.log(y))
```

An exponential of the form:
```
n_people = 150 x n_houses ^ 0.7
```
is probably a better fit, but maybe less practical for interpretation

```python
# Plot the data and best-fit line

y_linear = results_linear.x[0] * x_line + results_linear.x[1]
y_yint_zero = results_yint_zero.x[0] * x_line
y_log = np.exp(results_log.x[1]) * x_line ** results_log.x[0]

def plot_people_vs_houses(df, log=False, xlims=None, ylims=None):
    x = df['nhouses']
    y = df['npeople']

    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.3, c='Grey')
    x_line = np.linspace(100, x.max(), 100)
    ax.plot(x_line, y_linear, label='linear fit')
    ax.plot(x_line,  y_yint_zero, label='y-intercept 0')
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

y1 = results_linear.x[0] * x_line + results_linear.x[1]
y2 = results_yint_zero.x[0] * x_line
y3 = np.exp(results_log.x[1]) * x_line ** results_log.x[0]
fig, ax = plt.subplots()
ax.plot(x_line, y_linear / x_line, label='linear fit')
ax.plot(x_line, y_yint_zero / x_line, label='y-intercept 0')
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

We perform a liner fit both holding the y-intercept at 0 at allowing it to vary. 
In both cases we obtain a slope of 25, meaning that for a single totally damaged 
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

```python

```
