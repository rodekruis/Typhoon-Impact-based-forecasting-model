```python
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

IMPACT_FILENAME = 'impact.csv'
```

```python
# Read in the impact data
df_impact = pd.read_csv(IMPACT_FILENAME, parse_dates=['Year'], index_col='id')
# Convert date columnt to just the year
df_impact['Year'] = df_impact['Year'].dt.year

# Find the span of years to make a new index for the data
min_year = df_impact['Year'].min()
max_year = df_impact['Year'].max()
new_index = pd.date_range(str(min_year), str(max_year+1), freq='Y').year

df_impact
```

```python
# Limit df_impact to cyclones only
# For now, take events where totally damaged is > 1000
df_impact_limited = df_impact[df_impact['Totally'] > 1000]
df_impact_limited
```

```python
def get_events(df):
    return (df
      .groupby('Year')
      .max()[['Totally']]
      .reindex(new_index)
      .fillna(0)
           )

# Get the max number of totally damaged houses per year,
# and fill the empty years with 0
df_events = get_events(df_impact)
df_events_limited = get_events(df_impact_limited)
```

```python
def get_return_period_analytical(df_rp: pd.DataFrame, 
                                 rp_var: str, 
                                 show_plot: bool = False,
                                 plot_title: str = "") -> interp1d:
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values, loc=rp_var_values.median(), scale=rp_var_values.median() / 2
    )
    x = np.linspace(rp_var_values.min(), rp_var_values.max()*2, 100)
    if show_plot:
        fig, ax = plt.subplots()
        ax.hist(rp_var_values, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(plot_title)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)

def get_return_period_empirical(df_rp: pd.DataFrame, rp_var: str) -> pd.DataFrame:
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return df_rp
```

```python
# Get the analytical and empirical RPs
rp_analytical = get_return_period_analytical(df_events, rp_var='Totally', show_plot=True)
rp_empirical = get_return_period_empirical(df_events, rp_var='Totally')
rp_analytical_limited = get_return_period_analytical(df_events_limited, rp_var='Totally', show_plot=True)
rp_empirical_limited = get_return_period_empirical(df_events_limited, rp_var='Totally')
```

```python
# Plot them    
fig, ax = plt.subplots()

rp = np.linspace(1.1, 20, 100)
ax.plot(rp, rp_analytical(rp), label=f'GEV', lw=4, alpha=0.8)
ax.plot(rp, rp_analytical_limited(rp), label=f'GEV, limited', ls='--', lw=2)

ax.plot(rp_empirical['rp'], rp_empirical['Totally'], 'o', label=f'data')
ax.plot(rp_empirical_limited['rp'], rp_empirical_limited['Totally'], 'x', label=f'data, limited')

ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('Number of houses totally damaged')
```
