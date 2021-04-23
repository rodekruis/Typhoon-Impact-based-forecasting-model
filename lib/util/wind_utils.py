import os
import logging
import math
import numpy as np
 


SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 137, 1000]
"""Saffir-Simpson Hurricane Wind Scale in kn based on NOAA"""

def _change_max_wind_unit(wind, unit_orig, unit_dest):
    """Compute maximum wind speed in unit_dest

    Parameters:
        wind (np.array): wind
        unit_orig (str): units of wind
        unit_dest (str): units to change wind

    Returns:
        double
    """
    if unit_orig in ('kn', 'kt'):
        ur_orig = ureg.knot
    elif unit_orig == 'mph':
        ur_orig = ureg.mile / ureg.hour
    elif unit_orig == 'm/s':
        ur_orig = ureg.meter / ureg.second
    elif unit_orig == 'km/h':
        ur_orig = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_orig)
        raise ValueError
    if unit_dest in ('kn', 'kt'):
        ur_dest = ureg.knot
    elif unit_dest == 'mph':
        ur_dest = ureg.mile / ureg.hour
    elif unit_dest == 'm/s':
        ur_dest = ureg.meter / ureg.second
    elif unit_dest == 'km/h':
        ur_dest = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_dest)
        raise ValueError
    return (np.nanmax(wind) * ur_orig).to(ur_dest).magnitude

def set_category(max_sus_wind, wind_unit, saffir_scale=None):
    """Add storm category according to saffir-simpson hurricane scale

      - -1 tropical depression
      - 0 tropical storm
      - 1 Hurricane category 1
      - 2 Hurricane category 2
      - 3 Hurricane category 3
      - 4 Hurricane category 4
      - 5 Hurricane category 5

    Parameters:
        max_sus_wind (np.array): max sustained wind
        wind_unit (str): units of max sustained wind
        saffir_scale (list, optional): Saffir-Simpson scale in same units as wind

    Returns:
        double
    """
    if saffir_scale is None:
        saffir_scale = SAFFIR_SIM_CAT
        if wind_unit != 'kn':
            max_sus_wind = _change_max_wind_unit(max_sus_wind, wind_unit, 'kn')
    max_wind = np.nanmax(max_sus_wind)
    try:
        return (np.argwhere(max_wind < saffir_scale) - 1)[0][0]
    except IndexError:
        return -1


