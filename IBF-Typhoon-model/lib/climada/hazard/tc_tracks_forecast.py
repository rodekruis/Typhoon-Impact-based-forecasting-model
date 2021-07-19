"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define TCTracks auxiliary methods: BUFR based TC predictions (from ECMWF)
"""

__all__ = ['TCForecast']

# standard libraries
import datetime as dt
import fnmatch
import ftplib
import logging
import os
import tempfile

# additional libraries
import numpy as np
import pandas as pd
import pybufrkit
import tqdm
import xarray as xr
from pybufrkit.renderer import FlatTextRenderer
from io import StringIO
# climada dependencies
from climada.hazard.tc_tracks import (
    TCTracks, set_category, DEF_ENV_PRESSURE, CAT_NAMES
)
from climada.util.files_handler import get_file_names

# declare constants
ECMWF_FTP = 'dissemination.ecmwf.int'
ECMWF_USER = 'wmo'
ECMWF_PASS = 'essential'

BASINS = {
    'W': 'W - North West Pacific',
    'C': 'C - North Central Pacific',
    'E': 'E - North East Pacific',
    'P': 'P - South Pacific',
    'L': 'L - North Atlantic',
    'A': 'A - Arabian Sea (North Indian Ocean)',
    'B': 'B - Bay of Bengal (North Indian Ocean)',
    'U': 'U - Australia',
    'S': 'S - South-West Indian Ocean'
}
"""Gleaned from the ECMWF wiki at
https://confluence.ecmwf.int/display/FCST/Tropical+Cyclone+tracks+in+BUFR+-+including+genesis
and Wikipedia at https://en.wikipedia.org/wiki/Invest_(meteorology)
"""

SAFFIR_MS_CAT_ = np.array([18, 33, 43, 50, 59, 71, 1000])
SAFFIR_MS_CAT = np.array([34, 47, 64,80, 99, 1000])

"""Saffir-Simpson Hurricane Categories in m/s"""

SIG_CENTRE = 1
"""The BUFR code 008005 significance for 'centre'"""

LOGGER = logging.getLogger(__name__)


class TCForecast(TCTracks):
    """An extension of the TCTracks construct adapted to forecast tracks
    obtained from numerical weather prediction runs.

    Attributes:
        data (list(xarray.Dataset)): Same as in parent class, adding the
            following attributes
                - ensemble_member (int)
                - is_ensemble (bool)
    """

    def fetch_ecmwf(self, path=None, files=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server into instance. Use path argument to use local
        files instead.

        Parameters:
            path (str, list(str)): A location in the filesystem. Either a
                path to a single BUFR TC track file, or a folder containing
                only such files, or a globbing pattern. Passed to
                climada.util.files_handler.get_file_names
            files (file-like): An explicit list of file objects, bypassing
                get_file_names
        """
        if path is None and files is None:
            files = self.fetch_bufr_ftp()
        elif files is None:
            files = get_file_names(path)

        for i, file in tqdm.tqdm(enumerate(files, 1), desc='Processing',
                                 unit='files', total=len(files)):
            try:
                file.seek(0)  # reset cursor if opened file instance
            except AttributeError:
                pass

            self.read_one_bufr_tc(file, id_no=i)

            try:
                file.close()  # discard if tempfile
            except AttributeError:
                pass

    @staticmethod
    def fetch_bufr_ftp(target_dir=None, remote_dir=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server. If target_dir is set, the files get downloaded
        persistently to the given location. A list of opened file-like objects
        gets returned.

        Parameters:
            target_dir (str): An existing directory to write the files to. If
                None, the files get returned as tempfiles.
            remote_dir (str, optional): If set, search this ftp folder for
                forecast files; defaults to the latest. Format:
                yyyymmddhhmmss, e.g. 20200730120000

        Returns:
            [str] or [filelike]
        """
        con = ftplib.FTP(host=ECMWF_FTP, user=ECMWF_USER, passwd=ECMWF_PASS)

        try:
            if remote_dir is None:
                remote = pd.Series(con.nlst())
                remote = remote[remote.str.contains('120000|000000$')]
                remote = remote.sort_values(ascending=False)
                remote_dir = remote.iloc[0]

            con.cwd(remote_dir)

            remotefiles = fnmatch.filter(con.nlst(), '*tropical_cyclone*')
            if len(remotefiles) == 0:
                msg = 'No tracks found at ftp://{}/{}'
                msg.format(ECMWF_FTP, remote_dir)
                raise FileNotFoundError(msg)

            localfiles = []

            LOGGER.info('Fetching BUFR tracks:')
            for rfile in tqdm.tqdm(remotefiles, desc='Download', unit=' files'):
                if target_dir:
                    lfile = open(os.path.join(target_dir, rfile), 'w+b')
                else:
                    lfile = tempfile.TemporaryFile(mode='w+b')

                con.retrbinary('RETR ' + rfile, lfile.write)

                if target_dir:
                    localfiles.append(lfile.name)
                    lfile.close()
                else:
                    localfiles.append(lfile)

        except ftplib.all_errors as err:
            con.quit()
            LOGGER.error('Error while downloading BUFR TC tracks.')
            raise err

        _ = con.quit()

        return localfiles
    def label_en (row,co):
        if row['code'] == co :
            return int(row['Data'])
        return np.nan
    def read_one_bufr_tc(self, file, id_no=None, fcast_rep=None):
        """ Read a single BUFR TC track file.

        Parameters:
            file (str, filelike): Path object, string, or file-like object
            id_no (int): Numerical ID; optional. Else use date + random int.
            fcast_rep (int): Of the form 1xx000, indicating the delayed
                replicator containing the forecast values; optional.
        """

        decoder = pybufrkit.decoder.Decoder()
 


        if hasattr(file, 'read'):
            bufr = decoder.process(file.read())
        elif hasattr(file, 'read_bytes'):
            bufr = decoder.process(file.read_bytes())
        elif os.path.isfile(file):
            with open(file, 'rb') as i:
                bufr = decoder.process(i.read())
        else:
            raise FileNotFoundError('Check file argument')
        text_data = FlatTextRenderer().render(bufr)
        
        # setup parsers and querents
        #npparser = pybufrkit.dataquery.NodePathParser()
        #data_query = pybufrkit.dataquery.DataQuerent(npparser).query

        meparser = pybufrkit.mdquery.MetadataExprParser()
        meta_query = pybufrkit.mdquery.MetadataQuerent(meparser).query
        timestamp_origin = dt.datetime(
            meta_query(bufr, '%year'), meta_query(bufr, '%month'),
            meta_query(bufr, '%day'), meta_query(bufr, '%hour'),
            meta_query(bufr, '%minute'),
        )
        timestamp_origin = np.datetime64(timestamp_origin)

        orig_centre = meta_query(bufr, '%originating_centre')
        if orig_centre == 98:
            provider = 'ECMWF'
        else:
            provider = 'BUFR code ' + str(orig_centre)
            
        list1=[]
        with StringIO(text_data) as input_data:
            # Skips text before the beginning of the interesting block:
            for line in input_data:
                if line.startswith('<<<<<< section 4 >>>>>>'):  # Or whatever test is needed
                    break
            # Reads text until the end of the block:
            for line in input_data:  # This keeps reading the file
                if line.startswith('<<<<<< section 5 >>>>>>'):
                    break
                list1.append(line)
         

        list1=[li for li in list1 if li.startswith(" ") or li.startswith("##") ]
        list2=[]
        for items in list1:
            if items.startswith("######"):
                                list2.append([0,items.split()[1],items.split()[2]])
            else:
                list2.append([int(items.split()[0]),items.split()[1],items.split()[-1]])

        df_ = pd.DataFrame(list2,columns=['id','code','Data'])
        
        def label_en (row,co):
           if row['code'] == co :
              return int(row['Data'])
           return np.nan
        
        df_['subset'] = df_.apply (lambda row: label_en(row,co='subset'), axis=1)
        df_['subset'] =df_['subset'].fillna(method='ffill')
        df_['model_sgn'] = df_.apply (lambda row: label_en(row,co='008005'), axis=1)      
        df_['model_sgn'] =df_['model_sgn'].fillna(method='ffill')
        df_['model_sgn'] =df_['model_sgn'].fillna(method='bfill')

        for names, group in df_.groupby("subset"):
            pcen = list(group.query('code in ["010051"]')['Data'].values)
            latc =  list(group.query('code in ["005002"] and model_sgn in [1]')['Data'].values)
            lonc =  list(group.query('code in ["006002"] and model_sgn in [1]')['Data'].values)
            latm =  list(group.query('code in ["005002"] and model_sgn in [3]')['Data'].values)
            lonm =  list(group.query('code in ["006002"] and model_sgn in [3]')['Data'].values)
            wind =  list(group.query('code in ["011012"]')['Data'].values)
            vhr =  list(group.query('code in ["004024"]')['Data'].values)
            wind=[np.nan if value=='None' else float(value) for value in wind]
            pre=[np.nan if value=='None' else float(value)/100 for value in pcen]
            lonm=[np.nan if value=='None' else float(value) for value in lonm]
            lonc=[np.nan if value=='None' else float(value) for value in lonc]
            latm=[np.nan if value=='None' else float(value) for value in latm]
            latc=[np.nan if value=='None' else float(value) for value in latc]
            vhr=[np.nan if value=='None' else int(value) for value in vhr]
            
            timestep_int = np.array(vhr).squeeze() #np.array(msg['timestamp'].get_values(index)).squeeze()
            timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')
            year =  list(group.query('code in ["004001"]')['Data'].values)
            month =  list(group.query('code in ["004002"]')['Data'].values)
            day =  list(group.query('code in ["004003"]')['Data'].values)
            hour =  list(group.query('code in ["004004"]')['Data'].values)
            #forecs_agency_id =  list(group.query('code in ["001033"]')['Data'].values)
            storm_name =  list(group.query('code in ["001027"]')['Data'].values)
            storm_id =  list(group.query('code in ["001025"]')['Data'].values)
            frcst_type =  list(group.query('code in ["001092"]')['Data'].values)
            max_radius=np.sqrt(np.square(np.array(latc)-np.array(latm))+np.square(np.array(lonc)-np.array(lonm)))*111
            date_object ='%04d%02d%02d%02d'%(int(year[0]),int(month[0]),int(day[0]),int(hour[0]))
            date_object=dt.datetime.strptime(date_object, "%Y%m%d%H")
            #timestamp=[(date_object + dt.timedelta(hours=int(value))).strftime("%Y%m%d%H") for value in vhr]
            #timestamp=[dt.datetime.strptime(value, "%Y%m%d%H") for value in timestamp]
            track = xr.Dataset(
                    data_vars={
                            'max_sustained_wind': ('time', wind[1:]),
                            'central_pressure': ('time', pre[1:]),
                            'ts_int': ('time', timestep_int),
                            'max_radius': ('time', max_radius[1:]),
                            'lat': ('time', latc[1:]),
                            'lon': ('time', lonc[1:]),
                            'environmental_pressure':('time', np.full_like(timestamp, DEF_ENV_PRESSURE, dtype=float)),
                            'radius_max_wind':('time', np.full_like(timestamp, np.nan, dtype=float)),
                            },
                            coords={'time': timestamp,
                                    },
                                    attrs={
                                            'max_sustained_wind_unit': 'm/s',
                                            'central_pressure_unit': 'mb',
                                            'name': storm_name[0].strip("'"),
                                            'sid': storm_id[0].split("'")[1],
                                            'orig_event_flag': False,
                                            'data_provider': provider,
                                            'id_no': 'NA',
                                            'ensemble_number': int(names),
                                            'is_ensemble': ['TRUE' if frcst_type[0]!='0' else 'False'][0],
                                            'forecast_time': date_object,
                                            })
            track = track.set_coords(['lat', 'lon'])
            track['time_step'] = track.ts_int - track.ts_int.shift({'time': 1}, fill_value=0)
            #track = track.drop('ts_int')
            track.attrs['basin'] = BASINS[storm_id[0].split("'")[1][2].upper()]
            cat_name = CAT_NAMES[set_category(
            max_sus_wind=track.max_sustained_wind.values,
            wind_unit=track.max_sustained_wind_unit,
            saffir_scale=SAFFIR_MS_CAT)]          
            track.attrs['category'] = cat_name
            if track.sizes['time'] == 0:
                track= None
            
            if track is not None:
                self.append(track)
            else:
                LOGGER.debug('Dropping empty track, subset %s', names)
            

                
                
    def read_one_bufr_tc_old(self, file, id_no=None, fcast_rep=None):
        """ Read a single BUFR TC track file.

        Parameters:
            file (str, filelike): Path object, string, or file-like object
            id_no (int): Numerical ID; optional. Else use date + random int.
            fcast_rep (int): Of the form 1xx000, indicating the delayed
                replicator containing the forecast values; optional.
        """

        decoder = pybufrkit.decoder.Decoder()
 


        if hasattr(file, 'read'):
            bufr = decoder.process(file.read())
        elif hasattr(file, 'read_bytes'):
            bufr = decoder.process(file.read_bytes())
        elif os.path.isfile(file):
            with open(file, 'rb') as i:
                bufr = decoder.process(i.read())
        else:
            raise FileNotFoundError('Check file argument')


        # setup parsers and querents
        npparser = pybufrkit.dataquery.NodePathParser()
        data_query = pybufrkit.dataquery.DataQuerent(npparser).query

        meparser = pybufrkit.mdquery.MetadataExprParser()
        meta_query = pybufrkit.mdquery.MetadataQuerent(meparser).query

        if fcast_rep is None:
            fcast_rep = self._find_delayed_replicator(
                meta_query(bufr, '%unexpanded_descriptors')
            )

        # query the bufr message
        msg = {
            # subset forecast data
            'significance': data_query(bufr, fcast_rep + '> 008005'),
            'latitude': data_query(bufr, fcast_rep + '> 005002'),
            'longitude': data_query(bufr, fcast_rep + '> 006002'),
            'wind_10m': data_query(bufr, fcast_rep + '> 011012'),
            'pressure': data_query(bufr, fcast_rep + '> 010051'),
            'timestamp': data_query(bufr, fcast_rep + '> 004024'),

            # subset metadata
            'wmo_longname': data_query(bufr, '/001027'),
            'storm_id': data_query(bufr, '/001025'),
            'ens_type': data_query(bufr, '/001092'),
            'ens_number': data_query(bufr, '/001091'),
        }

        timestamp_origin = dt.datetime(
            meta_query(bufr, '%year'), meta_query(bufr, '%month'),
            meta_query(bufr, '%day'), meta_query(bufr, '%hour'),
            meta_query(bufr, '%minute'),
        )
        timestamp_origin = np.datetime64(timestamp_origin)

        if id_no is None:
            id_no = timestamp_origin.item().strftime('%Y%m%d%H') + \
                    str(np.random.randint(1e3, 1e4))

        orig_centre = meta_query(bufr, '%originating_centre')
        if orig_centre == 98:
            provider = 'ECMWF'
        else:
            provider = 'BUFR code ' + str(orig_centre)

        for i in msg['significance'].subset_indices():
            name = msg['wmo_longname'].get_values(i)[0].decode().strip()
            track = self._subset_to_track(
                msg, i, provider, timestamp_origin, name, id_no
            )
            if track is not None:
                self.append(track)
            else:
                LOGGER.debug('Dropping empty track %s, subset %d', name, i)

    @staticmethod
    def _subset_to_track(msg, index, provider, timestamp_origin, name, id_no):
        """Subroutine to process one BUFR subset into one xr.Dataset"""
        sig = np.array(msg['significance'].get_values(index), dtype='int')
        lat = np.array(msg['latitude'].get_values(index), dtype='float')
        lon = np.array(msg['longitude'].get_values(index), dtype='float')
        wnd = np.array(msg['wind_10m'].get_values(index), dtype='float')
        pre = np.array(msg['pressure'].get_values(index), dtype='float')

        sid = msg['storm_id'].get_values(index)[0].decode().strip()

        timestep_int = np.array(msg['timestamp'].get_values(index)).squeeze()
        timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')

        try:
            track = xr.Dataset(
                data_vars={
                    'max_sustained_wind': ('time', np.squeeze(wnd)),
                    'central_pressure': ('time', np.squeeze(pre)/100),
                    'ts_int': ('time', timestep_int),
                    'lat': ('time', lat[sig == 1]),
                    'lon': ('time', lon[sig == 1]),
                },
                coords={
                    'time': timestamp,
                },
                attrs={
                    'max_sustained_wind_unit': 'm/s',
                    'central_pressure_unit': 'mb',
                    'name': name,
                    'sid': sid,
                    'orig_event_flag': False,
                    'data_provider': provider,
                    'id_no': (int(id_no) + index / 100),
                    'ensemble_number': msg['ens_number'].get_values(index)[0],
                    'is_ensemble': msg['ens_type'].get_values(index)[0] != 0,
                    'forecast_time': timestamp_origin,
                }
            )
        except ValueError as err:
            LOGGER.warning(
                'Could not process track %s subset %d, error: %s',
                sid, index, err
                )
            return None

        track = track.dropna('time')

        if track.sizes['time'] == 0:
            return None

        # can only make latlon coords after dropna
        track = track.set_coords(['lat', 'lon'])
        track['time_step'] = track.ts_int - \
            track.ts_int.shift({'time': 1}, fill_value=0)

        # TODO use drop_vars after upgrading xarray
        track = track.drop('ts_int')

        track['radius_max_wind'] = np.full_like(track.time, np.nan,
                                                dtype=float)
        track['environmental_pressure'] = np.full_like(
            track.time, DEF_ENV_PRESSURE, dtype=float
        )

        # according to specs always num-num-letter
        track.attrs['basin'] = BASINS[sid[2]]

        cat_name = CAT_NAMES[set_category(
            max_sus_wind=track.max_sustained_wind.values,
            wind_unit=track.max_sustained_wind_unit,
            saffir_scale=SAFFIR_MS_CAT
        )]
        track.attrs['category'] = cat_name
        return track

    @staticmethod
    def _find_delayed_replicator(descriptors):
        """The current bufr tc tracks only use one delayed replicator,
        enclosing all forecast values. This finds it.

        Parameters:
            bufr_message: An in-memory pybufrkit BUFR message
        """
        if len(descriptors) == 1:
            delayed_replicators=descriptors
        else:
            delayed_replicators = [d for d in descriptors if 100000 < d < 200000 and d % 1000 == 0]

        if len(delayed_replicators) != 1:
            LOGGER.error('Could not find fcast_rep, please set manually.')
            raise ValueError('More than one delayed replicator in BUFR file')

        return str(delayed_replicators[0])
