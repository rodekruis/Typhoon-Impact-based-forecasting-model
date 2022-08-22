import os
import requests
from datetime import datetime
from pathlib import Path

from dateutil import rrule


save_dir = Path("/home/turnerm/sync/aa_repo_data/Data/public/exploration/phl/ecmwf_hindcast")

email = input('email: ')
pswd = input('password: ')

values = {'email' : email, 'passwd' : pswd, 'action' : 'login'}
login_url = 'https://rda.ucar.edu/cgi-bin/login'

ret = requests.post(login_url, data=values)
if ret.status_code != 200:
    print('Bad Authentication')
    print(ret.text)
    exit(1)

start_date = datetime(2006, 10, 1, 0, 0, 0)
start_date = datetime(2012, 9, 1, 0, 0, 0)
dspath = 'https://rda.ucar.edu/data/ds330.3/'
date_list = rrule.rrule(rrule.HOURLY, dtstart=start_date, interval=12)
verbose = True


for date in date_list:
    ymd = date.strftime("%Y%m%d")
    ymdhms = date.strftime("%Y%m%d%H%M%S")
    server = "test" if date < datetime(2008, 8, 1) else "prod"
    file = f'ecmf/{date.year}/{ymd}/z_tigge_c_ecmf_{ymdhms}_ifs_glob_{server}_all_glo.xml'
    filename = dspath + file
    outfile = save_dir / "xml" / os.path.basename(filename)
    # Don't download if exists already
    if outfile.exists():
        if verbose:
            print(f'{file} already exists')
        continue
    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True)
    if req.status_code != 200:
        if verbose:
            print(f'{file} invalid URL')
        continue
    if verbose:
        print(f'{file} downloading')
    open(outfile, 'wb').write(req.content)
