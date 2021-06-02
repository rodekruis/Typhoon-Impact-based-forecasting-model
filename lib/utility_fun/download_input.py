def download_ecmwf(Input_folder,filepatern):
    """
    Reads ecmwf forecast data and save to folder
    """    
    if not os.path.exists(os.path.join(Input_folder,'ecmwf/')):
        os.makedirs(os.path.join(Input_folder,'ecmwf/'))
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    ftp=FTP("dissemination.ecmwf.int")
    ftp.login("wmo","essential")
    files=ftp.nlst()
    
    ftp.cwd(os.path.join(ftp.pwd(),files[-1]))
    file_list=ftp.nlst()    
    for files in file_list:
        if filepatern in files:
            ftp.retrbinary("RETR " +files,open(os.path.join(path_ecmwf,files),'wb').write)
            

def download_rainfall(Input_folder):
    """
    download rainfall 
    
    """
    #s.makedirs(os.path.join(Input_folder,'rainfall/'))
    if not os.path.exists(os.path.join(Input_folder,'rainfall/')):
        os.makedirs(os.path.join(Input_folder,'rainfall/'))
    
    rainfall_path=os.path.join(Input_folder,'rainfall/')      
    download_day = datetime.today()
    year_=str(download_day.year)
    ftp = FTP('ftp.cdc.noaa.gov')
    ftp.login(user='anonymous', passwd = 'anonymous')
    path1='/Projects/Reforecast2/%s/' % year_
    ftp.cwd(path1)
    folderlist = ftp.nlst()
    path1_='%s/' % folderlist[-1]
    ftp.cwd(path1_)
    folderlist = ftp.nlst()
    try:
        path2='%s/c00/latlon/' % folderlist[-1]
        ftp.cwd(path2)
        filelist = ftp.nlst()
        for file in filelist:
            if ((file_pattern in file) and file.endswith('.grib2')):
                ftp.retrbinary("RETR " + file, open(os.path.join(rainfall_path,'rainfall_forecast.grib2'),"wb").write)
                print(file + " downloaded")
        #downloadRainfallFiles(rainfall_path,ftp)
        rainfall_error=False
    except:
        rainfall_error=True
        pass
    ftp.quit()
    
def download_rainfall_nomads(Input_folder,path,Alternative_data_point):
    """
    download rainfall 
    
    """
    if not os.path.exists(os.path.join(Input_folder,'rainfall/')):
        os.makedirs(os.path.join(Input_folder,'rainfall/'))
    
    rainfall_path=os.path.join(Input_folder,'rainfall/')
 
    url='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Input_folder.split('/')[-3][:-2] #datetime.now().strftime("%Y%m%d")
    url2='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Alternative_data_point #datetime.now().strftime("%Y%m%d")
    
    def listFD(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + node.get('href') for node in soup.find_all('a') if node.get('href').split('/')[-2] in ['00','06','12','18']]#.endswith(ext)]
    
    try:        
        base_url=listFD(url, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            print(rain_file)
            output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(path)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=path)
    except:
        base_url=listFD(url2, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(path)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=path)
        
    rain_files = [f for f in listdir(rainfall_path) if isfile(join(rainfall_path, f))]
    os.chdir(rainfall_path)
    pattern1='.pgrb2a.0p50.bc_06h'
    pattern2='.pgrb2a.0p50.bc_24h'
    for files in rain_files:
        if pattern2 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_24.nc'%files ,cwd=rainfall_path)
            os.remove(files)
        if pattern1 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_06.nc'%files ,cwd=rainfall_path)
            os.remove(files)


def hk_data(Input_folder):
    HKfeed =  BeautifulSoup(requests.get('https://www.weather.gov.hk/wxinfo/currwx/tc_list.xml').content,parser=parser,features="lxml")#'html.parser')
    trac_data=[]
    try:
        HK_track =  BeautifulSoup(requests.get(HKfeed.find('tropicalcycloneurl').text).content,parser=parser,features="lxml")#'html.parser')    
        for WeatherReport in HK_track.find_all('weatherreport'):
            for forecast in WeatherReport.find_all('pastinformation'):
                l2=[forecast.find('index').text,forecast.find('latitude').text,forecast.find('longitude').text,forecast.find('time').text]
                print(l2)
                trac_data.append(l2)
                last_item=forecast.find('time').text
            for forecast in WeatherReport.find_all('forecastinformation'):
                l2=[forecast.find('index').text,forecast.find('latitude').text,forecast.find('longitude').text,last_item]
                print(l2)
                trac_data.append(l2)
        trac_data.to_csv(os.path.join(Input_folder,'HK_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
    except:
        pass
def jtcw_data(Input_folder):
    parser = ET2.XMLParser(recover=True)#
    output=[] 
    index_list=[' WARNING POSITION:',' 12 HRS, VALID AT:',' 24 HRS, VALID AT:',' 36 HRS, VALID AT:',' 48 HRS, VALID AT:',' 72 HRS, VALID AT:']
    index_list_id=[]
    index_list_wd=[]
    
    jtwc_content = BeautifulSoup(requests.get('https://www.metoc.navy.mil/jtwc/rss/jtwc.rss').content,parser=parser,features="lxml")#'html.parser')
    try:    
        for channel in jtwc_content.find_all('channel'):
            for item in channel.find_all('item'):
                for li in item.find_all('li'):
                    for href_li in li.find_all('a',href=True):
                        if href_li.text =='TC Warning Text ':
                            output.append(href_li['href'])        
        jtwc_content = BeautifulSoup(requests.get(output[0]).content,'html.parser')#parser=parser,features="lxml")#'html.parser')
        jtwc_=re.sub(' +', ' ', jtwc_content.text)
        listt=jtwc_.split('\r\n')
        listt=listt[listt.index(' WARNING POSITION:'):]        
        for i in index_list:
            index_list_id.append(listt[listt.index(i)+1].replace("NEAR ", "").replace("---", ","))            
        for i in listt:
            if (' '.join(i.split()[0:3])=='MAX SUSTAINED WINDS'):
                i_l=i.replace(",", "").split()
                index_list_wd.append(','.join([i_l[-5],i_l[-2]]))              
        jtwc_df = pd.DataFrame(index_list_id)
        jtwc_df['wind']=index_list_wd
        #jtwc_.split('\r\n')[2].strip('/')  name of the event 
        jtwc_df.to_csv(os.path.join(Input_folder,'JTCW_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
        #jtwc_df.to_csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/philipiness/jtwc_df.csv',index=False)
    except:
        pass