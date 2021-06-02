import gzip
import xml.etree.ElementTree as ET
import csv
import re, os

def sum_date_time(date_time, days=0, hours=0, minutes=0, seconds=0):

    from datetime import datetime  
    from datetime import timedelta 
    
    date_time_year = int(date_time[:4])
    date_time_month = int(date_time[5:7])
    date_time_day = int(date_time[8:10])
    date_time_hour = int(date_time[12:14])
    date_time_minute = int(date_time[15:17])
    date_time_second = int(date_time[18:20])
    
    date_time_formatted = datetime(year=date_time_year, month=date_time_month, day=date_time_day, hour=date_time_hour, minute=date_time_minute, second=date_time_second)
    
    new_date_time_formatted = date_time_formatted + timedelta(days=int(days), hours=int(hours), minutes=int(minutes), seconds=int(seconds))  
    
    new_year = str(new_date_time_formatted)[:4]
    new_month = str(new_date_time_formatted)[5:7]
    new_day = str(new_date_time_formatted)[8:10]
    new_hour = str(new_date_time_formatted)[11:13]
    new_minute = str(new_date_time_formatted)[14:16]
    new_second = str(new_date_time_formatted)[17:19]
    
    return new_year+'/'+new_month+'/'+new_day+', '+new_hour+':'+new_minute+':'+new_second



def cyclone_in_sublists(cyclone_name, big_list):
    in_big_list = 0
    which_sublist = 0
    for s,sub_list in enumerate(big_list):
        if cyclone_name in sub_list:
            in_big_list = 1
            which_sublist = s
    return in_big_list, which_sublist



def CXML_to_csv(data, cyclone_possible_names, results_folder):
    
    list_failed1 = []
    # Institute name from filename
    institute_name = os.path.basename(data).split('_')[3].lower()

    # Read file
    if data.endswith('.xml'):
        try:
            tree = ET.parse(data)
            root = tree.getroot()
            file_valid = 1
        except:
            file_valid = 0
            pass

    elif data.endswith('.gz'):
        try:
            tree = ET.parse(gzip.open(data))  
            root = tree.getroot()
            file_valid = 1
        except:
            file_valid = 0
            pass

    # Check how many files could not be parsed
    if file_valid == 1:
    
        try:
            model_name=root.find('header/generatingApplication/model/name').text 
        except:
            model_name='NAN'
            pass
        # print(len(list_failed1))
        prod_center=root.find('header/productionCenter').text
        baseTime=root.find('header/baseTime').text

        ## Create one dictonary for each time point, and append it to a list

        for members in root.findall('data'):
    
            Mtype=members.get('type')
     
            for members2 in members.findall('disturbance'):
                try: 
                    cyclone_name = [name.text.lower().strip() for name in members2.findall('cycloneName')]
                except:
                    cyclone_name = [' ']
            
                if cyclone_name in cyclone_possible_names:
            
                    list_data = []
    
                    if Mtype in ['forecast','ensembleForecast']:
                        for members3 in members2.findall('fix'):
                            tem_dic={}
                            tem_dic['Mtype']=[Mtype]
                            tem_dic['institute_name']=[institute_name.lower()]
                            tem_dic['product']=[re.sub('\s+',' ',prod_center).strip().lower()]
                            if model_name != 'NAN':
                                tem_dic['model_name']=[model_name.lower()]
                            else:
                                tem_dic['model_name'] = tem_dic['product']
                            tem_dic['basin'] = [name.text for name in members2.findall('basin')]
                            tem_dic['cycloneName'] = cyclone_name
                            tem_dic['cycloneName2'] = [cyclone_names[cyclone_in_sublists(cyclone_name[0],cyclone_possible_names)[1]]]
                            tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                            tem_dic['ensemble']=[members.get('member')]#[member]
                            tem_dic['cyc_speed'] = [name.text for name in members3.findall('cycloneData/maximumWind/speed')]
                            #tem_dic['cyc_speed'] = [name.text for name in members3.findall('cycloneData/minimumPressure/pressure')]
                            tem_dic['cyc_cat'] = [name.text for name in members3.findall('cycloneData/development')]
                            time = [name.text for name in members3.findall('validTime')]
                            tem_dic['time'] = ['/'.join(time[0].split('T')[0].split('-'))+', '+time[0].split('T')[1][:-1]]
                            tem_dic['lat'] = [name.text for name in members3.findall('latitude')]
                            tem_dic['lon']= [name.text for name in members3.findall('longitude')]                
                            tem_dic['vhr']=[members3.get('hour')]
        #                     validt=tem_dic['validt'][0].split('-')[0]+tem_dic['validt'][0].split('-')[1]+tem_dic['validt'][0].split('-')[2][:2]+tem_dic['validt'][0].split('-')[2][3:5]
        #                     date_object = datetime.strptime(validt, "%Y%m%d%H")
        #                     s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
        #                     tem_dic['time']=[s1]
        #                     validt2=members2.get('ID').split('_')[0]
        #                     date_object = datetime.strptime(validt2, "%Y%m%d%H")
        #                     s2 = date_object.strftime("%m/%d/%Y, %H:00:00")
        #                     tem_dic['validt2']=[s2] 
                            tem_dic['forecast_time'] = ['/'.join(baseTime.split('T')[0].split('-'))+', '+baseTime.split('T')[1][:-1]]
                            tem_dic1 = dict( [(k,''.join(str(e).lower().strip() for e in v)) for k,v in tem_dic.items()])
                            list_data.append(tem_dic1)
                
                    elif Mtype=='analysis':

                        tem_dic={}
                        tem_dic['Mtype']=['analysis']
                        tem_dic['institute_name']=[institute_name.lower()]
                        tem_dic['product']=[re.sub('\s+',' ',prod_center).strip().lower()]
                        if model_name != 'NAN':
                            tem_dic['model_name']=[model_name.lower()]
                        else:
                            tem_dic['model_name'] = tem_dic['product']
                        tem_dic['basin']= [name.text for name in members2.findall('basin')]
                        tem_dic['cycloneName'] = cyclone_name
                        tem_dic['cycloneName2'] = [cyclone_names[cyclone_in_sublists(cyclone_name[0],cyclone_possible_names)[1]]]
                        tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                        tem_dic['ensemble']=['NAN']
                        tem_dic['cyc_speed'] = [name.text for name in members2.findall('cycloneData/maximumWind/speed')]
                        #tem_dic['cyc_speed'] = [name.text for name in members2.findall('cycloneData/minimumPressure/pressure')]
                        tem_dic['cyc_cat'] = [name.text for name in members2.findall('cycloneData/development')]
                        time = [name.text for name in members2.findall('fix/validTime')]
                        tem_dic['time'] = ['/'.join(time[0].split('T')[0].split('-'))+', '+time[0].split('T')[1][:-1]]
                        tem_dic['lat'] = [name.text for name in members2.findall('fix/latitude')]
                        tem_dic['lon']= [name.text for name in members2.findall('fix/longitude')]
                        tem_dic['vhr']=[members2.get('hour')]
        #                 validt=tem_dic['validt'][0].split('-')[0]+tem_dic['validt'][0].split('-')[1]+tem_dic['validt'][0].split('-')[2][:2]+tem_dic['validt'][0].split('-')[2][3:5]
        #                 date_object = datetime.strptime(validt, "%Y%m%d%H")
        #                 s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
        #                 tem_dic['validt']=[s1]
        #                 validt2=members2.get('ID').split('_')[0]
        #                 date_object = datetime.strptime(validt2, "%Y%m%d%H")
        #                 s2 = date_object.strftime("%m/%d/%Y, %H:00:00")
        #                 tem_dic['validt2']=[s2]
                        tem_dic['forecast_time'] = ['/'.join(baseTime.split('T')[0].split('-'))+', '+baseTime.split('T')[1][:-1]]
                        tem_dic1 = dict( [(k,''.join(str(e).lower().strip() for e in v)) for k,v in tem_dic.items()])
                        list_data.append(tem_dic1)

                    # Save the databases to the csv files (one for each institute)

                    # Define csv file
                    csv_file = results_folder+cyclone_name[0]+'_all.csv'

                    # Headers
                    csv_columns = tem_dic1.keys()

                    if os.path.exists(csv_file):
                        append_write = 'a' # append if already exists
                    else:
                        append_write = 'w' # make a new file if not

                    # Write data
                    try:
                        with open(csv_file, append_write) as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            if append_write == 'w':
                                writer.writeheader()
                            for row in list_data:
                                writer.writerow(row)
                    except IOError:
                        return "I/O error"