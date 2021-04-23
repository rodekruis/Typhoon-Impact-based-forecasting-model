import re
import shutil
import sys
import os
from lxml import etree
from os.path import relpath
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
from sys import platform
import xml.etree.ElementTree as ET
import lxml.etree as ET2
import pandas as pd

def jtcw_data(Input_folder):
    parser = ET2.XMLParser(recover=True)#
    output=[] 
    index_list=[' WARNING POSITION:',' 12 HRS, VALID AT:',' 24 HRS, VALID AT:',' 36 HRS, VALID AT:',' 48 HRS, VALID AT:',' 72 HRS, VALID AT:',' 96 HRS, VALID AT:',' 120 HRS, VALID AT:']
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
            if i in listt:
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

#%%
 

 