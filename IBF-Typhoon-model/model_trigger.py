# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 18:13:20 2021

@author: ATeklesadik
"""
import requests
import json

#URL = "https://prod-160.westeurope.logic.azure.com:443/workflows/5d45560f191b4dc5a425144065adea90/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=L58J5R7qEwDqLJ0-lojIw649WxHl8djdiKJKn5TJwNk"
URL = "https://prod-255.westeurope.logic.azure.com:443/workflows/6112d9aa0aba41f4a1f27a765f88a87a/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=bLyROGysu9vbeKkMfIC1zT3_pSloP60_h4Ai81bOY6U"
data= '{"address":{"FirstName":"string","Organization":"OCHA","passCode":"2365879236548568"}}'

y = json.loads(data)  

r = requests.post(URL, json=y)


print(r.content)
