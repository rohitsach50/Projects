import time
import os
import requests


url = 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key={API_KEY}&format=csv&offset=0&limit=all'
myfile = requests.get(url)

open('dex.csv', 'wb').write(myfile.content)


f_path = 'dex.csv'

t = os.path.getctime(f_path)

t_str = time.ctime(t)

t_obj = time.strptime(t_str)

form_t = time.strftime("%Y-%m-%d %H:%M:%S", t_obj)

form_t = form_t.replace(':', '˸') #Colon replaced with special unicode character U+02F8

os.rename(f_path, os.path.split(f_path)[0] + '' + form_t + os.path.splitext(f_path)[1])
