import pandas as pd
import glob
import os
import numpy as np
import mysql.connector as sqlcon
from geopy.geocoders import ArcGIS
from geopy.geocoders import Bing
from datetime import datetime


n=ArcGIS()
nom=Bing(api_key="AiCa2yYZihcaJZZOjc-O2p8PjoTvdkkGVk8fscrgkhyYgN7KMn0btM907NW8B9Rs")


# conn = mysql.connector.connect(user = '', password = '', host = '', port = '3306')
mydb = sqlcon.connect(user = 'root', password = '', host = 'localhost', port = '3306')
now=datetime.now()
dt_string = now.strftime("%Y/%m/%d %H:%M:%S")


dir = "/Users/Rock/Desktop/data/"
file_paths=[]
for file in glob.iglob(f'{dir}/*.csv'):
    file_paths.append(file)
print(file_paths)


def file_to_df(path):
    df = pd.read_csv(path, index_col=0)
    df.drop(columns=['pollutant_min','pollutant_max','last_update'],inplace=True)
    
    city_df = df[['city','station','state','country']]
    by_city = city_df.groupby(['city','station','state'])
    
    country = df['country'].unique()[0]
    Date = dt_string
    
    
    df2 = by_city['country'].unique()
    df2.to_csv("test.csv")
    df2 = pd.read_csv("test.csv")
    city = df2.drop(columns=['country'],inplace=True) 
# df2 contains data related to station,state,city and country
    df2['country']=country
    os.remove('test.csv')
    
    
    tsil = ['PM2.5','PM10','NO2','NH3','SO2','CO','OZONE']
    df3 = pd.DataFrame()

    df3['poll_id'] =df['pollutant_id']
    df3['poll_avg'] = df['pollutant_avg']

    df4 = pd.DataFrame()
    
    df4['poll_id'] = [x for x in tsil]*len(df2['station'])
    df4['poll_avg'] = 0
    df4['poll_avg'].replace(0,np.nan,inplace=True)
# df4 contains data related to pollutants
    
    df4_lnth = 0
    df3_lnth = 0
    tot= len(df4['poll_id'])
    while df4_lnth < tot:
        if df3['poll_id'].iloc[df3_lnth] == df4['poll_id'].iloc[df4_lnth]:

            data = df3['poll_avg'].iloc[df3_lnth]
            df4['poll_avg'].iloc[df4_lnth]=data

            df4_lnth = df4_lnth+1
            df3_lnth = df3_lnth+1
        else:
            df4_lnth = df4_lnth+1
            
            
    def df_to_sql():
        
        query = mydb.cursor()
        query.execute("USE test_db;")
       

        for city,station,state,contry in zip(df2['city'],df2['station'],df2['state'],df2['country']):
            try:
                query.execute(f"INSERT INTO city_info (City_name,Station_Name,State,Country_Name ) VALUES('{city}','{station}','{state}','{contry}');")
            except Exception as e:
                print(e)
            else:
                print("New Record Inserted into city_info Table")
            finally:
                mydb.commit()
                
        
        try:
            query.execute(f"INSERT INTO countries (Country_Name) VALUES('{country}');")
        except:
            print(f"Country Name:--> {country} already exist in countries table")
        else:
            print(f"Country Name:--> {country} inserted into countries table")
        finally:
            mydb.commit()

            
        
        by_poll_id = df4.groupby('poll_id')
        pm25 = by_poll_id.get_group('PM2.5')
        pm10 = by_poll_id.get_group('PM10')
        no2 = by_poll_id.get_group('NO2')
        nh3 = by_poll_id.get_group('NH3')
        so2 = by_poll_id.get_group('SO2')
        co = by_poll_id.get_group('CO')
        ozone = by_poll_id.get_group('OZONE')

        for p25,p10,n2,n3,s2,c,oz,st in zip(pm25['poll_avg'],pm10['poll_avg'],no2['poll_avg'],nh3['poll_avg'],so2['poll_avg'],co['poll_avg'],ozone['poll_avg'],df2['station']):            
            try:
                query.execute(f"INSERT INTO air_quality_info (Station_Name,Date,PM25,PM10, NO2,NH3,SO2,CO,OZONE) VALUES('{st}','{Date}','{p25}','{p10}','{n2}','{n3}','{s2}','{c}','{oz}');")
            except Exception as e:
                print(e)
            else:
                print("New Record Inserted into air_quality_info Table")
        
            finally:
                mydb.commit()
        
    
    return df_to_sql()

file_to_df(file_paths[0])

mydb.close()
