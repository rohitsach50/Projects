from bs4 import BeautifulSoup
import requests
from selenium import webdriver

driver=webdriver.Firefox()

driver.get("https://www.imdb.com/search/title/?groups=oscar_winner,best_picture_winner&sort=year,desc&count=250")

html_text = driver.execute_script("return document.documentElement.outerHTML")
soup = BeautifulSoup(html_text, 'lxml')

for q in soup.find_all('div', class_ = 'lister-item-content'):
    z=q.h3.text
    winner=z.replace("\n","")
    print(winner)
    
    certificate = q.p.find('span', class_='certificate').text
    print(f"Certificate: {certificate}")  
    
    runtime = q.p.find('span', class_='runtime').text
    print(f"Runtime: {runtime}")
    
    genre = q.p.find('span', class_='genre').text
    print(f"Genre: {genre}")
    
    print("---------------------------------------------------------------------------")
