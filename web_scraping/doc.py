import re
import scrapy
from scrapy import Request
import pandas as pd

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext




class DocSpider(scrapy.Spider):
    name = 'doc'
    # allowed_domains = ['https://indmedica.com/directory.php?keywords=']
    start_urls = ['https://indmedica.com/directory.php?keywords=&directory=doctor&action=search&search=Search&allspl=no&catid=25']

    def parse(self, response):
        link_list = response.css('.rslttbl a::attr(href)').extract()
        
        for link in link_list:
            url='https://indmedica.com'+link
            yield Request(url=url,callback=self.doc_link)
        c = 1
        for i in range(1,20):
            c+=10
            next_page = f"https://indmedica.com/directory.php?keywords=&directory=doctor&search=Search&allspl=no&catid=25&num=10&start={c}"
            yield Request(url=next_page,callback=self.n_p)

    def doc_link(self,response):
        html = response.css('.maincolumn').extract()
        cont = cleanhtml(str(html))
        e = cont.replace("[","").replace("]","").split("\\n")
        a = e[2:-2]
        idx=0
        Ph = ""
        Address = ""
        Speciality = ""
        Qualification = ""

        for i in a:

            if i == "Doctor Profile":
                Doctor_Profile=a[idx+1]
            elif "Qualifi" in i:
                Qualification=i
            elif "Phon" in i:
                Ph = i
            elif "Specia" in i:
                Speciality=i
            elif "Address" in i:
                Address = [a[idx+1],a[idx+2],a[idx+3]]
            

            idx+=1


        # nm = response.css('h4::text').extract()
        items={
        "doc_name" : [Doctor_Profile],
        "Qualifications": [Qualification],
        "Speciality": [Speciality],
        "Phone": [Ph], 
        "Address": [Address]
        }
        dt=pd.DataFrame.from_dict(items)
        with open('final.csv','a') as f:
            dt.to_csv(f, sep=",",header=False)
        print("+++++++++++++++++++++++++++++++++++++++++")
        yield items
        

    def n_p(self,response):
        link_list = response.css('.rslttbl a::attr(href)').extract()
        
        for link in link_list:
            url='https://indmedica.com'+link
            yield Request(url=url,callback=self.doc_link)
