# Import libraries
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# tutorial form https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460
# scrape this: https://www.folkhalsomyndigheten.se/nyheter-och-press/nyhetsarkiv?topic=covid-19
# https://www.folkhalsomyndigheten.se/nyheter-och-press/nyhetsarkiv?topic=covid-19&page=2
# page number 10000 (more than existing) goes to last page

# Set the URL you want to webscrape from
url = 'http://web.mta.info/developers/turnstile.html'

# Connect to the URL
response = requests.get(url)

# Parse HTML and save to BeautifulSoup object¶
soup = BeautifulSoup(response.text, "html.parser")

# To download the whole data set, let's do a for loop through all a tags
line_count = 1 #variable to track what line you are on
for one_a_tag in soup.findAll('a'):  #'a' tags are for links
    if line_count >= 36: #code for text files starts at line 36
        link = one_a_tag['href']
        
        r = requests.get(link)
        soup = BeautifulSoup(r.content)

        print(soup)
    #add 1 for next line
    line_count +=1