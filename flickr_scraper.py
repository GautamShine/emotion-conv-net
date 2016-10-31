#! /usr/bin/env python

from bs4 import BeautifulSoup
import requests
import re
import shutil

url ='https://www.flickr.com/photos/osucommons/3654953163/'
response = requests.get(url)
html = response.content
soup = BeautifulSoup(html, 'lxml')

target = soup.find_all(class_='low-res-photo')
target_url = re.findall('//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(target))
target_url = 'http://' + target_url[0][2:]

response = requests.get(target_url, stream=True)
with open(target_url[-10:], 'wb') as saved_img:
  shutil.copyfileobj(response.raw, saved_img)
del response
