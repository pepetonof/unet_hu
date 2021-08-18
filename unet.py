# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:15:33 2021

@author: josef
"""

# import urllib.request as req

# inputs='https://github.com/pepetonof/unet_hu/tree/main/input'
# masks ='https://github.com/pepetonof/unet_hu/tree/main/target'
    
# import requests
# from bs4 import BeautifulSoup
# import re

# # URL on the Github where the csv files are stored
# github_url = 'https://raw.githubusercontent.com/pepetonof/unet_hu/tree/main/input'
# result = requests.get(github_url)

# soup = BeautifulSoup(result.text,'html.parser')
# files = soup.find_all(title=re.compile("\.png$"))

# filename = []
# for i in files:
#         filename.append(i.extract().get_text())
        
# print(filename)



from skimage import io
url='https://github.com/pepetonof/unet_hu/blob/main/input/alejandra%20(1).png'

image = io.imread(url)
io.imshow(image)
io.show()