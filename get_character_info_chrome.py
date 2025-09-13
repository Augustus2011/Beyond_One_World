from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import json
import os
from typing import Optional, List, Dict, Any

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import json
import os
import re


import pandas as pd
# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.binary_location = '/usr/bin/' #chromedriver chromium-browser

# # Create a Service object
# service = Service(executable_path='/usr/local/bin/chromedriver')  # If chromedriver is in PATH

# #driver = webdriver.Chrome(service=service, options=options)

driver = webdriver.Chrome(options=options)


def get_character_id(character_name):
    url = 'https://www.personality-database.com/search?keyword='
    url += character_name.replace(" ", "%20")
    print(url)
    driver.get(url)
    driver.implicitly_wait(10)
    
    # Find elements with the specified class name
    link = driver.find_elements(By.CLASS_NAME, "profile-card-link")
    if not link:
        print("No character found.")
        return None
    
    target = []
    for item in link:
        html_content = item.get_attribute('outerHTML')
        soup = BeautifulSoup(html_content, 'html.parser')
        profile_link = soup.find('a', class_='profile-card-link')['href']
        print(f"Found profile link: {profile_link}")
        if character_name.split(" ")[0].lower() in profile_link:
            profile_number = profile_link.split('/')[2]
            target.append(profile_number)
    
    if not target:
        print("No matching character ID found.")
        return None
    return target


def clean_sub_category(txt):
    # ex txt = "448|||Injustice: Gods Among Us,,,11125|||Scribblenauts,,,31227|||Multiversus,,,43873|||LEGO Dimensions"
    match = re.search(r'\d+\|\|\|([^,|]+)', txt)

    if match:
        return match.group(1)
    else:
        return txt

def get_character_info(id):
    if id is None:
        print("No ID provided for character info.")
        return None
    
    total = {}
    url = f"https://api.personality-database.com/api/v1/profile/{id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for ID {id}.")
        return None
    
    json_data = response.json()
    name=json_data.get("mbti_profile", "N/A")
    if(len(name.split("."))>1):
        name=name.split(".")[0]
    
    re_subcategory=clean_sub_category(json_data.get("related_subcategories","N/A"))
    category=[]
    category.append(json_data.get("subcategory","N/A"))
    if len(re_subcategory)>2:
        category.append(re_subcategory)

    description=json_data.get("wiki_description", "N/A")
    result = {
        "character": name,
        "source": category,
        "description": description,
        "cid":id,
    }
    
    file_name = f"characters/{name.replace(' ', '')}_{id}.json"
    if len(file_name.split("/"))>=3:
        name=os.path.basename(file_name)
        file_name=f"characters/{name.replace(' ', '')}_{id}.json"
        with open(file_name, 'w', encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
    else:
        with open(file_name, 'w', encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
    
    print(f"Character information saved to {file_name}")
    return name,description,category

if __name__ == "__main__":
    import time
    df=pd.read_csv("filter_character_data.csv")
    all_names=[]
    all_descriptions=[]
    all_category=[]
    for i in df["CID"]:
        try:
            character_id=int(i)
            if character_id:
                name,description,category = get_character_info(character_id)
                all_names.append(name)
                all_descriptions.append(description)
                all_category.append(category)
                #get_character_id(character_info.get("character"))
                time.sleep(1)
                #print(f"Character Info: {character_info}")
                
        except Exception as e:
            print(f"Error in main execution: {str(e)}")
        finally:
            driver.quit()  # Always close the driver when done
    df["Source"]=all_category
    df["Name"]=all_names
    df["Description (only from pdb)"]=all_descriptions
    df.to_csv("new_character_data.csv",index=False)
    # try:
    #     character_id=2337
    #     if character_id:
    #         character_info = get_character_info(character_id)
    #         #get_character_id(character_info.get("character"))

    #         #print(f"Character Info: {character_info}")
            
    # except Exception as e:
    #     print(f"Error in main execution: {str(e)}")
    # finally:
    #     driver.quit()  # Always close the driver when done