import requests
from bs4 import BeautifulSoup
import time
import os
import asyncio
import aiohttp
import pandas as pd
import asyncio
from aiohttp import ClientSession, ClientResponseError
from tqdm import tqdm

def parser(path):
    """function to parse the website and scraping informations about
    the URL of the Msc Degrees

    Args:
        path (str): name of the .txt output file with the list of the 6000 URLs
    """
    if not os.path.exists(path):
        with open(path, 'w') as file:
            # scraping first page
            response = requests.get('https://www.findamasters.com/masters-degrees/msc-degrees')
            soup = BeautifulSoup(response.text, 'html.parser')
            course_link = soup.find_all('a', class_='courseLink')
            for link in course_link:
                file.write(f"www.findamasters.com{link.get('href')}\n") # writing URLs in the txt file

            # if response.status_code == 200:
            for i in tqdm(range(2, 401), desc="Processing"):
                # scraping pages from 2 to 400
                response = requests.get(f'https://www.findamasters.com/masters-degrees/msc-degrees/?PG={i}')
                soup = BeautifulSoup(response.text, 'html.parser')
                course_link = soup.find_all('a', class_='courseLink')
                # writing URL's in the txt file
                for link in course_link:
                    file.write(f"www.findamasters.com{link.get('href')}\n")

                time.sleep(1)
                # adding a time.sleep of 1 second is important to avoid sending too many requests to the website
    else:
        return("The file already exists.")
    

def populate_df(pth, path, fold, df):
    # Populate the dataframe
    with open(path, 'r') as file:
        urls = [line.strip() for line in file] # creating a list with the 6000 URLs from the lines of course_links.txt

    if not os.path.exists("files_tsv"):
        os.mkdir("files_tsv")

    count = (15 * fold) + 1
    count_na_rows = 0
    for folder in tqdm(range(fold + 1, 401), desc="Processing"): # loop for every folder
        for file in range(1, 16): # loop for every file
            file_path = os.path.join(pth, f"folder_{folder}", f"page_{file}.html")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as fl:
                soup = BeautifulSoup(fl, 'html.parser')
                if soup.title.text == r"FindAMasters | 500 Error : Internal Server Error":
                    courseName = universityName = facultyName = isItFullTime = description = startDate = fees = modality = duration = city = country = administration = ""
                    count_na_rows += 1
                else:
                    courseName = soup.find("h1", {"class": "course-header__course-title"})
                    if courseName is None:
                        courseName = ""
                    else:
                        courseName = courseName.get_text(strip = True)
                    universityName = soup.find("a", {"class": "course-header__institution"}).get_text(strip = True)
                    facultyName = soup.find("a", {"class": "course-header__department"}).get_text(strip = True)
                    extract = soup.find("span", {"class": "key-info__study-type"})
                    if extract is None:
                        isItFullTime = ""
                    else:
                        isItFullTime = extract.get_text(strip = True)
                    description = soup.find("div", {"class": "course-sections__description"}).find("div", {"class": "course-sections__content"}).get_text(strip = True)
                    startDate = soup.find("span", {"class": "key-info__start-date"}).get_text(strip = True)
                    # some entries do not have this field
                    extract = soup.find("div", {"class": "course-sections__fees"})
                    if extract is None:
                        fees = ""
                    else:
                        fees = extract.find("div", {"class": "course-sections__content"}).get_text(strip = True)
                    modality = soup.find("span", {"class": "key-info__qualification"}).get_text(strip = True)
                    duration = soup.find("span", {"class": "key-info__duration"}).get_text(strip = True)
                    city = soup.find("a", {"class": "course-data__city"}).get_text(strip = True)
                    country = soup.find("a", {"class": "course-data__country"}).get_text(strip = True)
                    extract1 = soup.find("a", {"class": "course-data__online"})
                    extract2 = soup.find("a", {"class": "course-data__on-campus"})
                    if extract1 is None and extract2 is None:
                        administration = ""
                    elif extract2 is None:
                        administration = extract1.get_text(strip = True)
                    elif extract1 is None:
                        administration = extract2.get_text(strip = True)
                    else:
                        administration = extract1.get_text(strip = True) + " & " + extract2.get_text(strip = True)
                        
                url = urls[count-1]
                new_row_data = {"courseName": courseName,
                                "universityName": universityName,
                                "facultyName": facultyName,
                                "isItFullTime": isItFullTime,
                                "description": description,
                                "startDate": startDate,
                                "fees": fees,
                                "modality": modality,
                                "duration": duration,
                                "city": city,
                                "country": country,
                                "administration": administration,
                                "url" : url
                                }
                df = pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
                file_path = os.path.join('./files_tsv', f'course{count}.tsv')
                pd.DataFrame([new_row_data]).to_csv(file_path, sep='\t', index=False)
                # print(f"Created course{count}.tsv, {count}/6000, folder = {folder}, file = {file}")
                count += 1

    print(f"There are {count_na_rows} links corrupted.")
    return(df)