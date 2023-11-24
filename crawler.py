import requests
from bs4 import BeautifulSoup
import time
import os
import asyncio
import aiohttp
import pandas as pd
import asyncio
from aiohttp import ClientSession, ClientResponseError

async def get_info(url, session, folder, page_number):
    try:
        resp = await session.request(method="GET", url="https://" + url)
        resp.raise_for_status()
        html = await resp.text(encoding='utf-8')
        
        # Create the path in the corresponding folder
        file_path = os.path.join(folder, f"page_{int(page_number)}.html")
        
        # Write the html page in the right folder
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return html
    
    except ClientResponseError as e:
        if e.status == 429: # Error 429: too many requests.
            print(f"Received 429 error. Too many requests. Waiting for a while...")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
            return await get_info(url, session, folder, page_number)
        else:
            raise e  # Re-raise other ClientResponseError
        
async def process_batch(urls, session, folder):
    tasks = [get_info(url, session, folder, page_number = count) for count, url in enumerate(urls, start=1)]
    return await asyncio.gather(*tasks)

async def crawler(urls, batch_size=15, starting_folder=0):
    main_directory = "master_programs_html"
    os.makedirs(main_directory, exist_ok=True)
    
    async with ClientSession() as session:
        count_folder = starting_folder + 1
        for i in range(starting_folder*batch_size, len(urls), batch_size):
            # selecting the URLs from urls variable
            batch_urls = urls[i:i + batch_size]
            # Creating a sub-folder for every batch
            folder_name = os.path.join(main_directory, f"folder_{count_folder}")
            os.makedirs(folder_name, exist_ok=True)
            count_folder += 1
            # Downloading and writing file HTML in the batch
            await process_batch(batch_urls, session, folder_name)
