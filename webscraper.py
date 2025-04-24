import requests
from bs4 import BeautifulSoup
import json
import os
import pandas as pd
from tqdm import tqdm
from utils import get_headers
import pytesseract
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import re

#Set Tesseract path 
pytesseract.pytesseract.tesseract_cmd = r'D:\hr round\tesseract.exe'  # Windows example


def preprocess_image_for_ocr(img):
    """Preprocess image to improve OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Remove noise
    denoised = cv2.medianBlur(thresh, 3)
    return Image.fromarray(denoised)

def extract_menu_data(image_url):
    """Extract structured menu data from a JPG image URL."""
    try:
        # Fetch and preprocess the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        processed_img = preprocess_image_for_ocr(img)
        
        # Custom Tesseract configuration (focus on menus)
        custom_config = r'--oem 3 --psm 6 -l eng'
        extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Parse text into structured dictionary
        menu_data = {}
        current_category = None
        
        for line in extracted_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Detect category (uppercase, bold, or followed by a colon)
            if (line.isupper() or 
                re.match(r'^[A-Z][A-Z\s&]+$', line) or 
                re.match(r'^.*[:：]$', line)):
                current_category = line.strip(':：').title()
                menu_data[current_category] = []
            elif current_category:
                # Detect menu items (ignore prices for now)
                item = re.sub(r'\s{2,}', ' ', line)  # Remove extra spaces
                if item:
                    menu_data[current_category].append(item)
        
        return {"Categories": menu_data}
    
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}



def fetch_html(url):
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url} due to {e}")
        return None
    
    
    
def parse_restaurant_info(html, name, url,jpg_list):
        soup = BeautifulSoup(html, 'html.parser')
       # Initialize the data structure
        restaurant_data = {
            'basic_info': {},
            'rating': {},
            'menu':[],
            }
        # Extract basic info
        basic_info = {}
        name_tag = soup.find('h1', class_='sc-7kepeu-0')
        basic_info['name'] = name_tag.get_text(strip=True) if name_tag else 'N/A'
        
        address_tag = soup.find('div', class_='sc-clNaTc ckqoPM')
        basic_info['address'] = address_tag.get_text(strip=True) if address_tag else 'N/A'
        
        restaurant_data['basic_info'] = basic_info
        
        # Extract highlights
        known_for = soup.find('div', string=lambda text: text and "People Say This Place Is Known For" in text)
        if known_for:
          parent = known_for.find_parent('div')
          highlights_div = parent.find('div', class_='sc-bFADNz inYxft')  
          restaurant_data['highlights'] = highlights_div.get_text(strip=True) if highlights_div else 'N/A'
        else:
          restaurant_data['highlights'] = 'N/A'
        
        # Extract average cost
        cost_tag = soup.select_one('p.sc-1hez2tp-0.sc-hacOGl.iVRrnK')

        if not cost_tag:

         cost_tag_candidates = soup.find_all('p')
         for tag in cost_tag_candidates:
            text = tag.get_text(strip=True)
            if 'for two' in text or 'approx.' in text:
              cost_tag = tag
              break
        restaurant_data['average_cost'] = cost_tag.get_text(strip=True) if cost_tag else 'N/A'

        
        # Extract timings
        timings_tag = soup.find('span', class_='sc-kasBVs dfwCXs')
        restaurant_data['timings'] = timings_tag.get_text(strip=True) if timings_tag else 'N/A'
        
        # Extract rating info
        rating_info = {}
        rating_tag = soup.find('div', class_='sc-1q7bklc-5')
        rating_info['average'] = rating_tag.get_text(strip=True) if rating_tag else 'N/A'
        
        rating_count_tag = soup.find('div', class_='sc-1q7bklc-8')
        if rating_count_tag:
            rating_info['total_ratings'] = rating_count_tag.get_text(strip=True).replace('(', '').replace(')', '')
        
        restaurant_data['rating'] = rating_info
        
      
               
        # Extract more info
        amenities = []
        amenity_items = soup.find_all('div', class_='sc-bke1zw-1')
        
        for item in amenity_items:
            text_element = item.find('p', class_='sc-1hez2tp-0')
            if text_element:
                amenity = text_element.get_text(strip=True)
                if amenity:
                    amenities.append(amenity)
        
        restaurant_data['more_info'] = amenities
        
        
        
        #Extract cuisines    
        cuisine_container = soup.find('div', class_='sc-bFADNz kFXYlm')
        if not cuisine_container:
            restaurant_data['cuisines_data'] = []
        else:
         cuisines = []
         cuisine_links = cuisine_container.find_all('a', class_='sc-bFADNz cWYoZb')
        
         for link in cuisine_links:
           
            cuisine = link.get('title')
            if cuisine:
                    cuisines.append(cuisine)
        
         restaurant_data['cuisines_data'] = cuisines
         
        #Extract Top dishes
        dishes_containers = soup.find_all('div', class_='sc-bFADNz jQsfZN')
        if not dishes_containers:
            restaurant_data['Topdishes_data'] = []
        restaurant_data['Topdishes_data'] = []
        if len(dishes_containers) > 1:
       
         dish_container = dishes_containers[1]  # Change index if needed
         if not dish_container:
            restaurant_data['Topdishes_data'] = []
         else:
          dishes = []
          dish_links = dish_container.find_all('a', class_='sc-bFADNz cWYoZb')
        
          for link in dish_links:
           
            dish = link.get('title')
            if dish:
                    dishes.append(dish)
        
         restaurant_data['Topdishes_data'] = dishes
         
        #Extract menu
         for x in jpg_list:
             menu_data = extract_menu_data(x)
             restaurant_data['menu'].append(menu_data) 
         

        return restaurant_data    
    
    
def main():
    with open('scraper/restaurant_list.json', 'r') as f:
        restaurants = json.load(f)
   

    results = []
    for r in tqdm(restaurants):
        html = fetch_html(r['url'])
        if html:
            data = parse_restaurant_info(html, r['name'], r['url'],r['jpg_list'])
            results.append(data)

    os.makedirs('data/processed',exist_ok=True)
    with open('data/processed/scraped_restaurants.json', 'w') as f:
        json.dump(results, f, indent=2)

    

if __name__ == "__main__":
    main()
    