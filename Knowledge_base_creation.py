import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import json

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Normalize and clean text for embedding generation."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars/OCR artifacts
    text = re.sub(r'[^\w\s]|[\u200b-\u200f]|[\ufffd]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def extract_prices(text):
    prices = re.findall(r'(?:₹|rs?\.?)\s*(\d{1,3}(?:,\d{3})*)', text, re.IGNORECASE)
    return [int(p.replace(',', '')) for p in prices]


        
def preprocess_restaurant_data(restaurant):

    documents = []
    
    # Basic Info (with error handling)
    name = clean_text(restaurant.get('basic_info', {}).get('name', 'Unknown'))
    address = clean_text(restaurant.get('basic_info', {}).get('address', ''))
    doc = f"Restaurant: {name}. Location: {address}."
    
    # Rating info (with error handling)
    average = clean_text(restaurant.get('average', {}).get('average', 'Unknown'))
    total_ratings = clean_text(restaurant.get('total_ratings', {}).get('total_ratings', ''))
    doc += f"Average rating: {average}. Total rating: {total_ratings}."
    
    # Cuisines (handle missing key)
    cuisines = restaurant.get('cuisines_data', [])
    doc += f" Cuisines: {', '.join([clean_text(c) for c in cuisines])}." if cuisines else ""
    
    # Topdishes (handle missing key)
    topdishes = restaurant.get('Topdishes_data', [])
    doc += f" Topdishes: {', '.join([clean_text(d) for d in topdishes])}." if topdishes else ""
    
     # Moreinfo (handle missing key)
    moreinfo = restaurant.get('more_info', [])
    doc += f"Facilities Provided by restaurant: {', '.join([clean_text(f) for f in  moreinfo ])}." if moreinfo else ""
    
    # Average cost
    averagecost = clean_text(restaurant['average_cost'])
    doc += f"Average cost for two people: {averagecost}."
    
    # Highlights
    highlights = clean_text(restaurant['highlights'])
    doc += f"People Say This Place Is Known For: {highlights}."
    
    # Timmings
    Timming = clean_text(restaurant['timings'])
    doc += f"Restaurant opening timings: {Timming}."
    
    # Menu Processing (fixed)
    menu_items = []
    for menu_page in restaurant.get('menu', []):
        categories = menu_page.get('Categories', {})  # Safe access
        for category, items in categories.items():
            if items and isinstance(items, list):  # Double-check type
                category_clean = clean_text(category)
                items_clean = ' '.join([clean_text(item) for item in items if isinstance(item, str)])
                menu_items.append(f"{category_clean}: {items_clean}")
    
    if menu_items:
        doc += " Menu: " + ' '.join(menu_items)
    
    # Dietary Info Extraction
    dietary_tags = set()
    for item in menu_items:
        if 'vegetarian' in item:
            dietary_tags.add('vegetarian')
        if 'gluten' in item:
            dietary_tags.add('gluten-free')
    
    if dietary_tags:
        doc += f" Dietary options: {', '.join(dietary_tags)}."
    
    documents.append(doc)
    return documents




from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB Setup
# Initialize ChromaDB (persistent)
client = chromadb.PersistentClient(path="data/vectordb")

# Create collection
collection = client.get_or_create_collection(
    name="restaurants",
    metadata={"hnsw:space": "cosine"}
)



def create_knowledge_base(restaurants_data):
   
    ids = []
    documents = []
    metadatas = []
    
    for idx, restaurant in enumerate(restaurants_data):
        docs = preprocess_restaurant_data(restaurant)
        if not docs:  # Skip if no valid documents
            continue
            
        embeddings = model.encode(docs).tolist()
        
        # Prepare metadata with defaults for missing fields
        metadata = {
            "name": restaurant.get('basic_info', {}).get('name', 'Unknown'),
            "cuisines": ", ".join(restaurant.get('cuisines_data', [])),
            "price_range": restaurant.get('average_cost', 'N/A'),
            "rating": safe_float_conversion(restaurant.get('rating', {}).get('average', '0'))
              }
        
        collection.add(
            ids=f"rest_{idx}",
            documents=docs,
            embeddings=embeddings,
            metadatas=metadata
        )
    
    

def safe_float_conversion(rating_str):
    """Convert messy rating strings to float safely"""
    try:
        # Extract first number from strings like "4.5star-fill2,918Dining Ratings"
        clean_num = ''.join(c for c in str(rating_str) if c.isdigit() or c == '.')
        return float(clean_num.split('.')[0][:3])  
    except:
        return 0.0 
    
print("Total entries:", collection.count())
print("Sample documents:", collection.peek())   
    
def query_restaurants(query_text, n_results=3):
    """Safe querying with metadata checks"""
    try:
        query_embedding = model.encode([query_text])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    
        # Safe result printing
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            name = meta.get('name', 'Unknown Restaurant')
            price = meta.get('price_range', 'Price not available')
            print(f"--- {name} ({price}) ---")
            print(doc[:200] + "...")
            
        return results
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return None
    


# Load scraped data
with open('data/processed/scraped_restaurants.json') as f:
    restaurants = json.load(f)

# Create knowledge base
create_knowledge_base(restaurants)

# Sample query
results = query_restaurants("vegetarian pasta under 500 rupees")

