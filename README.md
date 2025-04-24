# Restaurant-Data-Scraper-RAG-based-chatbot


This project is an end-to-end Generative AI solution combining web scraping with a Retrieval-Augmented Generation (RAG) chatbot. It simulates a real-world application for Zomato that allows users to ask natural language questions about restaurants and get accurate, contextual answers based on real data.

---

##  Features

-  Web scraper to extract restaurant data and menus from Zomato
-  OCR-powered extraction from menu images
-  Preprocessing and creation of a vector-based knowledge base using ChromaDB
-  RAG-based chatbot using HuggingFace FLAN-T5 and Sentence Transformers
-  Streamlined Gradio UI for interaction
-  Handles queries like:
  - *“Which restaurant has vegetarian pasta under ₹500?”*
  - *“Compare spice levels of dishes at two restaurants”*
  - *“Which cafes near Sector 144 have gluten-free desserts?”*

---

##  Project Structure
├── data/                  
│   ├── raw/                  # Raw HTML (optional but great for backup)
│   └── processed/            # Final structured JSON/CSV data
│
├── scraper/
│   ├── __init__.py
│   ├── s_main.py             # Main orchestrator for scraping
│   ├── utils.py              # Clean functions: headers, sanitizers, etc.
│   └── restaurant_list.json  # Target restaurant links
│
├── Knowledge_base_creation.py  # Indexing + Preprocessing logic
├── Rag_chatbot_&_UI.py         # Retrieval + Generation + UI (Streamlit/Gradio)
├── README.md                   # Setup + usage guide
└── requirements.txt            # All deps in one place


---

##  Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/yourusername/zomato-rag-chatbot.git
cd zomato-rag-chatbot
```

2. Install dependencies
 ```bash
pip install -r requirements.txt
```
Note: Make sure tesseract is installed and configured. Modify the path in s_main.py if you're on Windows:
 ```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

```

3.Scrape restaurant data
 ```bash
python scraper/s_main.py

```
This will fetch and store structured restaurant data into data/processed/scraped_restaurants.json.

4. Build the Knowledge Base
```bash
python Knowledge_base_creation.py
```
This step vectorizes and stores data in a persistent ChromaDB instance.

5.Implement the Chatbot and launch the user Interface
```bash
python Rag_chatbot_&_ui.py
```
This will implement the chatbot and launch a Gradio UI where you can interact with the chatbot.

##Sample Queries

Q: Which restaurants offer vegetarian biryani under ₹400?
Q: What are the top-rated cafes in Sector 135 with desserts?
Q: Do any restaurants near Aerocity have gluten-free pasta?

## Notes & Assumptions
Menu data is extracted from images via OCR; inaccuracies may occur due to image quality.

Only restaurants from restaurant_list.json are supported.

Free-tier Hugging Face models are used (FLAN-T5, MiniLM).

No paid APIs were used—entirely open-source stack.

## Acknowledgments
Hugging Face Transformers

Sentence Transformers

ChromaDB

Zomato for restaurant data





 





