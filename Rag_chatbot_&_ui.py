from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json
import os
import gradio as gr

torch.cuda.empty_cache() 


class RestaurantChatbot:
    def __init__(self):
        # Set custom cache location
        os.environ['HF_HOME'] = 'D:/huggingface_cache'
        # Initialize models and vector DB
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained( "google/flan-t5-base",cache_dir="D:/huggingface_cache",local_files_only=False)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",torch_dtype=torch.float16, device_map="auto")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="data/vectordb")
        self.collection = self.client.get_collection("restaurants")
        
        # Conversation history
        self.conversation_history = []
        self.max_history = 5  
        
        
        
    """Retrieve relevant documents from ChromaDB."""
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        
        query_embedding = self.retriever.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        
        # Format results with metadata
        retrieved_data = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_data.append({
                "content": doc,
                "name": meta.get("name", "Unknown Restaurant"),
                "price_range": meta.get("price_range", "N/A"),
                "rating": meta.get("rating", 0.0)
            })
        
        return retrieved_data
    
    """Generate answer using FLAN-T5."""
    def generate_response(self, query: str, context: str) -> str:
        
        prompt = f"Answer based on this restaurant information: {context}. Question: {query}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.generator.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    """Full RAG pipeline with conversation history."""
    def handle_query(self, query: str) -> str:
        
        # Retrieve relevant info
        retrieved_data = self.retrieve(query)
        
        if not retrieved_data:
            return "I couldn't find any relevant restaurant information."
        
        # Check retrieval confidence (simple version)
        if len(retrieved_data[0]["content"]) < 10:  # Low-content check
            return "I don't have enough information to answer that."
        
        # Generate response using top result
        context = retrieved_data[0]["content"]
        response = self.generate_response(query, context)
        
        # Update conversation history
        self.conversation_history.append({"query": query, "response": response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        return response

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []
    
  
        
    # Initialize chatbot
bot = RestaurantChatbot()

# Sample queries
queries = [
    "Which restaurants have vegetarian pasta under 500 rupees?",
    "Compare prices for desserts between cafes in Noida",
    "Find gluten-free options near Sector 144",
    "What is average price for two people in Music & Mountains - Hillside Cafe & Cocktail Garden restaurant? ",
    "What is rating of restaurant Roastery Coffee House? "
]

for query in queries:
    print(f"\nUser: {query}")
    response = bot.handle_query(query)
    print(f"Bot: {response}")

def chat_interface(query, history):
    response = bot.handle_query(query)
    return response
    
gr.ChatInterface(
    chat_interface,
    title="Zomato Restaurant Chatbot",
    description="Ask about menus, prices, and dietary options!").launch()