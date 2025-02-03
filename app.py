import streamlit as st
from openai import OpenAI
import os
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv
import re

# Load API keys securely from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YELP_API_KEY = st.secrets["YELP_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Cache the CLIP model to speed up performance
@st.cache_resource
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model, clip_processor = load_clip_model()

# List of 100 everyday items (balanced for MIT Media Lab + general household)
ITEMS = [
    "AirPods", "Backpack", "Badge", "Ballpoint pen", "Battery pack", "Belt", "Binder clip", "Bluetooth speaker",
    "Book", "Calculator", "Camera", "Coffee cup", "Cord", "Desk lamp", "Digital tablet", "Drone", "Earbuds",
    "Ethernet cable", "External hard drive", "Face mask", "Flash drive", "Flashlight", "Glasses", "Hand sanitizer",
    "Headphones", "Hoodie", "Ink pen", "Jacket", "Journal", "Keyboard", "Laptop", "Laser pointer", "Leather wallet",
    "Lipstick", "Magnifying glass", "Marker", "Mechanical pencil", "Microphone", "Multimeter", "Notebook",
    "Paperclip", "Patch cable", "Phone charger", "Portable projector", "Power bank", "Power strip", "Recorder",
    "Raspberry Pi", "Ring light", "Rubber band", "Safety goggles", "Scientific calculator", "Screwdriver", "SD card",
    "Shoes", "Smartwatch", "Soldering iron", "Soundproofing foam", "Sticker", "Stylus pen", "Tape measure",
    "Tote bag", "Tripod", "USB cable", "Whiteboard marker", "Wireless mouse", "Sunscreen", "Water bottle", "Plushie",
    "Pencil", "Plant", "Eraser", "Scissors", "Notebook stand", "Desk chair", "Coffee beans", "Yoga mat",
    "Smart lightbulb", "Bike helmet", "Portable fan", "Guitar pick", "Measuring spoon", "TV remote", "Scented candle",
    "Desk organizer", "Stress ball", "Wireless keyboard", "Phone stand", "Resistance bands", "Fidget spinner",
    "Keychain", "Reusable straw", "Travel mug", "Paper towel roll", "Sticky notes", "Charger cable", "Umbrella",
    "Shopping tote", "Shoe cleaner", "Lint roller", "Coaster", "Clip-on ring light"
]

# Function to recognize object using CLIP
def get_object_label(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = clip_processor(text=ITEMS, images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        best_match_idx = probs.argmax().item()
        return ITEMS[best_match_idx]
    except Exception as e:
        return f"Error using CLIP model: {e}"

# Function to generate a full prophecy while extracting a keyword for Yelp
def get_lunch_prophecy(object_label, user_response):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mystical oracle that provides symbolic lunch suggestions based on a user's object and reflections."},
                {"role": "user", "content": f"I presented an object: {object_label}. Here's what it means to me: {user_response}. What should I eat for lunch? Highlight what I 'trust' in, find 'comfort' in, and 'value'. Then, provide a two-sentence, poetic, mystical prophecy. Finally, return a single word (e.g., 'salad', 'ramen', 'pasta') indicating the kind of food, but do not include it in the response."}
            ]
        )
        oracle_response = response.choices[0].message.content

        # Extract keyword (last word in response)
        keyword_match = re.search(r"\b(salad|soup|sandwich|pizza|ramen|sushi|pasta|burger|tacos|burrito|noodles|rice|wrap|curry|steak|pancakes|smoothie|poke|bagel|falafel|dumplings|noodle|bbq|pho|dim sum|hotpot|teriyaki|laksa|bánh mì|pad thai|roti|shawarma)\b", oracle_response.lower())
        keyword = keyword_match.group(0) if keyword_match else "lunch"

        return oracle_response, keyword
    except Exception as e:
        return f"Error generating lunch prophecy: {e}", "lunch"

# Function to find personalized lunch spots using Yelp API
def find_personalized_lunch_spots(food_keyword):
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": f"{food_keyword} restaurant",
        "location": "MIT Media Lab, Cambridge, MA",
        "limit": 3,
        "price": "1,2"
    }
    try:
        response = requests.get("https://api.yelp.com/v3/businesses/search", headers=headers, params=params)
        businesses = response.json().get("businesses", [])
        if businesses:
            return [f"{biz['name']} - {biz['location']['address1']} ({biz.get('categories', [{'title': 'Unknown'}])[0]['title']})" for biz in businesses]
        else:
            return ["No matching lunch spots found nearby."]
    except Exception as e:
        return [f"Error fetching lunch spots: {e}"]

# Streamlit UI
st.title("🔮 The Lunch Oracle")
st.subheader("Reveal your lunch destiny by presenting an offering - a photo of an everyday item you use and love.")

# Upload image
uploaded_file = st.file_uploader("📸 Upload a photo or take one with your camera.", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Your sacred offering...", use_container_width=True)

    # Get object label using CLIP
    object_label = get_object_label(uploaded_file)
    st.write(f"🌀 The Oracle perceives... **{object_label}**")

    # Ask user mystical question
    user_response = st.text_input(f"How does your **{object_label}** guide you in life?")

    if user_response:
        with st.spinner("🔮 Consulting the ancient energies..."):
            lunch_prophecy, food_keyword = get_lunch_prophecy(object_label, user_response)

        # Display mystical lunch prophecy
        st.success(f"🌟 Your lunch destiny: {lunch_prophecy}")

        # Display personalized lunch spot recommendations
        st.subheader("🍽️ The Oracle has foreseen these offerings, aligned with your deepest values:")
        personalized_lunch_spots = find_personalized_lunch_spots(food_keyword)
        for spot in personalized_lunch_spots:
            st.write(f"🍴 {spot}")
