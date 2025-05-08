import streamlit as st
from openai import OpenAI
import requests
import torch
import time
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import re

# Load API keys
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Load CLIP model
@st.cache_resource
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model, clip_processor = load_clip_model()

# Everyday items
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
    "Shopping tote", "Shoe cleaner", "Lint roller", "Coaster", "Clip-on ring light", "Bicycle", "Monitor", "Lamp",
    "Cat", "Dog", "Tissues", "Pills", "Mat", "Notebook holder", "Mug warmer", "Gaming mouse", "Wireless charger",
    "Standing desk", "Laptop sleeve", "Portable speaker", "Drawing tablet", "E-reader", "Wrist rest", "Neck pillow",
    "Hand cream", "Back massager", "Shower speaker", "Sleep mask", "Pocket notebook", "Desk fan"
]

# Object label with CLIP
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

# Get lunch prophecy using OpenAI v1+
def get_lunch_prophecy(object_label, user_response):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mystical oracle that provides symbolic lunch suggestions based on a user's object and reflections."},
                {"role": "user", "content": f"I presented an object: {object_label}. The seeker tells me: {user_response}. What should they eat for lunch? Highlight what they 'trust' in, find 'comfort' in, and 'value'. Then, provide a poetic, mystical prophecy. Finally, return a single word (e.g., 'salad', 'ramen', 'pasta') indicating the kind of food, but do not include it in the response."}
            ]
        )
        oracle_response = response.choices[0].message.content
        keyword_match = re.search(r"\b(salad|soup|sandwich|pizza|ramen|sushi|pasta|burger|tacos|burrito|noodles|rice|wrap|curry|steak|pancakes|smoothie|poke|bagel|falafel|dumplings|noodle|bbq|pho|dim sum|hotpot|teriyaki|laksa|bánh mì|pad thai|roti|shawarma)\b", oracle_response.lower())
        keyword = keyword_match.group(0) if keyword_match else "lunch"
        return oracle_response, keyword
    except Exception as e:
        return f"Error generating lunch prophecy: {e}", "lunch"

# Google Places integration
def find_personalized_lunch_spots(food_keyword):
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{food_keyword} restaurant near National Design Centre Singapore"
    params = {
        "query": query,
        "key": GOOGLE_API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        results = response.json().get("results", [])[:3]
        if results:
            return [f"{r['name']} - {r['formatted_address']}" for r in results]
        else:
            return ["No matching lunch spots found nearby."]
    except Exception as e:
        return [f"Error fetching lunch spots: {e}"]

# Streamlit UI
st.title("🔮 The Lunch Oracle")
st.subheader("Reveal your lunch destiny by presenting an offering - a photo of an everyday item you use and love.")

uploaded_file = st.file_uploader("📸 Upload a photo or take one with your camera.", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Your sacred offering...", use_container_width=True)

    object_label = get_object_label(uploaded_file)
    user_response = st.text_input("How does this artifact guide your spirit?")

    if user_response:
        with st.spinner("🌿 The Oracle is infusing wisdom and care into its vision... 🔮"):
            time.sleep(5)
            lunch_prophecy, food_keyword = get_lunch_prophecy(object_label, user_response)

        st.success(f"🌟 Your lunch destiny: {lunch_prophecy}")

        st.subheader("🍽️ The Oracle has foreseen these offerings, aligned with your deepest values:")
        personalized_lunch_spots = find_personalized_lunch_spots(food_keyword)
        for spot in personalized_lunch_spots:
            st.write(f"🍴 {spot}")
