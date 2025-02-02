import streamlit as st
from openai import OpenAI
import os
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv
from functools import lru_cache

# Load API keys securely from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YELP_API_KEY = st.secrets["YELP_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Cache the CLIP model to speed up performance
@st.cache_resource
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model, clip_processor = load_clip_model()

# List of 75 research-related items (optimized for MIT Media Lab users)
ITEMS = [
    "AirPods", "Backpack", "Badge", "Ballpoint pen", "Battery pack", "Belt", "Binder clip", "Bluetooth speaker",
    "Book", "Calculator", "Camera", "Coffee cup", "Cord", "Cutting mat", "Desk lamp", "Digital tablet", "Drone",
    "Earbuds", "Ergonomic chair", "Ethernet cable", "External hard drive", "Fabric sample", "Face mask", "Flash drive",
    "Flashlight", "Glasses", "Graphic tablet", "Green screen", "Guitar pick", "Hand sanitizer", "Headphones",
    "Heat gun", "Hoodie", "ID badge", "Ink pen", "Jacket", "Journal", "Keyboard", "Laptop", "Laser pointer",
    "Leather wallet", "LED strip", "Lens cap", "Lipstick", "Magnifying glass", "Marker", "Mechanical keyboard",
    "Mechanical pencil", "Microphone", "Microprocessor", "Mug", "Multimeter", "Notebook", "Paperclip", "Patch cable",
    "Phone charger", "Phone stand", "Portable projector", "Power bank", "Power strip", "Prototyping board",
    "Recorder", "Raspberry Pi", "Resistor pack", "Ring light", "Rubber band", "Safety goggles", "Scientific calculator",
    "Screwdriver", "SD card", "Shoes", "Smartwatch", "Soldering iron", "Soundproofing foam", "Sticker", "Stylus pen",
    "Tape measure", "Tote bag", "Tripod", "USB cable", "Whiteboard marker", "Wireless mouse"
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

# Function to generate lunch prophecy
def get_lunch_prophecy(object_label, user_responses):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mystical oracle that provides symbolic lunch suggestions based on a user's object and reflections."},
                {"role": "user", "content": f"I presented an object: {object_label}. Here's what it means to me: {user_responses[0]} and {user_responses[1]}. What should I eat for lunch? Provide a mystical yet practical suggestion."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating lunch prophecy: {e}"

# Function to find personalized lunch spots using Yelp API
def find_personalized_lunch_spots(lunch_suggestion):
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": lunch_suggestion,
        "location": "MIT Media Lab, Cambridge, MA",
        "limit": 3,
        "price": "1,2"  # Affordable options only
    }
    try:
        response = requests.get("https://api.yelp.com/v3/businesses/search", headers=headers, params=params)
        businesses = response.json().get("businesses", [])
        if businesses:
            return [f"{biz['name']} - {biz['location']['address1']} (Price: {biz.get('price', 'N/A')})" for biz in businesses]
        else:
            return ["No matching lunch spots found nearby."]
    except Exception as e:
        return [f"Error fetching lunch spots: {e}"]

# Streamlit UI
st.title("🔮 The Lunch Oracle")
st.subheader("Reveal your lunch destiny by presenting a sacred offering...")

# Upload image
uploaded_file = st.file_uploader("📸 Present your offering (an object you use daily):", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Your sacred offering...", use_container_width=True)

    # Get object label using CLIP
    object_label = get_object_label(uploaded_file)
    st.write(f"🌀 The Oracle perceives... **{object_label}**")

    # Ask user mystical questions
    is_correct = st.radio(f"Is this a **{object_label}**?", ["Yes", "No"])
    if is_correct == "No":
        object_label = st.text_input("What is this object instead?")

    prompts = [
        f"Tell me, why is this {object_label} important to you?",
        "And what do you cherish most about it?"
    ]
    answer1 = st.text_input(prompts[0])
    answer2 = st.text_input(prompts[1])

    if answer1 and answer2:
        st.write("🔮 Consulting the ancient energies...")
        lunch_prophecy = get_lunch_prophecy(object_label, [answer1, answer2])

        # Display mystical lunch prophecy
        st.success(f"🌟 Your lunch destiny: {lunch_prophecy}")

        # Display personalized lunch spot recommendations
        st.subheader("🍽️ Nearby Offerings")
        personalized_lunch_spots = find_personalized_lunch_spots(lunch_prophecy)
        for spot in personalized_lunch_spots:
            st.write(f"🍴 {spot}")

