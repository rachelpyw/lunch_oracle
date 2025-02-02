import streamlit as st
from openai import OpenAI
import os
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv

# Load API keys securely from .env file
load_dotenv()
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
YELP_API_KEY = st.secrets("YELP_API_KEY")
GOOGLE_API_KEY = st.secrets("GOOGLE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to recognize object using CLIP
def get_object_label(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        text_inputs = ["a mug", "a wallet", "a fork", "a laptop", "a random object"]
        inputs = clip_processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        
        # Get probabilities
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Find the best match
        best_match_idx = probs.argmax().item()
        return text_inputs[best_match_idx]
    except Exception as e:
        return f"Error using CLIP model: {e}"

# Function to generate lunch prophecy
def get_lunch_prophecy(object_label, user_responses):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mystical oracle that provides symbolic lunch suggestions based on a user's object and reflections."},
                {"role": "user", "content": f"I presented an object: {object_label}. Here's what it means to me: {user_responses[0]} and {user_responses[1]}. What should I eat for lunch? Provide a mystical yet practical suggestion under $20, reflecting values such as trust, comfort, and frugality."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating lunch prophecy: {e}"

# Function to find affordable lunch spots using Yelp API
def find_affordable_lunch_spots():
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": "lunch",
        "location": "MIT Media Lab, Cambridge, MA",
        "limit": 5,
        "price": "1,2"  # 1: Cheapest, 2: Affordable
    }
    try:
        response = requests.get("https://api.yelp.com/v3/businesses/search", headers=headers, params=params)
        businesses = response.json().get("businesses", [])
        if businesses:
            return [f"{biz['name']} - {biz['location']['address1']} (Price: {biz.get('price', 'N/A')})" for biz in businesses]
        else:
            return ["No affordable lunch spots found nearby. The Oracle is uncertain..."]
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

        # Display affordable lunch spot recommendations
        st.subheader("🍽️ Neary oOfferings")
        affordable_lunch_spots = find_affordable_lunch_spots()
        for spot in affordable_lunch_spots:
            st.write(f"🍴 {spot}")

