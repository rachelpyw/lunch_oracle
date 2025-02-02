import streamlit as st
from openai import OpenAI
import os
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv

# Load API keys securely from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YELP_API_KEY = st.secrets["YELP_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 150 common objects found at MIT Media Lab + everyday items
clip_labels = [
    # Tech / Research Gear
    "a MacBook", "a Windows laptop", "a mechanical keyboard", "a wireless mouse", "a trackpad",
    "a tablet", "an iPad", "a stylus pen", "a drawing tablet", "a VR headset", "an AR headset",
    "a 3D printer", "a printed 3D model", "a Raspberry Pi", "an Arduino board", "a microcontroller",
    "a breadboard", "a multimeter", "an oscilloscope", "a soldering iron", "a thermal camera",
    "a spectrum analyzer", "a power supply", "a function generator", "a smart speaker", 
    "a smartwatch", "a drone", "a camera", "a DSLR camera", "a mirrorless camera",
    "a camcorder", "a GoPro", "a ring light", "a boom mic", "a sound mixer", "an audio interface",
    "a podcast microphone", "a studio monitor speaker", "a MIDI keyboard", "a synthesizer",
    
    # Everyday Personal Items
    "a smartphone", "a pair of AirPods", "a pair of over-ear headphones", "a smartwatch",
    "a pair of sunglasses", "a tote bag", "a backpack", "a hoodie", "a baseball cap",
    "a scarf", "a belt", "a pair of sneakers", "a pair of boots", "a wristwatch",
    "a thermos", "a reusable water bottle", "a disposable coffee cup", "a coffee mug",
    "a bottle of hand sanitizer", "a set of keys", "a lanyard", "a student ID card",
    "a transit card", "a wallet", "a credit card", "a lucky charm", "a plushie", "a small plant",
    "a desk lamp", "a candle", "a tube of lip balm", "a tube of lipstick", "a compact mirror",
    "a set of business cards", "a pack of gum", "a snack bar", "a protein bar",
    
    # Creative & Maker Tools
    "a sketchbook", "a notebook", "a bullet journal", "a set of sticky notes",
    "a whiteboard marker", "a set of Sharpies", "a mechanical pencil", "a ballpoint pen",
    "a fountain pen", "a set of highlighters", "a ruler", "a protractor",
    "a stack of research papers", "a roll of masking tape", "a roll of duct tape",
    "a glue gun", "a soldering kit", "a set of screwdrivers", "a set of hex keys",
    "a pair of pliers", "a multicolored LED strip", "a pack of resistors",
    
    # Media Lab-specific Tools
    "a laser cutter", "a CNC machine", "a robotic arm", "a digital fabrication toolkit",
    "a smart textile", "a wearable device", "a conductive fabric", "a flex sensor",
    "an EEG headset", "an eye-tracking device", "a pressure sensor",
    "a depth camera", "a motion capture suit", "a robotic pet",
    
    # Academic / Study Tools
    "a research journal", "a grant proposal", "a physics textbook",
    "a statistics textbook", "a printed research paper", "a lab notebook",
    "a conference badge", "a poster tube", "a citation guide",
    
    # Random Objects
    "a stress ball", "a fidget cube", "a fidget spinner", "a Rubik's cube",
    "a yoga mat", "a resistance band", "a jump rope", "a frisbee",
    "a skateboard", "a bike helmet", "a smart bike lock",
    
    # Office / Meeting Room Stuff
    "a whiteboard", "a conference microphone", "a projector remote",
    "a speakerphone", "a stack of business proposals", "a startup pitch deck"
]

# Function to recognize object using CLIP
def get_object_label(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = clip_processor(text=clip_labels, images=image, return_tensors="pt", padding=True)
        
        # Get probabilities
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Find the best match
        best_match_idx = probs.argmax().item()
        return clip_labels[best_match_idx]
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

# Function to find lunch spots based on inferred preferences
def find_lunch_spots(query):
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": query,
        "location": "MIT Media Lab, Cambridge, MA",
        "limit": 3,
        "price": "1,2"  # Only suggesting affordable places
    }
    try:
        response = requests.get("https://api.yelp.com/v3/businesses/search", headers=headers, params=params)
        businesses = response.json().get("businesses", [])
        if businesses:
            return [f"{biz['name']} - {biz['location']['address1']} (Price: {biz.get('price', 'N/A')})" for biz in businesses]
        else:
            return ["No suitable lunch spots found nearby."]
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
    
    # Ask user confirmation
    is_correct = st.radio(f"Is this a... {object_label}?", ["Yes", "No"])

    if is_correct == "Yes":
        answer1 = st.text_input("Tell me, why is this important to you?")
        answer2 = st.text_input("And what do you cherish most about it?")

        if answer1 and answer2:
            lunch_prophecy = get_lunch_prophecy(object_label, [answer1, answer2])
            st.success(f"🌟 Your lunch destiny: {lunch_prophecy}")

            # Find relevant lunch spots
            lunch_spots = find_lunch_spots(lunch_prophecy)
            st.subheader("🍽️ Nearby Offerings")
            for spot in lunch_spots:
                st.write(f"🍴 {spot}")
