import streamlit as st
from openai import OpenAI
import os
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv

# Load API keys securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YELP_API_KEY = st.secrets["YELP_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ 1. Cache the CLIP model to prevent reloading
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

clip_model, clip_processor = load_clip_model()

# ✅ 2. Define object categories (optimized for MIT Media Lab users)
object_labels = [
    "laptop", "headphones", "airpods", "mechanical keyboard", "smartphone", "tablet", 
    "stylus pen", "notebook", "calculator", "water bottle", "backpack", "sneakers", 
    "wallet", "ID badge", "USB drive", "portable hard drive", "coffee mug", "espresso cup", 
    "glasses", "smartwatch", "camera", "tripod", "microphone", "VR headset", "LED light strip", 
    "projector", "arduino board", "raspberry pi", "3D printed object", "circuit board", 
    "soldering iron", "breadboard", "multimeter", "robotic hand", "microcontroller", 
    "AI-generated artwork", "fashion prototype", "VR gloves", "motion capture suit", 
    "plushie", "lipstick", "plant", "belt", "earrings", "scarf", 
    "notebook", "sketchbook", "journal", "marker", "highlighter", "desk lamp", "mouse", 
    "mouse pad", "desk organizer", "sticky notes", "external monitor", "bluetooth speaker", 
    "drawing tablet", "pencil", "pen", "eraser", "mask", "safety goggles", "clipboard", 
    "textbook", "poster tube", "travel mug", "tote bag", "power bank", "umbrella", "keychain"
]

# ✅ 3. Optimize object recognition with caching
@st.cache_data
def get_object_label(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = clip_processor(text=object_labels, images=image, return_tensors="pt", padding=True)
        
        # Get probabilities
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Find the best match
        best_match_idx = probs.argmax().item()
        return object_labels[best_match_idx]
    except Exception as e:
        return f"Error using CLIP model: {e}"

# ✅ 4. Optimize Oracle Prophecy to minimize API calls
@st.cache_data
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

# ✅ 5. Optimize Yelp API calls (based on Oracle’s prophecy)
@st.cache_data
def find_affordable_lunch_spots(oracle_suggestion):
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {
        "term": oracle_suggestion,  # Use Oracle's suggestion to refine search
        "location": "MIT Media Lab, Cambridge, MA",
        "limit": 3,
        "price": "1,2"  # 1: Cheapest, 2: Affordable
    }
    try:
        response = requests.get("https://api.yelp.com/v3/businesses/search", headers=headers, params=params)
        businesses = response.json().get("businesses", [])
        if businesses:
            return [f"{biz['name']} - {biz['location']['address1']} (Price: {biz.get('price', 'N/A')})" for biz in businesses]
        else:
            return ["No matching affordable lunch spots found nearby. The Oracle is uncertain..."]
    except Exception as e:
        return [f"Error fetching lunch spots: {e}"]

# ✅ 6. Streamlit UI optimized for 8+ users
st.title("🔮 The Lunch Oracle")
st.subheader("Reveal your lunch destiny by presenting a sacred offering...")

# Store user state to avoid conflicts
if "responses" not in st.session_state:
    st.session_state.responses = {}

uploaded_file = st.file_uploader("📸 Present your offering (an object you use daily):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Your sacred offering...", use_container_width=True)

    # Get object label using CLIP
    object_label = get_object_label(uploaded_file)
    st.write(f"🌀 The Oracle perceives... **{object_label}**")

    # Confirmation button
    if st.button(f"Is this a {object_label}?"):
        st.write("The Oracle has spoken! Now share your thoughts...")

    # User input
    prompts = [
        f"Tell me, why is this {object_label} important to you?",
        "And what do you cherish most about it?"
    ]
    user_id = str(hash(uploaded_file))  # Unique user ID based on image hash
    st.session_state.responses[user_id] = {}
    
    answer1 = st.text_input(prompts[0], key=f"q1_{user_id}")
    answer2 = st.text_input(prompts[1], key=f"q2_{user_id}")

    if answer1 and answer2:
        st.write("🔮 Consulting the ancient energies...")
        if user_id not in st.session_state.responses:
            st.session_state.responses[user_id] = {}

        # Get cached Oracle Prophecy
        if "prophecy" not in st.session_state.responses[user_id]:
            st.session_state.responses[user_id]["prophecy"] = get_lunch_prophecy(object_label, [answer1, answer2])

        lunch_prophecy = st.session_state.responses[user_id]["prophecy"]
        st.success(f"🌟 Your lunch destiny: {lunch_prophecy}")

        # Get cached affordable lunch spots
        if "lunch_spots" not in st.session_state.responses[user_id]:
            st.session_state.responses[user_id]["lunch_spots"] = find_affordable_lunch_spots(lunch_prophecy)

        affordable_lunch_spots = st.session_state.responses[user_id]["lunch_spots"]
        st.subheader("🍽️ Nearby Offerings")
        for spot in affordable_lunch_spots:
            st.write(f"🍴 {spot}")
