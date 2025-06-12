import streamlit as st
import asyncio
from PIL import Image
import tempfile
import os
from main import CalorieEstimator # Import your CalorieEstimator class


estimator = CalorieEstimator()


# --- Streamlit Page Configuration and Initial State ---
st.set_page_config(page_title="DietGPT", layout="centered")

# Initialize page state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"
    
st.title("🍽️ DietGPT")
st.markdown("""
## Upload a **photo of your meal**
Then AI will analyze it and return an estimate calories
Response will be given in the format:

`CALORIES: [number]`

Invalid responses:
- "The total calories are 450"
- "CALORIES 450-500"
- "Calories: 450"

Valid response examples:
* CALORIES: 450
* CALORIES: 1249
* CALORIES: 740
* CALORIES: 320
""")
st.markdown("🔐 Please log in to use DietGPT **log in**.")


# --- Page Navigation Logic  ---
if st.session_state.page == "home":
    if st.button("🔐 Login"):
        st.session_state.page = "login"

elif st.session_state.page == "login":
    try:
        from pages.login import main as login_main
        login_main()
    except ImportError:
        st.error("Login module (pages/login.py) not found. Please create it.")

if st.session_state.page == "diet" or ("logged_in" in st.session_state and st.session_state.logged_in):
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Please log in to use DietGPT")

    # --- Sugar Rate Slider and Logic ---
sugar = st.slider('Define your sugar rate', min_value=0.0, max_value=12.0, step=0.5, value=5.0)
st.write("Your sugar is:", sugar)

    # Custom CSS for the Streamlit slider to match branding/aesthetics
st.markdown(f"""
        <style>
        div.stSlider > div[data-baseweb="slider"] > div > div {{
            background: linear-gradient(to right, rgb(1, 183, 158) 0%,
                                        rgb(1, 183, 158) {sugar/12*100}%,
                                        rgba(151, 166, 195, 0.25) {sugar/12*100}%,
                                        rgba(151, 166, 195, 0.25) 100%);
        }}
        div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {{
            background: rgb(1 1 1 / 0%);
        }}
        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {{
            background-color: rgb(14, 38, 74);
            box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;
        }}
        div.stSlider > div[data-baseweb="slider"] > div > div > div > div {{
            color: rgb(14, 38, 74);
        }}
        </style>
    """, unsafe_allow_html=True)

    # Logic based on sugar rate value
if 0 <= sugar <= 3.5:
    st.markdown(f"<span style='color: red;'>Your sugar is too low: {sugar}</span>", unsafe_allow_html=True)
    st.warning("🚨 **Call Ambulance!** Your sugar level is critically low.")

elif 3.6 <= sugar <= 3.8:
    st.markdown(f"<span style='color: blue;'>Your sugar is low: {sugar}</span>", unsafe_allow_html=True)
    st.info("""
    🟦 **1. Low Sugar Rate (Hypoglycemia)**

    When your blood sugar is too low, the goal is to raise it safely and maintain stability throughout the day.

    **What to Eat:**

    * **Frequent small meals:** Eat every 3–4 hours to prevent dips.
    * **Complex carbs + protein:** Whole grains (like oats or brown rice) with lean protein (chicken, tofu, eggs).
    * **Healthy snacks:** Apple slices with peanut butter, Greek yogurt with berries, or a handful of nuts.
    * **Soluble fiber:** Beans, lentils, and oats help slow sugar absorption.
    * **Natural sugars (in moderation):** Fresh fruits like bananas or oranges can give a quick boost.

    **What to Avoid:**

    * Sugary drinks or candy (unless treating an acute low).
    * Alcohol and caffeine on an empty stomach.
    * Skipping meals.
    """)
elif 3.9 <= sugar <=5.5:
    st.markdown(f"<span style='color: green;'>Your sugar is in a normal range: {sugar}</span>", unsafe_allow_html=True)
    st.success("✅ Your sugar level is currently in a healthy range. Keep up the good work!")

elif 5.6 <= sugar <= 7:
    st.markdown(f"<span style='color: red;'>Your sugar is too high: {sugar}</span>", unsafe_allow_html=True)
    st.info("""
    🟥 **2. High Sugar Rate (Hyperglycemia)**

    Here, the focus is on lowering and stabilizing blood sugar through balanced, low-glycemic meals.

    **What to Eat:**

    * **Non-starchy vegetables:** Broccoli, spinach, peppers, cauliflower.
    * **Whole grains:** Quinoa, barley, whole wheat pasta (in moderation).
    * **Lean proteins:** Fish, turkey, legumes, tofu.
    * **Healthy fats:** Avocados, olive oil, nuts.
    * **Low-GI fruits:** Berries, apples, pears.

    **What to Avoid:**

    * Refined carbs: White bread, pastries, sugary cereals.
    * Sweetened beverages: Soda, sweetened teas, energy drinks.
    * Processed snacks: Chips, cookies, and fast food.
    """)
elif 7 < sugar <= 12: # Changed from 7 <= sugar to 7 < sugar to avoid overlap with 6-7 range
    st.markdown(f"<span style='color: red;'>Your sugar is too high: {sugar}</span>", unsafe_allow_html=True)
    st.warning("🚨 **Call Ambulance!** Your sugar level is critically high.")


    # --- Meal Photo Upload and Calorie Estimation ---
uploaded_file = st.file_uploader(
    "📷 Upload or take a meal photo",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="visible"
    )

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Your Meal Photo", use_column_width=True)

    if st.button("Estimate Calories"):
        with st.spinner("🔍 Analyzing image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            try:
                result = asyncio.run(estimator.analyze_image(temp_path))
            finally:
                os.remove(temp_path)

            # Display the estimation result to the user
        if "total_calories" in result:
            st.success(f"✅ Calorie Estimation Result: {result['raw_output']}")
        else:
            st.error("❌ Could not extract calorie count. Please try another image or check the API key.")
            st.text_area("Raw Output from AI (for debugging):", result.get("raw_output", "No output returned."), height=100)

