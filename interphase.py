import streamlit as st
import asyncio
from PIL import Image
import tempfile
import os
from main import CalorieEstimator
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Retrieve the GEMINI_API_KEY from environment variables
# Ensure this exact variable name (GEMINI_API_KEY) is set in your Streamlit Cloud secrets
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize the CalorieEstimator with the actual API key
# The CalorieEstimator class itself will handle further configuration with genai.configure()
estimator = CalorieEstimator(api_key=GEMINI_API_KEY)

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "home"
    
st.set_page_config(page_title="DietGPT", layout="centered")
st.title("🍽️ DietGPT")
st.markdown("""
## Upload a **photo of your meal**   
Then AI will analyze it and return an estimate calories 
Response will be given in the format:  

CALORIES: [number]

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


if st.session_state.page == "home":
    if st.button("🔐 Login"):
        st.session_state.page = "login"
    

elif st.session_state.page == "login":
    # Ensure 'pages' directory exists and 'login.py' is inside it
    from pages.login import main
    main()

elif st.session_state.page == "diet":
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("🔒You must be logged in to use DietGPT.")
        st.stop()


sugar = st.slider('Define your sugar rate',min_value=0, max_value=12)

st.write("Your sugar is:", sugar)

# --- Custom Slider Styling ---
st.markdown(f"""
    <style>
    div.stSlider > div[data-baseweb="slider"] > div > div {{
        background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                    rgb(1, 183, 158)%, 
                                    rgba(151, 166, 195, 0.25)%, 
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

# -----Logic-----
if 0 <= sugar <= 3:
    st.markdown(f"<span style='color: red;'>Your sugar is too low: {sugar}</span>", unsafe_allow_html=True)
    st.write("Your sugar is too low:", sugar)
    st.warning("Call Ambulance!")

elif 4 <= sugar <= 4.5:
    st.markdown(f"<span style='color: blue;'>Your sugar is low: {sugar}</span>", unsafe_allow_html=True)
    st.write("Your sugar is low :", sugar)
    st.info("""
🟦 1. Low Sugar Rate (Hypoglycemia)

When your blood sugar is too low, the goal is to raise it safely and maintain stability throughout the day.

What to Eat:

    Frequent small meals: Eat every  3–4 hours to prevent dips.

    Complex carbs + protein: Whole grains (like oats or brown rice) with lean protein (chicken, tofu, eggs).

    Healthy snacks: Apple slices with peanut butter, Greek yogurt with berries, or a handful of nuts.

    Soluble fiber: Beans, lentils, and oats help slow sugar absorption.

    Natural sugars (in moderation): Fresh fruits like bananas or oranges can give a quick boost.

What to Avoid:

    Sugary drinks or candy (unless treating an acute low).

    Alcohol and caffeine on an empty stomach.

    Skipping meals.
""")
elif sugar == 5:
    st.markdown(f"<span style='color: green;'>Your sugar is in a normal range: {sugar}</span>", unsafe_allow_html=True)
    st.write("Your sugar is in a normal range: ", sugar)
elif 6 <= sugar <= 7:
    st.markdown(f"<span style='color: red;'>Your sugar is too high: {sugar}</span>", unsafe_allow_html=True)
    st.write("Your sugar is too high:", sugar)
    st.info("""
🟥 2. High Sugar Rate (Hyperglycemia)

Here, the focus is on lowering and stabilizing blood sugar through balanced, low-glycemic meals.

What to Eat:

    Non-starchy vegetables: Broccoli, spinach, peppers, cauliflower.

    Whole grains: Quinoa, barley, whole wheat pasta (in moderation).

    Lean proteins: Fish, turkey, legumes, tofu.

    Healthy fats: Avocados, olive oil, nuts.

    Low-GI fruits: Berries, apples, pears.

What to Avoid:

    Refined carbs: White bread, pastries, sugary cereals.

    Sweetened beverages: Soda, sweetened teas, energy drinks.

    Processed snacks: Chips, cookies, and fast food.
""")
elif 7 <= sugar <= 12:
    st.markdown(f"<span style='color: red;'>Your sugar is too high: {sugar}</span>", unsafe_allow_html=True)
    st.write("Your sugar is too high:", sugar)
    st.warning("Call Ambulance!")


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
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            try:
                # Run analysis
                result = asyncio.run(estimator.analyze_image(temp_path))
            finally:
                # Ensure the temporary file is deleted even if analysis fails
                os.remove(temp_path)

        # Display result
        if "total_calories" in result:
            st.success(f"✅ {result['raw_output']}")
        else:
            st.error("❌ Could not extract calorie count.")
            st.text_area("Raw Output", result.get("raw_output", "No output returned."), height=100)
