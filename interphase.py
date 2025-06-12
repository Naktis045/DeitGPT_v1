import streamlit as st
import asyncio
from PIL import Image
import tempfile
import os
from main import CalorieEstimator # Import your CalorieEstimator class
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
# This allows you to keep your GEMINI_API_KEY out of your codebase.
load_dotenv()

# Retrieve the GEMINI_API_KEY from environment variables.
# For Streamlit Cloud, ensure this exact variable name is set in your secrets.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize the CalorieEstimator with the API key.
# The CalorieEstimator class itself handles the `genai.configure()` call.
# Pass a dummy key if GEMINI_API_KEY is None, but the main.py will raise an error
# if it's the placeholder key. This ensures the app doesn't crash here immediately.
estimator = CalorieEstimator(api_key=GEMINI_API_KEY or "AIzaSyAJV2C-skymKknmkuusvGwma135kKPACns")


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


# --- Page Navigation Logic (Basic Example) ---
if st.session_state.page == "home":
    if st.button("🔐 Login"):
        # This will change the session state and re-run the script
        st.session_state.page = "login"

elif st.session_state.page == "login":
    # This assumes you have a 'pages' directory with a 'login.py' file.
    # The actual content of login.py is not provided here.
    try:
        from pages.login import main as login_main
        login_main()
    except ImportError:
        st.error("Login module (pages/login.py) not found. Please create it.")
        st.stop()
    # After login, you might set st.session_state.page = "diet"
    # For now, we'll assume a successful login implicitly allows access to the diet features.

# This block is executed if the page is 'diet' or if a successful login
# is assumed to have happened and the user is on the main app page.
# For demonstration, we'll proceed directly to the diet features if logged_in state is set.
# You might want to remove 'home' and 'login' pages once the login flow is complete.
if st.session_state.page == "diet" or ("logged_in" in st.session_state and st.session_state.logged_in):
    # Check if the user is logged in (if a login mechanism is in place)
    # If not logged in, show a warning and stop execution for this section.
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("🔒 You must be logged in to use DietGPT.")
        # If you want to force redirect to login, uncomment the next line and comment st.stop()
        # st.session_state.page = "login"
        st.stop() # Stops execution of the rest of the script if not logged in.

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
    if 0 <= sugar <= 3:
        st.markdown(f"<span style='color: red;'>Your sugar is too low: {sugar}</span>", unsafe_allow_html=True)
        st.warning("🚨 **Call Ambulance!** Your sugar level is critically low.")

    elif 4 <= sugar <= 4.5:
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
    elif sugar == 5:
        st.markdown(f"<span style='color: green;'>Your sugar is in a normal range: {sugar}</span>", unsafe_allow_html=True)
        st.success("✅ Your sugar level is currently in a healthy range. Keep up the good work!")

    elif 6 <= sugar <= 7:
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
                # Save the uploaded image to a temporary file
                # This is necessary because the CalorieEstimator expects a file path.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    temp_path = tmp.name

                try:
                    # Run the analysis using the CalorieEstimator.
                    # asyncio.run is used here because analyze_image is an async function.
                    result = asyncio.run(estimator.analyze_image(temp_path))
                finally:
                    # Ensure the temporary file is deleted after analysis,
                    # regardless of success or failure, to clean up resources.
                    os.remove(temp_path)

            # Display the estimation result to the user
            if "total_calories" in result:
                st.success(f"✅ Calorie Estimation Result: {result['raw_output']}")
            else:
                st.error("❌ Could not extract calorie count. Please try another image or check the API key.")
                st.text_area("Raw Output from AI (for debugging):", result.get("raw_output", "No output returned."), height=100)

