import logging
import os
import datetime as dt
import base64
import io
import re
import asyncio

from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import google.generativeai as genai
from tenacity import (
    retry, wait_exponential, stop_after_attempt, retry_if_exception_type
)
from tqdm import tqdm

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompt for the LLM
SYSTEM_PROMPT = """You are a Food analyst specializing in estimating the calories of a meal from a photo. Your task is to analyze a meal from an image and provide an estimate.

Follow these steps to complete your analysis:
1. Carefully examine the image and identify all food items present in the meal.
2. For each food item:
    - Determine the portion size (e.g., in grams).
    - Consider any hidden ingredients or preparation methods that affect calorie content.
    - Calculate the calorie content based on the portion size.
3. Sum up the calorie estimates to arrive at a single total calorie count for the entire meal.

Always provide a single number estimate and not a range. Prioritize accuracy over speed to ensure the most accurate estimate.
Respond in this format:
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
"""

# Load API key from environment variables (important for local development)
load_dotenv()
# Attempt to get the API key from environment variables first.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Define a specific placeholder that the code will check against.
# This value should NOT be your actual API key.
# It's just a unique string used for validation.
PLACEHOLDER_KEY = 'YOUR_UNIQUE_GEMINI_API_KEY_PLACEHOLDER_HERE'

# If the environment variable is not set, we will use this placeholder.
if not GEMINI_API_KEY:
    GEMINI_API_KEY = PLACEHOLDER_KEY

# Final check: if the key is still the placeholder, it means it hasn't been set.
if GEMINI_API_KEY == PLACEHOLDER_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found. Please set it as an environment variable or "
                            "uncomment and replace the placeholder on line 64 with your actual API key obtained from Google AI Studio.")

# Configure the genai library with your API key
# This global configuration is important for all subsequent calls
genai.configure(api_key=GEMINI_API_KEY)


class CalorieEstimator:
    def __init__(self, api_key):
        # api_key is now configured globally via genai.configure, but storing it for consistency
        self.api_key = api_key 
        self.system_prompt = SYSTEM_PROMPT
        # Use a compatible Gemini model for vision tasks
        self.model_name = "gemini-1.5-flash"
        self.model = genai.GenerativeModel(self.model_name)

    @staticmethod
    def encode_image_to_pil_image(image_path):
        """
        Loads an image from the given path and returns a PIL Image object.
        This method is modified to ensure the file handle is closed immediately,
        preventing 'PermissionError' on file deletion.
        """
        try:
            # Read the image bytes from the file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Create a BytesIO object from the image bytes
            img_buffer = io.BytesIO(image_bytes)
            
            # Open the PIL Image from the BytesIO object
            return Image.open(img_buffer)
        except Exception as e:
            logging.error(f"Error loading image from {image_path}: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5),
        # Retry for any general exception, or specific API errors if identified
        retry=retry_if_exception_type((Exception, asyncio.TimeoutError)),
        retry_error_callback=lambda retry_state: {
            "error": f"All retries failed: {retry_state.outcome.exception()}",
            "raw_output": str(retry_state.outcome.exception())
        }
    )
    async def analyze_image(self, image_path):
        """
        Analyzes the given image using the Gemini Pro Vision model.
        """
        logging.info(f"Analyzing image: {image_path}")
        try:
            img_pil = self.encode_image_to_pil_image(image_path)

            # Gemini API expects content as a list of parts (text and image)
            parts = [
                self.system_prompt, # System instruction can be a part
                "What is the total calorie count for this meal? Remember to format your final answer as CALORIES:[number]",
                img_pil # Pass the PIL Image object directly
            ]

            # The generate_content method automatically handles API calls and authentication
            # Set stream=False for non-streaming response.
            # Using asyncio.to_thread to run blocking genai.generate_content in an async context
            response = await asyncio.to_thread(self.model.generate_content, parts, stream=False)

            return self._parse_api_response(response)

        except genai.types.BlockedPromptException as e:
            logging.error(f"Prompt was blocked for {image_path}: {e.response.prompt_feedback}")
            return {"error": f"Prompt blocked: {e.response.prompt_feedback}", "raw_output": str(e.response.prompt_feedback)}
        except Exception as e:
            logging.error(f"API request failed for {image_path}: {e}")
            # Re-raise to trigger tenacity retry
            raise ValueError(f"Gemini API request failed: {e}")

    @staticmethod
    def _parse_api_response(result):
        """
        Parses the Gemini API response to extract the calorie count.
        """
        try:
            # Access the text from the candidate's content
            content = result.text
            if calorie_match := re.search(r'CALORIES:\s*(\d+(?:\.\d+)?)', content):
                return {"total_calories": float(calorie_match.group(1)), "raw_output": content}
            logging.warning(f"Could not extract calorie count from response: {content}")
            return {"error": "Could not extract calorie count response", "raw_output": content}
        except Exception as e:
            logging.error(f"Error parsing API response: {e}, Raw result: {result}")
            return {"error": f"Error parsing API response: {e}", "raw_output": str(result)}

    async def process_single_image(self, image_path, actual_calories):
        """
        Processes a single image file to estimate calories and record results.
        """
        if not os.path.exists(image_path):
            logging.error(f"File does not exist: {image_path}")
            return None
        try:
            result = await self.analyze_image(image_path)
            logging.info(f"API raw output for {os.path.basename(image_path)}:\n{result.get('raw_output')}")

            if estimated_calories := result.get('total_calories'):
                return {
                    'image': os.path.basename(image_path),
                    'actual_calories': float(actual_calories),
                    'estimated_calories': float(estimated_calories),
                    'calorie_difference': abs(float(estimated_calories) - float(actual_calories)),
                    'llm_output': result.get('raw_output', 'N/A'),
                }
            logging.error(f"Error processing {image_path}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logging.error(f"Exception processing {image_path}: {str(e)}")
        return None

async def main():
    """
    Main function to load the dataset, process images, and save results.
    This function is primarily for batch processing, not for the Streamlit UI flow.
    """
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, 'DATASET')
    results_dir = "estimation_results"

    # Initialize CalorieEstimator with the Gemini API key
    # Note: genai.configure() handles the key globally, but passing it is harmless.
    estimator = CalorieEstimator(api_key=GEMINI_API_KEY)

    csv_path = os.path.join(dataset_path, 'processed_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and preprocess CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'calories' not in df.columns:
        raise ValueError("❌ 'calories' column not found in the dataset.")
    df = df.dropna(subset=['calories'])

    logging.info(f"DataFrame head:\n{df.head()}")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"Columns in CSV: {df.columns.tolist()}")

    # Check for missing image files and log warnings
    missing_files = []
    for _, row in df.iterrows():
        full_path = os.path.join(dataset_path, row['img_path'])
        if not os.path.exists(full_path):
            missing_files.append(full_path)

    if missing_files:
        logging.warning(f"Missing {len(missing_files)} image files. Examples (first 10):")
        for mf in missing_files[:10]:
            logging.warning(mf)

    # Create tasks for all images
    tasks = [
        estimator.process_single_image(os.path.join(dataset_path, row['img_path']), row['calories'])
        for _, row in df.iterrows()
    ]

    results = []
    # Use tqdm for a progress bar while processing tasks
    with tqdm(total=len(tasks), desc="Processing images with Gemini") as pbar:
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)
            pbar.update(1)

    # Save results to a CSV file
    if results:
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, 'estimation_gemini.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        logging.info(f'Results are saved to {output_file}')
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    # This block only runs when main.py is executed directly, not when imported
    asyncio.run(main())
