import logging
import os
import datetime as dt # Not directly used in the provided snippet but often useful
import base64 # Not directly used for PIL image but common for other image handling
import io
import re
import asyncio

# Removed: from dotenv import load_dotenv # No longer needed as key is hardcoded

from PIL import Image
import pandas as pd # Used in the batch processing 'main' function, not the CalorieEstimator itself
import google.generativeai as genai
from tenacity import (
    retry, wait_exponential, stop_after_attempt, retry_if_exception_type
)
from tqdm import tqdm # Used for progress bar in batch processing

# Configure logging for better visibility and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- System Prompt for the LLM (Critical for consistent output) ---
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

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Check if the API key was successfully loaded. If not, raise an error.
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Please set it in your .env file "
        "or as an environment variable."
    )
genai.configure(api_key=GEMINI_API_KEY)

class CalorieEstimator:
    def __init__(self, api_key=None): # api_key parameter is now optional as it's hardcoded globally
        # api_key is now configured globally via genai.configure, so no need to store it in self.api_key.
        # This parameter can be removed entirely if desired, but kept for signature compatibility.
        self.system_prompt = SYSTEM_PROMPT
        # Use a compatible Gemini model for vision tasks (e.g., gemini-1.5-flash for speed)
        self.model_name = "gemini-1.5-flash"
        self.model = genai.GenerativeModel(self.model_name)

    @staticmethod
    def encode_image_to_pil_image(image_path):
        """
        Loads an image from the given path and returns a PIL Image object.
        This method ensures the file handle is properly closed, preventing 'PermissionError'
        when Streamlit tries to delete temporary files on Windows.
        """
        try:
            # Read the image bytes from the file to ensure the file handle is closed
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Create a BytesIO object from the image bytes, then open with PIL
            img_buffer = io.BytesIO(image_bytes)
            return Image.open(img_buffer)
        except Exception as e:
            logging.error(f"Error loading image from {image_path}: {e}")
            raise # Re-raise the exception to indicate failure

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20), # Exponential backoff for retries
        stop=stop_after_attempt(5), # Try up to 5 times
        # Retry for any general exception, or specific API errors if identified.
        # asyncio.TimeoutError is added for potential async operation timeouts.
        retry=retry_if_exception_type((Exception, asyncio.TimeoutError)),
        retry_error_callback=lambda retry_state: { # Callback if all retries fail
            "error": f"All retries failed: {retry_state.outcome.exception()}",
            "raw_output": str(retry_state.outcome.exception())
        }
    )
    async def analyze_image(self, image_path):
        """
        Analyzes the given image using the configured Gemini Vision model.
        It constructs the request with a system prompt and the image.
        """
        logging.info(f"Analyzing image: {image_path}")
        try:
            img_pil = self.encode_image_to_pil_image(image_path)

            # Gemini API expects content as a list of parts (text and image).
            # The system prompt and user query are combined here.
            parts = [
                self.system_prompt, # System instruction can be a part
                "What is the total calorie count for this meal? Remember to format your final answer as CALORIES:[number]",
                img_pil # Pass the PIL Image object directly
            ]

            # The generate_content method automatically handles API calls and authentication.
            # Using asyncio.to_thread to run blocking genai.generate_content in an async context,
            # which is necessary because Streamlit's event loop might be blocked otherwise.
            response = await asyncio.to_thread(self.model.generate_content, parts, stream=False)

            return self._parse_api_response(response)

        except genai.types.BlockedPromptException as e:
            # Handle cases where the prompt or content is blocked by safety filters
            logging.error(f"Prompt was blocked for {image_path}: {e.response.prompt_feedback}")
            return {"error": f"Prompt blocked: {e.response.prompt_feedback}", "raw_output": str(e.response.prompt_feedback)}
        except Exception as e:
            # Catch any other exceptions during the API request and re-raise to trigger tenacity retry
            logging.error(f"API request failed for {image_path}: {e}")
            raise ValueError(f"Gemini API request failed: {e}") # Re-raise as ValueError for retry mechanism

    @staticmethod
    def _parse_api_response(result):
        """
        Parses the Gemini API response (text content) to extract the calorie count.
        It uses a regular expression to find the "CALORIES: [number]" pattern.
        """
        try:
            # Access the text from the candidate's content
            content = result.text
            # Use regex to find "CALORIES: " followed by a number (integer or float)
            if calorie_match := re.search(r'CALORIES:\s*(\d+(?:\.\d+)?)', content):
                # Return the extracted number as a float and the raw output
                return {"total_calories": float(calorie_match.group(1)), "raw_output": content}
            logging.warning(f"Could not extract calorie count from response: {content}")
            # If no match found, return an error message with the raw output
            return {"error": "Could not extract calorie count from response", "raw_output": content}
        except Exception as e:
            # Handle errors during parsing the response
            logging.error(f"Error parsing API response: {e}, Raw result: {result}")
            return {"error": f"Error parsing API response: {e}", "raw_output": str(result)}

    async def process_single_image(self, image_path, actual_calories):
        """
        Processes a single image file to estimate calories and records the results.
        This function is primarily used for batch processing/evaluation, not the Streamlit UI directly.
        """
        if not os.path.exists(image_path):
            logging.error(f"File does not exist: {image_path}")
            return None # Return None if the file is not found
        try:
            result = await self.analyze_image(image_path) # Call the async analysis method
            logging.info(f"API raw output for {os.path.basename(image_path)}:\n{result.get('raw_output')}")

            if estimated_calories := result.get('total_calories'):
                # If calorie estimation is successful, return a dictionary with results
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
        return None # Return None if any error occurs during processing

async def main():
    """
    Main function to load the dataset, process images in a batch, and save results.
    This function is intended for standalone execution (e.g., for model evaluation),
    not typically part of the live Streamlit UI.
    """
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, 'DATASET') # Assuming a 'DATASET' folder exists
    results_dir = "estimation_results"

    # Initialize CalorieEstimator (API key is now hardcoded and configured globally)
    estimator = CalorieEstimator()

    csv_path = os.path.join(dataset_path, 'processed_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Please ensure it's in the DATASET folder.")

    # Load and preprocess CSV containing image paths and actual calorie labels
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower() # Clean column names
    if 'calories' not in df.columns:
        raise ValueError("❌ 'calories' column not found in the dataset CSV.")
    df = df.dropna(subset=['calories']) # Drop rows with missing calorie data

    logging.info(f"DataFrame head:\n{df.head()}")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"Columns in CSV: {df.columns.tolist()}")

    # Check for missing image files before processing and log warnings
    missing_files = []
    for _, row in df.iterrows():
        full_path = os.path.join(dataset_path, row['img_path'])
        if not os.path.exists(full_path):
            missing_files.append(full_path)

    if missing_files:
        logging.warning(f"Missing {len(missing_files)} image files. Examples (first 10):")
        for mf in missing_files[:10]:
            logging.warning(mf)

    # Create asynchronous tasks for all images to process them concurrently
    tasks = [
        estimator.process_single_image(os.path.join(dataset_path, row['img_path']), row['calories'])
        for _, row in df.iterrows()
    ]

    results = []
    # Use tqdm for a progress bar to visualize the processing status
    with tqdm(total=len(tasks), desc="Processing images with Gemini") as pbar:
        for task in asyncio.as_completed(tasks): # Process tasks as they complete
            result = await task
            if result:
                results.append(result)
            pbar.update(1) # Update the progress bar

    # Save the processed results to a new CSV file
    if results:
        os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist
        output_file = os.path.join(results_dir, 'estimation_gemini.csv')
        pd.DataFrame(results).to_csv(output_file, index=False) # Save as CSV without index
        logging.info(f'Results are saved to {output_file}')
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    # This block only runs when main.py is executed directly (e.g., `python main.py`),
    # not when it's imported by interphase.py. It's useful for batch evaluation.
    asyncio.run(main())