import logging
import os
import datetime as dt
import base64
import io
import re
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
from tenacity import (
    retry, wait_exponential, stop_after_attempt, retry_if_exception_type
)
from tqdm import tqdm

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
OPENAI_API_KEY = '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'


# Load API key from .env file
load_dotenv()
api_key = OPENAI_API_KEY
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
#print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))


class CalorieEstimator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def encode_image_to_base64(image_path):
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='WEBP', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def analyze_image(self, image_path):
        img_base64 = self.encode_image_to_base64(image_path)
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the total calorie count for this meal? Remember to format your final answer as CALORIES:[number]"},
                {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{img_base64}"}}
            ]}
        ]

        async with aiohttp.ClientSession(timeout=ClientTimeout(total=120)) as session:
            response = await self._make_api_request(session, {
                "model": self.model,
                "messages": message,
                "max_tokens": 4096,
                "temperature": 0.0
            })
            return await self._parse_api_response(response)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=20),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((aiohttp.ClientError, ValueError, asyncio.TimeoutError)),
        retry_error_callback=lambda retry_state: {
            "error": f"All retries failed: {retry_state.outcome.exception()}",
            "raw_output": str(retry_state.outcome.exception())
        }
    )
    async def _make_api_request(self, session, payload):
        async with session.post(self.api_url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(f"API request failed: {await response.text()}")
            return await response.json()

    @staticmethod
    async def _parse_api_response(result):
        content = result['choices'][0]['message']['content']
        if calorie_match := re.search(r'CALORIES:\s*(\d+(?:\.\d+)?)', content):
            return {"total_calories": float(calorie_match.group(1)), "raw_output": content}
        return {"error": "Could not extract calorie count response", "raw_output": content}


# Move this function OUTSIDE the class
async def process_single_image(estimator, image_path, actual_calories):
    try:
        result = await estimator.analyze_image(image_path)
        if estimated_calories := result.get('total_calories'):
            return {
                'image': os.path.basename(image_path),
                'actual_calories': float(actual_calories),
                'estimated_calories': float(estimated_calories),
                'calorie_difference': abs(float(estimated_calories) - float(actual_calories)),
                'llm_output': result.get('raw_output', 'N/A'),
            }
        logging.error(f"Error processing {image_path}: {result.get('error', 'Unknown error')}\nLLM Output: {result.get('raw_output', 'No output')}")
    except Exception as e:
        logging.error(f"Exception processing {image_path}: {str(e)}")
    return None


async def main():
    # Change this if your dataset is in a different location
    dataset_path = os.path.join(os.path.dirname(__file__), 'DATASET')
    results_dir = "estimation_results"

    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in file")

    estimator = CalorieEstimator(api_key=api_key)

    #csv_path = os.path.join(dataset_path, 'processed_labels.csv')
    #if not os.path.exists(csv_path):
    #   raise FileNotFoundError(f"CSV file not found: {csv_path}")

    #df = pd.read_csv(csv_path).dropna(subset=['calories'])
    #df = pd.read_csv(csv_path)
    #df.columns = df.columns.str.strip().str.lower()
    #df = df.dropna(subset=['calories']) 

    #df = pd.read_csv(csv_path)
    #if 'calories' in df.columns:
        #df = df.dropna(subset=['calories'])
    #else:
        #print("Column 'calories' not found in the dataset.")
        
        #df = pd.read_csv(csv_path, encoding='utf-8', error_bad_lines=False)
    #print(df.columns)
    #print("Columns in CSV:", df.columns.tolist())

    #df = pd.read_csv(csv_path)
    #df.columns = df.columns.str.strip().str.lower()  # removes spaces, makes all lowercase
    
    #if 'calories' not in df.columns:
        #print("❌ 'calories' column not found. Available columns:", df.columns.tolist())
    #else:
        #df = df.dropna(subset=['calories'])


    #tasks = [
        #process_single_image(estimator, os.path.join(dataset_path, row['img_path']), row['calories'])
        #for _, row in df.iterrows()
        #if os.path.exists(os.path.join(dataset_path, row['img_path']))
    #]

    results = []
    #with tqdm(total=len(tasks), desc="Processing images") as pbar:
        #for task in asyncio.as_completed(tasks):
            #result = await task
            #if result:
                #results.append(result)
            #pbar.update(1)

    if results:
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, 'estimation_openai.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        logging.info(f'Results are saved to {output_file}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


