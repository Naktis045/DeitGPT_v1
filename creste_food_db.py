import pandas as pd
import logging
import json
import os
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # fixed from datetime
)

def load_data():
    path = r'C:\Users\Mukhammad Ali\DietGPT\usda'  # Fixed path using raw string
    extension = 'csv'
    os.chdir(path)
    result = glob.glob(f'*.{extension}')
    print(result)
    food_df = pd.read_csv('food.csv', low_memory=False)
    food_nutrients_df = pd.read_csv('food_nutrient.csv', low_memory=False)
    nutrients_df = pd.read_csv('nutrient.csv', low_memory=False) 
    categories_df = pd.read_csv('food_category.csv', low_memory=False)
    valid_types = ['foundation_food']
    food_df = food_df[food_df['data_type'].isin(valid_types)]
    return food_df, food_nutrients_df, nutrients_df, categories_df

def process_nutrients(food_df, food_nutrients_df, nutrients_df):
    nutrients_merge = pd.merge(
        food_nutrients_df,
        nutrients_df,
        left_on='nutrient_id',
        right_on='id',
    ).drop_duplicates(subset=['fdc_id', 'nutrient_id', 'unit_name'])

    nutrients_merge = nutrients_merge[nutrients_merge['fdc_id'].isin(food_df['fdc_id'])]
    clean_nutrients = nutrients_merge[
        ~((nutrients_merge['name'] == 'Energy') & (nutrients_merge['unit_name'] != 'KCAL'))]
    return clean_nutrients

def analyze_nutrient_coverage(nutrients_merge, food_df):
    total_foods = len(food_df)
    nutrient_counts = nutrients_merge.groupby('name').size()
    threshold = 0.005 * total_foods
    nutrient_counts = nutrient_counts[nutrient_counts >= threshold]
    logging.info(f"\nFound {len(nutrient_counts)} unique nutrients in {total_foods} foods")
    logging.info("\nNutrient Coverage (sorted by frequency):")
    logging.info("-" * 80)

    for nutrient, count in sorted(nutrient_counts.items(), key=lambda x: (x[1] / total_foods), reverse=True):
        percentage = (count / total_foods) * 100
        logging.info(f"{nutrient:<50} {percentage:>6.3f}% ({count:>8} occurrences)")
    return nutrient_counts

def process_food_items(food_df, nutrients_merge, nutrient_counts):
    data = []
    total_items = len(food_df)

    for idx, (_, food_row) in enumerate(food_df.iterrows(), 1):
        if idx % 1000 == 0:
            logging.info(f"Processing food item {idx}/{total_items}")

        food_nutrients = nutrients_merge[
            (nutrients_merge['fdc_id'] == food_row['fdc_id']) &
            (nutrients_merge['name'].isin(nutrient_counts.index))
        ]

        food_item = {
            'name': food_row['description'],
            'fdc_id': food_row['fdc_id'],
            'nutrients': {}
        }

        for _, nutrient in food_nutrients.iterrows():
            if pd.notna(nutrient['amount']):
                column_name = f"{nutrient['name'].strip()} ({nutrient['unit_name']})"
                food_item['nutrients'][column_name] = nutrient['amount']

        data.append(food_item)
    return data

def main():
    food_df, food_nutrients_df, nutrients_df, categories_df = load_data()
    nutrients_merge = process_nutrients(food_df, food_nutrients_df, nutrients_df)
    nutrient_counts = analyze_nutrient_coverage(nutrients_merge, food_df)
    data = process_food_items(food_df, nutrients_merge, nutrient_counts)
    output_file = 'food_db.json'
    logging.info(f"Exporting {len(data)} food items to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info("Export complete")

if __name__ == '__main__':
    main()
