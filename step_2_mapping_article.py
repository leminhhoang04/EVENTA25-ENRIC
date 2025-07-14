import json
import csv
import os

# Load the caption and article mappings
with open("./assemble_result/cot_5_things_fact_more_event.json", 'r', encoding='utf-8') as cap_f:
    matching_captions_list = json.load(cap_f)  # {query_id: caption}
matching_captions = {}
for cap in matching_captions_list:
    matching_captions[cap['query_id']] = cap['enhanced_caption']

# with open('./assemble_result/cot_5_things_fact_more_event_2_llama.json', 'r', encoding='utf-8') as cap_f:
#     matching_captions = json.load(cap_f)


with open('./data/database/matching_articles.json', 'r', encoding='utf-8') as f:
    matching_image_article = json.load(f)  # {image_id: article_id}

# Input and output CSV paths
updated_rerank_path = 'final_csv_result/updated_first_result.csv'
output_file = './final_csv_result/surround_merging_final_results.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Process and write new CSV
with open(updated_rerank_path, 'r', newline='', encoding='utf-8') as f1, \
    open(output_file, 'w', newline='', encoding='utf-8') as csvfile:

    reader = csv.DictReader(f1)
    writer = csv.writer(csvfile)

    header = ['query_id'] + [f'article_id_{i+1}' for i in range(10)] + ['generated_caption']
    writer.writerow(header)

    for row in reader:
        query_id = row['query_id']
        
        article_ids = []
        for i in range(1, 11):
            image_key = row.get(f'image_id_{i}')
            article_id = matching_image_article.get(image_key, "N/A")
            article_ids.append(article_id)

        caption = matching_captions.get(query_id, "N/A")

        writer.writerow([query_id] + article_ids + [caption])

print(f"Finished writing to: {output_file}")
