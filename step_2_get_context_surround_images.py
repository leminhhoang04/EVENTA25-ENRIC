import os
import torch
from tqdm import tqdm
import argparse
import json
import csv
from llama3 import Llama  # Make sure Llama class is correctly implemented
import re
def main(args):
    retrieval_results_folder = args.result_folder
    database_file = args.database_file
    output_file = "./private_test_final_elements_json/reranking_query_first_article_context_surround_images_second_version.json"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    llms_bot = Llama(device='cuda:5')

    # Read CSV
    with open(retrieval_results_folder, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Load JSON database

    with open('private_test_final_elements_json/final_rerank_private_test_detail_top1_caption.json', 'r') as caption_f:
        caption_data = json.load(caption_f)
        
    
    # Read input JSON file
    with open("private_test_final_elements_json/result-hoang.json", 'r') as f:
        new_database = json.load(f)

    # Extract query_id and crawl_caption into a new dictionary
    newdatabase_data = {}
    for value in new_database:
        temp = {
            'position': value['article_position'],
            'content' : value['article']
        }
        newdatabase_data[value['query_id']] = temp


    
    result = {}

    for row in tqdm(data):
        query = row["query_id"]
        first_article = row["article_id_1"]
        image_content_caption = caption_data[query]
        content = newdatabase_data[query]['content']
        position = newdatabase_data[query]['position']
        
        # FOLLOW CHARACTERS PUT THE IMAGE_CONTETNT_CAPTION with a brackets <> inside the full content of the article
        if position >= 0 and position <= len(content):
            full_article_content_with_caption = (
                content[:position] + f" <<{image_content_caption}>> " + content[position:]
            )
        else:
            print(f"Invalid position {position} for article {first_article}, inserting at end.")
            full_article_content_with_caption = content + f" <{image_content_caption}>"

        # if first_article not in database_data:
        #     print(f"Warning: Article {first_article} not found in database.")
        #     continue

        # article_content = database_data[first_article]["content"]

        summarized_content = llms_bot.provided_context(full_article_content_with_caption)
        result[query] = summarized_content
        
        with open(output_file, 'w') as outfile:
            json.dump(result, outfile, indent=2)

    print(f"Summarized results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, default="./final_csv_result/private_final_retrieval_merging_final_results.csv", help="CSV file with retrieval results.")
    parser.add_argument('--database_file', type=str, default="./data/database/database.json", help="JSON file containing article contents.")
    args = parser.parse_args()
    main(args)