import csv
import json

# File paths
first_result = 'final_csv_result/private_test_image_first_step_retrieval_results.csv'
reranking_csv = '/solved_with_entropy.csv'
output_csv = 'final_csv_result/updated_first_result.csv'


# Step 1: Read first CSV to extract top 10 image_ids per query_id
query_image_ids = {}

with open(first_result, 'r', newline='', encoding='utf-8') as f1:
    reader = csv.DictReader(f1)
    header = reader.fieldnames  # Preserve original headers
    for row in reader:
        query_id = row['query_id']
        image_ids = [row[f'image_id_{i}'] for i in range(1, 11) if f'image_id_{i}' in row]
        query_image_ids[query_id] = image_ids

# Step 2: Read reranking CSV to get better-ranked images
reranked_results = {}

with open(reranking_csv, 'r', newline='', encoding='utf-8') as f2:
    reader = csv.DictReader(f2)
    for row in reader:
        query_id = row['query_id']
        chose_rank = row.get('chose_rank')
        image_id = row.get('image_id')
        if chose_rank and image_id:
            try:
                rank_int = int(chose_rank)
                reranked_results[query_id] = {
                    "num_select": rank_int,
                    "image_id": image_id
                }
            except ValueError:
                continue  


# Step 3: Merge reranked result into the first result
for query_id, rerank_data in reranked_results.items():
    if query_id in query_image_ids:
        selected_image = rerank_data['image_id']
        num_select = rerank_data['num_select']
        current_list = query_image_ids[query_id]

        if num_select != 1:
            if selected_image in current_list:
                current_list.remove(selected_image)

            # Insert at the top
            current_list.insert(0, selected_image)

            # Trim to 10 items
            query_image_ids[query_id] = current_list[:10]

# Step 4: Save the updated list to a new CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.DictWriter(fout, fieldnames=['query_id'] + [f'image_id_{i}' for i in range(1, 11)])
    writer.writeheader()

    for query_id, image_list in query_image_ids.items():
        row = {'query_id': query_id}
        for i, image_id in enumerate(image_list, start=1):
            row[f'image_id_{i}'] = image_id
        writer.writerow(row)

print(f"Updated results saved to: {output_csv}")
