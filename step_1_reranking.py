import step_1_reranking_SIFT as sift
import step_1_reranking_ensemble as ensemble
# import step_1_reranking_text_ensemble as text_ensemble
import argparse
from pathlib import Path
import pandas as pd
import re
import time
from environment import *
from helper import *

strats = {
    'sift': sift.rerank,
    'ensemble': ensemble.rerank,
    # 'text_ensemble': text_ensemble.rerank
}
DEFAULT_RETRIEVAL_PATH = Path('./final_csv_result/first_step_retrieval_results.csv')
DEFAULT_RES_DIR = Path('./rerank_results')

def get_input(input_path):
    """
    Return a dictionary where keys are the query_id and the values are the list of image_ids, in order [top1_image_id, top_2_image_id,...]
    """
    input_df = pd.read_csv(input_path)

    def extract_number(col_name):
        match = re.search(r'image_id_(\d+)', col_name)
        return int(match.group(1)) if match else float('inf')

    image_id_cols = sorted(
        [col for col in input_df.columns if col.startswith('image_id_')],
        key=extract_number
    )

    result = {}
    for _, row in input_df.iterrows():
        query_id = row['query_id']
        image_ids = [row[col] for col in image_id_cols if pd.notnull(row[col])]
        result[query_id] = image_ids

    return result

def save_result(result, res_dir: Path):
    rows = []
    for query_id, image_ids in result.items():
        row = {"query_id": query_id}
        for i, image_id in enumerate(image_ids):
            row[f"image_id_{i}"] = image_id
        rows.append(row)

    df = pd.DataFrame(rows)
    to_save_path = res_dir / f'output_{int(time.time())}.csv'
    df.to_csv(to_save_path, index=False)
    print(f"Output saved to \"{res_dir / f'output_{int(time.time())}.csv'}\"")

def input_insights(input):
    print(f"{'='*20} Input insights {'='*20}")
    print(f"{len(input) = }")
    print(f"{all([len(lst)==10 for lst in input.values()]) = }")
    
    all_image_ids = list(set(x for lst in input.values() for x in lst))
    print(f"{len(all_image_ids) = }")
    
    all_image_path = images_from_ids(all_image_ids)
    print(f"All image suffix: {set([path.suffix for path in all_image_path])}")
    
    all_query_path = queries_from_ids(list(input.keys()))
    print(f"All query suffix: {set([path.suffix for path in all_query_path])}")
    
    print(f"{len(set(all_image_ids) - set(database_image_map.keys())) = }")
    print(f"{'='*58}\n\n")

def main(args):
    res_dir = Path(args.res_dir)
    retrieval_path = Path(args.retrieval_path)
    rerank_fn = strats[args.strategy]
    
    input = get_input(retrieval_path)
    input_insights(input)
    result = rerank_fn(input)
    save_result(result, res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_path', type=str, default=DEFAULT_RETRIEVAL_PATH, help="Retrival csv file to be reranked")
    parser.add_argument('--res_dir', type=str, default=DEFAULT_RES_DIR, help="Resulting path")
    parser.add_argument('--strategy', type=str, choices=strats.keys(), required=True,
                        help="Reranking strategy to use")
    args = parser.parse_args()
    main(args)