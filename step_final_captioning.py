import os
import json
import argparse
from tqdm import tqdm
from llama3 import Llama
# from inference_llama_finetuning import load_finetuned_model, llama_generate_caption

def main(args):
    CUDA_VISIBLE_DEVICES=4

    # Load model
    if args.finetuning:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        adapter_path = "experiments/checkpoint-20"
        # model , tokenizer = load_finetuned_model(model_name, adapter_path)
    
    else:
        bot = Llama(device=args.device)

    # Load summaries
    with open(args.summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    # Load captions
    with open(args.caption_file, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join("private_test_final_elements_json", "reranking_query_first_article_question_answer.json")

    # Process and enrich captions
    final_result = {}
    for query, caption in tqdm(caption_data.items(), desc="Enriching captions"):
        if (query == "index"):
            continue
        
        summary = summary_data.get(query, {}).get("summary", "")
        if not summary:
            print(f"Warning: No summary found for query '{query}'")
            continue

        if args.finetuning:
            # enriched_caption = llama_generate_caption(model, tokenizer, caption, summary)
            print("Finetuning")
        else:
            enriched_caption = bot.question_answer(caption, summary)
        
        final_result[query] = enriched_caption

        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_result, out_f, indent=2, ensure_ascii=False)

    print(f"✅ Enriched captions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich image captions using news summaries.")
    parser.add_argument("--summary_file", type=str, default="./private_test_final_elements_json/reranking_query_first_article_summary.json", help="Path to JSON file with news summaries.")
    parser.add_argument("--caption_file", type=str, default="./private_test_final_elements_json/final_rerank_private_test_detail_top1_caption.json", help="Path to JSON file with image captions.")
    parser.add_argument("--output_dir", type=str, default="./private_enrich_caption_results", help="Directory to save enriched captions.")
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the Llama model on (e.g., 'cuda:0').")
    CUDA_VISIBLE_DEVICES=4

    args = parser.parse_args()
    main(args)
