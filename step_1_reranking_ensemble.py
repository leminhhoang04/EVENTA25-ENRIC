import rerank_models

def rerank(data: dict[str, list[str]], k=60) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for query_id, image_ids in data.items():
        rrf_scores = {
            image_id: 1 / (k + rank) 
            for rank, image_id in enumerate(image_ids, 1)
        }
        for rerank_model in rerank_models.models:
            ranks = rerank_model.rerank(query_id, image_ids)
            for rank, image_id in enumerate(ranks, 1): 
                rrf_scores[image_id] += 1 / (k + rank)

        sorted_images = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result[query_id] = [img_id for img_id, _ in sorted_images]
    return result
