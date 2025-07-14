# import cv2
# import numpy as np
# import os

# def sift_descriptors(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Warning: Could not load {image_path}")
#         return None
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(img, None)
#     return descriptors

# def match_descriptors(desc1, desc2, ratio_thresh=0.75):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(desc1, desc2, k=2)
#     good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
#     return good

# def compute_score(matches):
#     if not matches:
#         return float('inf')  # Bad match
#     return 1 / len(matches)  # Inverse: more matches = better score

def rerank(data):
    pass

# # --- Load Descriptors ---
# image_paths = {
#     'img1.jpg': './data/track1_public/pub_images/0a1feed226ff315d.jpg',
#     'img2.jpg': './data/track1_public/pub_images/0a7f22148940991c.jpg',
#     'img3.jpg': './data/track1_public/pub_images/0a3651bbeb6d5a05.jpg'
# }

# database_descriptors = {
#     name: sift_descriptors(path)
#     for name, path in image_paths.items()
# }

# # Query image (same as img2)
# query_desc = sift_descriptors(image_paths['img2.jpg'])

# # --- Match and Score ---
# results = []

# for img_name, desc in database_descriptors.items():
#     if desc is None or query_desc is None:
#         continue
#     good_matches = match_descriptors(query_desc, desc)
#     score = compute_score(good_matches)
#     results.append((img_name, score))

# # --- Sort and Rank ---
# results = sorted(results, key=lambda x: x[1])
# top_k = [img for img, _ in results[:5]]

# # Optional: Rerank results
# # results = rerank(results)

# print("All results (sorted):", results)
# print("Top-K results:", top_k)
