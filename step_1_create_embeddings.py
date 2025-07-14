# from clip_model import ImageProjection, ImagePathDataset
# from internvl import CustonInternVLRetrievalModel
# import os
# from tqdm import tqdm
# import torch

# output_folder = "./embeddings/maching_new_database_internvlg/"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# with torch.no_grad():
#     embedding_model = CustonInternVLRetrievalModel()
    

#     image_paths = os.listdir("./web_crawling/imgs")
#     for image in tqdm(image_paths):
#         if not image.endswith(('.jpg', '.jpeg', '.png')):
#             continue
#         name = os.path.splitext(image)[0]
#         image_path = os.path.join("./data/track1_private/query", image)
#         if not os.path.isfile(image_path):
#             continue
#         embedding = embedding_model.encode_image([image_path], is_path= True)
#         embedding = embedding.squeeze(0)   
#         # embedding = embedding_model.prediction(image_path)
#         output_path = os.path.join(output_folder, f"{name}.pt")
#         torch.save(embedding, output_path)
from internvl import CustonInternVLRetrievalModel
import os
from tqdm import tqdm
import torch

# Set this to 1, 2, 3, or 4
part_number = 3  # Change this to the part you want to run
total_parts = 4
assert 1 <= part_number <= total_parts, "part_number must be between 1 and 4"

# Optional: set device
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

output_folder = "./embeddings/maching_new_database_internvlg/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with torch.no_grad():
    embedding_model = CustonInternVLRetrievalModel(device=device)

    image_paths = [f for f in os.listdir("./web_crawling/imgs") if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths.sort()
    total = len(image_paths)
    split_size = total // total_parts

    start = (part_number - 1) * split_size
    end = total if part_number == total_parts else start + split_size
    image_subset = image_paths[start:end]

    for image in tqdm(image_subset, desc=f"Processing Part {part_number}/{total_parts}"):
        name = os.path.splitext(image)[0]
        image_path = os.path.join("./web_crawling/imgs", image)
        if not os.path.isfile(image_path):
            continue
        embedding = embedding_model.encode_image([image_path], is_path=True).to(device)
        embedding = embedding.squeeze(0)
        output_path = os.path.join(output_folder, f"{name}.pt")
        torch.save(embedding.cpu(), output_path)

