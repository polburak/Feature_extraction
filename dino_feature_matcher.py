import os
import torch
import timm
from PIL import Image
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
import time

# Device selection (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load DINO model (e.g., ViT-B/16)
model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
model.eval().to(device)

# Preprocessing transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Feature extraction using DINO
def extract_dino_features(image_path):
    try:
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {image_path}: {e}")

# Paths
target_path = "input/reference.jpg"
dataset_folder = "dataset/"
results_folder = "results_dino/"
os.makedirs(results_folder, exist_ok=True)

# List all .jpg images in the dataset
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".jpg")]

# Start timing
start = time.time()

# Extract features for target image
print("Extracting DINO features for the target image...")
target_features = extract_dino_features(target_path)

# Compare dataset images to the target image
results = []
for img_name in image_files:
    img_path = os.path.join(dataset_folder, img_name)
    try:
        features = extract_dino_features(img_path)
        similarity = cosine_similarity([target_features], [features])[0][0]
        results.append((img_name, similarity))
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

# Sort results by similarity
results.sort(key=lambda x: x[1], reverse=True)

# Print top matching results
print("\nTop Matching Results using DINO:")
for name, score in results:
    print(f"{name}: Similarity = {score * 100:.2f}%")

# Copy similar images and save to output.txt
threshold = 0.90
output_txt_path = os.path.join(results_folder, "output.txt")

print(f"\nCopying similar images to '{results_folder}' and saving results to output.txt:")
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Top Matching Images (Similarity ≥ {int(threshold * 100)}%):\n\n")
    index = 1
    for name, score in results:
        if score >= threshold:
            src = os.path.join(dataset_folder, name)
            dst = os.path.join(results_folder, name)
            shutil.copy(src, dst)
            percent = f"{score * 100:.2f}%"
            line = f"{index:>2}. {name:<15} {percent}"
            f.write(line + "\n")
            print(line)
            index += 1

# End timing
end = time.time()
print(f"\n Done using DINO! Total processing time: {end - start:.2f} seconds")
