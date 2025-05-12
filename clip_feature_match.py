import os
import torch
import clip
from PIL import Image
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# Select device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract feature vector from an image using CLIP
def extract_clip_features(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {image_path}: {e}")

# Define file paths
target_path = "input/reference.jpg"
dataset_folder = "dataset/"
results_folder = "results_clip/"
os.makedirs(results_folder, exist_ok=True)

# List all .jpg images in the dataset folder
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".jpg")]

# Extract features from the reference (target) image
print("Extracting CLIP features for target image...")
target_features = extract_clip_features(target_path)

# Extract features and compute similarity for all dataset images
results = []
for img_name in image_files:
    img_path = os.path.join(dataset_folder, img_name)
    try:
        features = extract_clip_features(img_path)
        similarity = cosine_similarity([target_features], [features])[0][0]
        results.append((img_name, similarity))
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

# Sort results by similarity score (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Print top matches to console
print("\nTop Matching Results using CLIP:")
for name, score in results:
    print(f"{name}: Similarity = %{score * 100:.2f}")

# Copy similar images to results folder and save to output.txt
threshold = 0.85
output_txt_path = os.path.join(results_folder, "output.txt")

print("\nCopying similar images to 'results_clip/' folder and saving results to output.txt:")
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Top Matching Images (Similarity ≥ {int(threshold * 100)}%):\n\n")
    index = 1
    for name, score in results:
        if score >= threshold:
            src = os.path.join(dataset_folder, name)
            dst = os.path.join(results_folder, name)
            shutil.copy(src, dst)  # Copy matching image to results folder
            percent = f"%{score * 100:.2f}"
            line = f"{index:>2}. {name:<15} {percent}"
            f.write(line + "\n")
            print(line)
            index += 1

print("\nDone with CLIP!")
