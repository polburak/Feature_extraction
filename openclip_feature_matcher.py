import os
import torch
import open_clip
from PIL import Image
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# Select device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OpenCLIP model and preprocessing pipeline
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",  # Alternatives: ViT-L-14, ViT-H-14
    pretrained="laion2b_s34b_b79k",  # Trained on a large dataset
    device=device
)

model.eval()

# Feature extraction function using OpenCLIP
def extract_openclip_features(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {image_path}: {e}")

# Define paths
target_path = "input/reference.jpg"
dataset_folder = "dataset/"
results_folder = "results_openclip/"
os.makedirs(results_folder, exist_ok=True)

# Collect all .jpg images from dataset
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".jpg")]

# Extract features from the target image
print("Extracting OpenCLIP features for target image...")
target_features = extract_openclip_features(target_path)

# Compare dataset images to the target image
results = []
for img_name in image_files:
    img_path = os.path.join(dataset_folder, img_name)
    try:
        features = extract_openclip_features(img_path)
        similarity = cosine_similarity([target_features], [features])[0][0]
        results.append((img_name, similarity))
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

# Sort results by similarity (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Print top matching results
print("\nTop Matching Results using OpenCLIP:")
for name, score in results:
    print(f"{name}: Similarity = %{score * 100:.2f}")

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
            percent = f"%{score * 100:.2f}"
            line = f"{index:>2}. {name:<15} {percent}"
            f.write(line + "\n")
            print(line)
            index += 1

print("\nDone with OpenCLIP!")
