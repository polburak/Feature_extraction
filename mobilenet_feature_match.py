import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import shutil
import pillow_avif
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Load the MobileNetV2 model (excluding classification layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Feature extraction function
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data, verbose=0)
        return features.flatten()
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {img_path}: {e}")

# Define file paths
target_path = "input/reference.jpg"
dataset_folder = "dataset/"
results_folder = "results_mobilenet/"
os.makedirs(results_folder, exist_ok=True)

# List dataset images (.jpg only)
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".jpg")]

# Extract features from the target image
print("Extracting features from target image using MobileNetV2...")
target_features = extract_features(target_path)

# Compare features from each dataset image
results = []
for img_name in image_files:
    img_path = os.path.join(dataset_folder, img_name)
    try:
        features = extract_features(img_path)
        similarity = cosine_similarity([target_features], [features])[0][0]
        results.append((img_name, similarity))
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

# Sort results by similarity (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Print results to console
print("\nTop Matching Results:")
for name, score in results:
    print(f"{name}: Similarity = %{score * 100:.2f}")

# Copy similar images to output folder and write to output.txt
threshold = 0.85
output_txt_path = os.path.join(results_folder, "output.txt")

print("\nCopying similar images to 'results_mobilenet/' folder and saving results to output.txt:")
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

print("\nDone using MobileNetV2!")
