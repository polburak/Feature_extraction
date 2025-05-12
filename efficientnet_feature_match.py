import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import shutil
import pillow_avif
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# EfficientNetB0 modelini yükle (sınıflandırma katmanı hariç)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Özellik çıkarma fonksiyonu
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

# Dosya yolları
target_path = "input/reference.jpg"
dataset_folder = "dataset/"
results_folder = "results_efficientnet/"
os.makedirs(results_folder, exist_ok=True)

# Dataset görsellerini listele
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(".jpg")]

# Hedef görselin özelliklerini çıkar
print("Extracting features from target image using EfficientNetB0...")
target_features = extract_features(target_path)

# Karşılaştırma
results = []
for img_name in image_files:
    img_path = os.path.join(dataset_folder, img_name)
    try:
        features = extract_features(img_path)
        similarity = cosine_similarity([target_features], [features])[0][0]
        results.append((img_name, similarity))
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")

# Sonuçları sırala
results.sort(key=lambda x: x[1], reverse=True)

# Sonuçları yazdır
print("\nTop Matching Results:")
for name, score in results:
    print(f"{name}: Similarity = %{score * 100:.2f}")

# Benzer görselleri kopyala ve output.txt'ye yaz
threshold = 0.85
output_txt_path = os.path.join(results_folder, "output.txt")

print("\nCopying similar images to 'results_efficientnet/' folder and saving results to output.txt:")
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Top Matching Images (Similarity ≥ {int(threshold * 100)}%):\n\n")
    index = 1
    for name, score in results:
        if score >= threshold:
            src = os.path.join(dataset_folder, name)
            dst = os.path.join(results_folder, name)
            shutil.copy(src, dst)
            yuzde = f"%{score * 100:.2f}"
            line = f"{index:>2}. {name:<15} {yuzde}"
            f.write(line + "\n")
            print(line)
            index += 1

print("\nDone using EfficientNetB0!")
