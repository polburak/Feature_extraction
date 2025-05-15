# Feature Extraction with Deep Learning Models

A comparative image feature extraction system using MobileNetV2, ResNet50, EfficientNetB0, and CLIP (including OpenCLIP) to compute both visual and semantic similarities between images.

## This project extracts feature vectors from images using different pre-trained deep learning models and compares the similarities between images. It includes:

- Visual similarity models: MobileNetV2, ResNet50, EfficientNetB0
- Semantic similarity model: CLIP (ViT-B/32), OpenCLIP (ViT-B/32, ViT-L/14, vb.)
- Cosine similarity scoring
- Auto-copying of top matches to result folders

## Sample Data

You can download sample reference and dataset images from the links below:

- 📥 [Download `input/` folder (reference image)](https://drive.google.com/drive/folders/1n3GDFoQeQnrIUr1jWmqPmk-Ma1oX2laO?usp=sharing)
- 📥 [Download `dataset/` folder (images to compare)](https://drive.google.com/drive/folders/1VcItX9HWAGZfJxiO_DKOUbtOPEcAq0lZ?usp=sharing)


## Installation

Install required dependencies:

```bash
pip install -r requirements.txt

To run the CLIP version, install additional libraries:

pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex tqdm

## (Usage)

1. Place your reference image in the `input/` folder (e.g., `input/reference.jpg`)
2. Add your dataset images in the `dataset/` folder
3. Run one of the scripts:
4. Similar images will be copied to results_*/ folders
5. Matching results will be saved in output.txt

```bash
python mobilenet_feature_match.py
python resnet_feature_match.py
python efficientnet_feature_match.py
python clip_feature_match.py
python openclip_feature_matcher.py
```

## Model Comparison

| Model          | Type               | Strength                              |
|----------------|--------------------|---------------------------------------|
| MobileNetV2    | CNN (Lightweight)  | Fast, Mobile-friendly                 |
| ResNet50       | CNN (Residual)     | Balanced performance                  |
| EfficientNetB0 | CNN (Optimized)    | Good accuracy, small size             |
| CLIP           | Vision Transformer | Semantic understanding                |
| OPENCLIP       | Vision Transformer |Large-scale pretrained, multimodal use |

## File Structure
```
feature_extraction/
├── input/ # Reference image
├── dataset/ # Images to compare
├── results_mobilenet/
├── results_resnet/
├── results_efficientnet/
├── results_clip/
├── results_openclip/
├── mobilenet_feature_match.py
├── resnet_feature_match.py
├── efficientnet_feature_match.py
├── clip_feature_match.py
├── openclip_feature_match.py
├── requirements.txt
└── README.md
```
