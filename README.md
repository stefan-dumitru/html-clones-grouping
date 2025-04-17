# HTML Clone Clustering by Visual Similarity

This project groups visually similar HTML documents using full-page screenshots, ResNet50 features, and DBSCAN clustering.

## Workflow

1. Take full-page screenshots of each HTML file.
2. Extract image features using ResNet50.
3. Cluster screenshots using DBSCAN.
4. Organize results into folders.

## Requirements

- Python 3.x
- Playwright
- TensorFlow / Keras
- scikit-learn
- Pillow
