# PotholeVision: Automating Pothole Detection and GeoMapping

## Overview
The road inspection process for cities is highly manual, slow, and expensive. Municipalities desperately need a scalable way to identify road damage faster and prioritize repairs. To solve this, we developed PotholeVision, an automated pothole detection and geo-mapping solution. We built a big data pipeline that ingests video and image data in batches, extracts frames, and evaluates them using a ResNet18-based Convolutional Neural Network. This technical pipeline drives direct business value by pushing positive pothole predictions into a dynamic ArcGIS dashboard. Through this dashboard, Public Works planners can view pothole counts, exact locations, and street traffic volumes to prioritize repairs on the busiest roads. Ultimately, PotholeVision drives faster repairs, better allocation of limited maintenance resources, and a significant reduction in vehicle accidents.

## Project Structure
- `model_artifacts/`: Contains the trained model weights (`weights.pth`), configuration (`config.json`), and a TorchScript version (`model_v3_traced.pt`).
- `notebooks/`: 
  - `Pothole_CNN.ipynb`: Initial training and baseline experiments.
  - `Pothole_version2.ipynb`: Fine-tuning, hard negative mining, and domain adaptation experiments.
  - `pothole_pipeline_final.ipynb`: Databricks-ready inference pipeline (Bronze/Silver/Gold architecture).
- `src/`: 
  - `pothole_classifier.py`: A reusable Python wrapper for model inference.
- `reports/`: 
  - `team_pothole_summary_report.md/pdf`: Comprehensive project summary and findings.
- `requirements.txt`: Python dependencies.

## Key Findings
- **Domain Shift:** High false-negative rates on external datasets were primarily driven by domain shift (visual style differences) rather than label noise.
- **Strict Boundary:** Removing ambiguous classes like alligator cracks from the main negative training pool helped stabilize the decision boundary.
- **Model V3:** Our final model achieved a **90.6% recall** and only a **9.4% miss rate** on an independent external benchmark (Neha dataset), a significant improvement over earlier versions.

## Usage (Inference)
The `src/pothole_classifier.py` provides a clean interface for predicting potholes in images:

```python
from src.pothole_classifier import PotholeClassifier

# Initialize with the model artifacts directory
clf = PotholeClassifier(r"./model_artifacts")

# Predict on a single image
result = clf.predict("path/to/image.jpg")
print(result)
```

## Datasets Used
- **RDD 2020:** Multi-national road damage dataset (7 countries).
- **Sovit Pothole Dataset:** Dashcam-based dataset for vehicle-perspective adaptation.
- **YOLOv8 Pothole Segmentation:** Used for strengthening the positive class in Model V3.
- **Neha Normal/Pothole Dataset:** Used as an independent external benchmark.

## Team
Team 3: Chunfang Wang, James Pashek, Joseph Sheehan, Moses Akoto, Madhu Damani, Tao Fang.

* This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.
