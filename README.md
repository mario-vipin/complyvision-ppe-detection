# 🦺 ComplyVision

**Real-time PPE and face mask compliance detection using deep learning.**

ComplyVision is a computer vision system that detects personal protective equipment (PPE) and face mask compliance in images and live webcam feed. Built by fine-tuning YOLOv8 on a custom-labeled dataset, it identifies 10 classes — including hardhats, masks, safety vests, and their corresponding non-compliance states — and is deployed as a live web application.

![Uploading image.png…]()


🔗 **[Live Demo](https://complyvision-ppe-detection-awb7ujhvaxwpaqtlcgu9mh.streamlit.app/)**

---

## Overview

Manual monitoring of PPE and mask compliance in workplaces is labor-intensive and inconsistent. ComplyVision automates this using object detection — given an image or video frame, it locates people and safety equipment, then classifies compliance status per item with bounding boxes and confidence scores.

## Features

- Real-time object detection using a fine-tuned YOLOv8 model
- Detects **10 classes**: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle
- Web interface built with Streamlit — supports both image upload and live webcam input
- Adjustable confidence threshold for detections
- Compliance summary showing detected object counts

## Model Performance

Fine-tuned YOLOv8n on the [Construction Site Safety dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) (Roboflow, v27).

| Metric | Score |
|---|---|
| mAP@0.5 (overall) | 0.780 |
| mAP@0.5-0.95 | 0.467 |
| Precision | 0.871 |
| Recall | 0.724 |

**Per-class mAP@0.5:**

| Class | mAP@0.5 |
|---|---|
| Mask | 0.886 |
| machinery | 0.946 |
| Safety Vest | 0.872 |
| Safety Cone | 0.869 |
| Hardhat | 0.826 |
| Person | 0.815 |
| NO-Safety Vest | 0.736 |
| NO-Hardhat | 0.652 |
| NO-Mask | 0.635 |
| vehicle | 0.561 |

## Tech Stack

- **Model**: YOLOv8 (Ultralytics) — transfer learning from pretrained weights
- **Framework**: PyTorch
- **Deployment**: Streamlit
- **Training**: Google Colab (Tesla T4 GPU)
- **Dataset tooling**: Roboflow
