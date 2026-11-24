# AIRI – AI-Based Safety Helmet Detection

An AI-powered computer vision project developed during the PITB AI/ML internship task.

## Overview
This project uses YOLOv8 to detect people, safety helmets, and individuals without helmets in construction environments.

## Classes
- Person
- Helmet
- No Helmet

## Features
- Image-based helmet detection
- Video-based helmet detection
- YOLOv8 object detection
- Annotated detection results
- Streamlit web interface

## Technologies
- Python
- YOLOv8
- OpenCV
- Streamlit
- PyTorch

## Project Structure
```text
AIRI_PITB_CV_Task1/
├── app.py
├── best.pt
├── requirements.txt
├── audit_dataset.py
├── prepare_dataset.py
├── preview_dataset.py
└── cv_project/
    ├── project_brief.md
    └── src/
```

## Installation

```bash
git clone https://github.com/Ali-Husnain-AI-ML/AIRI-Safety-Helmet-Detection.git
cd AIRI-Safety-Helmet-Detection
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Model
The project uses a trained YOLOv8 model (`best.pt`) to detect safety-related objects.

## Note
This project is intended for educational and demonstration purposes. Detection performance may vary depending on image quality, lighting, and environmental conditions.