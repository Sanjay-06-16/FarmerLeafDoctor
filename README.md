# Farmer Leaf Doctor (உழவர் இலை மருத்துவர்)

Farmer Leaf Doctor is a full-stack, mobile-first web application designed for farmers to instantly diagnose diseases in tomato and potato leaves using AI.

🌐 **Live Demo:** https://farmerleafdoctor-2.onrender.com  
*(Hosted on free tier — may take 30-60 seconds to wake up on first load)*

## Features

- **Bilingual interface** (English & Tamil) togglable with a single tap
- **Mobile-first design** with high-contrast, large tap targets, and clean icons
- **Direct camera integration** for snapping leaf pictures in the field, alongside gallery/drag-and-drop uploads
- **Treatment advice** (causes, symptoms, and actionable remedies) translated fully into Tamil and English
- **Robust TFLite inference** with a confidence threshold safeguard for uncertain predictions

## Tech Stack

- **Model:** MobileNetV2 (transfer learning), trained on the PlantVillage dataset using TensorFlow/Keras in Google Colab
- **Backend:** FastAPI + TensorFlow Lite for inference
- **Frontend:** HTML, CSS, JavaScript — bilingual, mobile-responsive
- **Deployment:** Render

## Supported Classes (7)
Tomato: Healthy, Late Blight, Early Blight, Bacterial Spot  
Potato: Healthy, Late Blight, Early Blight

## Folder Structure
...
farmer-leaf-doctor/
├── backend/
│   ├── main.py              # FastAPI server (runs inference & serves static files)
│   ├── remedies.json        # Bilingual remedies database for all 7 classes
│   ├── requirements.txt     # Python dependencies
│   ├── test_predict.py      # Standalone backend verification tests
│   └── model/
│       └── leaf_disease_model.tflite
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
└── README.md
...
## Setup & Run Locally

```bash
cd backend
pip install -r requirements.txt
pip install tensorflow
python -m uvicorn main:app --reload --port 8000
```
Open http://localhost:8000

## Verification
```bash
python backend/test_predict.py
```

## Challenges Solved

- **Class imbalance:** The training dataset had significant imbalance (e.g., Tomato Late Blight had ~12x more images than Potato Healthy), which biased the model toward over-predicting the majority class. Fixed using `sklearn.utils.class_weight.compute_class_weight` during training.
- **Preprocessing mismatch:** Training normalized pixels to [-1, 1] (MobileNetV2 standard) but the initial backend implementation used [0, 1] scaling, causing incorrect predictions in production despite a working model. Traced and fixed by aligning backend preprocessing with training exactly.
- **Windows/TFLite Unicode path bug:** TFLite's interpreter failed to load the model when the project directory path contained non-ASCII characters (a Chinese-localized OneDrive folder name). Resolved by relocating the project to an ASCII-only path.
- **Overfitting during fine-tuning:** An attempt to fine-tune the pretrained base model caused severe overfitting (99% train accuracy vs. ~10-50% validation accuracy). Reverted to the stable base model and prioritized class balancing over fine-tuning given the dataset size.

## Limitations
- Trained on the PlantVillage dataset (lab-conditioned, plain-background images); accuracy may be lower on outdoor/field photos with cluttered backgrounds
- Free-tier hosting has a cold-start delay after inactivity
