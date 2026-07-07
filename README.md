# Farmer Leaf Doctor (உழவர் இலை மருத்துவர்)

Farmer Leaf Doctor is a full-stack, mobile-first web application designed for farmers to instantly diagnose diseases in tomato and potato leaves. 

It provides:
- **Bilingual interface** (English & Tamil) togglable with a single tap.
- **Mobile-first design** with high-contrast, large tap targets, and clean icons.
- **Direct Camera integration** for snapping leaf pictures in the field, alongside gallery/drag-and-drop uploads.
- **Treatment advice** (Causes, Symptoms, and Actionable Remedies) translated fully into Tamil and English.
- **Robust TFLite inference** with quantized/float support and an automatic fallback **Mock Mode** for developers.

---

## Folder Structure

```text
farmer-leaf-doctor/
├── backend/
│   ├── main.py              # FastAPI server (runs inference & serves static files)
│   ├── remedies.json        # Bilingual remedies database for all 7 classes
│   ├── requirements.txt     # Python dependencies
│   ├── test_predict.py      # Standalone backend verification tests
│   └── model/
│       └── [Place leaf_disease_model.tflite here]
├── frontend/
│   ├── index.html           # Main bilingual user interface
│   ├── style.css            # Custom organic earth styling
│   ├── script.js            # Frontend logic & bilingual toggle coordinator
│   └── assets/
│       ├── good_leaf.png    # Example of a correct leaf photo
│       └── bad_leaf.png     # Example of what to avoid (blurry, far)
└── README.md
```

---

## 1. Setup & Installation

### Step 1: Install Python Dependencies
Open PowerShell or your terminal and run:
```powershell
cd backend
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements.txt
```
*(Note: If you have a working SSL connection, you can omit the `--trusted-host` flags).*

### Step 2: Install TensorFlow or TFLite Runtime
The backend dynamically loads the model using `tensorflow` or the lighter `tflite-runtime`. You can install either:
```powershell
# Option A: Install full TensorFlow (heavier, standard)
pip install tensorflow

# Option B: Install tflite-runtime (much lighter, faster install)
pip install tflite-runtime
```

---

## 2. Upload the Model File

Please place your `leaf_disease_model.tflite` file inside the `backend/model/` folder:
- **Target File Path**: `backend/model/leaf_disease_model.tflite`

### Mock Mode Fallback (No Model Needed to Test)
If the model file is not uploaded yet, the backend automatically runs in **Mock Mode**! 
- Uploading a image file with `"tomato_early"` in its name will mock a **Tomato Early Blight** positive result.
- Uploading a file with `"potato_late"` will mock **Potato Late Blight**.
- Uploading a file with `"bad"`, `"low"`, or `"blur"` in its name will mock a **Low Confidence Match (< 60%)** error card, allowing you to test the retake warning card in the UI.
- All other uploads will select a random disease with a realistic high-confidence score.

---

## 3. Running the Application

To start the local FastAPI web server:
```powershell
# From the backend directory
python -m uvicorn main:app --reload --port 8000
```

Once started, open your web browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

The index page will load directly. Scale down your browser window or open it on a mobile device to see the premium mobile-responsive layout.

---

## 4. Verification

To verify that the backend is fully operational (checks file loading, preprocessing, and remedy database outputs):
```powershell
python backend/test_predict.py
```
This runs standalone assertions in-process and prints the result JSON.
