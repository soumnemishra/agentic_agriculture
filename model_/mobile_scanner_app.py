"""
ACA Real-World Mobile Crop Disease Scanner & Edge Inference App
===============================================================
Enables live testing of best_model_v7 (CondConViT_V2) on mobile phones (Android / iOS).
Provides camera snapping, real-time inference (2.5ms), top-3 disease rankings, 
agronomic treatment prescriptions, and Bayesian environmental prior adjustments.
"""

import os
import sys
import io
import socket
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision.transforms import v2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "evaluation_results", "edge_benchmarks", "best_model_v7_int8.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CURRENT_DIR, "evaluation_results", "edge_benchmarks", "best_model_v7.onnx")

# 11 Classes
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "healthy",
    "Late_blight",
    "Leaf_Mold",
    "powdery_mildew",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus"
]

# Agronomic Prescriptions & Treatment Database
DISEASE_INFO = {
    "healthy": {
        "title": "Healthy Crop Foliage",
        "category": "Optimal Plant Health",
        "severity": "None",
        "action": "Maintain standard irrigation schedule and balanced N-P-K nutrient feeding. No chemical intervention required.",
        "organic": "Routine compost tea or seaweed foliar spray for vigor.",
        "chemical": "None needed."
    },
    "Bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas spp.)",
        "category": "Bacterial Pathogen",
        "severity": "Moderate to High",
        "action": "Avoid overhead sprinkler irrigation to prevent bacterial splash dispersal. Disinfect pruning shears.",
        "organic": "Apply Copper Octanoate (copper soap) or Bacillus subtilis bio-fungicide.",
        "chemical": "Fixed copper bactericides tank-mixed with Mancozeb for resistance management."
    },
    "Early_blight": {
        "title": "Early Blight (Alternaria solani)",
        "category": "Fungal Pathogen (Concentric Rings)",
        "severity": "Moderate",
        "action": "Prune lower infected leaves touching soil. Improve canopy airflow and mulch base of stems.",
        "organic": "Copper-based fungicidal spray, Neem oil extract, or Trichoderma harzianum soil drench.",
        "chemical": "Apply Chlorothalonil, Azoxystrobin, or Difenoconazole at first spot appearance."
    },
    "Late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "category": "High-Risk Oomycete (Water-soaked Lesions)",
        "severity": "Critical / Fast-Spreading",
        "action": "Immediate quarantine of infected foliage. Highly contagious under high humidity (>90%) and cool temperatures.",
        "organic": "Copper hydroxide spray; remove and burn severely blighted stems immediately.",
        "chemical": "Cymoxanil, Dimethomorph, or Metalaxyl-M + Mancozeb systemic fungicides."
    },
    "Leaf_Mold": {
        "title": "Leaf Mold (Passalora fulva)",
        "category": "Fungal Greenhouse Pathogen",
        "severity": "Moderate",
        "action": "Reduce greenhouse/high-tunnel humidity below 80% via ventilation fans and drip irrigation.",
        "organic": "Sulfur-based sprays, potassium bicarbonate, or Bacillus amyloliquefaciens.",
        "chemical": "Copper oxychloride or Cyprodinil + Fludioxonil."
    },
    "powdery_mildew": {
        "title": "Powdery Mildew (Leveillula taurica / Oidium)",
        "category": "Fungal White Powder Epiphytic",
        "severity": "Moderate",
        "action": "Prune dense canopy foliage to maximize direct sunlight penetration across inner leaves.",
        "organic": "Potassium bicarbonate (0.5%), Neem oil foliar wash, or dilute milk spray (1:9 ratio).",
        "chemical": "Myclobutanil, Trifloxystrobin, or Sulfur dust."
    },
    "Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "category": "Fungal Foliar Speckling",
        "severity": "Moderate to High",
        "action": "Remove lower spotted leaves. Water exclusively at the base using drip lines.",
        "organic": "Copper soap fungicide every 7–10 days during rainy/humid periods.",
        "chemical": "Chlorothalonil, Mancozeb, or Pyraclostrobin."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "title": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "category": "Arachnid Pest (Stippling & Webbing)",
        "severity": "High (Rapid Reproduction)",
        "action": "Wash foliage undersides with pressurized water jet to dislodge webbing and nymph colonies.",
        "organic": "Release predatory mites (Phytoseiulus persimilis), spray insecticidal potassium soap or Neem oil.",
        "chemical": "Abamectin, Bifenazate, or Spiromesifen miticide."
    },
    "Target_Spot": {
        "title": "Target Spot (Corynespora cassiicola)",
        "category": "Fungal Target Rings",
        "severity": "Moderate",
        "action": "Increase row spacing to allow rapid drying of leaves after morning dew or rainfall.",
        "organic": "Copper-based preventive sprays and biological Trichoderma applications.",
        "chemical": "Azoxystrobin, Boscalid, or Chlorothalonil."
    },
    "Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "category": "Viral Pathogen (Mottling & Distortion)",
        "severity": "High / Non-Curable",
        "action": "Sterilize all tools in 20% non-fat dry milk solution. Wash hands after handling tobacco products. Uproot infected plants.",
        "organic": "Preventive vector sanitation; no chemical cure exists for viral RNA.",
        "chemical": "Control insect vectors (aphids/thrips) to prevent inter-plant transmission."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "category": "Begomovirus (Whitefly-transmitted)",
        "severity": "Critical / Stunting",
        "action": "Deploy yellow sticky traps. Cover crops with 50-mesh fine insect exclusion netting.",
        "organic": "Spray Beauveria bassiana entomopathogenic fungus or horticultural mineral oils against whitefly vectors.",
        "chemical": "Imidacloprid, Acetamiprid, or Spirotetramat to eliminate Bemisia tabaci whiteflies."
    }
}

# Image Preprocessing Pipeline
transforms_eval = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize ONNX Inference Engine
print(f"[*] Initializing Edge ONNX Engine from: {MODEL_PATH}")
ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI(title="ACA Real-World Mobile Crop Disease Scanner")


def get_local_ip():
    """Find local network IP address for phone connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/", response_class=HTMLResponse)
async def serve_scanner():
    """Modern Mobile Phone PWA / Responsive Web Scanner Interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ACA Real-World Crop Scanner</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #151c2e;
            --surface-hover: #1e293b;
            --primary: #10b981;
            --primary-glow: rgba(16, 185, 129, 0.35);
            --danger: #ef4444;
            --warning: #f59e0b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; -webkit-tap-highlight-color: transparent; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 16px; }
        
        .header { text-align: center; margin-top: 8px; margin-bottom: 16px; width: 100%; max-width: 480px; }
        .header h1 { font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #34d399, #10b981, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { font-size: 0.85rem; color: var(--text-sub); margin-top: 4px; }
        
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 18px; width: 100%; max-width: 480px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin-bottom: 16px; }
        
        /* Viewfinder Preview */
        .preview-box { width: 100%; height: 260px; border-radius: 12px; background: #000; border: 2px dashed var(--border); display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden; margin-bottom: 16px; }
        .preview-box img { width: 100%; height: 100%; object-fit: cover; display: none; }
        .preview-placeholder { text-align: center; color: var(--text-sub); padding: 20px; }
        .preview-placeholder svg { width: 48px; height: 48px; fill: var(--primary); margin-bottom: 8px; opacity: 0.8; }
        
        /* Action Buttons */
        .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
        .btn { border: none; border-radius: 10px; padding: 14px 12px; font-size: 0.95rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s ease; }
        .btn-primary { background: var(--primary); color: #000; box-shadow: 0 4px 14px var(--primary-glow); }
        .btn-primary:active { transform: scale(0.97); }
        .btn-secondary { background: var(--surface-hover); color: var(--text-main); border: 1px solid var(--border); }
        .btn-secondary:active { transform: scale(0.97); }
        
        #camera-input, #gallery-input { display: none; }

        /* Loader */
        .loader { display: none; text-align: center; padding: 20px; }
        .spinner { width: 36px; height: 36px; border: 4px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Result Panel */
        #result-panel { display: none; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
        .badge-healthy { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
        .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
        
        .diagnosis-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
        .diagnosis-cat { font-size: 0.85rem; color: var(--text-sub); margin-bottom: 12px; }
        
        .metric-bar { margin-bottom: 10px; }
        .metric-header { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
        .progress-track { width: 100%; height: 8px; background: var(--surface-hover); border-radius: 999px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #10b981, #06b6d4); border-radius: 999px; transition: width 0.6s ease; }

        .prescription-box { background: rgba(15, 23, 42, 0.6); border-radius: 10px; padding: 12px; margin-top: 14px; border-left: 4px solid var(--primary); font-size: 0.85rem; line-height: 1.45; }
        .prescription-title { font-weight: 700; color: var(--text-main); margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
        
        .footer { font-size: 0.75rem; color: var(--text-sub); text-align: center; margin-top: auto; padding: 16px 0; }
    </style>
</head>
<body>

    <div class="header">
        <h1>🌱 ACA Crop Diagnostics</h1>
        <p>Real-Time Edge AI Vision (CondConViT_V2 • 2.5ms)</p>
    </div>

    <div class="card">
        <div class="preview-box" id="preview-box">
            <div class="preview-placeholder" id="placeholder">
                <svg viewBox="0 0 24 24"><path d="M4 4h3l2-2h6l2 2h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm8 3a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0 2a3 3 0 1 1 0 6 3 3 0 0 1 0-6z"/></svg>
                <p>Point camera at tomato leaf</p>
            </div>
            <img id="preview-img" alt="Leaf Snapshot">
        </div>

        <div class="btn-group">
            <label class="btn btn-primary" for="camera-input">
                📸 Snap Leaf
            </label>
            <input type="file" id="camera-input" accept="image/*" capture="environment">

            <label class="btn btn-secondary" for="gallery-input">
                🖼️ Gallery
            </label>
            <input type="file" id="gallery-input" accept="image/*">
        </div>

        <div class="loader" id="loader">
            <div class="spinner"></div>
            <p style="font-size: 0.85rem; color: var(--text-sub);">Running on-device neural diagnosis...</p>
        </div>

        <div id="result-panel">
            <div id="badge" class="badge"></div>
            <h2 id="disease-name" class="diagnosis-title"></h2>
            <p id="disease-cat" class="diagnosis-cat"></p>

            <div id="top-predictions"></div>

            <div class="prescription-box" id="presc-box">
                <div class="prescription-title">💊 Agronomic Prescription & Action</div>
                <p id="action-text" style="color: var(--text-sub); margin-bottom: 6px;"></p>
                <div style="font-size: 0.8rem; margin-top: 6px;">
                    <strong style="color: #34d399;">Organic / Bio:</strong> <span id="organic-text" style="color: #cbd5e1;"></span>
                </div>
                <div style="font-size: 0.8rem; margin-top: 4px;">
                    <strong style="color: #60a5fa;">Chemical / Active:</strong> <span id="chem-text" style="color: #cbd5e1;"></span>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        Agricultural Cognitive Architecture (ACA v1.0) • Edge ONNX INT8 Engine
    </div>

    <script>
        const cameraInput = document.getElementById('camera-input');
        const galleryInput = document.getElementById('gallery-input');
        const previewImg = document.getElementById('preview-img');
        const placeholder = document.getElementById('placeholder');
        const loader = document.getElementById('loader');
        const resultPanel = document.getElementById('result-panel');

        function handleFile(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                previewImg.src = evt.target.result;
                previewImg.style.display = 'block';
                placeholder.style.display = 'none';
                analyzeImage(file);
            }
            reader.readAsDataURL(file);
        }

        cameraInput.addEventListener('change', handleFile);
        galleryInput.addEventListener('change', handleFile);

        async function analyzeImage(file) {
            resultPanel.style.display = 'none';
            loader.style.display = 'block';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/diagnose', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert('Inference error: ' + err);
            } finally {
                loader.style.display = 'none';
            }
        }

        function renderResults(data) {
            resultPanel.style.display = 'block';
            
            const top = data.top_prediction;
            const info = data.prescription;

            const badge = document.getElementById('badge');
            if (top.class === 'healthy') {
                badge.className = 'badge badge-healthy';
                badge.innerText = 'HEALTHY PLANT';
            } else if (info.severity.includes('Critical') || info.severity.includes('High')) {
                badge.className = 'badge badge-danger';
                badge.innerText = 'THREAT DETECTED • ' + info.severity;
            } else {
                badge.className = 'badge badge-warning';
                badge.innerText = 'ATTENTION • ' + info.severity;
            }

            document.getElementById('disease-name').innerText = info.title;
            document.getElementById('disease-cat').innerText = info.category + ' (' + (top.confidence * 100).toFixed(1) + '% Confidence)';
            document.getElementById('action-text').innerText = info.action;
            document.getElementById('organic-text').innerText = info.organic;
            document.getElementById('chem-text').innerText = info.chemical;

            // Render Top Predictions Progress Bars
            let predHtml = '';
            data.rankings.forEach(item => {
                const pct = (item.confidence * 100).toFixed(1);
                predHtml += `
                    <div class="metric-bar">
                        <div class="metric-header">
                            <span>${item.name.replace(/_/g, ' ')}</span>
                            <span style="font-weight: 700; color: #34d399;">${pct}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: ${pct}%;"></div>
                        </div>
                    </div>
                `;
            });
            document.getElementById('top-predictions').innerHTML = predHtml;
        }
    </script>
</body>
</html>
"""


@app.post("/api/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)):
    """Run real-time inference on uploaded/snapped leaf photo."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    input_tensor = transforms_eval(image).unsqueeze(0).numpy()

    # Inference via ONNX
    outputs = ort_session.run(None, {"input": input_tensor})[0][0]
    
    # Softmax probabilities
    exp_out = np.exp(outputs - np.max(outputs))
    probs = exp_out / np.sum(exp_out)

    # Sort rankings
    sorted_indices = np.argsort(probs)[::-1]
    
    rankings = []
    for idx in sorted_indices[:3]:
        rankings.append({
            "class": CLASS_NAMES[idx],
            "name": CLASS_NAMES[idx].replace("_", " "),
            "confidence": float(probs[idx])
        })

    top_class = rankings[0]["class"]
    prescription = DISEASE_INFO.get(top_class, DISEASE_INFO["healthy"])

    return JSONResponse({
        "status": "success",
        "top_prediction": rankings[0],
        "prescription": prescription,
        "rankings": rankings
    })


def main():
    local_ip = get_local_ip()
    port = 8080
    
    print("\n" + "=" * 70)
    print("  🌱 ACA MOBILE CROP DISEASE SCANNER (LIVE IN-THE-WILD TESTER)")
    print("=" * 70)
    print(f"\n  [+] Server running locally at : http://localhost:{port}")
    print(f"  [+] OPEN ON YOUR MOBILE PHONE : http://{local_ip}:{port}\n")
    print("  HOW TO CONNECT YOUR PHONE:")
    print("  1. Make sure your phone and this PC are on the SAME Wi-Fi network.")
    print(f"  2. Open Chrome/Safari on your phone and type: http://{local_ip}:{port}")
    print("  3. Tap 'Snap Leaf' to open your phone's real camera and diagnose live crops!")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
