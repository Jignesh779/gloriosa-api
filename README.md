# Gloriosa Seed Classifier API

**AI-Powered Smart Gloriosa Seed Monitoring System**  
EfficientNet-B0 · 97.5% Accuracy · FastAPI + ONNX Runtime · Final Year ECE Project

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Keep-alive ping |
| `/predict` | POST | Image file upload |
| `/predict_base64` | POST | Base64 JSON (for ESP32-CAM) |

## ESP32-CAM Usage

```
POST https://<your-app>.onrender.com/predict
Content-Type: multipart/form-data
Field: file = <image>
```
