from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import io

# ===============================
# 1️⃣ Khai báo model giống như khi train
# ===============================
class MaxViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


# ===============================
# 2️⃣ Load model và chuẩn bị transform
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaxViTModel(num_classes=2).to(device)
model.load_state_dict(torch.load("maxvit_trained_with_acer.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Gắn nhãn nếu cần
CLASS_NAMES = ["Fake", "Real"]


app = FastAPI(title="MaxViT Anti-Spoofing API", version="1.0")


@app.get("/")
def home():
    return {"message": "Welcome to MaxViT Anti-Spoofing API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Tiền xử lý
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        result = {
            "filename": file.filename,
            "predicted_label": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4)
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
