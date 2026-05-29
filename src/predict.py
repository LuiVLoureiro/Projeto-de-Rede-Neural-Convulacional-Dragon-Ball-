import torch
import cv2
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/goku_e_vegeta.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


def load_model(model_path):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class


def classify_frame(frame, model, transform, idx_to_class):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_transformed = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_transformed).view(-1)
        prob = torch.sigmoid(output).item()

    if prob >= 0.80:
        label = "Vegeta"
    elif prob <= 0.10:
        label = "Goku"
    else:
        label = "Classe Desconhecida"

    return label, prob


model, idx_to_class = load_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame da webcam.")
        break

    label, prob = classify_frame(frame, model, transform, idx_to_class)

    color = (0, 255, 0) if label != "Classe Desconhecida" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Classificação em Tempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
