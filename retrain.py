from ultralytics import YOLO
from pathlib import Path

# Configurações (ajustadas para treino com nova classe)
BASE_DIR = Path("C:/Users/Julliano Rodrigues/Desktop/IA")
DATA_YAML = BASE_DIR / "dataset/data.yaml"  
MODEL_PATH = BASE_DIR / "runs/detect/epi_treinamento/weights/best.pt"

# Verificações críticas
if not DATA_YAML.exists():
    raise FileNotFoundError(f"❌ data.yaml não encontrado em: {DATA_YAML}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Modelo pré-treinado não encontrado em: {MODEL_PATH}")

model = YOLO(MODEL_PATH)  # Carrega o modelo existente

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=8,  
    device='cpu',
    project=str(BASE_DIR / "runs/detect"),
    exist_ok=True,
    workers=0,  # Necessário para Windows
    patience=10,  # Para early stopping se a métrica não melhorar
    lr0=0.001,  # Taxa de aprendizado inicial 
    pretrained=True  
)

print("✅ Treinamento concluído! Verifique os resultados em:")
