from PIL import Image
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
NEW_SIZE = (224, 224)

for class_name in os.listdir(RAW_DIR):
    input_folder = os.path.join(RAW_DIR, class_name)
    output_folder = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    img_resized = img.resize(NEW_SIZE, Image.Resampling.LANCZOS)
                    img_resized.save(os.path.join(output_folder, filename))
                    print(f"Salvo: {output_folder}/{filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

print("Pré-processamento concluído!")
