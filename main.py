import os
import sys
import subprocess


def run_step(label, command):
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}\n")
    result = subprocess.run(command, check=True)
    return result


def processed_data_exists():
    for class_name in ("goku", "vegeta"):
        folder = os.path.join("data", "processed", class_name)
        if not os.path.exists(folder) or not os.listdir(folder):
            return False
    return True


def model_exists():
    return os.path.exists(os.path.join("models", "goku_e_vegeta.pth"))


def main():
    print("\n  CNN — Classificador Goku vs. Vegeta")
    print("  ResNet50 com Transfer Learning\n")

    # Etapa 1: Pré-processamento
    if processed_data_exists():
        print("[OK] Imagens processadas já existem — pulando pré-processamento.")
    else:
        run_step("ETAPA 1/2 — Pré-processamento das imagens",
                 [sys.executable, "src/preprocess.py"])

    # Etapa 2: Treinamento
    run_step("ETAPA 2/2 — Treinamento da CNN",
             [sys.executable, "src/train.py"])

    print("\n" + "=" * 55)
    print("  Pipeline concluído com sucesso!")
    print(f"  Modelo salvo em: models/goku_e_vegeta.pth")
    print("=" * 55)
    print("\n  Para classificação em tempo real via webcam:")
    print("    python src/predict.py\n")


if __name__ == "__main__":
    main()
