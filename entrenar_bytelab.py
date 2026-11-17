import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import multiprocessing
from tqdm import tqdm
from PIL import ImageFile

# Esta es la línea mágica que evita que el entrenamiento colapse por fotos corruptas o a medias
ImageFile.LOAD_TRUNCATED_IMAGES = True

def entrenar_modelo():
    # 1. Configuración de Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inicializando entrenamiento ByteLab en: {device.upper()} (RTX 5070 Ti)")

    ruta_dataset = r"C:\onco_oral_ia\dataset_final"
    batch_size = 16 
    epocas = 10

    # 2. Transformaciones y Data Augmentation
    transformaciones = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(ruta_dataset):
        print(f"[!] Error: No se encontró la ruta {ruta_dataset}")
        return

    # 3. Carga y División de Datos
    dataset_completo = datasets.ImageFolder(root=ruta_dataset, transform=transformaciones)
    print(f"[*] Clases detectadas: {dataset_completo.classes}")

    train_size = int(0.8 * len(dataset_completo))
    val_size = len(dataset_completo) - train_size
    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"[*] Imágenes para entrenar: {train_size}")
    print(f"[*] Imágenes para validar (testeo real): {val_size}")

    # 4. Arquitectura del Modelo
    print("[*] Descargando/Cargando pesos preentrenados de EfficientNet-B4...")
    modelo = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    num_ftrs = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(num_ftrs, len(dataset_completo.classes))
    modelo = modelo.to(device)

    # 5. Optimizador y Función de Pérdida
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=1e-4)

    # 6. Bucle de Entrenamiento
    print("\n[+] ¡Comenzando el entrenamiento!")
    for epoca in range(epocas):
        print(f"\n--- Época {epoca+1}/{epocas} ---")
        
        # ENTRENAMIENTO
        modelo.train()
        train_loss = 0.0
        train_corrects = 0
        
        # Barra de progreso para el entrenamiento
        train_bar = tqdm(train_loader, desc="Entrenando", unit="batch", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizador.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        train_loss = train_loss / train_size
        train_acc = train_corrects.double() / train_size
        
        # VALIDACIÓN
        modelo.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # Barra de progreso para la validación
        val_bar = tqdm(val_loader, desc="Validando", unit="batch", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / val_size
        val_acc = val_corrects.double() / val_size
        
        print(f" -> ENTRENAMIENTO | Pérdida: {train_loss:.4f} | Precisión: {train_acc:.4f}")
        print(f" -> VALIDACIÓN    | Pérdida: {val_loss:.4f} | Precisión: {val_acc:.4f}")

    # 7. Guardar Modelo
    ruta_modelo = "modelo_bytelab_b4.pth"
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f"\n[+] ¡Entrenamiento exitoso! El modelo se guardó como '{ruta_modelo}'.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    entrenar_modelo()