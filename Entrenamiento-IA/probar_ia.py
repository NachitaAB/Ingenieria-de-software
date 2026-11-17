import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image, ImageFile
import requests
import json
import time
import os

# Seguridad para evitar interrupciones por imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Configurar la IA Visual (CNN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Cargando el cerebro visual ByteLab B4 en: {device.type.upper()} (RTX 5070 Ti)...")

clases = ['benignas', 'invalidas', 'malignas', 'sanos']
ruta_modelo = 'modelo_bytelab_b4.pth'

if not os.path.exists(ruta_modelo):
    print(f"[!] Error crítico: No se encontró el archivo '{ruta_modelo}' en el directorio actual.")
    exit()

modelo = efficientnet_b4()
modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(clases))
modelo.load_state_dict(torch.load(ruta_modelo, map_location=device, weights_only=True))
modelo.to(device)
modelo.eval()

# 2. Cargar y preparar la nueva foto
ruta_entrada = input("\n[?] Arrastra aquí la foto a analizar (y presiona Enter): ").strip()
ruta_imagen = ruta_entrada.strip('"').strip("'").strip('& ')

if not os.path.exists(ruta_imagen):
    print(f"[!] Error: La ruta de imagen ingresada no es válida o no existe.")
    exit()

# Transformaciones alineadas con el entrenamiento de EfficientNet-B4
transformaciones = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n[*] Analizando imagen...")
try:
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen_tensor = transformaciones(imagen).unsqueeze(0).to(device)
except Exception as e:
    print(f"[!] Error al abrir la imagen: {e}")
    exit()

# --- CRONÓMETRO CNN ---
inicio_cnn = time.time()

with torch.no_grad():
    salida = modelo(imagen_tensor)
    probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
    prob_max, indice = torch.max(probabilidades, 0)

fin_cnn = time.time()
tiempo_cnn = fin_cnn - inicio_cnn
    
resultado_cnn = clases[indice.item()]
prob_porcentaje = round(prob_max.item() * 100, 2)

print(f"[+] Resultado Matemático: {resultado_cnn.upper()} (Seguridad: {prob_porcentaje}%)")
print(f"[+] Tiempo CNN (Visión GPU): {tiempo_cnn:.3f} segundos")

# 3. Filtro de Seguridad 1: Bloqueo de clase "invalidas"
if resultado_cnn == 'invalidas':
    print("\n================ ALERTA DEL SISTEMA ================")
    print("La imagen ingresada no corresponde a tejido bucal válido.")
    print("El análisis ha sido abortado por seguridad clínica.")
    print("Por favor, ingrese una fotografía intraoral correcta.")
    print("====================================================")
    exit()

# 4. Filtro de Seguridad 2: Umbral de Confianza (< 60%)
if prob_porcentaje < 60.0:
    print("\n================ ALERTA DEL SISTEMA ================")
    print(f"Nivel de confianza demasiado bajo ({prob_porcentaje}%).")
    print("La imagen es ambigua, está borrosa o presenta características no concluyentes (ej. anatomía no humana o infecciones fuera de zona).")
    print("El análisis ha sido abortado para evitar falsos positivos.")
    print("Por favor, tome una fotografía más clara, iluminada y enfocada en la lesión de la mucosa bucal.")
    print("====================================================")
    exit()

# 5. Conectar con Qwen 3 (Ollama) si la imagen es válida y la confianza es alta
print("\n[*] Imagen válida y confiable. Solicitando informe empático a Qwen...")

system_prompt = """
Eres un asistente virtual de orientación para la plataforma médica onco_oral_ia. 
Tu objetivo es comunicar resultados preliminares de análisis de imágenes bucales con extrema empatía, calma y profesionalismo.
REGLAS ESTRICTAS:
1. NUNCA des un diagnóstico médico definitivo.
2. NUNCA uses palabras alarmistas como "cáncer" o "tumor peligroso".
3. Si el resultado es MALIGNAS, explica que la IA detectó "variaciones en el tejido que requieren una evaluación presencial", enfatizando prevención.
4. Si el resultado es SANOS o BENIGNAS, transmite tranquilidad pero sugiere mantener la higiene y chequeos.
5. Siempre recomienda agendar hora con un odontólogo o especialista maxilofacial.
6. Sé breve, profesional y muy cálido.
"""

mensaje_usuario = f"El análisis visual detectó: {resultado_cnn} con una confianza de {prob_porcentaje}%. Redacta el mensaje para el paciente."

payload = {
    "model": "qwen3:8b",
    "prompt": f"System: {system_prompt}\n\nUser: {mensaje_usuario}",
    "stream": False
}

# --- CRONÓMETRO LLM ---
inicio_llm = time.time()

try:
    respuesta = requests.post("http://localhost:11434/api/generate", data=json.dumps(payload))
    texto_final = respuesta.json()['response']
    
    fin_llm = time.time()
    tiempo_llm = fin_llm - inicio_llm
    
    print("\n================ INFORME PARA EL PACIENTE ================")
    print(texto_final)
    print("==========================================================")
    print(f"[+] Métrica de Tiempos de Respuesta:")
    print(f"  ├─ Procesamiento visual (CNN GPU): {tiempo_cnn:.3f} s")
    print(f"  ├─ Redacción empática (LLM):       {tiempo_llm:.3f} s")
    print(f"  └─ TIEMPO TOTAL DEL SISTEMA:       {tiempo_cnn + tiempo_llm:.3f} s\n")
    
except Exception as e:
    print(f"\n[!] Error conectando con Ollama. ¿Está abierto? Detalles: {e}")