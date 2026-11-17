from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image, ImageFile
import requests
import json
import os
import time

# Seguridad para evitar interrupciones por imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app) # Permite la comunicación segura con la web de Cristian

# 1. Configurar y cargar la IA Visual (CNN) en la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Inicializando servidor puente ByteLab en: {device.type.upper()}...")

clases = ['benignas', 'invalidas', 'malignas', 'sanos']
ruta_modelo = 'modelo_bytelab_b4.pth'

if not os.path.exists(ruta_modelo):
    print(f"[!] Error crítico: No se encontró '{ruta_modelo}'. Asegúrate de tenerlo en la carpeta.")
    exit()

modelo = efficientnet_b4()
modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(clases))
modelo.load_state_dict(torch.load(ruta_modelo, map_location=device, weights_only=True))
modelo.to(device)
modelo.eval()

# Transformaciones de EfficientNet-B4
transformaciones = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/analizar', methods=['POST'])
def analizar_imagen():
    inicio_total = time.time()
    print("\n--- NUEVA PETICIÓN RECIBIDA ---")
    
    # Validar si viene algún archivo en el FormData
    if not request.files:
        return jsonify({"error": "No se recibió ningún archivo. Asegúrate de enviar un FormData."}), 400
    
    # Tomar el primer archivo sin importar si se llamó 'imagen', 'file' o 'foto'
    llave_archivo = list(request.files.keys())[0]
    file = request.files[llave_archivo]
    
    if file.filename == '':
        return jsonify({"error": "El archivo adjunto está vacío."}), 400
    
    try:
        # Procesar imagen con PyTorch y GPU
        imagen = Image.open(file.stream).convert("RGB")
        imagen_tensor = transformaciones(imagen).unsqueeze(0).to(device)
        
        with torch.no_grad():
            salida = modelo(imagen_tensor)
            probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
            prob_max, indice = torch.max(probabilidades, 0)
            
        resultado_cnn = clases[indice.item()].upper()
        prob_porcentaje = round(prob_max.item() * 100, 2)
        
        # Espiar el resultado matemático en la consola local
        print(f">>> DIAGNÓSTICO IA: {resultado_cnn} | CONFIANZA: {prob_porcentaje}% <<<")
        
        # 1. Filtro de Seguridad: Clase INVÁLIDAS
        if resultado_cnn == 'INVALIDAS':
            tiempo_segundos = round(time.time() - inicio_total, 3)
            return jsonify({
                "estado": "rechazado",
                "diagnostico_cnn": resultado_cnn,
                "confianza": prob_porcentaje,
                "informe_paciente": "Error clínico: La imagen ingresada no corresponde a tejido bucal válido o presenta demasiado ruido. Por favor, tome una nueva fotografía.",
                "tiempo_respuesta_segundos": tiempo_segundos
            })
            
        # 2. Filtro de Seguridad: Confianza menor al 60%
        elif prob_porcentaje < 60.0:
            tiempo_segundos = round(time.time() - inicio_total, 3)
            return jsonify({
                "estado": "rechazado",
                "diagnostico_cnn": resultado_cnn,
                "confianza": prob_porcentaje,
                "informe_paciente": f"Filtro de seguridad activado: La IA detectó características de la clase {resultado_cnn}, pero con un nivel de confianza muy bajo ({prob_porcentaje}%). Por prevención clínica, se requiere una imagen más clara y enfocada.",
                "tiempo_respuesta_segundos": tiempo_segundos
            })
        
        # 3. Consulta al LLM (Qwen 3) para resultados válidos (SANOS, BENIGNAS, MALIGNAS)
        system_prompt = """
        Eres un asistente virtual de orientación para la plataforma médica ByteLab. 
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
        
        resp_llm = requests.post("http://localhost:11434/api/generate", data=json.dumps(payload), timeout=30)
        texto_llm = resp_llm.json().get('response', 'Informe generado con éxito por ByteLab.')
        
        tiempo_segundos = round(time.time() - inicio_total, 3)
        
        # 4. Respuesta exitosa estructurada exactamente como la pide la base de datos de Cristian
        return jsonify({
            "estado": "exito",
            "diagnostico_cnn": resultado_cnn,
            "confianza": prob_porcentaje,
            "informe_paciente": texto_llm,
            "tiempo_respuesta_segundos": tiempo_segundos
        })
        
    except Exception as e:
        print(f"[!] Error procesando la imagen: {str(e)}")
        return jsonify({"error": f"Error interno procesando la imagen: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)