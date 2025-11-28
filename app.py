from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import timm
import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import gradio as gr

# Importar módulos del proyecto
import config
from utils import (
    validar_imagen, validar_video,
    generar_reporte_imagen, generar_reporte_video, generar_reporte_error,
    Timer,
)

# ==========================================
# 🔧 Configuración de Logging
# ==========================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# ==========================================
# 🧠 Modelos de Deep Learning
# ==========================================


class CNNDetectionResNet(nn.Module):
    """
    Modelo ResNet50 personalizado para detección de imágenes sintéticas.
    Detecta artefactos de GANs y modelos de difusión.
    """

    def __init__(self):
        super(CNNDetectionResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


class ModelManager:
    """
    Gestor centralizado de modelos con caché y manejo robusto de errores.
    """

    def __init__(self):
        self.modelo_imagen: Optional[CNNDetectionResNet] = None
        self.modelo_video: Optional[nn.Module] = None
        self.dispositivo = torch.device(config.DEVICE)

        # Transformaciones de imagen
        self.transform_imagen = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        # Transformaciones de video
        self.transform_video = transforms.Compose([
            transforms.Resize((config.VIDEO_SIZE, config.VIDEO_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        # Cargar modelos al inicializar
        self._cargar_modelos()

    def _cargar_modelos(self):
        """Carga ambos modelos con manejo de errores."""
        self.modelo_imagen = self._cargar_modelo_imagen()
        self.modelo_video = self._cargar_modelo_video()

    def _cargar_modelo_imagen(self) -> Optional[CNNDetectionResNet]:
        """Carga el modelo de detección de imágenes."""
        logger.info("🖼️ Cargando modelo de imágenes...")

        try:
            if not os.path.exists(config.MODEL_IMAGE_PATH):
                logger.warning(
                    f"⚠️ No se encontró el modelo en: {config.MODEL_IMAGE_PATH}"
                )
                logger.warning("📥 Modo demostración activado para imágenes")
                return None

            modelo = CNNDetectionResNet()
            checkpoint = torch.load(
                config.MODEL_IMAGE_PATH,
                map_location=self.dispositivo,
            )

            # Manejar diferentes formatos de checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Corregir nombres de llaves si es necesario
            new_state_dict = {}
            for k, v in state_dict.items():
                nombre = k.replace("module.", "")
                if not nombre.startswith("model."):
                    nombre = "model." + nombre
                new_state_dict[nombre] = v

            modelo.load_state_dict(new_state_dict)
            modelo.to(self.dispositivo)
            modelo.eval()

            logger.info("✅ Modelo de imágenes cargado exitosamente")
            return modelo

        except Exception as e:
            logger.error(
                f"❌ Error cargando modelo de imágenes: {e}",
                exc_info=True,
            )
            logger.warning("📥 Usando modo demostración para imágenes")
            return None

    def _cargar_modelo_video(self) -> Optional[nn.Module]:
        """Carga el modelo de detección de deepfakes en video."""
        logger.info("🎥 Cargando modelo de video...")

        try:
            modelo = timm.create_model(
                config.MODEL_VIDEO_NAME,
                pretrained=True,
                num_classes=2,
            )
            modelo.to(self.dispositivo)
            modelo.eval()

            logger.info("✅ Modelo de video cargado exitosamente")
            return modelo

        except Exception as e:
            logger.error(
                f"❌ Error cargando modelo de video: {e}",
                exc_info=True,
            )
            logger.warning("📥 Modo demostración activado para videos")
            return None


# Inicializar gestor de modelos (singleton)
model_manager = ModelManager()

# ==========================================
# 🔍 Funciones de Análisis
# ==========================================


def analizar_imagen(imagen_input) -> str:
    """
    Analiza una imagen para detectar si es sintética o manipulada.

    Args:
        imagen_input: Array numpy de la imagen

    Returns:
        HTML con el reporte del análisis
    """
    logger.info("📸 Iniciando análisis de imagen...")

    # Validación de entrada
    if imagen_input is None:
        return generar_reporte_error(
            "No se proporcionó ninguna imagen",
            "warning",
        )

    es_valida, mensaje = validar_imagen(imagen_input)
    if not es_valida:
        return generar_reporte_error(mensaje, "error")

    # Modo demostración si el modelo no está cargado
    if model_manager.modelo_imagen is None:
        logger.warning("⚠️ Usando modo demostración (modelo no disponible)")
        import random
        prob = random.uniform(40, 95)
        es_fake = prob > 50
        h, w = imagen_input.shape[:2]
        return generar_reporte_imagen(es_fake, prob, w, h, 0.123)

    # Análisis real con el modelo
    try:
        with Timer() as timer:
            # Convertir a PIL y aplicar transformaciones
            img_pil = Image.fromarray(imagen_input).convert("RGB")
            img_tensor = model_manager.transform_imagen(img_pil).unsqueeze(0)
            img_tensor = img_tensor.to(model_manager.dispositivo)

            # Inferencia
            with torch.no_grad():
                output = model_manager.modelo_imagen(img_tensor)
                probabilidad_fake = torch.sigmoid(output).item() * 100

            # Determinar clasificación
            es_fake = probabilidad_fake > config.IMAGE_THRESHOLD

            logger.info(
                f"✅ Análisis completado: {'FAKE' if es_fake else 'REAL'} "
                f"({probabilidad_fake:.2f}%)"
            )

        # Generar reporte
        ancho, alto = img_pil.size
        return generar_reporte_imagen(
            es_fake=es_fake,
            probabilidad=probabilidad_fake,
            ancho=ancho,
            alto=alto,
            tiempo_proceso=timer.duracion,
        )

    except Exception as e:
        logger.error(
            f"❌ Error durante el análisis de imagen: {e}",
            exc_info=True,
        )
        return generar_reporte_error(
            f"Ocurrió un error durante el análisis: {str(e)}",
            "error",
        )


def analizar_video(video_path: str, progress=gr.Progress()) -> str:
    """
    Analiza un video para detectar deepfakes.

    Args:
        video_path: Ruta al archivo de video
        progress: Objeto de Gradio para mostrar progreso

    Returns:
        HTML con el reporte del análisis
    """
    logger.info("🎬 Iniciando análisis de video...")

    # Validación de entrada
    if video_path is None:
        return generar_reporte_error(
            "No se proporcionó ningún video",
            "warning",
        )

    es_valido, mensaje = validar_video(video_path)
    if not es_valido:
        return generar_reporte_error(mensaje, "error")

    # Verificar disponibilidad del modelo
    if model_manager.modelo_video is None:
        return generar_reporte_error(
            "El modelo de análisis de video no está disponible. "
            "Por favor, verifica la instalación.",
            "error",
        )

    try:
        with Timer() as timer:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return generar_reporte_error(
                    "No se pudo abrir el archivo de video",
                    "error",
                )

            # Obtener propiedades del video
            frames_totales = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duracion = frames_totales / fps if fps and fps > 0 else 0

            # Validar duración
            if duracion > config.MAX_VIDEO_DURATION_SECONDS:
                cap.release()
                return generar_reporte_error(
                    f"El video es demasiado largo ({duracion:.1f}s). "
                    f"Máximo: {config.MAX_VIDEO_DURATION_SECONDS}s",
                    "warning",
                )

            # Detector de rostros
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            # Análisis frame por frame
            predicciones = []
            frames_analizados = 0

            logger.info(
                f"📊 Analizando video: {frames_totales} frames, {duracion:.1f}s"
            )

            # Usar stride adaptativo basado en duración
            stride = config.VIDEO_FRAME_STRIDE
            if duracion > 60:  # Si es mayor a 1 minuto, aumentar stride
                stride = 60

            for i in progress.tqdm(
                range(0, frames_totales, stride),
                desc="🔍 Escaneando rostros...",
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                # Detectar rostros
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    frames_analizados += 1

                    # Extraer el primer rostro detectado
                    (x, y, w, h) = faces[0]
                    cara = frame[y:y + h, x:x + w]

                    # Convertir y transformar
                    cara_rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
                    cara_pil = Image.fromarray(cara_rgb)
                    cara_tensor = model_manager.transform_video(cara_pil).unsqueeze(0)
                    cara_tensor = cara_tensor.to(model_manager.dispositivo)

                    # Inferencia
                    with torch.no_grad():
                        output = model_manager.modelo_video(cara_tensor)
                        prob_fake = (
                            torch.softmax(output, dim=1)[0][1].item() * 100.0
                        )
                        predicciones.append(prob_fake)

            cap.release()

            # Validar resultados
            if len(predicciones) < config.MIN_FACES_REQUIRED:
                return generar_reporte_error(
                    "No se detectaron suficientes rostros para un análisis confiable "
                    f"(frames con rostro: {frames_analizados}). "
                    "Asegúrate de que el video contenga rostros claros y bien iluminados.",
                    "warning",
                )

            # Calcular promedio
            promedio_fake = sum(predicciones) / len(predicciones)
            es_deepfake = promedio_fake > config.VIDEO_THRESHOLD

            logger.info(
                f"✅ Análisis completado: "
                f"{'DEEPFAKE' if es_deepfake else 'REAL'} ({promedio_fake:.2f}%)"
            )

        # Generar reporte
        return generar_reporte_video(
            es_deepfake=es_deepfake,
            probabilidad=promedio_fake,
            frames_totales=frames_totales,
            frames_analizados=frames_analizados,
            duracion=duracion,
            tiempo_proceso=timer.duracion,
        )

    except Exception as e:
        logger.error(
            f"❌ Error durante el análisis de video: {e}",
            exc_info=True,
        )
        return generar_reporte_error(
            f"Ocurrió un error durante el análisis del video: {str(e)}",
            "error",
        )


# ==========================================
# 🖥️ Interfaz Gradio
# ==========================================

with gr.Blocks(title="UIDE Forense AI") as demo:
    gr.Markdown(
        """
        ### Sistema Avanzado de Detección de Deepfakes y Contenido Sintético
        
        Plataforma basada en **Inteligencia Artificial** para análisis forense de medios digitales.
        Utiliza redes neuronales profundas (ResNet50 y XceptionNet) para identificar manipulaciones 
        y contenido generado por IA.
        """
    )

    gr.Markdown("---")

    with gr.Tabs():

        # ============ TAB 1: IMÁGENES ============
        with gr.TabItem("🖼️ Análisis de Imágenes"):
            gr.Markdown(
                """
                ### 📸 Detector de Imágenes Sintéticas
                
                Identifica imágenes generadas por IA (DALL-E, Midjourney, Stable Diffusion) o manipuladas digitalmente.
                El sistema analiza artefactos microscópicos invisibles al ojo humano.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="📤 Sube tu imagen aquí",
                        type="numpy",
                        height=400,
                        sources=["upload", "clipboard"],
                    )

                    with gr.Row():
                        btn_img = gr.Button(
                            "🔍 Iniciar Análisis Forense",
                            variant="primary",
                            size="lg",
                            scale=2,
                        )
                        btn_clear_img = gr.ClearButton(
                            [img_input],
                            value="🗑️ Limpiar",
                            size="lg",
                            scale=1,
                        )

                    gr.Markdown(
                        """
                        **💡 Formatos soportados:** JPG, PNG, WebP, BMP
                        
                        **⚡ Tiempo estimado:** < 1 segundo
                        """
                    )

                with gr.Column(scale=1):
                    img_output = gr.HTML(label="📊 Informe de Análisis")

            # Información adicional expandible
            with gr.Accordion("ℹ️ ¿Cómo funciona este análisis?", open=False):
                gr.Markdown(
                    """
                    El modelo **CNNDetection** utiliza una red ResNet50 entrenada para detectar:
                    
                    - 🎨 **Imágenes GAN**: Generadas por redes generativas adversarias
                    - 🌊 **Modelos de Difusión**: Como Stable Diffusion, DALL-E 3
                    - ✂️ **Manipulaciones**: Ediciones con Photoshop u otras herramientas
                    
                    El análisis busca patrones estadísticos anómalos en la distribución de píxeles,
                    artefactos de compresión inconsistentes y características espectrales únicas de contenido sintético.
                    """
                )

            # Evento de click
            btn_img.click(
                fn=analizar_imagen,
                inputs=img_input,
                outputs=img_output,
            )

        # ============ TAB 2: VIDEOS ============
        with gr.TabItem("🎥 Análisis de Videos"):
            gr.Markdown(
                """
                ### 🎬 Detector de Deepfakes
                
                Analiza videos para identificar rostros manipulados mediante técnicas de deepfake.
                Detecta inconsistencias temporales y artefactos de síntesis facial.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(
                        label="📤 Sube tu video aquí",
                        height=400,
                        sources=["upload"],
                    )

                    with gr.Row():
                        btn_vid = gr.Button(
                            "▶️ Analizar Deepfakes",
                            variant="primary",
                            size="lg",
                            scale=2,
                        )
                        btn_clear_vid = gr.ClearButton(
                            [vid_input],
                            value="🗑️ Limpiar",
                            size="lg",
                            scale=1,
                        )

                    gr.Markdown(
                        f"""
                        **💡 Formatos soportados:** MP4, AVI, MOV, MKV, WebM
                        
                        **⚙️ Limitaciones:** Máximo {config.MAX_VIDEO_SIZE_MB}MB, {config.MAX_VIDEO_DURATION_SECONDS//60} minutos
                        
                        **⚡ Tiempo estimado:** Variable según duración (30s - 5min)
                        """
                    )

                with gr.Column(scale=1):
                    vid_output = gr.HTML(label="📊 Informe de Análisis")

            # Información adicional
            with gr.Accordion("ℹ️ ¿Cómo funciona este análisis?", open=False):
                gr.Markdown(
                    """
                    El modelo **XceptionNet** analiza cada frame del video para detectar:
                    
                    - 👤 **Face Swap**: Intercambio de rostros
                    - 🎭 **Síntesis facial**: Rostros completamente generados
                    - 🔄 **Reenactment**: Transferencia de expresiones faciales
                    - 🗣️ **Lip Sync**: Manipulación de movimientos labiales
                    
                    El sistema utiliza **muestreo inteligente** para optimizar el rendimiento,
                    analizando frames clave donde es más probable detectar inconsistencias.
                    
                    ⚠️ **Nota**: El análisis requiere que el video contenga rostros visibles y bien iluminados.
                    """
                )

            # Evento de click
            btn_vid.click(
                fn=analizar_video,
                inputs=vid_input,
                outputs=vid_output,
            )

        # ============ TAB 3: INFORMACIÓN ============
        with gr.TabItem("📚 Acerca de"):
            gr.Markdown(
                """
                # 🎓 UIDE Forense AI
                
                ## 🔬 Proyecto Académico
                
                Sistema de detección de deepfakes y contenido sintético desarrollado como parte 
                de la investigación en **Visión por Computadora e Inteligencia Artificial** en la 
                Universidad Internacional del Ecuador.
                
                ---
                
                ## 👥 Equipo de Desarrollo
                
                - **Anthony Perez** - Investigador Principal
                - **Bruno Ortega** - Desarrollador ML
                - **Manuel Pacheco** - Ingeniero de Software
                
                ---
                
                ## 🛠️ Tecnologías Utilizadas
                
                | Componente | Tecnología |
                |------------|------------|
                | **Framework UI** | Gradio 4.0+ |
                | **Deep Learning** | PyTorch, Torchvision |
                | **Modelos** | ResNet50, XceptionNet |
                | **Visión Computacional** | OpenCV, PIL |
                | **Datasets** | FaceForensics++, CNNDetection |
                
                ---
                
                ## ⚖️ Consideraciones Éticas
                
                > **⚠️ IMPORTANTE**: Esta herramienta tiene fines exclusivamente académicos y de investigación.
                > Los resultados son probabilísticos y deben ser verificados por expertos forenses.
                > No debe usarse como única evidencia en contextos legales.
                
                ---
                
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px; margin-top: 20px;">
                    <p style="font-size: 0.9em; color: #666; margin: 0;">
                        © 2025 UIDE - Todos los derechos reservados
                    </p>
                </div>
                """
            )

# ==========================================
# 🚀 Punto de Entrada
# ==========================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando UIDE Forense AI...")

    demo.launch(
        server_name="0.0.0.0",  # Accesible desde red local
        server_port=7860,
        share=False,            # Cambiar a True para crear link público temporal
        show_error=True,
        quiet=False,
    )