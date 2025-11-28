"""
Utilidades y funciones auxiliares para UIDE Forense AI
"""

import os
import time
import logging
from typing import Optional, Tuple
from PIL import Image
import config

# Configurar logging
logger = logging.getLogger(__name__)

# ==========================================
# 📊 Validación de Archivos
# ==========================================

def validar_imagen(imagen_array, tamaño_max_mb: int = config.MAX_IMAGE_SIZE_MB) -> Tuple[bool, str]:
    """
    Valida que una imagen cumpla los requisitos.
    
    Args:
        imagen_array: Array numpy de la imagen
        tamaño_max_mb: Tamaño máximo en MB
        
    Returns:
        Tupla (es_valida, mensaje_error)
    """
    try:
        if imagen_array is None:
            return False, "No se proporcionó ninguna imagen"
        
        # Validar dimensiones mínimas
        if imagen_array.shape[0] < 32 or imagen_array.shape[1] < 32:
            return False, "La imagen es demasiado pequeña (mínimo 32x32 píxeles)"
        
        # Validar dimensiones máximas (para evitar OOM)
        if imagen_array.shape[0] > 8192 or imagen_array.shape[1] > 8192:
            return False, "La imagen es demasiado grande (máximo 8192x8192 píxeles)"
        
        return True, ""
    except Exception as e:
        logger.error(f"Error validando imagen: {e}")
        return False, f"Error al validar imagen: {str(e)}"


def validar_video(video_path: str) -> Tuple[bool, str]:
    """
    Valida que un video cumpla los requisitos.
    
    Args:
        video_path: Ruta al archivo de video
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    try:
        if not os.path.exists(video_path):
            return False, "El archivo de video no existe"
        
        # Validar tamaño del archivo
        tamaño_mb = os.path.getsize(video_path) / (1024 * 1024)
        if tamaño_mb > config.MAX_VIDEO_SIZE_MB:
            return False, f"El video es demasiado grande ({tamaño_mb:.1f}MB). Máximo: {config.MAX_VIDEO_SIZE_MB}MB"
        
        # Validar extensión
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in config.SUPPORTED_VIDEO_FORMATS:
            return False, f"Formato no soportado. Use: {', '.join(config.SUPPORTED_VIDEO_FORMATS)}"
        
        return True, ""
    except Exception as e:
        logger.error(f"Error validando video: {e}")
        return False, f"Error al validar video: {str(e)}"


# ==========================================
# 🎨 Generación de Reportes HTML
# ==========================================

def generar_gauge_svg(probabilidad: float, color: str, tamaño: int = 200) -> str:
    """
    Genera un medidor circular SVG animado.
    
    Args:
        probabilidad: Valor de 0 a 100
        color: Color en formato hex
        tamaño: Tamaño del medidor en píxeles
    """
    # Calcular el stroke-dashoffset para la animación circular
    circunferencia = 2 * 3.14159 * 70  # radio = 70
    offset = circunferencia - (probabilidad / 100 * circunferencia)
    
    return f"""
    <svg width="{tamaño}" height="{tamaño}" viewBox="0 0 200 200" style="transform: rotate(-90deg);">
        <circle cx="100" cy="100" r="70" fill="none" stroke="#e5e7eb" stroke-width="12"/>
        <circle cx="100" cy="100" r="70" fill="none" stroke="{color}" stroke-width="12"
                stroke-dasharray="{circunferencia}" stroke-dashoffset="{offset}"
                stroke-linecap="round" style="transition: stroke-dashoffset 1s ease;">
        </circle>
        <text x="100" y="110" text-anchor="middle" font-size="32" font-weight="bold" 
              fill="{color}" style="transform: rotate(90deg); transform-origin: center;">
            {probabilidad:.1f}%
        </text>
    </svg>
    """


def generar_barra_progreso(probabilidad: float, color: str) -> str:
    """Genera una barra de progreso animada."""
    return f"""
    <div style="background-color: #e5e7eb; height: 20px; border-radius: 10px; width: 100%; overflow: hidden; margin: 15px 0;">
        <div style="background: linear-gradient(90deg, {color}, {color}dd); height: 100%; border-radius: 10px; 
                    width: {probabilidad}%; transition: width 1.5s ease; box-shadow: 0 0 10px {color}88;">
        </div>
    </div>
    """


def generar_stat_card(valor, etiqueta: str, icono: str = "📊") -> str:
    """Genera una tarjeta de estadística."""
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 12px; text-align: center; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 150px;">
        <div style="font-size: 2.5em; margin-bottom: 5px;">{icono}</div>
        <div style="font-size: 2em; font-weight: bold; color: white; margin-bottom: 5px;">{valor}</div>
        <div style="font-size: 0.9em; color: rgba(255,255,255,0.9); text-transform: uppercase; letter-spacing: 1px;">
            {etiqueta}
        </div>
    </div>
    """


def generar_reporte_imagen(es_fake: bool, probabilidad: float, 
                          ancho: int, alto: int, tiempo_proceso: float) -> str:
    """
    Genera el reporte HTML completo para análisis de imagen.
    """
    color = config.COLOR_FAKE if es_fake else config.COLOR_REAL
    icono = "🚨" if es_fake else "✅"
    diagnostico = "POSIBLE MANIPULACIÓN DETECTADA" if es_fake else "CONTENIDO AUTÉNTICO"
    confianza = probabilidad if es_fake else (100 - probabilidad)
    
    gauge = generar_gauge_svg(confianza, color)
    barra = generar_barra_progreso(confianza, color)
    
    # Cards de estadísticas
    stats_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0;">
        {generar_stat_card(f"{ancho}x{alto}", "Resolución", "🖼️")}
        {generar_stat_card(f"{tiempo_proceso:.2f}s", "Tiempo", "⚡")}
        {generar_stat_card(f"{confianza:.1f}%", "Confianza", "🎯")}
    </div>
    """
    
    return f"""
    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                padding: 30px; border-radius: 15px; border-left: 5px solid {color}; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.1); animation: slideIn 0.5s ease;">
        
        <h2 style="color: {color}; margin-top: 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5em;">{icono}</span>
            <span>Análisis de Imagen: {diagnostico}</span>
        </h2>
        
        <div style="display: flex; align-items: center; justify-content: center; margin: 30px 0;">
            {gauge}
        </div>
        
        {barra}
        {stats_html}
        
        <div style="background: rgba(255,255,255,0.5); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; color: #1f2937;">🔍 Detalles Técnicos</h3>
            <ul style="line-height: 1.8; color: #374151;">
                <li><strong>Modelo:</strong> CNNDetection (ResNet50)</li>
                <li><strong>Método:</strong> Detección de artefactos de GANs y Difusión</li>
                <li><strong>Resolución analizada:</strong> {ancho} × {alto} píxeles</li>
                <li><strong>Tiempo de procesamiento:</strong> {tiempo_proceso:.3f} segundos</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: rgba(59, 130, 246, 0.1); 
                    border-radius: 8px; border-left: 3px solid #3b82f6;">
            <p style="margin: 0; font-size: 0.9em; color: #1f2937;">
                ℹ️ <strong>Nota:</strong> Este análisis es probabilístico y debe ser verificado por un experto forense. 
                Los resultados pueden variar según la calidad y origen de la imagen.
            </p>
        </div>
    </div>
    
    <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """


def generar_reporte_video(es_deepfake: bool, probabilidad: float,
                         frames_totales: int, frames_analizados: int,
                         duracion: float, tiempo_proceso: float) -> str:
    """
    Genera el reporte HTML completo para análisis de video.
    """
    color = config.COLOR_FAKE if es_deepfake else config.COLOR_REAL
    icono = "🚨" if es_deepfake else "✅"
    diagnostico = "DEEPFAKE DETECTADO" if es_deepfake else "VIDEO AUTÉNTICO"
    confianza = probabilidad if es_deepfake else (100 - probabilidad)
    
    gauge = generar_gauge_svg(confianza, color)
    barra = generar_barra_progreso(confianza, color)
    
    # Cards de estadísticas
    stats_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0;">
        {generar_stat_card(f"{duracion:.1f}s", "Duración", "🎬")}
        {generar_stat_card(frames_analizados, "Rostros", "👤")}
        {generar_stat_card(f"{tiempo_proceso:.1f}s", "Tiempo", "⚡")}
        {generar_stat_card(f"{confianza:.1f}%", "Confianza", "🎯")}
    </div>
    """
    
    return f"""
    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                padding: 30px; border-radius: 15px; border-left: 5px solid {color}; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.1); animation: slideIn 0.5s ease;">
        
        <h2 style="color: {color}; margin-top: 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5em;">{icono}</span>
            <span>Análisis de Video: {diagnostico}</span>
        </h2>
        
        <div style="display: flex; align-items: center; justify-content: center; margin: 30px 0;">
            {gauge}
        </div>
        
        {barra}
        {stats_html}
        
        <div style="background: rgba(255,255,255,0.5); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; color: #1f2937;">📹 Análisis Forense</h3>
            <ul style="line-height: 1.8; color: #374151;">
                <li><strong>Modelo:</strong> XceptionNet (Pre-entrenado en FaceForensics++)</li>
                <li><strong>Método:</strong> Análisis de consistencia facial frame-a-frame</li>
                <li><strong>Frames totales:</strong> {frames_totales}</li>
                <li><strong>Rostros detectados:</strong> {frames_analizados}</li>
                <li><strong>Tasa de muestreo:</strong> 1 frame cada {config.VIDEO_FRAME_STRIDE} frames</li>
                <li><strong>Tiempo total:</strong> {tiempo_proceso:.2f} segundos</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: rgba(59, 130, 246, 0.1); 
                    border-radius: 8px; border-left: 3px solid #3b82f6;">
            <p style="margin: 0; font-size: 0.9em; color: #1f2937;">
                ℹ️ <strong>Nota:</strong> El análisis de videos es computacionalmente intensivo. 
                Se utiliza muestreo inteligente para optimizar el rendimiento manteniendo la precisión.
            </p>
        </div>
    </div>
    
    <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """


def generar_reporte_error(mensaje: str, tipo: str = "error") -> str:
    """Genera un reporte de error con estilo."""
    colores = {
        "error": ("#ef4444", "❌"),
        "warning": ("#f59e0b", "⚠️"),
        "info": ("#3b82f6", "ℹ️")
    }
    color, icono = colores.get(tipo, colores["error"])
    
    return f"""
    <div style="background: {color}15; padding: 25px; border-radius: 12px; 
                border-left: 5px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="color: {color}; margin-top: 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5em;">{icono}</span>
            <span>{"Error" if tipo == "error" else "Aviso"}</span>
        </h3>
        <p style="font-size: 1.1em; color: #1f2937; margin: 10px 0;">{mensaje}</p>
    </div>
    """


# ==========================================
# ⏱️ Utilidades de Tiempo
# ==========================================

def formatear_tiempo(segundos: float) -> str:
    """Formatea segundos a formato legible."""
    if segundos < 1:
        return f"{segundos*1000:.0f}ms"
    elif segundos < 60:
        return f"{segundos:.2f}s"
    else:
        minutos = int(segundos // 60)
        segs = int(segundos % 60)
        return f"{minutos}m {segs}s"


class Timer:
    """Context manager para medir tiempo de ejecución."""
    
    def __init__(self):
        self.inicio = None
        self.fin = None
        
    def __enter__(self):
        self.inicio = time.time()
        return self
        
    def __exit__(self, *args):
        self.fin = time.time()
        
    @property
    def duracion(self) -> float:
        if self.inicio and self.fin:
            return self.fin - self.inicio
        return 0.0
