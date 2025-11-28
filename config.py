"""
Configuración centralizada del proyecto UIDE Forense AI
"""

import os

# ==========================================
# 📁 Rutas de Archivos
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Rutas de modelos
MODEL_IMAGE_PATH = os.path.join(WEIGHTS_DIR, "blur_jpg_prob0.1.pth")
MODEL_VIDEO_NAME = "xception"

# ==========================================
# 📊 Límites y Validación
# ==========================================
MAX_IMAGE_SIZE_MB = 15
MAX_VIDEO_SIZE_MB = 200
MAX_VIDEO_DURATION_SECONDS = 300  # 5 minutos

# Formatos soportados
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# ==========================================
# 🎯 Parámetros de Análisis
# ==========================================
# Imágenes
IMAGE_THRESHOLD = 50.0  # Umbral de clasificación (%)
IMAGE_SIZE = 224

# Video
VIDEO_FRAME_STRIDE = 30  # Analizar 1 frame cada N frames
VIDEO_SIZE = 299
VIDEO_THRESHOLD = 50.0
MIN_FACES_REQUIRED = 3  # Mínimo de rostros para análisis válido

# ==========================================
# 🎨 Configuración UI
# ==========================================
DEFAULT_THEME = "soft"
PRIMARY_COLOR = "blue"
SECONDARY_COLOR = "slate"

# Colores para reportes
COLOR_FAKE = "#ef4444"      # Rojo
COLOR_REAL = "#22c55e"      # Verde
COLOR_WARNING = "#f59e0b"   # Ámbar
COLOR_INFO = "#3b82f6"      # Azul

# ==========================================
# 🔧 Configuración Técnica
# ==========================================
DEVICE = "cpu"  # Cambiar a 'cuda' si hay GPU disponible
NUM_WORKERS = 4
ENABLE_CACHE = True

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
