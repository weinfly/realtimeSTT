"""
实时语音转文字+翻译应用
支持中文和英文
"""

__version__ = "0.1.0"

from .config import AudioSource, ROOT_DIR, MODEL_DIR, OUT_DIR, CONFIG_FILE
from .models import OnnxModel, create_recognizer
from .audio import create_input_stream_safe
from .workers import (
    TranslationWorker,
    OptimizationWorker,
    MeetingMinutesWorker,
    RealtimeTranslationWorker,
    Worker
)
from .ui import RealTimeWindow

__all__ = [
    'AudioSource',
    'ROOT_DIR',
    'MODEL_DIR',
    'OUT_DIR',
    'CONFIG_FILE',
    'OnnxModel',
    'create_recognizer',
    'create_input_stream_safe',
    'TranslationWorker',
    'OptimizationWorker',
    'MeetingMinutesWorker',
    'RealtimeTranslationWorker',
    'Worker',
    'RealTimeWindow',
]
