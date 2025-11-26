"""
配置和常量模块
包含应用程序的所有配置、路径和常量定义
"""

import sys
import os
from pathlib import Path
from enum import Enum


# 音频源类型枚举
class AudioSource(Enum):
    MICROPHONE = "仅麦克风"
    SYSTEM = "仅系统声音"  
    BOTH = "麦克风+系统声音"


# 路径配置
if getattr(sys, 'frozen', False):
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    ROOT_DIR = Path(os.getcwd()).as_posix()

MODEL_DIR = f'{ROOT_DIR}/onnx'
OUT_DIR = f'{ROOT_DIR}/output'
CONFIG_FILE = f'{ROOT_DIR}/config.json'
LOG_DIR = f'{ROOT_DIR}/Log'

# 创建必要的目录
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(OUT_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

# 模型文件路径
CTC_MODEL_FILE = f"{MODEL_DIR}/ctc.model.onnx"
PAR_ENCODER = f"{MODEL_DIR}/encoder-epoch-99-avg-1.onnx"
PAR_DECODER = f"{MODEL_DIR}/decoder-epoch-99-avg-1.onnx"
PAR_JOINER = f"{MODEL_DIR}/joiner-epoch-99-avg-1.onnx"
PAR_TOKENS = f"{MODEL_DIR}/tokens.txt"

# 环境变量设置
if sys.platform == 'win32':
    os.environ['PATH'] = f'{ROOT_DIR};{ROOT_DIR}/ffmpeg;' + os.environ['PATH']
