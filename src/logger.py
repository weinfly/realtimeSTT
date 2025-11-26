"""
日志模块
处理应用程序的日志记录和标准输出重定向
"""

import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from .config import LOG_DIR

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass

def setup_logging():
    """配置日志系统并重定向stdout/stderr"""
    log_file = os.path.join(LOG_DIR, f'stt.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = TimedRotatingFileHandler(
        log_file, when='D', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # 重定向 stdout 和 stderr
    # 保存原始 stdout/stderr 以便需要时恢复
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)
    
    return logger
