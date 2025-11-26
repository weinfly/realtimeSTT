import sys
from PySide6.QtWidgets import QApplication
from src.config import ROOT_DIR
from src.logger import setup_logging
from src.ui import RealTimeWindow

# 设置日志
setup_logging()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 加载样式表
    try:
        with open(f'{ROOT_DIR}/data/style.qss', 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("警告：未找到样式表文件")
    
    window = RealTimeWindow()
    window.show()
    sys.exit(app.exec())