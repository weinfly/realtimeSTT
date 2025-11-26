"""
UI模块
包含应用程序的主窗口和界面逻辑
"""

import sys
import os
import time
import json
import re
import sounddevice as sd
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QPushButton, QPlainTextEdit, QFileDialog, QLabel, QSplitter, 
                               QGroupBox, QMessageBox, QCheckBox)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QIcon, QCloseEvent, QDesktopServices

from .config import (
    AudioSource, ROOT_DIR, OUT_DIR, CONFIG_FILE, MODEL_DIR,
    CTC_MODEL_FILE, PAR_ENCODER, PAR_DECODER
)
from .workers import (
    TranslationWorker,
    OptimizationWorker,
    MeetingMinutesWorker,
    RealtimeTranslationWorker,
    Worker
)


class RealTimeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('实时语音转文字+翻译 - 支持中文和英文v0.1.0 - WHL')
        self.setMinimumSize(1000, 700)
        self.layout = QVBoxLayout(self)
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/data/icon.ico"))

        self.config = self.load_config()

        self.mic_layout = QHBoxLayout()
        self.combo = QComboBox()
        self.populate_devices()
        self.mic_layout.addWidget(self.combo)
        
        self.mode_label = QLabel('音频模式:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(AudioSource.MICROPHONE.value, AudioSource.MICROPHONE)
        self.mode_combo.addItem(AudioSource.SYSTEM.value, AudioSource.SYSTEM)
        self.mode_combo.addItem(AudioSource.BOTH.value, AudioSource.BOTH)
        self.mode_combo.setCurrentIndex(2)  # 默认选择"麦克风+系统声音"
        self.mic_layout.addWidget(self.mode_label)
        self.mic_layout.addWidget(self.mode_combo)

        self.start_button = QPushButton('启动实时语音转文字')
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setMinimumHeight(30)
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self.toggle_transcription)
        self.mic_layout.addWidget(self.start_button)
        self.layout.addLayout(self.mic_layout)

        # 实时识别文本窗口
        self.realtime_text = QPlainTextEdit()
        self.realtime_text.setReadOnly(True)
        self.realtime_text.setStyleSheet("background: transparent; border: none;font-size:14px")
        self.realtime_text.setMaximumHeight(80)
        self.layout.addWidget(self.realtime_text)

        self.realtime_translation = QPlainTextEdit()
        self.realtime_translation.setReadOnly(True)
        self.realtime_translation.setStyleSheet("background: transparent; border: none; font-size:14px; color: #888888; font-style: italic;")
        self.realtime_translation.setMaximumHeight(60)
        self.realtime_translation.setPlaceholderText("实时译文...")
        self.layout.addWidget(self.realtime_translation)

        self.splitter = QSplitter(Qt.Horizontal)
        
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.original_label = QLabel('识别的原文')
        self.original_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.left_layout.addWidget(self.original_label)
        
        self.textedit = QPlainTextEdit()
        self.textedit.setReadOnly(True)
        self.textedit.setMinimumHeight(400)
        self.textedit.setStyleSheet("color:#ffffff")
        self.left_layout.addWidget(self.textedit)
        
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.trans_control_layout = QHBoxLayout()
        self.translation_label = QLabel('翻译结果')
        self.translation_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.trans_control_layout.addWidget(self.translation_label)
        
        self.auto_translate_checkbox = QCheckBox('自动优化翻译')
        self.auto_translate_checkbox.setToolTip("开启后，每句话识别完成后会自动进行优化翻译并追加到右侧窗口")
        self.auto_translate_checkbox.setChecked(self.config.get('auto_translate', True))
        self.auto_translate_checkbox.stateChanged.connect(self.on_auto_translate_changed)
        self.trans_control_layout.addWidget(self.auto_translate_checkbox)
        
        self.smart_optimize_checkbox = QCheckBox('智能优化识别')
        self.smart_optimize_checkbox.setToolTip("使用AI根据上下文优化识别结果，提高准确性")
        self.smart_optimize_checkbox.setChecked(self.config.get('smart_optimize', False))
        self.smart_optimize_checkbox.stateChanged.connect(self.on_smart_optimize_changed)
        self.trans_control_layout.addWidget(self.smart_optimize_checkbox)
        
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(['中文','英语', '日语', '韩语', '法语', '德语', '西班牙语', '俄语', '阿拉伯语'])
        default_lang = self.config.get('default_target_language', '中文')
        index = self.target_lang_combo.findText(default_lang)
        if index >= 0:
            self.target_lang_combo.setCurrentIndex(index)
        self.target_lang_combo.currentIndexChanged.connect(self.on_language_changed)
        self.trans_control_layout.addWidget(self.target_lang_combo)
        
        self.translate_button = QPushButton('优化翻译')
        self.translate_button.setToolTip("对左侧识别原文进行优化翻译")
        self.translate_button.setCursor(Qt.PointingHandCursor)
        self.translate_button.setMinimumHeight(30)
        self.translate_button.clicked.connect(self.start_translation)
        self.trans_control_layout.addWidget(self.translate_button)
        self.trans_control_layout.addStretch()
        
        self.right_layout.addLayout(self.trans_control_layout)
        
        self.translation_textedit = QPlainTextEdit()
        self.translation_textedit.setReadOnly(True)
        self.translation_textedit.setMinimumHeight(400)
        self.translation_textedit.setStyleSheet("color:#ffffff")
        self.right_layout.addWidget(self.translation_textedit)
        
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        
        # 会议纪要窗口
        self.minutes_widget = QWidget()
        self.minutes_layout = QVBoxLayout(self.minutes_widget)
        self.minutes_layout.setContentsMargins(0, 0, 0, 0)
        
        self.minutes_control_layout = QHBoxLayout()
        self.minutes_label = QLabel('会议纪要')
        self.minutes_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.minutes_control_layout.addWidget(self.minutes_label)
        
        self.minutes_lang_combo = QComboBox()
        self.minutes_lang_combo.addItems(['中文','英语', '日语', '韩语', '法语', '德语', '西班牙语', '俄语', '阿拉伯语'])
        default_minutes_lang = self.config.get('default_target_language', '中文')
        index = self.minutes_lang_combo.findText(default_minutes_lang)
        if index >= 0:
            self.minutes_lang_combo.setCurrentIndex(index)
        self.minutes_control_layout.addWidget(self.minutes_lang_combo)
        
        self.generate_minutes_button = QPushButton('生成会议纪要')
        self.generate_minutes_button.setToolTip("根据识别的原文生成结构化的会议纪要")
        self.generate_minutes_button.setCursor(Qt.PointingHandCursor)
        self.generate_minutes_button.setMinimumHeight(30)
        self.generate_minutes_button.clicked.connect(self.generate_meeting_minutes)
        self.minutes_control_layout.addWidget(self.generate_minutes_button)
        self.minutes_control_layout.addStretch()
        
        self.minutes_layout.addLayout(self.minutes_control_layout)
        
        self.minutes_textedit = QPlainTextEdit()
        self.minutes_textedit.setReadOnly(True)
        self.minutes_textedit.setMinimumHeight(400)
        self.minutes_textedit.setStyleSheet("color:#ffffff")
        self.minutes_layout.addWidget(self.minutes_textedit)
        
        self.splitter.addWidget(self.minutes_widget)
        self.splitter.setSizes([400, 300, 300])

        self.layout.addWidget(self.splitter)

        self.button_layout = QHBoxLayout()
        self.export_button = QPushButton('导出原文')
        self.export_button.clicked.connect(self.export_txt)
        self.export_button.setCursor(Qt.PointingHandCursor)
        self.export_button.setMinimumHeight(35)
        self.button_layout.addWidget(self.export_button)

        self.export_translation_button = QPushButton('导出译文')
        self.export_translation_button.clicked.connect(self.export_translation)
        self.export_translation_button.setCursor(Qt.PointingHandCursor)
        self.export_translation_button.setMinimumHeight(35)
        self.button_layout.addWidget(self.export_translation_button)


        self.export_minutes_button = QPushButton('导出会议纪要')

        self.export_minutes_button.clicked.connect(self.export_minutes)

        self.export_minutes_button.setCursor(Qt.PointingHandCursor)

        self.export_minutes_button.setMinimumHeight(35)

        self.button_layout.addWidget(self.export_minutes_button)


        self.copy_button = QPushButton('复制原文')
        self.copy_button.setMinimumHeight(35)
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_textedit)
        self.button_layout.addWidget(self.copy_button)

        self.clear_button = QPushButton('清空')
        self.clear_button.setMinimumHeight(35)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear_textedit)
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)
        self.btn_opendir=QPushButton(f"录音文件保存到: {OUT_DIR}")
        self.btn_opendir.setStyleSheet("background-color:transparent;border:0;")
        self.btn_opendir.clicked.connect(self.open_dir)
        self.layout.addWidget(self.btn_opendir)

        self.worker = None
        self.translation_worker = None
        self.realtime_translation_worker = None
        self.optimization_worker = None
        self.minutes_worker = None
        self.transcribing = False
        self.last_translated_text = ""
        self.translation_queue = []
        self.is_translating = False
        self.is_optimizing = False
        self.is_generating_minutes = False
        self.recent_segments = []
        self.current_realtime_translation = ""

    def load_config(self):
        try:
            if Path(CONFIG_FILE).exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                default_config = {
                    "api_base_url": "https://api.openai.com/v1",
                    "api_key": "",
                    "model": "gpt-3.5-turbo",
                    "default_target_language": "英语",
                    "timeout": 30,
                    "auto_translate": True,
                    "smart_optimize": False
                }
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def on_auto_translate_changed(self, state):
        is_checked = state == Qt.Checked
        self.config['auto_translate'] = is_checked
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def on_smart_optimize_changed(self, state):
        is_checked = state == Qt.Checked
        self.config['smart_optimize'] = is_checked
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def start_translation(self, text=None, is_auto=False):
        if isinstance(text, bool):
            text = None
            
        if text is None:
            raw_text = self.textedit.toPlainText().strip()
            original_text = re.sub(r'\[\d{2}:\d{2}:\d{2}\] ', '', raw_text)
        else:
            original_text = text.strip()
            
        if not original_text:
            if not is_auto:
                QMessageBox.warning(self, '提示', '没有可翻译的内容')
            return
        
        if not self.config.get('api_key'):
            if not is_auto:
                QMessageBox.warning(self, '配置错误', f'请先在 {CONFIG_FILE} 中配置 API 密钥（api_key）')
            else:
                print(f'警告：未配置 API 密钥，无法进行自动翻译')
            return
        
        if self.is_translating:
            self.translation_queue.append((original_text, is_auto))
            return
        
        self.is_translating = True
        target_language = self.target_lang_combo.currentText()
        
        self.translate_button.setEnabled(False)
        self.translate_button.setText('翻译中...')
        
        if not is_auto:
            self.translation_textedit.setPlainText('正在翻译，请稍候...')
            self.last_translated_text = ""
        
        self.translation_worker = TranslationWorker(original_text, target_language, self.config)
        self.translation_worker.translation_ready.connect(lambda t: self.on_translation_ready(t, is_auto))
        self.translation_worker.translation_error.connect(lambda e: self.on_translation_error(e, is_auto))
        self.translation_worker.start()
    
    def on_translation_ready(self, translation, is_auto=False):
        timestamp = time.strftime("[%H:%M:%S]")
        
        if is_auto:
            # 自动翻译：追加到现有内容
            current_text = self.translation_textedit.toPlainText()
            if current_text and not current_text.startswith('正在翻译'):
                self.translation_textedit.appendPlainText(f"{timestamp} {translation}")
            else:
                self.translation_textedit.setPlainText(f"{timestamp} {translation}")
        else:
            # 手动优化翻译：替换所有内容，每行带时间戳
            lines = translation.split('\n')
            result_lines = []
            for line in lines:
                if line.strip():
                    result_lines.append(f"{timestamp} {line}")
            self.translation_textedit.setPlainText('\n'.join(result_lines))
        
        scrollbar = self.translation_textedit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.translate_button.setEnabled(True)
        self.translate_button.setText('优化翻译')
        self.is_translating = False
        
        if self.translation_queue:
            next_text, next_is_auto = self.translation_queue.pop(0)
            self.start_translation(next_text, next_is_auto)

    def on_translation_error(self, error_msg, is_auto=False):
        if is_auto:
            print(f'自动翻译失败：{error_msg}')
        else:
            self.translation_textedit.setPlainText(f'翻译失败：\n{error_msg}')
            QMessageBox.warning(self, '翻译错误', error_msg)
        
        self.translate_button.setEnabled(True)
        self.translate_button.setText('优化翻译')
        self.is_translating = False
        
        if self.translation_queue:
            next_text, next_is_auto = self.translation_queue.pop(0)
            self.start_translation(next_text, next_is_auto)

    def check_model_exist(self):
        if not Path(PAR_ENCODER).exists() or not Path(CTC_MODEL_FILE).exists() or not Path(PAR_DECODER).exists():
            reply = QMessageBox.information(self,'缺少实时语音转文字所需模型，请去下载',
            f'模型下载地址已复制到剪贴板内，请到浏览器地址栏中粘贴下载\n\n为减小软件包体积，默认未内置模型，下载解压后，将其内的4个文件放到 {MODEL_DIR}  文件夹内'
            )
            QApplication.clipboard().setText('https://github.com/jianchang512/stt/releases/download/0.0/realtimestt-models.7z')
            return False
        return True

    def open_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(OUT_DIR))

    def populate_devices(self):
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print("未找到任何可用麦克风")
            sys.exit(0)

        default_idx = sd.default.device[0]
        default_item = 0
        
        for i, d in enumerate(input_devices):
            self.combo.addItem(f"[麦克风] {d['name']}", ('mic', d['index']))
            if d['index'] == default_idx:
                default_item = i
        
        if sys.platform == 'win32':
            for i, d in enumerate(devices):
                if any(keyword in d['name'].lower() for keyword in ['立体声混音', 'stereo mix', 'wave out', 'loopback', '您听到的声音', 'what u hear']):
                    if d['max_input_channels'] > 0:
                        self.combo.addItem(f"[系统声音] {d['name']}", ('system', i))
        
        self.combo.setCurrentIndex(default_item)

    def toggle_transcription(self):
        if self.check_model_exist() is not True:
            return
        if not self.transcribing:
            self.realtime_text.setPlainText('请稍等，正在加载模型，请在出现"请说话"后开始说话...')
            device_data = self.combo.currentData()
            audio_mode = self.mode_combo.currentData()
            
            if isinstance(device_data, tuple):
                device_type, device_idx = device_data
            else:
                device_type, device_idx = 'mic', device_data
            
            mic_idx = None
            sys_idx = None
            
            if audio_mode == AudioSource.MICROPHONE:
                mic_idx = device_idx if device_type == 'mic' else self._get_first_mic_idx()
            elif audio_mode == AudioSource.SYSTEM:
                sys_idx = device_idx if device_type == 'system' else self._get_first_system_idx()
            elif audio_mode == AudioSource.BOTH:
                mic_idx = self._get_first_mic_idx()
                sys_idx = self._get_first_system_idx()
            
            self.worker = Worker(device_idx, audio_mode, mic_idx, sys_idx)
            self.worker.new_word.connect(self.update_realtime)
            self.worker.new_segment.connect(self.append_segment)
            self.worker.ready.connect(self.update_realtime_ready)
            self.worker.start()
            
            self.realtime_translation_worker = RealtimeTranslationWorker(self.config)
            self.realtime_translation_worker.set_target_language(self.target_lang_combo.currentText())
            self.realtime_translation_worker.translation_ready.connect(self.update_realtime_translation)
            self.realtime_translation_worker.start()
            
            self.start_button.setText('正在语音转文字中...')
            self.transcribing = True
            self.translate_button.setEnabled(False)
        else:
            if self.worker:
                self.worker.running = False
                self.worker.wait()
                self.worker = None
            
            if self.realtime_translation_worker:
                self.realtime_translation_worker.stop()
                self.realtime_translation_worker = None
                
            self.start_button.setText('启动实时转录')
            self.transcribing = False
            self.translate_button.setEnabled(True)
            remaining_text = self.realtime_text.toPlainText().strip()
            if remaining_text:
                # 通过正常的段处理流程处理最后一段话
                self.append_segment(remaining_text)
            self.realtime_text.clear()
            self.realtime_translation.clear()

    def update_realtime(self, text):
        self.realtime_text.setPlainText(text)
        scrollbar = self.realtime_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        if self.realtime_translation_worker:
            self.realtime_translation_worker.update_text(text)
            
    def update_realtime_translation(self, text):
        """更新实时翻译显示（仅更新实时翻译预览区域）"""
        self.realtime_translation.setPlainText(text)
        # 保存当前实时翻译结果，供段落完成时使用
        self.current_realtime_translation = text

    def update_realtime_ready(self):
        self.realtime_text.setPlainText('请说话...')

    def append_segment(self, text):
        # 过滤不规则内容，但保留标点符号
        # 使用更精确的过滤，只移除 <unk> 标记和多余空格
        text = re.sub(r'<unk>', '', text)  # 移除 <unk> 标记
        text = re.sub(r'\s+', ' ', text)   # 将多个空格替换为单个空格
        text = text.strip()                # 移除首尾空格
        
        if not text:
            return
        
        # 如果启用智能优化，先优化文本
        if self.smart_optimize_checkbox.isChecked():
            context = self.get_recent_context()
            self.start_optimization(text, context)
        else:
            self.display_segment(text)
    
    def get_recent_context(self, max_segments=3):
        """获取最近的文本作为上下文"""
        if not self.recent_segments:
            return ""
        return " ".join(self.recent_segments[-max_segments:])
    
    def start_optimization(self, text, context):
        """启动文本优化"""
        if self.is_optimizing:
            # 如果正在优化，直接显示原文
            self.display_segment(text)
            return
        
        self.is_optimizing = True
        self.optimization_worker = OptimizationWorker(text, context, self.config)
        self.optimization_worker.optimization_ready.connect(self.on_optimization_ready)
        self.optimization_worker.optimization_error.connect(lambda e: self.on_optimization_error(e, text))
        self.optimization_worker.start()
    
    def on_optimization_ready(self, optimized_text):
        """优化完成"""
        self.is_optimizing = False
        self.display_segment(optimized_text)
    
    def on_optimization_error(self, error_msg, original_text):
        """优化失败，使用原文"""
        print(f"智能优化失败: {error_msg}，使用原文")
        self.is_optimizing = False
        self.display_segment(original_text)
    
    def display_segment(self, text):
        """显示文本段落（带时间戳）并同时写入翻译结果"""
        print(f"[UI DEBUG] Displaying segment: '{text}'")
        
        # 添加到最近段落列表
        self.recent_segments.append(text)
        if len(self.recent_segments) > 10:
            self.recent_segments.pop(0)
        
        # 确保文本以适当的标点符号结尾
        if text and text[-1] not in ['。', '！', '？', '，', '；', '：', '.', '!', '?', ',', ';', ':']:
            # 如果文本没有以标点结尾，添加句号
            text += '。'
            print(f"[UI DEBUG] Added period: '{text}'")
        
        # 添加时间戳（更精确的格式，包含毫秒）
        timestamp = time.strftime("[%H:%M:%S]")
        display_text = f"{timestamp} {text}"
        
        print(f"[UI DEBUG] Final display text: '{display_text}'")
        
        # 写入原文到左侧窗口
        self.textedit.appendPlainText(display_text)
        scrollbar = self.textedit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 同时写入实时翻译结果到右侧窗口（如果有）
        if self.current_realtime_translation.strip():
            translation_text = f"{timestamp} {self.current_realtime_translation}"
            self.translation_textedit.appendPlainText(translation_text)
            scrollbar = self.translation_textedit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            # 清空当前实时翻译
            self.current_realtime_translation = ""
        
        # 如果启用了自动翻译，翻译当前段落（这会替换实时翻译结果）
        if self.auto_translate_checkbox.isChecked():
            self.start_translation(text, is_auto=True)

    def export_txt(self):
        text=self.textedit.toPlainText().strip()
        if not text:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存原文", "", "Text files (*.txt)")
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)

    def export_translation(self):
        text = self.translation_textedit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, '提示', '没有可导出的译文')
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存译文", "", "Text files (*.txt)")
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)

    def copy_textedit(self):
        text = self.textedit.toPlainText()
        QApplication.clipboard().setText(text)

    def generate_meeting_minutes(self):
        """生成会议纪要"""
        raw_text = self.textedit.toPlainText().strip()
        original_text = re.sub(r'\[\d{2}:\d{2}:\d{2}\] ', '', raw_text)
        
        if not original_text:
            QMessageBox.warning(self, '提示', '没有可生成会议纪要的内容')
            return
        
        if not self.config.get('api_key'):
            QMessageBox.warning(self, '配置错误', f'请先在 {CONFIG_FILE} 中配置 API 密钥（api_key）')
            return
        
        if self.is_generating_minutes:
            return
        
        self.is_generating_minutes = True
        target_language = self.minutes_lang_combo.currentText()
        
        self.generate_minutes_button.setEnabled(False)
        self.generate_minutes_button.setText('生成中...')
        self.minutes_textedit.setPlainText('正在生成会议纪要，请稍候...')
        
        self.minutes_worker = MeetingMinutesWorker(original_text, target_language, self.config)
        self.minutes_worker.minutes_ready.connect(self.on_minutes_ready)
        self.minutes_worker.minutes_error.connect(self.on_minutes_error)
        self.minutes_worker.start()
    
    def on_minutes_ready(self, minutes):
        """会议纪要生成完成"""
        self.minutes_textedit.setPlainText(minutes)
        
        scrollbar = self.minutes_textedit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.generate_minutes_button.setEnabled(True)
        self.generate_minutes_button.setText('生成会议纪要')
        self.is_generating_minutes = False
    
    def on_minutes_error(self, error_msg):
        """会议纪要生成失败"""
        print(f'会议纪要生成失败：{error_msg}')
        self.minutes_textedit.setPlainText(f'生成失败：\n{error_msg}')
        QMessageBox.warning(self, '生成错误', f'会议纪要生成失败：\n{error_msg}')
        
        self.generate_minutes_button.setEnabled(True)
        self.generate_minutes_button.setText('生成会议纪要')
        self.is_generating_minutes = False
    
    def export_minutes(self):
        """导出会议纪要"""
        text = self.minutes_textedit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, '提示', '没有可导出的会议纪要')
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存会议纪要", "", "Text files (*.txt)")
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)
    
    def clear_textedit(self):
        # 自动导出临时文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exported_files = []
        
        # 导出原文
        original_text = self.textedit.toPlainText().strip()
        if original_text:
            filename = f"{OUT_DIR}/{timestamp}-原文-tmp.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(original_text)
            exported_files.append(f"原文: {filename}")
        
        # 导出译文
        translation_text = self.translation_textedit.toPlainText().strip()
        if translation_text:
            filename = f"{OUT_DIR}/{timestamp}-译文-tmp.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(translation_text)
            exported_files.append(f"译文: {filename}")
        
        # 导出会议纪要
        minutes_text = self.minutes_textedit.toPlainText().strip()
        if minutes_text:
            filename = f"{OUT_DIR}/{timestamp}-会议纪要-tmp.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(minutes_text)
            exported_files.append(f"会议纪要: {filename}")
        
        # 如果有导出文件，显示提示
        if exported_files:
            print(f"已自动导出临时文件:")
            for file in exported_files:
                print(f"  - {file}")
        
        # 清空所有窗口
        self.textedit.clear()
        self.translation_textedit.clear()
        self.minutes_textedit.clear()
        self.last_translated_text = ""
        self.translation_queue.clear()
        self.recent_segments.clear()
        self.current_realtime_translation = ""

    def on_language_changed(self, index):
        if self.realtime_translation_worker:
            self.realtime_translation_worker.set_target_language(self.target_lang_combo.currentText())

    def _get_first_mic_idx(self):
        for i in range(self.combo.count()):
            data = self.combo.itemData(i)
            if isinstance(data, tuple) and data[0] == 'mic':
                return data[1]
        return 0
    
    def _get_first_system_idx(self):
        for i in range(self.combo.count()):
            data = self.combo.itemData(i)
            if isinstance(data, tuple) and data[0] == 'system':
                return data[1]
        return None
    
    def closeEvent(self, event: QCloseEvent):
        if self.transcribing:
            self.toggle_transcription()
        if self.translation_worker and self.translation_worker.isRunning():
            self.translation_worker.wait()
        if self.realtime_translation_worker and self.realtime_translation_worker.isRunning():
            self.realtime_translation_worker.stop()
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.wait()
        if self.minutes_worker and self.minutes_worker.isRunning():
            self.minutes_worker.wait()
        super().closeEvent(event)
