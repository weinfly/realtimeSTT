"""
工作线程模块 - 备份版本
包含所有后台处理线程：语音识别、翻译、优化和会议纪要生成
"""

import time
import threading
import requests
import wave
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from .config import AudioSource, OUT_DIR
from .models import OnnxModel, create_recognizer
from .audio import create_input_stream_safe


class TranslationWorker(QThread):
    translation_ready = Signal(str)
    translation_error = Signal(str)
    
    def __init__(self, text, target_language, config, parent=None):
        super().__init__(parent)
        self.text = text
        self.target_language = target_language
        self.config = config
    
    def detect_language(self, text):
        """简单的语言检测：基于字符判断是中文还是英文"""
        if not text:
            return "unknown"
        
        # 统计中文字符和英文字符
        chinese_chars = 0
        english_chars = 0
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                chinese_chars += 1
            elif 'a' <= char.lower() <= 'z':  # 英文字符
                english_chars += 1
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return "unknown"
        
        # 如果中文字符占比超过30%，认为是中文
        if chinese_chars / total_chars > 0.3:
            return "中文"
        # 如果英文字符占比超过50%，认为是英文
        elif english_chars / total_chars > 0.5:
            return "英语"
        else:
            return "unknown"
    
    def run(self):
        try:
            # 检测源语言
            detected_lang = self.detect_language(self.text)
            
            # 如果检测到的语言与目标语言相同，跳过翻译
            if detected_lang == self.target_language:
                self.translation_ready.emit(f"[提示] 原文已经是{self.target_language}，无需翻译")
                return
            
            api_base_url = self.config.get('api_base_url', 'http://192.68.11.84:11434/v1')
            api_key = self.config.get('api_key', 'ollama')
            model = self.config.get('model', 'translate')
            timeout = self.config.get('timeout', 30)
            
            if not api_key:
                self.translation_error.emit('错误：未配置 API 密钥')
                return
            
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': f'你是一个专业的翻译助手，请将用户提供的文本翻译成{self.target_language}。只返回翻译结果，不要添加任何解释。'},
                    {'role': 'user', 'content': self.text}
                ]
            }
            
            url = f"{api_base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                translation = result['choices'][0]['message']['content']
                self.translation_ready.emit(translation)
            else:
                error_msg = f'API 错误 ({response.status_code}): {response.text}'
                self.translation_error.emit(error_msg)
        except requests.exceptions.Timeout:
            self.translation_error.emit('错误：请求超时')
        except requests.exceptions.ConnectionError:
            self.translation_error.emit('错误：无法连接到 API 服务器')
        except Exception as e:
            self.translation_error.emit(f'翻译错误: {str(e)}')


class OptimizationWorker(QThread):
    """智能优化Worker - 使用LLM根据上下文优化识别文本"""
    optimization_ready = Signal(str)
    optimization_error = Signal(str)
    
    def __init__(self, text, context, config, parent=None):
        super().__init__(parent)
        self.text = text
        self.context = context
        self.config = config
    
    def run(self):
        try:
            api_base_url = self.config.get('api_base_url', 'http://192.68.11.84:11434/v1')
            api_key = self.config.get('api_key', 'ollama')
            model = self.config.get('model', 'translate')
            timeout = self.config.get('timeout', 30)
            
            if not api_key:
                self.optimization_error.emit('未配置API密钥')
                return
            
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
            
            prompt = f"""你是一个语音识别优化助手。请根据上下文修正识别错误，使文本更通顺准确。

上下文：{self.context}

当前识别文本：{self.text}

请只返回优化后的文本，不要添加任何解释或标点。保持原意，只修正明显的识别错误。"""
            
            data = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3
            }
            
            url = f"{api_base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                optimized_text = result['choices'][0]['message']['content'].strip()
                self.optimization_ready.emit(optimized_text)
            else:
                self.optimization_error.emit(f'API错误: {response.status_code}')
        except Exception as e:
            self.optimization_error.emit(f'优化失败: {str(e)}')


class MeetingMinutesWorker(QThread):
    """会议纪要生成Worker - 使用LLM生成结构化的会议纪要"""
    minutes_ready = Signal(str)
    minutes_error = Signal(str)
    
    def __init__(self, text, target_language, config, parent=None):
        super().__init__(parent)
        self.text = text
        self.target_language = target_language
        self.config = config
    
    def run(self):
        try:
            api_base_url = self.config.get('api_base_url', 'http://192.68.11.84:11434/v1')
            api_key = self.config.get('api_key', 'ollama')
            model = self.config.get('model', 'translate')
            timeout = self.config.get('timeout', 60)
            
            if not api_key:
                self.minutes_error.emit('未配置API密钥')
                return
            
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
            
            prompt = f"""你是一个专业的会议记录助手。请根据以下会议内容生成结构化的会议纪要。

会议内容：
{self.text}

请用{self.target_language}生成会议纪要，包含以下部分：
1. 会议概要
2. 讨论要点
3. 待办事项（包括负责人和截止日期）
4. 决策事项

格式要求：
- 使用清晰的标题和列表
- 待办事项格式：【待办】任务描述 | 负责人：XXX | 截止日期：YYYY-MM-DD
- 如果内容中没有明确的负责人或截止日期，请标注为"待定"
"""
            
            data = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3
            }
            
            url = f"{api_base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                minutes = result['choices'][0]['message']['content'].strip()
                self.minutes_ready.emit(minutes)
            else:
                self.minutes_error.emit(f'API错误: {response.status_code}')
        except Exception as e:
            self.minutes_error.emit(f'生成失败: {str(e)}')


class RealtimeTranslationWorker(QThread):
    translation_ready = Signal(str)
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_text = ""
        self.last_translated_text = ""
        self.running = True
        self.target_language = "中文"
        self._lock = threading.Lock()
        
    def update_text(self, text):
        with self._lock:
            self.current_text = text
            
    def set_target_language(self, lang):
        self.target_language = lang
        
    def run(self):
        while self.running:
            text_to_translate = ""
            with self._lock:
                if self.current_text != self.last_translated_text and self.current_text.strip():
                    text_to_translate = self.current_text
            
            if text_to_translate:
                try:
                    api_base_url = self.config.get('api_base_url', 'http://192.68.11.84:11434/v1')
                    api_key = self.config.get('api_key', '')
                    model = self.config.get('model', 'translate')
                    
                    if api_key:
                        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
                        data = {
                            'model': model,
                            'messages': [
                                {'role': 'system', 'content': f'Translate to {self.target_language}. Concise.'},
                                {'role': 'user', 'content': text_to_translate}
                            ],
                            'max_tokens': 100
                        }
                        
                        url = f"{api_base_url.rstrip('/')}/chat/completions"
                        response = requests.post(url, headers=headers, json=data, timeout=5)
                        
                        if response.status_code == 200:
                            result = response.json()
                            translation = result['choices'][0]['message']['content']
                            self.translation_ready.emit(translation)
                            self.last_translated_text = text_to_translate
                except Exception as e:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.running = False
        self.wait()


class Worker(QThread):
    new_word = Signal(str)
    new_segment = Signal(str)
    ready = Signal()

    def __init__(self, device_idx, audio_mode=AudioSource.MICROPHONE, mic_idx=None, sys_idx=None, parent=None):
        super().__init__(parent)
        self.device_idx = device_idx
        self.audio_mode = audio_mode
        self.mic_idx = mic_idx if mic_idx is not None else device_idx
        self.sys_idx = sys_idx
        self.running = False
        self.sample_rate = 48000
        self.samples_per_read = int(0.1 * self.sample_rate)

    def commit_sentence(self, text, txt_file):
        """提交句子到文件和UI"""
        try:
            punctuated = PUNCT_MODEL(text)
            print(f"[DEBUG] Punctuated: '{punctuated}'")
            
            # 确保文本以换行符结尾，便于阅读
            if punctuated and not punctuated.endswith('\n'):
                punctuated += '\n'
            
            txt_file.write(punctuated)
            txt_file.flush()  # 立即写入文件
            self.new_segment.emit(punctuated.strip())  # 发送给UI时去掉换行符
        except Exception as e:
            print(f"[DEBUG] Punctuation model failed: {e}, using original text")
            # 如果标点符号模型失败，直接使用原文
            punctuated = text + '\n'
            txt_file.write(punctuated)
            txt_file.flush()
            self.new_segment.emit(punctuated.strip())

    def run(self):
        devices = sd.query_devices()
        if len(devices) == 0:
            return

        if self.audio_mode == AudioSource.MICROPHONE:
            print(f'使用麦克风: {devices[self.device_idx]["name"]}')
        elif self.audio_mode == AudioSource.SYSTEM:
            if self.sys_idx is not None:
                print(f'使用系统声音: {devices[self.sys_idx]["name"]}')
        elif self.audio_mode == AudioSource.BOTH:
            print(f'使用麦克风: {devices[self.mic_idx]["name"]}')
            if self.sys_idx is not None:
                print(f'使用系统声音: {devices[self.sys_idx]["name"]}')
        
        PUNCT_MODEL = OnnxModel()
        recognizer = create_recognizer()
        stream = recognizer.create_stream()

        mic_stream = None
        sys_stream = None
        
        if self.audio_mode == AudioSource.MICROPHONE:
            mic_stream = create_input_stream_safe(self.mic_idx, self.sample_rate)
            if mic_stream:
                mic_stream.start()
        elif self.audio_mode == AudioSource.SYSTEM:
            if self.sys_idx is not None:
                sys_stream = create_input_stream_safe(self.sys_idx, self.sample_rate)
                if sys_stream:
                    sys_stream.start()
        elif self.audio_mode == AudioSource.BOTH:
            if self.mic_idx is not None:
                mic_stream = create_input_stream_safe(self.mic_idx, self.sample_rate)
                if mic_stream:
                    mic_stream.start()
            if self.sys_idx is not None and self.sys_idx != self.mic_idx:
                sys_stream = create_input_stream_safe(self.sys_idx, self.sample_rate)
                if sys_stream:
                    sys_stream.start()
            elif self.sys_idx is None:
                print("警告：未找到系统音频设备，仅使用麦克风")
                
        timestamp = time.strftime("%Y%m%d_%H-%M-%S")
        txt_file = open(f"{OUT_DIR}/{timestamp}.txt", 'a')
        wav_file = wave.open(f"{OUT_DIR}/{timestamp}.wav", 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.sample_rate)

        self.ready.emit()
        self.running = True
        last_result = ""
        current_text = ""  # 当前累积的文本
        
        # 极简的句子检测参数
        frames_since_commit = 0  # 距离上次提交的帧数
        min_frames_between_commits = 30  # 最小间隔3秒
        
        print(f"[DEBUG] Worker started with minimal sentence detection")
        print(f"[DEBUG] Minimum frames between commits: {min_frames_between_commits}")
        print(f"[DEBUG] Will commit when text is stable and pauses occur")
        
        while self.running:
            if self.audio_mode == AudioSource.MICROPHONE:
                if mic_stream is not None:
                    samples, _ = mic_stream.read(self.samples_per_read)
                else:
                    break
            elif self.audio_mode == AudioSource.SYSTEM:
                if sys_stream is not None:
                    samples, _ = sys_stream.read(self.samples_per_read)
                else:
                    break
            elif self.audio_mode == AudioSource.BOTH:
                if mic_stream is not None and sys_stream is not None:
                    mic_samples, _ = mic_stream.read(self.samples_per_read)
                    sys_samples, _ = sys_stream.read(self.samples_per_read)
                    samples = (mic_samples + sys_samples) / 2.0
                elif mic_stream is not None:
                    samples, _ = mic_stream.read(self.samples_per_read)
                elif sys_stream is not None:
                    samples, _ = sys_stream.read(self.samples_per_read)
                else:
                    break
            
            samples_int16 = (samples * 32767).astype(np.int16)
            wav_file.writeframes(samples_int16.tobytes())

            samples = samples.reshape(-1)
            stream.accept_waveform(self.sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            result = recognizer.get_result(stream)
            
            # 更新计数器
            frames_since_commit += 1
            
            # 极简逻辑：
            if result and len(result.strip()) > 0:
                # 有识别结果
                if result != last_result:
                    # 结果变化，更新显示
                    self.new_word.emit(result)
                    current_text = result
                    print(f"[DEBUG] Text updated: '{result}'")
            else:
                # 没有识别结果 - 这是提交的机会
                if current_text and len(current_text.strip()) > 0 and frames_since_commit >= min_frames_between_commits:
                    cleaned_text = current_text.replace('<unk>', '').strip()
                    if len(cleaned_text) >= 3:  # 至少3个字符
                        print(f"[DEBUG] === COMMITTING TEXT ===")
                        print(f"[DEBUG] Text: '{cleaned_text}', Frames since last: {frames_since_commit}")
                        self.commit_sentence(cleaned_text, txt_file)
                        
                        # 重置状态
                        recognizer.reset(stream)
                        current_text = ""
                        frames_since_commit = 0
                        print(f"[DEBUG] === RESET FOR NEXT SENTENCE ===")

            last_result = result

        if mic_stream is not None:
            mic_stream.stop()
        if sys_stream is not None:
            sys_stream.stop()
        wav_file.close()
        txt_file.close()