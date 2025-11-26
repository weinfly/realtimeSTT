"""
音频处理模块
处理音频输入流的创建和设备管理
"""

import sounddevice as sd

def create_input_stream_safe(device_idx, sample_rate, channels=1):
    try:
        stream = sd.InputStream(device=device_idx, channels=channels, dtype="float32", samplerate=sample_rate)
        stream.start()
        stream.stop()
        return stream
    except sd.PortAudioError as e:
        if "WDM-KS" in str(e) or "-9999" in str(e):
            print(f"警告：设备 {device_idx} 不支持当前API，尝试使用其他设置...")
            try:
                stream = sd.InputStream(device=device_idx, channels=channels, dtype="float32", samplerate=sample_rate, latency='high')
                stream.start()
                stream.stop()
                return stream
            except:
                print(f"错误：无法打开设备 {device_idx}")
                return None
        else:
            raise
