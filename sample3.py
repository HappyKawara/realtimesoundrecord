import sounddevice as sd
import numpy as np

sd.default.device = [1, 6] # Input, Outputデバイス指定
duration = 10  # 10秒間収音する

def callback(indata, frames, time, status):
    # indata.shape=(n_samples, n_channels)
    # print root mean square in the current frame
    print(indata)

with sd.InputStream(
        channels=1,
        dtype='float32',
        callback=callback
    ):
    sd.sleep(int(duration * 1000))
