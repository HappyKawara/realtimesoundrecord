import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import wave
import whisper
import threading
import time

device_list = sd.query_devices()
print(device_list)

save_data = []
i = 0
x = 0
#sd.default.device = [12, 12] # Input, Outputデバイス指定

def callback(indata, frames, ti, status):
    time.sleep(0.01)
    # indata.shape=(n_samples, n_channels)
    global plotdata
    global save_data
    global i
    global x
    data = indata[::downsample, 0]
    save_data =  np.append(save_data,data)
    shift = len(data)
    plotdata = np.roll(plotdata, -shift, axis=0)
    plotdata[-shift:] = data
    avg = np.mean(np.abs(plotdata[:1000]))
    if avg < 0.05:
        i = i + 1
        print(i,avg)
    else:
        print("listen",avg)
        i = 0
        x = x + 1
    if i == 200 and x > 10:
        print("save")
        save_data = save_data / save_data.max() * np.iinfo(np.int16).max
        save_data = save_data.astype(np.int16)
        with wave.open('./wav_file/test1.wav', mode='w') as wb:
            wb.setnchannels(1)  # モノラル
            wb.setsampwidth(2)  # 16bit=2byte
            wb.setframerate(44100)
            wb.writeframes(save_data.tobytes())  # バイト列に変換
        thread = threading.Thread(target=func)
        thread.start()
        thread.join()
        i = 0
        save_data = []


    #if data[-1] <
    #print(save_data.shape)
def func():
    model = whisper.load_model(name='small')#,in_memory=True)
    result = model.transcribe('./wav_file/test1.wav', verbose=False, language="ja")
    print(result["text"])



def update_plot(frame):
    """This is called by matplotlib for each plot update.
    """
    global plotdata
    line.set_ydata(plotdata)
    return line,

downsample = 1
length = int(1000 * 44100 / (1000 * downsample))
plotdata = np.zeros((length))

fig, ax = plt.subplots()
line, = ax.plot(plotdata)
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([0, length])
ax.yaxis.grid(True)
fig.tight_layout()

stream = sd.InputStream(
        channels=1,
        dtype='float32',
        callback=callback
    )
ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

with stream:
    plt.show()
