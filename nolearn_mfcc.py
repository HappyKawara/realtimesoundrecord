import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import wave
import time
import threading
import whisper
import librosa
import soundfile as sf
from sklearn import svm
import pickle

device_list = sd.query_devices()
print(device_list)

save_data = []
i = 0
x = 0
t = -21
tim = 0
tim_sta = 0
num = 0
mfcc_dic = {}
ls = [10]
mfccs_t = [np.full(12,0)]
#sd.default.device = [12, 12] # Input, Outputデバイス指定

def callback(indata, frames, ti, status):
    # indata.shape=(n_samples, n_channels)
    global plotdata
    global save_data
    global i
    global tim
    global time_sta
    global x
    global t
    global num
    data = indata[::downsample, 0]
    save_data =  np.append(save_data,data)
    shift = len(data)
    plotdata = np.roll(plotdata, -shift, axis=0)
    plotdata[-shift:] = data
    time_end = time.time()
    #print((time_end-time_sta)%1)
    if (time_end - time_sta) > 0.2 * (i+1):
        avg = np.mean(np.abs(plotdata[:-1000]))
        if avg < 0.3:
            i = i + 1
            print(i,avg,data.shape)
        else:
            print("listen",avg)
            i = i + 1
            t = i
            #x = x + 1
        if (i - t) == 5:# and x > 1:
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
            #thread.join()
            t = 0
            save_data = []
            num = num + 1


    #if data[-1] <
    #print(save_data.shape)
def func():
    global num
    x, fs = sf.read('./wav_file/test1.wav')#BASIC5000_0641.wav')#test1.wav')
    mfccs = librosa.feature.mfcc(y = x, sr=fs,n_mfcc=12,dct_type=3)
    with open('sample_test.pkl','rb') as f:
        classifier = pickle.load(f)
    data = classifier.predict(mfccs.T)
    print(data)
    ls = [0,0,0,0,0]
    for d in data:
        ls[d] += 1
    if ls[1] > ((ls[2] + ls[3]) or 5):
        print(1)
    if ls[2] > ((ls[1] + ls[3]) or 5):
        print(2)
    elif ls[3] > ((ls[1] + ls[2]) or 5):
        print(3)
    else:
        print(False)

'''
        i = 0
        time_sta = time.time()
    if i == 500:
        # 時間計測終了
        time_end = time.time()
        # 経過時間（秒）
        tim = time_end - time_sta
        print(tim)
        sys.exit()
'''
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
time_sta = time.time()
with sd.InputStream(
        channels=1,
        dtype='float32',
        callback=callback
        ):
    sd.sleep(int(1000 * 1000))
#ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

#with stream:
    #plt.show()
