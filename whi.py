import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import wave
import time
import threading
import whisper


device_list = sd.query_devices()
print(device_list)


class Recod():
    def __init__(self):
        self.save_data = []
        self.i = 0
        self.x = 0
        self.t = -21
        self.tim = 0
        self.tim_sta = 0
        #sd.default.device = [12, 12] # Input, Outputデバイス指定



    def callback(self,indata,frames,ti,status):
        # indata.shape=(n_samples, n_channels)
        data = indata[::self.downsample, 0]
        self.save_data =  np.append(self.save_data,data)
        shift = len(data)
        self.plotdata = np.roll(self.plotdata, -shift, axis=0)
        self.plotdata[-shift:] = data
        self.time_end = time.time()
        #print((self.time_end-self.time_sta)%1)
        if (self.time_end - self.time_sta) > 0.2 * (self.i+1):
            avg = np.mean(np.abs(self.plotdata[:-1000]))
            if avg < 0.3:
                self.i = self.i + 1
                print(self.i,avg,data.shape)
            else:
                print("listen",avg)
                self.i = self.i + 1
                self.t = self.i
                #x = x + 1
            if (self.i - self.t) == 20:# and x > 1:
                print("save")
                self.save_data = self.save_data / self.save_data.max() * np.iinfo(np.int16).max
                self.save_data = self.save_data.astype(np.int16)
                with wave.open('./wav_file/test1.wav', mode='w') as wb:
                    wb.setnchannels(1)  # モノラル
                    wb.setsampwidth(2)  # 16bit=2byte
                    wb.setframerate(44100)
                    wb.writeframes(self.save_data.tobytes())  # バイト列に変換
                thread = threading.Thread(target=func)
                thread.start()
                thread.join()
                self.t = 0
                self.save_data = []


        #if data[-1] <
        #print(self.save_data.shape)
    def func(self):
        model = whisper.load_model(name='small',in_memory=True)
        result = model.transcribe('./wav_file/test1.wav', verbose=False, language="ja")
        print(result["text"])

    '''
            i = 0
            self.time_sta = time.time()
        if i == 500:
            # 時間計測終了
            self.time_end = time.time()
            # 経過時間（秒）
            tim = self.time_end - self.time_sta
            print(tim)
            sys.exit()
    '''
    def update_plot(self,frame):
        """This is called by matplotlib for each plot update.
        """
        self.plotdata
        self.line.set_ydata(self.plotdata)
        return self.line,

    def main(self):
        self.downsample = 1
        length = int(1000 * 44100 / (1000 * self.downsample))
        self.plotdata = np.zeros((length))

        fig, ax = plt.subplots()
        self.line, = ax.plot(self.plotdata)
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlim([0, length])
        ax.yaxis.grid(True)
        fig.tight_layout()
        self.time_sta = time.time()
        with sd.InputStream(
                channels=1,
                dtype='float32',
                callback=self.callback
                ):
            sd.sleep(int(1000 * 1000))
#ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

#with stream:
            #plt.show()
if __name__ == '__main__':
    Recod().main()
    print("a")
