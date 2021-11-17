import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
v_ref = 13e-9
p_ref = 2e-5
audio = None

# use a Blackman window
WINDOW = np.blackman(CHUNK)


def calc_spl(data):
    spl_data = 20 * np.log(abs(np.fft.fft(np.divide(data, p_ref))) / RATE)
    return spl_data


def capture():
    global audio
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=0)

    print("recording...")
    frames = []
    plot_data = []
    thing = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        np_data = np.fromstring(data, dtype=np.int16)
        windowed_data = WINDOW * np_data
        frames.append(data)
        thing.append(str(data))
        plot_data.extend(windowed_data)

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("finished recording")

    spl_data = calc_spl(plot_data)
    plt.plot(spl_data)
    plt.xscale('log')
    plt.xlim(1, RATE / 2)
    plt.show()

    with open('output.txt', 'w+') as file:
        lines = []
        for datum in spl_data:
            lines.append(str(f'{datum}\n'))
        file.writelines(lines)

    # thing = ''.join(thing)
    # fig = plt.figure()
    # s = fig.add_subplot(111)
    # amplitude = np.fromstring(thing, np.int16)
    # s.plot(amplitude)
    # plt.show()

    # spl_data = spl_data[3000:]
    peaks, _ = signal.find_peaks(spl_data, height=(250, 300))
    plt.plot(spl_data)
    plt.xscale('log')
    plt.xlim(1, RATE / 2)
    plt.plot(peaks, spl_data[peaks], "x")
    plt.plot(np.zeros_like(spl_data), "--", color="gray")
    plt.show()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == '__main__':
    capture()
