# Not tested since I dont have access to a microphone
emotion = input("Enter emotion(will be used as filename): ")
import sounddevice as sd
from scipy.io.wavfile import write

freq = 44100
duration = 5

recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
sd.wait()

write("{emotion}.wav", freq, recording)