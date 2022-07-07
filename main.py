'''
Guitar tuner script based on the Harmonic Product Spectrum (HPS)

MIT License
Copyright (c) 2021 chciken
'''

import copy
import os
import sys
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import threading
from playsound import playsound
import glob
import random
import PySimpleGUI as sg


drone_choice = random.choice(glob.glob(u"C:\Program Files\DroneIntervalTraining\Drone_wave_files\*.wav"))
interval = ""

def loopSound():
  while True:
     playsound(drone_choice)

# providing a name for the thread improves usefulness of error messages.
loopThread = threading.Thread(target=loopSound, name='backgroundMusicThread')
loopThread.daemon = True  # shut down music thread when the rest of the program exits
loopThread.start()
# General settings that can be changed by the user
SAMPLE_FREQ = 48000 # sample frequency in Hz
WINDOW_SIZE = 48000 # window size of the DFT in samples
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
CONCERT_PITCH = 440 # defining a1
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A_","A#","B_","C_","C#","D_","D#","E_","F_","F#","G_","G#"]
def find_closest_note(pitch):
  """
  This function finds the closest note for a given pitch
  Parameters:
    pitch (float): pitch given in hertz
  Returns:
    closest_note (str): e.g. a, g#, ..
    closest_pitch (float): pitch of the closest note in hertz
  """
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)
def callback(indata, frames, time, status):
  """
  Callback function of the InputStream method.
  That's where the magic happens ;)
  """
  # define static variables
  if not hasattr(callback, "window_samples"):
    callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
  if not hasattr(callback, "noteBuffer"):
    callback.noteBuffer = ["1","2"]

  if status:
    print(status)
    return
  if any(indata):
    callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0])) # append new samples
    callback.window_samples = callback.window_samples[len(indata[:, 0]):] # remove old samples

    # skip if signal power is too low
    signal_power = (np.linalg.norm(callback.window_samples, ord=2)**2) / len(callback.window_samples)
    if signal_power < POWER_THRESH:
      os.system('cls' if os.name=='nt' else 'clear')
      print(f"Drone {drone_choice[-6:]}")
      return

    # avoid spectral leakage by multiplying the signal with a hann window
    hann_samples = callback.window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

    # supress mains hum, set everything below 62Hz to zero
    for i in range(int(62/DELTA_FREQ)):
      magnitude_spec[i] = 0

    # calculate average energy per frequency for the octave bands
    # and suppress everything below it
    for j in range(len(OCTAVE_BANDS)-1):
      ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
      ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
      ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
      avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end-ind_start)
      avg_energy_per_freq = avg_energy_per_freq**0.5
      for i in range(ind_start, ind_end):
        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH*avg_energy_per_freq else 0

    # interpolate spectrum
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2) #normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculate the HPS
    for i in range(NUM_HPS):
      tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
      if not any(tmp_hps_spec):
        break
      hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)
    Drone_Number = 0
    Pitch_Number = 0

    if "C_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 1
    elif "C#.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 2
    elif "D_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 3
    elif "D#.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 4
    elif "E_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 5
    elif "F_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 6
    elif "F#.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 7
    elif "G_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 8
    elif "G#.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 9
    elif "A_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 10
    elif "A#.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 11
    elif  "B_.wav" in {drone_choice[-6:]}:
        Drone_Number = Drone_Number + 12
    else : ""
    #print(Drone_Number)
    if "C_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 1
    elif "C#" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 2
    elif "D_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 3
    elif "D#" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 4
    elif "E_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 5
    elif "F_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 6
    elif "F#" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 7
    elif "G_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 8
    elif "G#" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 9
    elif "A_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 10
    elif "A#" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 11
    elif "B_" in {closest_note[:2]}:
        Pitch_Number = Pitch_Number + 12
    else : ""



    if Pitch_Number < Drone_Number :
      Pitch_Number = Pitch_Number + 12
    else : ""

    n = Pitch_Number - Drone_Number
    if n == 0:
      interval = "1    (1:1)"
    elif n == 1:
      interval = "b2   (16:15)"
    elif n == 2:
      interval = "2    (9:8)"
    elif n == 3:
      interval = "b3   (6:5)"
    elif n == 4:
      interval = "3    (5:4)"
    elif n == 5:
      interval = "4    (4:3)"
    elif n == 6:
      interval = "b5   (7:5)"
    elif n == 7:
      interval = "5    (3:2)"
    elif n == 8:
      interval = "b6   (8:5)"
    elif n == 9:
      interval = "6    (5:3)"
    elif n == 10:
      interval = "b7   (7:4)"
    elif n == 11:
      interval = "7    (15:8)"
    else : ""

    callback.noteBuffer.insert(0, closest_note) # note that this is a ringbuffer
    callback.noteBuffer.pop()

    os.system('cls' if os.name=='nt' else 'clear')
    if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
      #print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
      window['-OUT-'].update(interval)
 #     print(interval)
 #   else:
 #     print(f"Drone {drone_choice[-6:]}")

  else:
    print('no input')

"""
def Interval_Answer():
    if True:
        sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ),
        while True:
                time.sleep(0.5)


loopAnswer = threading.Thread(target=Interval_Answer, name='testing')
loopAnswer.daemon = True
loopAnswer.start()
"""

Key_Font = ("Arial", 30)
Interval_Font = ("Arial", 90)
layout = [
  [sg.Text("Key of " + drone_choice[-6:-4], font=Key_Font)],
  [sg.Text(interval, size=(20,1), key='-OUT-', font=Interval_Font, text_color="Black", justification="center")],
    [sg.Button("Reroll?")],
  ]

window = sg.Window("Drone Interval reader", layout, size=(800,800))


while True:
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            event, values = window.read()
            window['-OUT-'].update(interval)
            time.sleep(0.5)
            if event == sg.WIN_CLOSED:
                sys.exit()
            if event == sg.Button:
                sys.stdout.flush()
                os.execl(sys.executable, 'python', __file__, *sys.argv[1:])






"""
try:
    print("Drone and Interval reader")
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))
"""