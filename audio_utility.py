
import io ,os
import numpy as np


from io import BytesIO

from pydantic import BaseModel

from uuid import UUID, uuid4
from typing import Dict
from pydub import AudioSegment

# 1 - Import library
from pydub import AudioSegment, silence

import whisper
# from audiorecorder import audiorecorder
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

# import pygame
import wave
from google.cloud import texttospeech
from pydub.playback import play

import shutil
from pathlib import Path,PosixPath

# from decouple import config
# from gtts import gTTS
# import winreg
# import win32com.client
# import pythoncom
# import pyttsx3
from dotenv import load_dotenv
from pydub.playback import play
from dotenv import load_dotenv

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import traceback
# import utility

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

from faster_whisper import WhisperModel

import speech_recognition as sr
import soundfile as sf
import fleep


SILENCE_DELETE_SEC=1

def get_audio_record_format(filename):
    with open(filename, "rb") as file:
        info = fleep.get(file.read(128))
        if info.extension:
            file_format = info.extension[0]
        else:
            file_format = None # or any other default value
        return file_format
    
# convert unsupported formats to wav
def check_and_convert(filename, supported_formats=['wav', 'aiff', 'aifc', 'flac']):
    # Check the file format
    # with open(filename, "rb") as file:
    #     info = fleep.get(file.read(128))
    #     if info.extension:
    #         file_format = info.extension[0]
    #     else:
    #         file_format = None # or any other default value
    file_format=get_audio_record_format(filename)
    # print("file_format",file_format)
    # If the file format is not supported, convert it
    if file_format not in supported_formats:
        
        audio = AudioSegment.from_file(filename, format=file_format)
        # if isinstance(filename, PosixPath):
        #     # filename = Path(filename)
        #     filename=filename.as_uri
        
        # new_file_name = filename / 'wav'
        new_file_name=str(filename)+'.wav'
        # new_file_name="output.wav"
        audio.export(new_file_name, format='wav')
        return new_file_name
    else:
        return filename

# @staticmethod
# def segment(file_path,segment_file_path,minuts_from=0,minutes_to=10):
#     song = AudioSegment.from_mp3(file_path)

#     # PyDub handles time in milliseconds
#     time_from = minuts_from * 60 * 1000
#     time_to = minutes_to * 60 * 1000

#     all_what_you_need_minutes = song[time_from:time_to]

#     all_what_you_need_minutes.export(segment_file_path, format="mp3")
    

def save_wav_from_byteio(filename, audio_data, sample_rate= 16000, num_channels=1, byte_width=2):
    # Create a wave file
    try:
        # Create a wave file
        with wave.open(filename, 'wb') as wav_file:
            # Set the parameters of the wave file
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(byte_width)
            wav_file.setframerate(sample_rate)

            # Write the audio data to the wave file
            wav_file.writeframes(audio_data)

        # Return the absolute path of the saved file
        return True, os.path.abspath(filename)
    except Exception as e:
        print(f"Failed to save audio: {e}")
        return False, None
    

def save_audio_from_audio_data(audio:sr.AudioData,file_path):
    try:
        with open(file_path, 'wb') as file:
            wav_data = audio.get_wav_data()
            file.write(wav_data)
            file.close()
            return True, os.path.abspath(file_path)
    except Exception as e:
        print(f"Failed to save audio: {e}")
        return False, None


def get_wisper_audio_array_from_file(audio_file_path) : 
    audio = Path(audio_file_path)
    audio = whisper.load_audio(audio)
    # print(audio)
    audio = whisper.pad_or_trim(audio)
    # print(audio)
    return audio


def get_wisper_audio_array_from_audio_data(audio_data:sr.AudioData) : 
    wav_bytes = audio_data.get_wav_data(convert_rate=16000)
    
    audio = np.frombuffer(wav_bytes, np.int16).flatten().astype(np.float32) / 32768.0
    # audio_array = whisper.pad_or_trim(audio) 
    # print(audio)
    audio = whisper.pad_or_trim(audio)
    # print(audio)
    return audio


def get_audio_data_from_wav_data(wav_data) :
    segment =AudioSegment.from_wav(BytesIO(wav_data))
    audio_data = sr.AudioData(segment.raw_data, segment.frame_rate,
                            segment.sample_width)
    return audio_data



def get_wav_data_from_audio_data(audio:sr.AudioData,convert_rate=None, convert_width=None) :
    return audio.get_wav_data(convert_rate=convert_rate, convert_width=convert_width)

def audio_to_audioAray(audio:sr.AudioData):
    wav_bytes = audio.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    return audio_array


def load_audioSource_from_file(file_path):
    segment =AudioSegment.from_file(file_path)
    audio_data = sr.AudioData(segment.raw_data, segment.frame_rate,
                            segment.sample_width)
    return audio_data
    
def load_audioSource_from_file2(file_path):
    r=sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.record(source)
        return audio
    
def load_byteio_from_file(file_path):
    audio=AudioSegment.from_file(file_path)
    bytes_audio = BytesIO()
    audio.export(bytes_audio, format="wav")
    return bytes_audio
    
    r=sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.record(source)
        return audio

def getSilenceStartStopTime(audio):
        silenc = silence.detect_nonsilent(audio, min_silence_len=250, silence_thresh=-50,seek_step=125)
        # silenc = [((start/1000),(stop/1000)) for start,stop in silenc]
        # silenc = [((start),(stop)) for start,stop in silenc]
        silenc = [(start, stop) for start, stop in silenc if stop - start >= SILENCE_DELETE_SEC*1000]
        print(silenc)
        return silenc