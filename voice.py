from abc import ABC, abstractmethod
import base64
import io ,time ,os
import pyaudio
import numpy as np

# import whisperx

from io import BytesIO

from pydantic import BaseModel

from uuid import UUID, uuid4
from typing import Dict
from pydub import AudioSegment

# 1 - Import library

import whisper
# from audiorecorder import audiorecorder
import magic
import re
from dotenv import load_dotenv
import shutil
from tempfile import NamedTemporaryFile
import subprocess
from gtts import gTTS
import pyttsx3
import objc
# import pygame
from pydub import AudioSegment

from pydub.playback import play

import shutil
from pathlib import Path
import voice

import openai
# from decouple import config
from gtts import gTTS
# import winreg
# import win32com.client
# import pythoncom
import pyttsx3
from dotenv import load_dotenv
from pydub.playback import play
from dotenv import load_dotenv

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import traceback
import utility

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

from faster_whisper import WhisperModel
import wave
import speech_recognition as sr
import soundfile as sf
from bidi.algorithm import get_display


load_dotenv()

openai.api_key =os.getenv('OPENAI_API_KEY')

# openai.api_key = config("OPENAI_API_KEY")

# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         import time
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
#         return result
#     return wrapper


class VoiceGenerator():
    def __init__(self, modelType=1):
        self.ttsVoices = {}
        for line in open('language-tts-voice-mapping.txt', 'rt').readlines():
            if len(line.strip().split(',')) == 3:
                language, langCode, voiceName = line.strip().split(',')
                self.ttsVoices[langCode.strip()] = voiceName.strip()
                
    def speak_to_file(self,text,speech_file_path=None,voice="alloy",model="tts-1"):
        if speech_file_path is None:
            speech_file_path = Path(__file__).parent / "speech.mp3"
        response = openai.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )

        response.stream_to_file(speech_file_path)
        
    def audiofile_as_stream(self,speech_file_path):
        
        with open(speech_file_path, "rb") as audio_file:
            audio_content = audio_file.read()
            audio_stream = io.BytesIO(audio_content)
    
        return audio_stream
    
    def audiofile_as_stream2(self,speech_file_path,format="mp3"):
        
       audio = AudioSegment.from_file(speech_file_path, format=format)
       return audio
       

    
    def audio_as_stream(self,text,voice="alloy",model="tts-1"):
        speech_file_path = Path(__file__).parent / "speech.mp3"
        self.speak_to_file(text,speech_file_path=speech_file_path,voice=voice,model=model)
        
        return self.audiofile_as_stream(speech_file_path)

    

            
    def speak(self,text,voice="alloy",model="tts-1"):
        speech_file_path = Path(__file__).parent / "speech.mp3"
        self.speak_to_file(text,speech_file_path=speech_file_path,voice=voice,model=model)
        
        #
        self.playfromFile(speech_file_path)
        os.remove(speech_file_path)
    
    
    def playfromFile(self,speech_file_path="path/to/audio.mp3"):
        #  Play the speech file using an appropriate media player library or command
        # Example using pygame (you'll need to install pygame library first)
        import pygame
        # from pygame.locals import *
        pygame.mixer.init()
        pygame.mixer.music.load(speech_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for the playback to finish
            pygame.time.Clock().tick(10)
        
    def play2(self,file="path/to/audio.mp3",format="mp3"):
        audio = AudioSegment.from_file(file, format=format)
        play(audio)


    def play(self,audio_data:sr.AudioData):
        import sounddevice as sd
        raw_data = audio_data.get_wav_data()
        sample_rate = audio_data.sample_rate
        channels = audio_data.sample_width

        # Convert raw data to numpy array
        data = np.frombuffer(raw_data, dtype=np.int16)
        sd.play(data, sample_rate)
        sd.wait()
        
    # # Define a function to generate TTS voices using the Web Speech API
    # def say_via_web(self,   st:streamlit,text:str, languageCode:str):
    #     # Define the JavaScript code to generate the voice
    #     voice=self.ttsVoices[languageCode]
    #     js_code = f"""
    #         const synth = window.speechSynthesis;
    #         const utterance = new SpeechSynthesisUtterance("{text}");
    #         utterance.voice = speechSynthesis.getVoices().filter((v) => v.name === "{voice}")[0];
    #         synth.speak(utterance);
    #     """
    #     # Use the components module to embed the JavaScript code in the web page
    #     st.components.v1.html(f"<script>{js_code}</script>", height=0)


    

class Trnscriber(ABC):
    

    def getVoiceFilePath(self,audioname, recordFormat):
        return audioname.replace("/./", "/") + recordFormat
    
    def getVoiceFile(self,audioname, recordFormat):
        return open(self.getVoiceFilePath(audioname, recordFormat), "rb")

    def get_audio_record_format(self,orgfile):
        info = magic.from_file(orgfile).lower()
        print(f'\n\n Recording file info is:\n {info} \n\n')
        if 'webm' in info:
            return '.webm'
        elif 'iso media' in info:
            return '.mp4'
        elif 'wave' in info:
            return '.wav'
        else:
            return '.mp4'
    @staticmethod
    def segment(file_path,segment_file_path,minuts_from=0,minutes_to=10):
        song = AudioSegment.from_mp3(file_path)

        # PyDub handles time in milliseconds
        time_from = minuts_from * 60 * 1000
        time_to = minutes_to * 60 * 1000

        all_what_you_need_minutes = song[time_from:time_to]

        all_what_you_need_minutes.export(segment_file_path, format="mp3")
        
    @staticmethod
    def save_wav_audio(filename, audio_data, sample_rate= 16000, num_channels=1, byte_width=2):
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
    @staticmethod
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
    
    @staticmethod
    def get_wisper_audio_array_from_file(audio_file_path) : 
        audio = Path(audio_file_path)
        audio = whisper.load_audio(audio)
        # print(audio)
        audio = whisper.pad_or_trim(audio)
        # print(audio)
        return audio
    
    @staticmethod
    def get_wisper_audio_array_from_audio_data(audio_data:sr.AudioData) : 
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        
        audio = np.frombuffer(wav_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        audio_array = whisper.pad_or_trim(audio) 
        # print(audio)
        audio = whisper.pad_or_trim(audio)
        # print(audio)
        return audio
    
    @staticmethod
    def get_audio_data_from_wav_data(wav_data) :
        # data = audio.get_wav_data()
        # segment = AudioSegment._from_safe_wav(data)
        segment =AudioSegment.from_wav(BytesIO(wav_data))
        # segment = AudioSegment._from_safe_wav(data)
        # play(segment)
        audio_data = sr.AudioData(segment.raw_data, segment.frame_rate,
                               segment.sample_width)
        return audio_data
    
    
    @staticmethod
    def get_wav_data_from_audio_data(audio:sr.AudioData,convert_rate=None, convert_width=None) :
      return audio.get_wav_data(convert_rate=convert_rate, convert_width=convert_width)
    @staticmethod
    def audio_to_audioAray(audio:sr.AudioData):
        wav_bytes = audio.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)
        return audio_array
    
    
    @staticmethod
    def load_audioSource_from_file(file_path):
        r=sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.record(source)
            return audio
        
    
    @abstractmethod
    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :
        pass
    
    @abstractmethod
    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        pass

    @abstractmethod
    def transcribeLang(self,audio,language='he') -> [str, str] :
        pass



class WhisperAsrTrnscriber(Trnscriber):
    def __init__(self, modelType="small",in_memory=True):#"large-v2"):
        # setup asr engine
            self.asrmodel =  whisper.load_model(name=modelType, download_root='asrmodel' ,in_memory=in_memory)

    
    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :  
        print("audio.name",audio_file_path,'Language',language)
        audio = Trnscriber.get_wisper_audio_array_from_file(audio_file_path)
        
        return self.transcribeLang(audio,language)
        
    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        
        return self.transcribeLang(self.audio_to_audioAray(audio),language)
    
    def transcribeLang(self,audio,language='he') -> [str, str] :
        if language is None:
            asr_result = self.asrmodel.transcribe( audio )
        else:
            asr_result = self.asrmodel.transcribe( audio ,language=language)
        transcript = asr_result["text"]
        languageCode = asr_result["language"]
        return transcript,languageCode
        
    # def transcribe(self, audio):
    #     return self.transcribeFile(audio.name )
        
  


class WhisperRegTrnscriber(Trnscriber):
    def __init__(self,modelType='small',in_memory=True):#'large-v2'):
        # setup asr engine  
        print("init")
        self.model = whisper.load_model(name=modelType,download_root="asrmodel",in_memory=in_memory)
        # options = whisper.DecodingOptions(language="he")
        # options = whisper.DecodingOptions(language= 'he', fp16=False)
        print("end init")

    
    
    # def detect_language(self,model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, input_features,
    #                 possible_languages: Optional[Collection[str]] = None) -> List[Dict[str, float]]:
    #     # hacky, but all language tokens and only language tokens are 6 characters long
    #     language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    #     if possible_languages is not None:
    #         language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
    #         if len(language_tokens) < len(possible_languages):
    #             raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

    #     language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    #     # 50258 is the token for transcribing
    #     logits = model(input_features,
    #                 decoder_input_ids = torch.tensor([[50258] for _ in range(input_features.shape[0])])).logits
    #     mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    #     mask[language_token_ids] = False
    #     logits[:, :, mask] = -float('inf')

    #     output_probs = logits.softmax(dim=-1).cpu()
    #     return [
    #         {
    #             lang: output_probs[input_idx, 0, token_id].item()
    #             for token_id, lang in zip(language_token_ids, language_tokens)
    #         }
    #         for input_idx in range(logits.shape[0])
    #     ]
        

    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :  
        print("audio.name",audio_file_path,'Language',language)
        import speech_recognition as sr
        import soundfile as sf
        from io import BytesIO
        
        audio = Trnscriber.getfileStreamWisper(audio_file_path)
        
        return self.transcribeLang(audio,language)
    
        
    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        
        return self.transcribeLang(self.get_wisper_audio_array_from_audio_data(audio),language)
    
    def transcribeLang(self,audio,language='he') -> [str, str] :
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        if language is None :
        # detect the spoken language
            _, probs = self.model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")
            options = whisper.DecodingOptions(fp16 = False)
        else:
            # decode the audio
            probs=language
            options = whisper.DecodingOptions(language=language,fp16 = False)
            
        result = whisper.decode(self.model, mel, options)
        # print the recognized text if available
        try:
            if hasattr(result, "text"):
                return result.text,probs
        except Exception as e:
            print(f"Error while printing transcription: {e}")
            traceback.print_exc()
            raise  e

   
    
class OpenAITrnscriber(Trnscriber):
    


    def __init__(self):
        # setup engine if need
            pass

    
    
    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :  
        # print("audio.name",audio_file_path,'Language',language)
        audio_file= open(audio_file_path, "rb")
        
        
        if language is None:
                transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text",temperature=0,
                timeout=30
                )
        else:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text",
                language=language,
                temperature=0,
                timeout=30
                )
        
        return transcript,language

    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        # wav_bytes = audio.get_wav_data(convert_rate=16000)
        # wav_stream = io.BytesIO(wav_bytes)
        # audio_array, sampling_rate = sf.read(wav_stream)
        # audio_array = audio_array.astype(np.float32)
        curr_start_time = time.time()
        file_name='tmp_'+str(curr_start_time)+'.wav'
        done,file_path=self.save_audio_from_audio_data(audio,file_name)
        if done :
            transcribe= self.transcribeFileLang(file_name,language)
            os.remove(file_path)
            return transcribe
        else:
            raise Exception("can't save the file!")
        
    def transcribeLang(self,audio,language='he') -> [str, str] :
        
        curr_start_time = time.time()
        file_name='tmp_'+str(curr_start_time)+'.wav'
        done,file_path=self.save_wav_audio(file_name,audio)
        # # Open the file in binary write mode
        # with open(file_name, 'wb') as f:
        #     f.write(audio)
        # f.close()
        # format=self.get_audio_record_format(file_name)
        # new_file_name=file_name+format
        # os.rename(file_name, new_file_name)
        # file_path=os.path.abspath(new_file_name)
        if done :
            transcribe= self.transcribeFileLang(file_path,language)
            os.remove(file_path)
            return transcribe
        else:
            raise Exception("can't save the file!")
        
    def transcribe(self, audio):
        pass

    
    def transcribeHe(self, audio,language="he"):
        pass
    




class TransformersTrnscriber(Trnscriber):
    
   
    
    # def __init__(self,modelType='openai/whisper-small'):#'openai/whisper-large-v3'):
    def __init__(self,modelType='openai/whisper-base'):#'openai/whisper-large-v3'):
        # setup engine if need
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        use_flash_attention=True if torch.cuda.is_available() else False
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            modelType, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,use_flash_attention_2=use_flash_attention
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(modelType)


    
    
    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :
        
        audio = Trnscriber.getfileStreamWisper(audio_file_path)
        
        return self.transcribeLang(audio,language)
    
    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        
        return self.transcribeLang(self.audio_to_audioAray(audio),language)
    
      
    def transcribeLang(self,audio,language='he') -> [str, str] : 

        # speech_file = whisper.load_audio(audio_file_path)

        if language is  None :
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        else:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
                generate_kwargs = {"task":"transcribe", "language":"<|"+language+"|>"}
            )
            
        result = pipe(audio)


            # print(result)
        transcript = result["text"]

        return transcript,language

        

    # def transcribeFileLangWork(self,audio_file_path,language=None) ->  [str, str] :
        
    #     from transformers import pipeline
    #     import torch
    #     from transformers import WhisperForConditionalGeneration, WhisperProcessor

    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #     # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
        
    #     # processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
    #     pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
    #     speech_file = whisper.load_audio(audio_file_path)
    #     if language is  None :
    #         result=pipe(speech_file, return_timestamps=True, chunk_length_s=30, stride_length_s=[6,0], batch_size=32)
    #     else:
    #         result=pipe(speech_file, return_timestamps=True, chunk_length_s=30, stride_length_s=[6,0], batch_size=32,generate_kwargs = {"task":"transcribe", "language":"<|he|>"})
       
    #     # print(result)
    #     transcript = result["text"]
        
    #     return transcript,language
        
   

class QuickWhisperTrnscriber(Trnscriber):
      
    NUM_WORKERS = 10
    LANGUAGE_CODE = "auto"
    CPU_THREADS = 4
    VAD_FILTER = True
    WHISPER_THREADS = 4
    
    def __init__(self,modelType='base',compute_type="int8"):
        # setup engine if need
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model  = WhisperModel(modelType,
                            device=self.device,
                            compute_type=compute_type,
                            num_workers=self.NUM_WORKERS,
                            cpu_threads=self.CPU_THREADS,
                            download_root="./models")
        
        # self.model  =WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=self.WHISPER_THREADS, download_root="./models")

        
    def read_wav_file(self,filename):
        
        
        # import numpy as np
        # import wavio
        # import base64
        # import io

        # rate = 16000  # samples per second
        # T = 3         # sample duration (seconds)
        # f = 440.0     # sound frequency (Hz)
        # t = np.linspace(0, T, T*rate, endpoint=False)
        # x = np.sin(2*np.pi * f * t)

        # file_in_memory = io.BytesIO()

        # wavio.write(file_in_memory, x, rate, sampwidth=3)

        # file_in_memory.seek(0)

        # encode_output = base64.b64encode(file_in_memory.read())

        # print(encode_output)

        import wavio
        import wave

        # Open the WAV file
        with wave.open(filename, 'rb') as wav_file:
            # Read the data
            # data =  b"" +wav_file.readframes(wav_file.getnframes())
            print(wav_file.getnframes())
            data =   wav_file.readframes(5000)
            
            audio_data_array: np.ndarray = np.frombuffer(data, np.int16).astype(np.float32) / 255.0
       

            print(audio_data_array,audio_data_array.shape)
            # print(memory_file)
            # encode_output = base64.b64encode(file_in_memory.read())
            # print(encode_output)
            # print(audio_data_array.shape,audio_data_array)
            return  audio_data_array


        # print(filename)
        # file_data=io.BytesIO()
        # wav_file = wave.open(file_data, 'rb')

        # # Get the number of frames in the file
        # num_frames = wav_file.getnframes()

        # # Get the frame width in bytes
        # frame_width = wav_file.getsampwidth()

        # # Read the frames into a byte array
        # data = wav_file.readframes(num_frames)

        # # Convert the byte array to a NumPy array
        # data = np.frombuffer(data, dtype=np.int16)

        # file_data.seek(0)
        # # # Reshape the array if the file is stereo
        # # if wav_file.getnchannels() == 2:
        # #     data = data.reshape(-1, 2)
        # print(file_data.shape,data.shape)
        # return file_data

    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :
    
        # from pydub import AudioSegment

        # # Load the sound file
        # sound = AudioSegment.from_file(audio_file_path)

        # new_wav_file_path=str(audio_file_path)+".wav"
        # # Export as WAV
        # sound.export(new_wav_file_path, format="wav")
        
        audio=self.read_wav_file(audio_file_path)

        
        return self.transcribeLang(audio,language)
    
    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        
        return self.transcribeLang(self.audio_to_audioAray(audio),language)
    
        
    def transcribeLang(self,audio,language='he') -> [str, str] : 
        # speech_file = whisper.load_audio(audio_file_path)

        segments, _ = self.model.transcribe(audio,
                                    language=language,
                                    beam_size=5,
                                    vad_filter=self.VAD_FILTER,
                                    vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        transcription = transcription.strip()
        return transcription,language




class fasterWhisperTrnscriber_not_work(Trnscriber):
      
    NUM_WORKERS = 10
    LANGUAGE_CODE = "auto"
    CPU_THREADS = 4
    VAD_FILTER = True
    def __init__(self,modelType='large-v2',compute_type="int8"):
        # setup engine if need
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model  = WhisperModel(modelType,
                            device=self.device,
                            compute_type=compute_type,
                            num_workers=self.NUM_WORKERS,
                            cpu_threads=self.CPU_THREADS,
                            download_root="./models")

    def transcribeFileLang(self,audio_file_path,language=None) ->  [str, str] :
    
        audio = Trnscriber.getfileStreamWisper(audio_file_path)
        
        return self.transcribeLang(audio,language)
    
        
    def transcribeLang(self,audio,language='he') -> [str, str] : 
        # speech_file = whisper.load_audio(audio_file_path)

        segments, _ = self.model.transcribe(audio,
                                    language=language,
                                    beam_size=5,
                                    vad_filter=self.VAD_FILTER,
                                    vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        transcription = transcription.strip()
        return transcription,language

    def transcribeADLang(self,audio:sr.AudioData,language='he') -> [str, str] :
        
        return self.transcribeLang(self.audio_to_audioAray(audio),language)
    


if __name__ == '__main__':
    
    # f1="/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/tmp_1700480401.106842.wav"
    # print(QuickWhisperTrnscriber().transcribeFileLang(f1))
    # exit()
    ##################
    # engine = pyttsx3.init()
    # engine.say("This is Text-To-Speech Engine Pyttsx3")
    # engine.runAndWait()
    # engine.stop()
    
    ##############################
    
    ask="נשיא טורקיה רג'פ טאיפ ארדואן ממשיך בקו הניצי שלו מול ישראל, ונפגש היום עם נשיא איראן איברהים ראיסי בפסגה כלכלית שנערכה בטשקנט, בירת אוזבקיסטן. בלשכת ארדואן אמרו כי הנשיא הטורקי אמר לעמיתו האיראני כי יש להגביר את הלחץ על ישראל על מנת לעצור את ההתקפות שלה ברצועת עזה במסגרת הניסיון להשמיד את חמאס. בנוסף, הצהיר ארדואן כי הוא מוכן לקבל על עצמו תפקיד של \"נותן ערבות\" על מנת לפתור את המשבר."
    speech_file_path = Path(__file__).parent / "speech.mp3"
    
    # vg=VoiceGenerator()
    
    # vg.speak_to_file(ask)
    # vg.play(speech_file_path)
        
    #################################
    # vg.say("חברים יקרים. אני מאוד שמח היום.. סוף-סוף קול עובד טוב!")
    # vg.play2('/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech.mp3')
    # vg.speak_to_file(ask,voice="nova")
    # https://platform.openai.com/docs/guides/text-to-speech/nova
    # https://platform.openai.com/docs/guides/text-to-speech/shimmer
    # https://platform.openai.com/docs/guides/text-to-speech/Onyx
    # https://platform.openai.com/docs/guides/text-to-speech/Fable
    # https://platform.openai.com/docs/guides/text-to-speech/Echo
    # https://platform.openai.com/docs/guides/text-to-speech/Alloy
    
    # vg.speak(ask,voice="nova") 
    
    
    # exit()
    # filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech.wav'
    # filename='/Users/dmitryshlymovich/workspace/wisper/tmp/voice-assistant-chatgpt/recording.tmp.mp4'
    filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/save_1702280862.193871.wav'
    # filename=speech_file_path
    # print(ask)
    instances = [WhisperAsrTrnscriber(),WhisperRegTrnscriber(),TransformersTrnscriber(),OpenAITrnscriber()]
    # instances=[QuickWhisperTrnscriber()]
    # instances=[QuickWhisperTrnscriber()]
    # instances=[WhisperRegTrnscriber()]

    @utility.timing_decorator
    def check_transcriber_file(transcriber,filename):
        
        # audio=transcriber.getVoiceFile( filename, "")
        transcription_result, languageCode = transcriber.transcribeFileLang(filename,language='he')
        print('1',transcriber.__class__.__name__,transcription_result, languageCode)
        # print(find_string_differences_html(ask,transcription_result))
        # vg.speak_to_file(ask)

    @utility.timing_decorator
    def check_transcriber(transcriber,audio):
        
        # audio=transcriber.getVoiceFile( filename, "")
        transcription_result, languageCode = transcriber.transcribeADLang(audio,language='he')
        print('1',transcriber.__class__.__name__,get_display(transcription_result), languageCode)
        # print(find_string_differences_html(ask,transcription_result))
        # vg.speak_to_file(ask)
        
    
    for instance in instances:
        audio=instance.load_audioSource_from_file(filename)
        # VoiceGenerator().play(audio)
        try:
            for i in range(3):
                check_transcriber(instance,audio)
    
        except Exception as e:
            print(f"An exception occurred: {e}")
            traceback.print_exc()
            # Resume to the next instance
            continue
    
       