# https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst

import speech_recognition as sr
import numpy as np
import soundfile as sf
import torch
import whisper
import time
from typing import Dict, List
import os,io 
from io import BytesIO

import speech_recognition as sr
import voice



class LabRecognizer(sr.Recognizer):
    

    # def __init__(self):
    #     super().__init__()
    
    
    def recognize_Transformer(self, audio_data, model="openai/whisper-base",  load_options=None, language=None, translate=False,  **transcribe_options):
        
        
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
        """

        if load_options or not hasattr(self, "Transformer_model") or self.Transformer_model.get(model) is None:
            self.Transformer_model = getattr(self, model, {})
            self.Transformer_model[model] = voice.TransformersTrnscriber(modelType=model)


        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
       

        return  self.Transformer_model[model].transcribeADLang(audio_data,language=language)[0]
        
    
 
    def recognize_asr(self, audio_data, model="large-v2",  load_options=None, language=None, translate=False,  **transcribe_options):
        
        
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
        """

        if load_options or not hasattr(self, "asr_model") or self.asr_model.get(model) is None:
            self.asr_model = getattr(self, model, {})
            self.asr_model[model] = voice.WhisperAsrTrnscriber(modelType=model)


        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        

        return  self.asr_model[model].transcribeADLang(audio_data,language=language)[0]
           
    
    
    def recognize_whisper_full(self, audio_data, model="small",  load_options=None, language=None, translate=False,  **transcribe_options):
        
        
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
        """

        if load_options or not hasattr(self, "whisper_full_model") or self.whisper_full_model.get(model) is None:
            self.whisper_full_model = getattr(self, model, {})
            self.whisper_full_model[model] = voice.WhisperRegTrnscriber(modelType=model)


        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        
        
        return  self.whisper_full_model[model].transcribeADLang(audio_data,language=language)[0]
 
    def recognize_openAI(self, audio_data, language=None, translate=False, **transcribe_options):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using openAI.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        """

        if  not hasattr(self, "openAI_model") :
            self.openAI_model = voice.OpenAITrnscriber()
            

        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        

        return  self.openAI_model.transcribeADLang(audio_data,language=language)[0]
           
           
        

class Microphone(sr.Microphone):
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024):
        super().__init__(device_index, sample_rate, chunk_size)

class Speak():
    def __init__(self):
        pass

class AudioData(sr.AudioData):
    def __init__(self, frame_data, sample_rate, sample_width):
         super().__init__(frame_data, sample_rate, sample_width)
         
         

 