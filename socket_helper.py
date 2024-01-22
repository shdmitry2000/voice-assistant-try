from pathlib import Path,PosixPath
from aiohttp import web
import aiohttp
import concurrent.futures
import asyncio
# from speech_recognition import AudioData
# from asyncio import Awaitable
import labSpeachrecognitionImpl
import librosa
import voice
import time
import io
from io import BytesIO
import traceback
# from fastapi import Depends
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment, silence
import json


import speech_recognition as sr
import os ,time
import voice,labSpeachrecognitionImpl
from typing import Dict, List
import numpy as np
from bidi.algorithm import get_display
import multiprocessing
import utility
import asyncio
import multiprocessing
import voice
from abc import ABC, abstractmethod
from audio_utility import *
class MultiprocessingHelper:
   def __init__(self,pool = None):
        if pool is None:
           self.pool=multiprocessing.Pool(processes=2)
        else:
            self.pool=pool
        
   async def perform_multiprocess(self, *args, **kwargs):
       loop = asyncio.get_event_loop()
       future = loop.run_in_executor(None, self.perform_task, *args, kwargs)
       return await future

   @abstractmethod
   def perform_task(self, *args, **kwargs):
        pass

class MultiprocessingTranscriberHelper(MultiprocessingHelper):
                                            # "recognize_google"
    def __init__(self,recognizer,method_name="recognize_openAI",savetofile=False,pool = None ):
                 
        super(MultiprocessingTranscriberHelper,self).__init__(pool)
        self.method_name=method_name
        self.recognizer=recognizer
        self.savetofile=savetofile

    
    def perform_task(self, *args, **kwargs):
        r = sr.Recognizer()
        audio_data = args[0]['audio_data'] # Accessing audio_data
        language = args[0]['language'] # Accessing language
        
        if self.savetofile:
                transcription_start_time = time.time()
                file_name="save_webm_"+str(transcription_start_time) +".wav"
                save_audio_from_audio_data(audio_data,file_name)
        
        
        try:
            if self.method_name is None:
                    text=self.recognizer.recognize_openAI(audio_data,language='he')
                    # transcription=recognizer.recognize_Transformer(audio,language='he')
                    # transcription=recognizer.recognize_azure(audio,language='he-IL',key=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_API_KEY'),location=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_SPEECH_REGION'))
                
            else:
                method_name= self.method_name   
                text=utility.run_method(self.recognizer,method_name,audio_data=audio_data,language=language)
                # text = r.recognize_google(audio_data=audio_data,language=language)
        
        
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return "Could not request results from Google Speech Recognition service; {0}".format(e)


    
   
class audio_handler_helper():
    format="wav"
 
    def __init__(self,recognizer,pool = None,near_real_time=False,savetofile=False,method_name="recognize_openAI"):
        self._near_real_time=near_real_time
        self.mpthelper = MultiprocessingTranscriberHelper(recognizer=recognizer,savetofile=savetofile,method_name=method_name,pool=pool)
        self.opus_data = io.BytesIO()
        self.audio_data= asyncio.Queue()
        self.silence_times=[]
        self.usedElements = []
        self.last_element = None
        # self.finalized=False
        self.text=[]
        self.finish_event = asyncio.Event()
        
    
    async def perform_pause_parce(self,final=False):
        try:   
            print("perform_pause_parce",self._near_real_time,self.finish_event.is_set(),final)
            if self._near_real_time:
            
                if self.finish_event.is_set():
                    return
                
                the_segment = self.get_segment(data=self.opus_data)
                
                if not final:
                    self.silence_times = getSilenceStartStopTime(the_segment)
                    for silence_time in self.silence_times[:-1]:
                        if silence_time not in self.usedElements:
                            #handle it
                            self.usedElements.append(silence_time)
                            self.last_element=silence_time
                            print(" self.audio_data.put ")
                            await self.audio_data.put( labSpeachrecognitionImpl.AudioData(the_segment[ 
                                    self.last_element[0]:self.last_element[1]].raw_data, the_segment.frame_rate,the_segment.sample_width))
                    #handle last element end
                    if len(self.silence_times) >= 1:
                        if  self.last_element==self.silence_times[-1] and self.last_element not in self.usedElements:
                            self.last_element=self.silence_times[-1]
                            print(" self.audio_data.put ")
                            await self.audio_data.put( labSpeachrecognitionImpl.AudioData(the_segment[ 
                                    self.last_element[0]:self.last_element[1]].raw_data, the_segment.frame_rate,the_segment.sample_width))
                
                            self.usedElements.append(self.last_element)
                        else:
                            self.last_element=self.silence_times[-1]  
                else:
                    # self.finalized=True
                    self.finish_event.set()
                    if len(self.silence_times) >= 1:
                        if   self.last_element not in self.usedElements:
                            begin_of_last_handle=self.last_element[0]
                        else:
                            begin_of_last_handle=self.last_element[1]
                        print(" self.audio_data.put ")    
                        await self.audio_data.put( labSpeachrecognitionImpl.AudioData(the_segment[ 
                                    begin_of_last_handle:].raw_data, the_segment.frame_rate,the_segment.sample_width))
                
                    else:
                        print(" self.audio_data.put ")
                        await self.audio_data.put( labSpeachrecognitionImpl.AudioData(the_segment.raw_data, the_segment.frame_rate,the_segment.sample_width))
            else:
                if  final:
                    # self.finalized=True 
                    self.finish_event.set()
                    the_segment = self.get_segment(data=self.opus_data)
                    print(" self.audio_data.put ")
                    await self.audio_data.put(labSpeachrecognitionImpl.AudioData(the_segment.raw_data, the_segment.frame_rate,the_segment.sample_width))
            print("perform_pause_parce finished")
        except Exception as e:
            print(f"Error while printing transcription: {e}")
            traceback.print_exc()
            raise  e
            
      
    @property
    def near_real_time(self):
       return self._near_real_time

    @near_real_time.setter
    def near_real_time(self, value):
       self._near_real_time = value
       
       
    async def finalize_data(self):
        try:
            print("finalize_data")
            await self.perform_pause_parce(final=True)
            
            # # await asyncio.wait(2) 
            # if self._near_real_time:
            #     await self.perform_pause_parce(final=True)
            # else:
            #     # self.opus_data.seek(0)
            #     the_segment=self.get_segment(data=self.opus_data)
            #     await self.audio_data.put(labSpeachrecognitionImpl.AudioData(the_segment.raw_data, the_segment.frame_rate,the_segment.sample_width))
            print("finalize_data finished")
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            traceback.print_exc()
            raise

    
    def get_segment(self,data,format=format):
        data.seek(0)
        return AudioSegment.from_file(data,format=format)
        
        
    # @utility.run_async(perform_pause_parce)
    async def add_chank(self,chank_data):
        self.opus_data.write(chank_data) 
        print("add_chank",self.opus_data.getbuffer().nbytes)
        asyncio.create_task(self.perform_pause_parce())
        
        
    async def add_raw_as_chank(self,seg_chank:AudioSegment):
        seg1=self.get_segment(self.opus_data)    
        concat_seg:AudioSegment=seg1+seg_chank
        self.opus_data= BytesIO()
        concat_seg.export(self.opus_data,format=self.format)
        print("add_raw_as_chank",self.opus_data.getbuffer().nbytes)
        asyncio.create_task(self.perform_pause_parce())
    
    
    
    async def get_next_audio(self) :
        # -> labSpeachrecognitionImpl.AudioData :
        """Get an item from the queue. Block if the queue is empty."""
        print('await to get')
        data = await self.audio_data.get()
        # Don't forget to remove the item from the queue after retrieving it
        self.audio_data.task_done()
        print('end of  get',data)
        return data
  

    async def get_current_text(self):
        audio_data=await self.get_next_audio()
        
        # async def perform_multi_process(self, *args, **kwarg):
        #     return await self.helper.perform_multiprocess(*args, **kwarg)
        try:
            the_trans_txt = await self.mpthelper.perform_multiprocess(method_name="recognize_openAI", audio_data=audio_data, language="he")
   
            self.text.append(the_trans_txt)
            return the_trans_txt
            # print(the_trans_txt)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            traceback.print_exc()
            return "didn't catch you! Unexpected {err=}"
            
            
    
    def reset(self):
        self.opus_data =BytesIO() 
        self.silence_times=[]
        self.usedElements = []
        self.last_element = None
        # self.finalized=False
        self._near_real_time=False
        self.text=[]   
        self.finish_event.clear()  
        
class socket_helper_webm(audio_handler_helper):     
    
    format='webm'
    
    def __init__(self,recognizer,pool =None,near_real_time=False,savetofile=False,method_name="recognize_openAI"):
        super(socket_helper_webm,self).__init__(recognizer,pool,near_real_time,savetofile,method_name)
    
    def get_segment(self,data,format=format):
        data.seek(0)
        return AudioSegment.from_file(data,format=format)
        
    
  
        
async def main():
       
    file_path="/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_k7xJngMTUo.wav"
    file_path2="/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_YaTiIvfTNC.wav"
    # file_path2='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech-heb.wav'
    pool = multiprocessing.Pool(processes=5) # limit to 10 processes
    recognizer= labSpeachrecognitionImpl.LabRecognizer()
    
    
    # audio=load_audioSource_from_file(file_path)
    # binary_data=get_wav_data_from_audio_data(audio,convert_rate=16000)
    
    # audio2=load_audioSource_from_file(file_path2)
    # binary_data2=get_wav_data_from_audio_data(audio2,convert_rate=16000)
    
    audio:AudioSegment=AudioSegment.from_file(file_path)
    audio2:AudioSegment=AudioSegment.from_file(file_path2)
    
    # audio=load_byteio_from_file(file_path)
    # audio2=load_byteio_from_file(file_path2)
    bytes_audio = BytesIO()
    audio.export(bytes_audio, format="wav")
    
    bytes_audio2 = BytesIO()
    audio2.export(bytes_audio2, format="wav")
    
    
    
    # concatenated_audio=audio+audio2
    # bytes_concatenated_audio = BytesIO()
    # concatenated_audio.export(bytes_concatenated_audio, format="wav")
    
    
    # audio_data = voice.Transcriber.get_audio_data_from_wav_data(data)
            
            
    # wav_data=voice.Transcriber.get_wisper_audio_array_from_file(file_path)
    # audio=voice.Transcriber.load_audioSource_from_file(file_path)
    
    # voice.Transcriber.save_audio_from_audio_data(audio,"test_wave.wav")
    
    # wav_data=voice.Transcriber.get_wav_data_from_audio_data(audio,convert_rate=16000)
    # voice.Transcriber.save_wav_from_byteio("test_wave.wav",binary_data)
    # audio_data = voice.Transcriber.get_audio_data_from_wav_data(wav_data)
            
    # voice.Transcriber.save_wav_audio("test_wave.wav",audio)       
    # print(wav_data)   
    # wav_bytes = audio.get_wav_data(convert_rate=16000)
    # chank= audio.get_raw_data
    # print(wav_bytes)
    try:
        
        
        # sereal
        sh=audio_handler_helper(pool=pool,recognizer=recognizer,near_real_time=True,savetofile=False,method_name='recognize_openAI')
        
        
        await sh.add_chank(bytes_audio.getvalue())
        await sh.add_raw_as_chank(audio2)
        print(str(await sh.get_current_text()))
        print(str(await sh.get_current_text()))
        await sh.finalize_data()
        print(str(await sh.get_current_text()))
        
        #paralel
        sh=audio_handler_helper(pool=pool,recognizer=recognizer,near_real_time=True,savetofile=False,method_name='recognize_openAI')
        
        async def print_current_text(func):
            print(await func())
            
        async def delayed_execution(func):
            await asyncio.sleep(2)
            await func()
   
        async with asyncio.TaskGroup() as tg:
            tg.create_task(print_current_text( sh.get_current_text))
            tg.create_task(print_current_text( sh.get_current_text))
            tg.create_task(print_current_text( sh.get_current_text))
            
            tg.create_task(sh.add_chank(bytes_audio.getvalue()))
            tg.create_task(sh.add_raw_as_chank(audio2))
            
            tg.create_task(delayed_execution( sh.finalize_data))
            

        # sh=audio_handler_helper(pool=pool,recognizer=recognizer,near_real_time=True,savetofile=False,method_name='recognize_openAI')
        
        
       
        
            
   
        
        # await sh.get_current_text()
            
        # done, pending = await asyncio.wait([sh.add_chank(chank), print(await sh.get_current_text()),sh.finalize_data()])
        # done, pending = await asyncio.wait([alive_task])

        # for task in pending:
        
            # task.cancel()
    except (asyncio.exceptions.CancelledError,TypeError) as e:
        print("task cancel ,error",e)
        
    
    
    print('end')
    
if __name__ == "__main__":
    # pool = multiprocessing.Pool(initializer=init_worker,processes=5) # limit to 10 processes
    # main()
    asyncio.run(main())

  