
# https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst

import speech_recognition as sr
import numpy as np
import soundfile as sf
import time
from typing import Dict, List
import os,io 
from io import BytesIO

import labSpeachrecognitionImpl
import voice
import traceback

   
file_counter=0
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}


def use_file(file_path,method_name,*args, **kwargs):
    
    def run_method(class_instance, method_name, *args, **kwargs):
        method = getattr(class_instance, method_name, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            print(f"No method named {method_name} found in the class")

    r = labSpeachrecognitionImpl.LabRecognizer()
    r.energy_threshold = 4000
    
    # file_path='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech.wav'
    file_path=voice.Transcriber.check_and_convert(file_path)
    with labSpeachrecognitionImpl.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.record(source)
    
    # with sr.Microphone() as source:
    #     print("Say something!")
    #     # Listen for the first phrase and extract it into audio data
    #     audio = r.listen(source)
    
    try:
    
        transcription_start_time = time.time()
                 
        # message = r.recognize_whisper(audio)
        # (message,_)=r.recognize_azure(audio,language='en-US',key=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_API_KEY'),location=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_SPEECH_REGION'))
        message = run_method(r,method_name,audio,*args, **kwargs)
        # message=r.recognize_google(audio,language='he')
        print("You said: " , message)
        transcription_postprocessing_end_time = time.time()
        
        transcription_end_time = time.time()

        # print(transcription, end='\r', flush=True)

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)
        return message

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    


if __name__ == "__main__":
    
        # filename='/Users/dmitryshlymovich/Downloads/Recording_erez.m4a'  
        # filename='/Users/dmitryshlymovich/Downloads/sentence_two.wav'
        filename='/Users/dmitryshlymovich/Downloads/test.wav'
        filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_k7xJngMTUo.wav'
        filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_OAWaqg5kDs.wav'
        filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_YaTiIvfTNC.wav'
        # filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech.-en.wav'
        
        def runRecognize(method_name,*args, **kwarg):
            

            try:
        
                replay=use_file(filename,method_name=method_name,*args,**kwarg)
                
        
        
                print("short:",method_name,"\t",replay,"\t",f"{np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s")
                # print out the statistics
                print("\nused function:",method_name,*args)
                print("Number of processed chunks: ", len(stats["overall"]))
                print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
                print(
                    f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
                )
                print(
                    f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
                )
                # We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
                print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")
                stats["overall"].clear()
                stats['transcription'].clear()
                stats['postprocessing'].clear()
                


            except Exception as e:
                print(f"An exception occurred: {e} in func:",method_name)
                traceback.print_exc()
                # Resume to the next instance
                
    
 
        runRecognize(method_name="recognize_google",language='he')
        
        runRecognize(method_name="recognize_azure",language='he-IL',key=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_API_KEY'),location=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_SPEECH_REGION'))
        
        runRecognize(method_name="recognize_openAI",language='he')
        
        runRecognize(method_name="recognize_whisper",model="small",language='he')
        
        runRecognize(method_name="recognize_whisper",model="large-v3",language='he')
        
        runRecognize(method_name="recognize_asr",model="small",language='he')
        
        runRecognize(method_name="recognize_asr",language='he')
        
        runRecognize(method_name="recognize_Transformer",model="openai/whisper-small",language='he')
        
        runRecognize(method_name="recognize_Transformer",model="openai/whisper-large-v3",language='he')
            
        
    
        # use_file(filename,method_name="recognize_google",language='he')
        # use_file(filename,method_name="recognize_Transformer",language='he')
        # use_file(filename,method_name="recognize_Transformer",model="openai/whisper-small",language='he')
        # use_file(filename,method_name="recognize_Transformer",model="openai/whisper-large-v3",language='he')
        # use_file(filename,method_name="recognize_openAI",language='he')
        # use_file(filename,method_name="recognize_azure",language='he-IL',key=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_API_KEY'),location=os.environ.get('MICROSOFT_SPEACH_TO_TEXT_SPEECH_REGION'))
        # runlenth=25
        # use_mic_on_background(runlenth)
        # use_mic_on_background_not_connected(runlenth)
        # use_mic_on_background_test(runlenth)
        print("Exiting...")
    
        
        
      
    