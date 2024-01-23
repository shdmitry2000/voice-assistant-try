
import speech_recognition as sr
import os ,time
import labSpeachrecognitionImpl
from typing import Dict, List
import numpy as np
from bidi.algorithm import get_display

import multiprocessing
import signal
from  audio_utility import *


os.environ['KMP_DUPLICATE_LIB_OK']='True'

transcription_tasks=[]


# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 29    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

pool = None
method_name=None

def init_worker():
   signal.signal(signal.SIGINT, signal.SIG_IGN)
   
def transcribe(recognizer,audio):
    try:
        transcription_start_time = time.time()
        file_name="save_reg_"+str(transcription_start_time) +".wav"
        save_audio_from_audio_data(audio,file_name)
        # transcription="test"
        if method_name is None:
            transcription=recognizer.recognize_Transformer(audio,language='he')
        else:
            method = getattr(recognizer, method_name)
            transcription= method(audio,language='he')
               
        transcription_postprocessing_end_time = time.time()
        # print(get_display(transcription))
        transcription_end_time = time.time()
        
        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)
        return transcription
    except (LookupError,sr.exceptions.UnknownValueError):
        print("Oops! Didn't catch that")
        return "Oops! Didn't catch that"
    


def callbackFunc(recognizer,audio):                          # this is called from the background thread

        
            
            # #paralel perform
            transcription_task = pool.apply_async(transcribe, (recognizer,audio) )#, callback=print_result)
            transcription_tasks.append(transcription_task)
            transcription_task=transcription_tasks[0]
            if(transcription_task.ready()):
                transcription=str(transcription_task.get()) 
                print(get_display(transcription))
                if transcription_task in transcription_tasks :
                    transcription_tasks.remove(transcription_task)
            
            # cerial
            # transcription=transcribe(recognizer,audio,transcription_start_time)
    
            
            
def use_mic_on_background(runlenth):
    
    
    r = labSpeachrecognitionImpl.LabRecognizer()
    m = labSpeachrecognitionImpl.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.4  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source,duration=3)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(source=m, callback=callbackFunc,phrase_time_limit=LENGHT_IN_SEC)
   
    
    try:
        print("speak please")
        # do some unrelated computations for 5 seconds
        while True: 
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Exiting...")
        try:
            stop_listening()
            print("stop_listening...")
            for task in transcription_tasks:
                task.wait() # wait for the task to complete
        
            for task in transcription_tasks:
                print(task.get(timeout=2)) # wait for the task to complete
            pool.terminate()
            pool.join()
        except :
            pass
        print("finish...")
        # print out the statistics
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


def main():
    runlenth=25
    # method_name="recognize_google"
    # method_name="recognize_whisper"
    use_mic_on_background(runlenth)

if __name__ == "__main__":
    pool = multiprocessing.Pool(initializer=init_worker,processes=10) # limit to 10 processes
    main()
    