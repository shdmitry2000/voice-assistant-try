
# import speech_recognition as sr
import os ,time
import voice,labSpeachrecognitionImpl
from typing import Dict, List
import numpy as np
from bidi.algorithm import get_display


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

# transcriberregTrans=voice.WhisperRegTrnscriber()
# transcribertransTrans=voice.TransformersTrnscriber()

def transcribe(recognizer,audio,transcription_start_time):
    file_name="save_reg_"+str(transcription_start_time) +".wav"
    voice.Trnscriber.save_audio_from_audio_data(audio,file_name)
    transcription=recognizer.recognize_google(audio,language='he')
    # transcription,_ =transcribertransTrans.transcribeADLang(audio,language='he')
    # transcription=recognizer.recognize_whisper(audio,language='he')
    print(get_display(transcription))
    return transcription

def callbackFunc(recognizer,audio):                          # this is called from the background thread

        try:
            
            transcription_start_time = time.time()
            transcription=transcribe(recognizer,audio,transcription_start_time)
            transcription_end_time = time.time()

            # transcription = " ".join(segments)
            
            transcription = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

            transcription_postprocessing_end_time = time.time()

            # print(transcription, end='\r', flush=True)

            overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
            transcription_elapsed_time = transcription_end_time - transcription_start_time
            postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
            stats["overall"].append(overall_elapsed_time)
            stats["transcription"].append(transcription_elapsed_time)
            stats["postprocessing"].append(postprocessing_elapsed_time)

                    
                    # print("You said " + recognizer.recognize_whisper(audio,language='he'))  # received audio data, now need to recognize it
        except LookupError:
            print("Oops! Didn't catch that")
            
            
            
            
def use_mic_on_background(runlenth):
    
    
    r = labSpeachrecognitionImpl.LabRecognizer()
    m = labSpeachrecognitionImpl.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.5  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.3  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source,duration=3)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(source=m, callback=callbackFunc,phrase_time_limit=LENGHT_IN_SEC)
    # do some unrelated computations for runlenth seconds
    
    try:
        print("speak please")
        # do some unrelated computations for 5 seconds
        while True: 
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Exiting...")
        stop_listening()
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




if __name__ == "__main__":
    runlenth=25
    # print(get_display("שלום"))
    use_mic_on_background(runlenth)
    