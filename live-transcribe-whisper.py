
import speech_recognition as sr
import os ,time
import voice
from typing import Dict, List
import numpy as np

transcriberregTrans=voice.WhisperRegTrnscriber()

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



def callbackFunc(recognizer,audio):                          # this is called from the background thread

        try:
            
            transcription_start_time = time.time()
            # transcription,_ =transcriberregTrans.transcribeLang(audio.)
            transcription=recognizer.recognize_whisper(audio,language='he')
            print(transcription)
            transcription_end_time = time.time()

            # transcription = " ".join(segments)
            
            # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
            # transcription = re.sub(r"\[.*\]", "", transcription)
            # transcription = re.sub(r"\(.*\)", "", transcription)
            # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
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
    
    
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.250  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

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
    use_mic_on_background(runlenth)
    