
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
   
file_counter=0
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}





def use_mic_on_background_test(runlenth):
    
    
    def callbackFunc(recognizer,audio):                          # this is called from the background thread
        try:
            print("in  callback")
            transcription_start_time = time.time()
            # transcription,_ =transcriberregTrans.transcribeLang(audio.)
            # transcription=recognizer.recognize_whisper( audio,  language='he')
            transcription=recognizer.recognize_whisper_full( audio,  language='he')
            
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
            
           
        # global file_counter
        # try:
            # print("in  callback")
            # with open('test_wav'+str(file_counter)+'.wav', 'wb') as file:
            #     wav_data = audio.get_wav_data()
            #     file.write(wav_data)
            #     file_counter=file_counter+1
        #     transcription_start_time = time.time()
        #     print("You said " + recognizer.recognize_whisper(audio,language='he'))  # received audio data, now need to recognize it
        # except LookupError:
        #     print("Oops! Didn't catch that")
        
        
    r = labSpeachrecognitionImpl.LabRecognizer()
    m = labSpeachrecognitionImpl.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.4  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        print("Say something!")
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(source=m, callback=callbackFunc,phrase_time_limit=LENGHT_IN_SEC)
    # `stop_listening` is now a function that, when called, stops background listening
    print("speak please")
    # do some unrelated computations for 5 seconds
    for _ in range(runlenth): time.sleep(1)  # we're still listening even though the main thread is doing other things

    # stop_listening=True
    stop_listening()
    # calling this function requests that the background listener stop listening
    # stop_listening(wait_for_stop=False)

    # do some more unrelated things
    # while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping







    

def use_mic_on_background_not_connected(runlenth):
    
    def callbackFunc(recognizer,audio):                          # this is called from the background thread
        from pydub import AudioSegment
        from pydub.playback import play #needs simpleaudio to work idk...
        try:
            # AudioSegment._from_safe_wav(audio.)
            # segment = AudioSegment.from_raw(from_file(wav_file)
            # audio_data = AudioData(segment.raw_data, segment.frame_rate,
            #                    segment.sample_width)
            print("in  callback",audio.sample_rate,audio.sample_width)
            transcription_start_time = time.time()
            data=voice.Trnscriber.get_wav_data_from_audio_data(audio,convert_rate=16000)
            audio_data = voice.Trnscriber.get_audio_data_from_wav_data(data)
            
            
            transcription=recognizer.recognize_whisper(audio_data,language='he')
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
            
           
        # global file_counter
        # try:
            # print("in  callback")
            # with open('test_wav'+str(file_counter)+'.wav', 'wb') as file:
            #     wav_data = audio.get_wav_data()
            #     file.write(wav_data)
            #     file_counter=file_counter+1
        #     transcription_start_time = time.time()
        #     print("You said " + recognizer.recognize_whisper(audio,language='he'))  # received audio data, now need to recognize it
        # except LookupError:
        #     print("Oops! Didn't catch that")
        
        
    r = labSpeachrecognitionImpl.LabRecognizer()
    m = labSpeachrecognitionImpl.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.4  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(source=m, callback=callbackFunc,phrase_time_limit=LENGHT_IN_SEC)
    # `stop_listening` is now a function that, when called, stops background listening
    print("Say something!")
    # do some unrelated computations for 5 seconds
    for _ in range(runlenth): time.sleep(1)  # we're still listening even though the main thread is doing other things

    # stop_listening=True
    stop_listening()
    # calling this function requests that the background listener stop listening
    # stop_listening(wait_for_stop=False)

    # do some more unrelated things
    # while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping




def use_mic_on_background(runlenth):
    
    def callbackFunc(recognizer,audio):                          # this is called from the background thread
        try:
            print("in  callback")
            transcription_start_time = time.time()
            # transcription,_ =transcriberregTrans.transcribeLang(audio.)
            transcription=recognizer.recognize_whisper( audio,  language='he')
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
            
           
        # global file_counter
        # try:
            # print("in  callback")
            # with open('test_wav'+str(file_counter)+'.wav', 'wb') as file:
            #     wav_data = audio.get_wav_data()
            #     file.write(wav_data)
            #     file_counter=file_counter+1
        #     transcription_start_time = time.time()
        #     print("You said " + recognizer.recognize_whisper(audio,language='he'))  # received audio data, now need to recognize it
        # except LookupError:
        #     print("Oops! Didn't catch that")
        
        
    r = labSpeachrecognitionImpl.LabRecognizer()
    m = labSpeachrecognitionImpl.Microphone()
    with m as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.4  # seconds of non-speaking audio to keep on both sides of the recording
        r.dynamic_energy_threshold = True
        print("Say something!")
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(source=m, callback=callbackFunc,phrase_time_limit=LENGHT_IN_SEC)
    # `stop_listening` is now a function that, when called, stops background listening
    print("speak please")
    # do some unrelated computations for 5 seconds
    for _ in range(runlenth): time.sleep(1)  # we're still listening even though the main thread is doing other things

    # stop_listening=True
    stop_listening()
    # calling this function requests that the background listener stop listening
    # stop_listening(wait_for_stop=False)

    # do some more unrelated things
    # while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping

    

     

def use_microfon():
    r = labSpeachrecognitionImpl.LabRecognizer()
    # r.energy_threshold = 4000
    # Use the microphone as the audio source
    
    with labSpeachrecognitionImpl.Microphone() as source:
        # r.energy_threshold = 270
        
        r.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete 
        r.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        r.non_speaking_duration = 0.4  # seconds of non-speaking audio to keep on both sides of the recording
        # r.energy_threshold = 500
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

        # r.dynamic_energy_threshold = True
        print("Say something!")
        # Listen for the first phrase and extract it into audio data
        
        # file_counter=0
        # for x in range(3):
        
        audio = r.listen(source,timeout=10,phrase_time_limit=6) #,phrase_time_limit=5
        # audio = r.listen(source)
        
        # with open('test_wav'+str(file_counter)+'.wav', 'wb') as file:
        #     wav_data = audio.get_wav_data()
        #     file.write(wav_data)
            
        # Recognize speech using Google Speech Recognition
    try:
        print("recognize_whisper Speech Recognition thinks you said " + r.recognize_whisper(audio,language='he'))
    except sr.UnknownValueError:
        print("recognize_whisper Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))




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
    file_path=voice.Trnscriber.check_and_convert(file_path)
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.record(source)
    
    # with sr.Microphone() as source:
    #     print("Say something!")
    #     # Listen for the first phrase and extract it into audio data
    #     audio = r.listen(source)
    
    try:
    
        transcription_start_time = time.time()
                 
                    
        # message = r.recognize_whisper(audio)
        message = run_method(r,method_name,audio,*args, **kwargs)
        # message=r.recognize_google(audio,language='he')
        print("You said: " + message)
        transcription_postprocessing_end_time = time.time()
        
        transcription_end_time = time.time()

        # print(transcription, end='\r', flush=True)

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    


if __name__ == "__main__":
    try:
        # filename='/Users/dmitryshlymovich/Downloads/Recording_erez.m4a'  
        filename='/Users/dmitryshlymovich/Downloads/sentence_two.wav'
        # use_file(filename,method_name="recognize_whisper")
        use_file(filename,method_name="recognize_google",language='he')
        # use_file(filename,method_name="recognize_Transformer",language='he')
        # use_file(filename,method_name="recognize_openAI",language='he')
        
        # runlenth=25
        # use_mic_on_background(runlenth)
        # use_mic_on_background_not_connected(runlenth)
        # use_mic_on_background_test(runlenth)
        
        # use_microfon()
        
        # print(sr.__version__)

    except KeyboardInterrupt:
        pass
    
    
    print("Exiting...")
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
        
        
        
      
    