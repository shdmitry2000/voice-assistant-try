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

file_counter=0
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

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
 
 

class Microphone(sr.Microphone):
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024):
        super().__init__(device_index, sample_rate, chunk_size)

class Speak():
    def __init__(self):
        pass

class AudioData(sr.AudioData):
    def __init__(self, frame_data, sample_rate, sample_width):
         super().__init__(frame_data, sample_rate, sample_width)
         
         

    
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
        
        
    r = LabRecognizer()
    m = Microphone()
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
        
        
    r = LabRecognizer()
    m = Microphone()
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
        
        
    r = LabRecognizer()
    m = Microphone()
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
    r = LabRecognizer()
    # r.energy_threshold = 4000
    # Use the microphone as the audio source
    
    with Microphone() as source:
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
def use_file():
    
    r = LabRecognizer()
    r.energy_threshold = 4000
    
    file_path='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/speech.wav'
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.record(source)
    
    # with sr.Microphone() as source:
    #     print("Say something!")
    #     # Listen for the first phrase and extract it into audio data
    #     audio = r.listen(source)
    
    try:
        message = r.recognize_whisper(audio)
        print("You said: " + message)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    


if __name__ == "__main__":
    try:
        runlenth=25
        # use_mic_on_background(runlenth)
        use_mic_on_background_not_connected(runlenth)
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
        
        
        
        
    