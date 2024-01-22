import asyncio
import os
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel
# import labSpeachrecognitionImpl
import speech_recognition as sr
import voice
import traceback
from pydub import AudioSegment
from io import BytesIO

app = FastAPI()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

NUM_WORKERS = 10
# MODEL_TYPE = "large-v2"
MODEL_TYPE = "base"
LANGUAGE_CODE = "en"
# LANGUAGE_CODE = None
CPU_THREADS = 4
VAD_FILTER = True
# recognizer= labSpeachrecognitionImpl.LabRecognizer()
recognizer= sr.Recognizer()

model= WhisperModel(MODEL_TYPE,
                           device="cpu",
                           compute_type="int8",
                           num_workers=NUM_WORKERS,
                           cpu_threads=4,
                           download_root="./models")


print("Loaded model")


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


def execute_blocking_whisper_prediction1(model: WhisperModel, audio_data:sr.AudioData) -> str:
    audio_data_array: np.ndarray = np.frombuffer(audio_data.get_wav_data(convert_rate=16000), np.int16).astype(np.float32) / 255.0
    segments, _ = model.transcribe(audio_data_array,
                                   language=LANGUAGE_CODE,
                                   beam_size=5,
                                   vad_filter=VAD_FILTER,
                                   vad_parameters=dict(min_silence_duration_ms=1000))
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    return transcription

# def execute_blocking_whisper_prediction1(model: WhisperModel, audio_data:sr.AudioData) -> str:
#     audio_data_array: np.ndarray = np.frombuffer(audio_data.get_wav_data(convert_rate=16000), np.int16).astype(np.float32) / 255.0
#     segments, _ = model.transcribe(audio_data_array,
#                                    language=LANGUAGE_CODE,
#                                    beam_size=5,
#                                    vad_filter=VAD_FILTER,
#                                    vad_parameters=dict(min_silence_duration_ms=1000))
#     segments = [s.text for s in segments]
#     transcription = " ".join(segments)
#     transcription = transcription.strip()
#     return transcription


# @app.post("/predict")
# async def predict(audio_data: bytes = Depends(parse_body)):
#     # Convert the audio bytes to a NumPy array
#     audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0

#     try:
#         # print(audio_data_array)
#         # Run the prediction on the audio data
#         result = await asyncio.get_running_loop().run_in_executor(None, execute_blocking_whisper_prediction, model,
#                                                                   audio_data_array)

#     except Exception as e:
#         print(e)
#         result = "Error"

#     return {"prediction": result}


def execute_blocking_whisper_prediction(recognizer,audio_data:sr.AudioData) -> str:
    print("sent data to recognition")
    transcription=recognizer.recognize_whisper(audio_data,language='he')
    return transcription


@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/predict")
async def predict(wav_data: bytes = Depends(parse_body)):
    print("in post")
    try:
        # print(audio_data_array)
        # Run the prediction on the audio data

        # data=voice.Trnscriber.get_wav_data_from_audio_data(data,convert_rate=16000)
        
         # data = audio.get_wav_data()
        # segment = AudioSegment._from_safe_wav(data)
        segment =AudioSegment.from_wav(BytesIO(wav_data))
        # segment = AudioSegment._from_safe_wav(data)
        # play(segment)
        audio_data = sr.AudioData(segment.raw_data, segment.frame_rate,
                               segment.sample_width)
        
        # audio_data = voice.Trnscriber.get_audio_data_from_wav_data(data)
        
        print("run executor") 
        result = await asyncio.get_running_loop().run_in_executor(None, execute_blocking_whisper_prediction,
                                                                 recognizer, audio_data)
    except Exception as error:
                result = "Error"
                print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                traceback.print_tb(error.__traceback__)

                
   
        

    return {"prediction": result}

if __name__ == "__main__":
    # Run the FastAPI app with multiple threads
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")