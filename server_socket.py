# https://www.section.io/engineering-education/how-to-set-up-a-python-web-socket-with-aiohttp/

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


import speech_recognition as sr
import os ,time
import voice,labSpeachrecognitionImpl
from typing import Dict, List
import numpy as np
from bidi.algorithm import get_display

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import multiprocessing
import signal

os.environ['KMP_DUPLICATE_LIB_OK']='True'

recognizer= labSpeachrecognitionImpl.LabRecognizer()

transcription_tasks=[]

SILENCE_DELETE_SEC=1

routes = web.RouteTableDef()



@routes.post('/request_post_test')
# https://stackoverflow.com/questions/65371754/aiohttp-web-post-method-get-params
async def post_handler_test(request):
    
    
    print('\n--- request ---\n')

    # ----------------------------------------------------------------------

    # print('ARGS string:', request.query_string)  # arguments in URL as string
    # print('ARGS       :', request.query)         # arguments in URL as dictionary

    # ----------------------------------------------------------------------

    # >> it can't use at the same time: `content.read()`, `text()`, `post()`, `json()`, `multiparts()` 
    # >> because they all read from the same stream (not get from variable) 
    # >> and after first read this stream is empty

    # ----------------------------------------------------------------------

    print('BODY bytes :', await request.content.read())  # body as bytes  (post data as bytes, json as bytes)
    # print('BODY string:', await request.text())          # body as string (post data as string, json as string)

    # ----------------------------------------------------------------------

    print('POST       :', await request.post())         # POST data

    # ----------------------------------------------------------------------

    try:
       print('JSON:', await request.json())  # json data as dictionary/list
    except Exception as ex:
       print('JSON: ERROR:', ex)

    # ----------------------------------------------------------------------

    try:
        #print('MULTIPART:', await request.multipart())  # files and forms
        reader = await request.multipart()
        print('MULTIPART:', reader)
        while True:
            part = await reader.next()
            if part is None: 
                break
            print('filename:', part.filename)
            print('>>> start <<<')
            print(await part.text())
            print('>>> end <<<')
    except Exception as ex:
        print('MULTIPART: ERROR:', ex)

    # ----------------------------------------------------------------------

    return web.Response(text='Testing...')



@routes.post('/predict')
async def predict_handler(request):
    wav_data: bytes = await request.content.read()
    try:
        
        segment =AudioSegment.from_wav(BytesIO(wav_data))
        # segment = AudioSegment._from_safe_wav(data)
        # play(segment)
        
        #closed temprory
        audio = labSpeachrecognitionImpl.AudioData(segment.raw_data, segment.frame_rate,
                               segment.sample_width)
        
        # audio_data = voice.Trnscriber.get_audio_data_from_wav_data(data)
        
        # #serial  
        # result = transcribe(recognizer, audio)
        # return web.json_response({"prediction":str(result)})
    
        #paralel
        transcription_task = pool.apply_async(transcribe, (recognizer,audio) )#, callback=print_result)
        return web.json_response({"prediction":str(transcription_task.get())})
        

        
    except Exception as error:
                result = "Error"
                print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                traceback.print_tb(error.__traceback__)


@routes.get('/test')
async def test(request):
   return web.FileResponse('./templates/index_test.html')


#index
async def index(request):
    return web.FileResponse('./templates/index.html')
   


# Audio settings
LENGHT_IN_SEC: int = 30    # We'll process this amount of audio data together maximum
# STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
# LENGHT_IN_SEC: int = 15    # We'll process this amount of audio data together maximum
# NB_CHANNELS = 1
# RATE = 16000
# CHUNK = RATE
# # Visualization (expected max number of characters for LENGHT_IN_SEC audio)
# MAX_SENTENCE_CHARACTERS = 80


pool = None
method_name=None


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def transcribe(recognizer,audio):
    try:
        transcription_start_time = time.time()
        file_name="save_webm_"+str(transcription_start_time) +".wav"
        voice.Trnscriber.save_audio_from_audio_data(audio,file_name)
        # transcription="test"
        if method_name is None:
            transcription=recognizer.recognize_Transformer(audio,language='he')
        else:
            method = getattr(recognizer, method_name)
            transcription= method(audio,language='he')
        return transcription
    except (LookupError,sr.exceptions.UnknownValueError):
        print("Oops! Didn't catch that")
        return "Oops! Didn't catch that"
    

async def websocket_handler(request):

    print('Websocket connection starting')
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection ready')

    # await asyncio.gather(
    #         close_handler(ws),
    #         stream_handler(ws)
    #     )

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                print("close")
                await ws.close()
            else:
                print(msg,msg.data)
                print("replay sent")
                await ws.send_str(msg.data + '/answer')
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' %
                  ws.exception())
        elif msg.type == aiohttp.WSMsgType.BINARY:
            print(msg,msg.data)
            message="onopen"
            await ws.send_str(message)
        else:
            print(msg,msg.data)

    print('websocket connection closed')

    return ws

# async def process_incoming_audio_messages(ws):
#     async for msg in ws:
        
#         print(msg.data)
#         message="onopen"
#         await ws.send_str(message)



def getSilenceStartStopTime(audio):
    silenc = silence.detect_nonsilent(audio, min_silence_len=250, silence_thresh=-50,seek_step=125)
    # silenc = [((start/1000),(stop/1000)) for start,stop in silenc]
    # silenc = [((start),(stop)) for start,stop in silenc]
    silenc = [(start, stop) for start, stop in silenc if stop - start >= SILENCE_DELETE_SEC*1000]
    print(silenc)
    return silenc


    
async def  perform_webm(ws,msg):
    usedElements = []
    last_element = None
    print("perform_webm start")
    # data_all=b""
    opus_data = io.BytesIO()
    if msg is not None:
        data_all=msg.data
        opus_data.write(data_all)
    while True:
        msg = await ws.receive()
        if msg.type == web.WSMsgType.BINARY:
            data_all=msg.data
            opus_data.write(data_all)
            opus_data.seek(0)
            # audio_data = opus_data.read()
            the_segment = AudioSegment.from_file(opus_data,format="webm")
            silence_times = getSilenceStartStopTime(the_segment)
            if len(silence_times) >= 1:
                if  last_element==silence_times[-1] and last_element not in usedElements:
                    last_element=silence_times[-1]
                    (start_time, stop_time)=last_element
                    chunk = the_segment[start_time:stop_time]
                    try:
                        print("perform chank",start_time, stop_time)
                        audio = labSpeachrecognitionImpl.AudioData(chunk.raw_data, chunk.frame_rate,chunk.sample_width)
                        # #paralel perform
                        transcription_task = pool.apply_async(transcribe, (recognizer,audio) )#, callback=print_result)
                        transcription_tasks.append(transcription_task)
                        transcription_task=transcription_tasks[0]
                        if(transcription_task.ready()):
                            transcription=str(transcription_task.get(timeout=1)) 
                            print("recive trans:",get_display(transcription))
                            await ws.send_str(str(transcription) ) 
                            print("sent to client")
                            if transcription_task in transcription_tasks :
                                print("remove",transcription_task)
                                transcription_tasks.remove(transcription_task)

                    except Exception as e:
                        print(f"Error while printing transcription: {e}")
                        traceback.print_exc()
                        raise  e
                    usedElements.append(last_element)
                else:
                    last_element=silence_times[-1]       
            # if len(silence_times) > 1 and silence_times[-2] not in usedElements:
            #     (start_time, stop_time)=silence_times[-2]
            #     chunk = the_segment[start_time:stop_time]
            #     try:
            #         print("perform chank",start_time, stop_time+150)
            #         audio = labSpeachrecognitionImpl.AudioData(chunk.raw_data, chunk.frame_rate,chunk.sample_width)
            #         # #paralel perform
            #         transcription_task = pool.apply_async(transcribe, (recognizer,audio) )#, callback=print_result)
            #         transcription_tasks.append(transcription_task)
            #         transcription_task=transcription_tasks[0]
            #         if(transcription_task.ready()):
            #             transcription=str(transcription_task.get(timeout=1)) 
            #             print("recive trans:",get_display(transcription))
            #             await ws.send_str(str(transcription) ) 
            #             print("sent to client")
            #             if transcription_task in transcription_tasks :
            #                 print("remove",transcription_task)
            #                 transcription_tasks.remove(transcription_task)

            #     except Exception as e:
            #         print(f"Error while printing transcription: {e}")
            #         traceback.print_exc()
            #         raise  e
            #     usedElements.append(silence_times[-2])
        else:
            if msg.data == 'stop':
                print("stop")
                silence_times = getSilenceStartStopTime(the_segment)
                if last_element!=silence_times[-1]:
                    (start_time, stop_time)=silence_times[-1]
                    print("perform chank",start_time)
                    chunk = the_segment[start_time:]
                    audio = labSpeachrecognitionImpl.AudioData(chunk.raw_data, chunk.frame_rate,chunk.sample_width)
                    transcription_task = pool.apply_async(transcribe, (recognizer,audio) )#, callback=print_result)
                    transcription_tasks.append(transcription_task)
                        
                for transcription_task in transcription_tasks:
                    transcription_task.wait() # wait for the task to complete
                    transcription=str(transcription_task.get(timeout=1)) 
                    print("recive trans:",get_display(transcription))
                    await ws.send_str(str(transcription) ) 
                    print("sent to client")
                    if transcription_task in transcription_tasks :
                        print("remove",transcription_task)
                        transcription_tasks.remove(transcription_task)
                
                break
            else:
                print("other msg",msg)
                if msg.type == web.WSMsgType.CLOSE:
                    print("close")
                    await ws.close()
                    break
                
    


async def audio_handler(request):
    
    ws = web.WebSocketResponse()
    await ws.prepare(request) 

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "stop":
                pass
            elif msg.data == "start":
                await perform_webm(ws,None)
            elif msg.data == 'close':
                print("close")
                await ws.close()
            else:
                print(msg,msg.data)
                print("replay sent")
                await ws.send_str(msg.data + '/answer')
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' %
                    ws.exception())
        if msg.type == web.WSMsgType.BINARY:
            print("in binary")
            await perform_webm(ws,msg)
               
    return ws


def main():
    method_name="recognize_google"
    app = web.Application()
    app.router.add_get('/', index)
    app.add_routes(routes)
    # app.add_routes([web.get('/ws', websocket_handler)])
    app.router.add_route('GET', '/echo', websocket_handler)
    app.router.add_route('GET', '/listen', audio_handler)
    # app.router.add_route('GET', '/predict',prediction_handler)
    web.run_app(app,host='localhost')
    
    

if __name__ == "__main__":
    pool = multiprocessing.Pool(initializer=init_worker,processes=20) # limit to 10 processes
    main()
    
