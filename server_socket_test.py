# https://www.section.io/engineering-education/how-to-set-up-a-python-web-socket-with-aiohttp/

from aiohttp import web
import aiohttp
import concurrent.futures
import asyncio
from speech_recognition import AudioData
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
pool_exc = ThreadPoolExecutor(max_workers=20)
# loop = asyncio.get_event_loop()


# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE
idle_time=1

recognizer= labSpeachrecognitionImpl.LabRecognizer()

# import sys
# from datetime import datetime
# from typing import Optional, List, Union, Any, Dict , Callable , Awaitable ,Tuple ,Metadata , TypedDict


# from enum import Enum

# EventHandler = Union[Callable[[Any], None], Callable[[Any], Awaitable[None]]]


# def cast(typ, val):
#     """Cast a value to a type.

#     This returns the value unchanged.  To the type checker this
#     signals that the return value has the designated type, but at
#     runtime we intentionally don't check anything (we want this
#     to be as fast as possible).
#     """
#     return val

# class LiveTranscriptionEvent(Enum):
#     OPEN = 'open'
#     CLOSE = 'close'
#     TRANSCRIPT_RECEIVED = 'transcript_received'
#     ERROR = 'error'




routes = web.RouteTableDef()


@routes.get('/test')
async def index(request):
   return web.FileResponse('./templates/index.html')

async def perform_chank_save(chunk):
    print("perform chank")
    audio_data = labSpeachrecognitionImpl.AudioData(chunk.raw_data, chunk.frame_rate,
            chunk.sample_width)
    # text = recognizer.recognize_google(audio_data)
    file_name="speach-"+str(time.time())+".webm"
    voice.Trnscriber.save_audio_from_audio_data(audio_data,file_name+".wav")
    return "file saved"

async def perform_chank(chunk):
    print("perform chank")
    audio_data = labSpeachrecognitionImpl.AudioData(chunk.raw_data, chunk.frame_rate,
            chunk.sample_width)
    # text = recognizer.recognize_google(audio_data)
    # return recognizer.recognize_whisper(audio_data,language='he')
    return await asyncio.sleep(60, result=" perform_chank exit"+str(time.time()))

    # callback(recognizer.recognize_whisper(audio_data,language='he'))


# async def execute_blocking_whisper_prediction1(model: WhisperModel, audio_data:sr.AudioData) -> str:
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







# async def long_thing(request):
#   try:
#     body = await request.json()

#     return web.json_response({"status": "success"}), await spawn(
#         request,
#         await long_stuff.really_long(body["field"]),
#     )
#     except Exception as e:

#       return web.json_response(
#           {"status": "failure", "error": str(e), "type": f"{type(e)}"}
#       )
      
      
      
      

@routes.post('/predict')
async def predict_handler(request):
    wav_data: bytes = await request.content.read()
    # async def run_in_paralell_predict(wav_data: bytes):
    try:
        
        segment =AudioSegment.from_wav(BytesIO(wav_data))
        result = await  perform_chank(segment)
        
        return web.json_response({"prediction":str(result)})
    
        
    except Exception as error:
                result = "Error"
                print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                traceback.print_tb(error.__traceback__)

    # task = asyncio.create_task(run_in_paralell_predict(await request.content.read()))
    # loop.call_exception_handler()
    # return await loop.run_in_executor(pool_exc, task)
    
    

async def index(request):
   return web.FileResponse('./templates/index_test.html')


async def handler(request):
    return web.Response()



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

    # print('websocket connection closed')

    # return ws

async def process_incoming_audio_messages(ws):
    async for msg in ws:
        
        print(msg.data)
        message="onopen"
        await ws.send_str(message)

def perform_audio_data1(b_data):
    
#     [pip3] numpy==1.23.4
# [pip3] torch==1.12.1
# [pip3] torchaudio==0.12.1
# [conda] numpy 1.23.4 pypi_0 pypi
    try:
        chunk_size = 1000
        while True:
            f = BytesIO(b_data)
            s = torchaudio.io.StreamReader(f)
            s.add_basic_audio_stream(chunk_size)
            array = torch.concat([chunk[0] for chunk in s.stream()])
            return array
            
            
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')


def perform_audio_data(b_data):
    
    
    try:
        return voice.Trnscriber.get_audio_data_from_wav_data(b_data)
        
            
            
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')





async def  perform_webm_save_webm(ws,msg):
    print("perform_webm")
    opus_data = BytesIO(msg.data)
    transcription_time = time.time()
    file_name="speach-"+str(transcription_time)+".webm"
    print("open file")
    with open(file_name, 'wb') as f:
        data_all=msg.data
        f.write(data_all)
        opus_data.write(data_all)
        while True:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.BINARY:
                data_all=msg.data
                f.write(data_all)
                opus_data.write(data_all)
            else:
                print("break")
                break
    # opus_data.seek(0)
    # segment = AudioSegment.from_file(opus_data,format="webm")
    # audio_data = labSpeachrecognitionImpl.AudioData(segment.raw_data, segment.frame_rate,
    #                            segment.sample_width)
    # voice.Trnscriber.save_audio_from_audio_data(audio_data,file_name+".wav")
    
async def  perform_webm_save_wav(ws,msg):
    print("perform_webm")
    opus_data = io.BytesIO(msg.data)
    data_all=msg.data
    opus_data.write(data_all)
    while True:
        msg = await ws.receive()
        if msg.type == web.WSMsgType.BINARY:
            data_all=msg.data
            opus_data.write(data_all)
        else:
            if msg == "stop":
                print("stop")
                break
            else:
                print("other msg",msg)
                if msg.type == web.WSMsgType.CLOSE:
                    print("close")
                    await ws.close()
                    break
                
    opus_data.seek(0)
    the_segment = AudioSegment.from_file(opus_data,format="webm")
    # silenc = silence.detect_silence(the_segment, min_silence_len=250, silence_thresh=-45)
    # silenc = [((start/1000),(stop/1000)) for start,stop in silenc]
    # print(silenc)
    
    audio_data = labSpeachrecognitionImpl.AudioData(the_segment.raw_data, the_segment.frame_rate,
                               the_segment.sample_width)
    file_name="speach-"+str(time.time())+".webm"
    voice.Trnscriber.save_audio_from_audio_data(audio_data,file_name+".wav")

def getSilenceStartStopTime(audio):
    silenc = silence.detect_nonsilent(audio, min_silence_len=250, silence_thresh=-45)
    # silenc = [((start/1000),(stop/1000)) for start,stop in silenc]
    silenc = [((start),(stop+100)) for start,stop in silenc]
    print(silenc)
    return silenc


    
async def  perform_webm(ws,msg):
    usedElements = []
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
           
            if len(silence_times) > 1 and silence_times[-2] not in usedElements:
                (start_time, stop_time)=silence_times[-2]
                chunk = the_segment[start_time:stop_time]
                try:
                    print("perform chank",start_time, stop_time)
                    
                    # result = await perform_chank(chunk)
                    # result = await asyncio.get_running_loop().run_in_executor(None, perform_chank, segment)
                    # audio_data = labSpeachrecognitionImpl.AudioData(chunk.raw_data, the_segment.frame_rate,
                    #         the_segment.sample_width)
                    # # text = recognizer.recognize_google(audio_data)
                    # file_name="speach-"+str(time.time())+".webm"
                    # voice.Trnscriber.save_audio_from_audio_data(audio_data,file_name+".wav")
                    await ws.send_str(str(perform_chank(chunk)) ) 
                    # await ws.send_str(str(result) ) 
                    # opus_data.seek(0)
                    # opus_data.truncate(0)
                except Exception as e:
                    print(f"Error while printing transcription: {e}")
                    traceback.print_exc()
                    raise  e
                usedElements.append(silence_times[-2])
        else:
            if msg.data == 'stop':
                print("stop")
                silence_times = getSilenceStartStopTime(the_segment)
                (start_time, stop_time)=silence_times[-1]
                print("perform chank",start_time)
                chunk = the_segment[start_time:]
                # result = await asyncio.get_running_loop().run_in_executor(None, perform_chank, chunk)
                
                await ws.send_str(str(perform_chank(chunk)) )  
                # result = await perform_chank(chunk)
                # await ws.send_str(str(result) )  
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
   
    # buf = io.BytesIO()
    # bufferLenth=0
        
    # audio_data = b''
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

app = web.Application()
app.router.add_get('/', index)
app.add_routes(routes)
# app.add_routes([web.get('/ws', websocket_handler)])
app.router.add_route('GET', '/echo', websocket_handler)
app.router.add_route('GET', '/listen', audio_handler)
# app.router.add_route('GET', '/predict',prediction_handler)
web.run_app(app,host='localhost')
