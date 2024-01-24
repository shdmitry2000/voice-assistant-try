import asyncio
from aiohttp import web
from fastapi import FastAPI ,WebSocket , Request ,Depends ,Response ,UploadFile,File ,Query ,Body
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import json
from socket_helper import *
from audio_utility import *
import uvicorn
import signal
import labSpeachrecognitionImpl
import multiprocessing
from enum import Enum
from pydantic import Field
    
from typing import Any

# from deepgram import Deepgram

os.environ['KMP_DUPLICATE_LIB_OK']='True'
LENGHT_IN_SEC: int = 30    # We'll process this amount of audio data together maximum
SILENCE_DELETE_SEC=1

pool = None
recognizer= labSpeachrecognitionImpl.LabRecognizer()

    
# app = FastAPI()
# method_name="recognize_google"
# is_operating = True # Global variable to keep track of operation status
# app = FastAPI()
# audio_data= asyncio.Queue()
# templates = Jinja2Templates(directory="templates")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    method_name="recognize_google"
    # is_operating = True # Global variable to keep track of operation status
    app = FastAPI()
    audio_data= asyncio.Queue()
    templates = Jinja2Templates(directory="templates")
    
    
        
    async def _socket_hndle_task(websocket: WebSocket,helper:socket_helper_webm):
        # global is_operating
        try:
            while True:
                data = await websocket.receive()
                
                # print("data arived>",data)
                # print("recived",data)
                if  data["type"] == "websocket.disconnect":
                    
                    print("close socket")
                    # is_operating=False
                    break
                elif data["type"] == "websocket.receive":
                    bin_data = data.get("bytes")
                    if bin_data is not None:
                        print("perform add_chank")
                        await helper.add_chank(bin_data)
                        print(" add_chank finished")
                    else:
                        text_data = data.get("text")
                        print("text_data is not None",text_data is not None)
                        if text_data is not None:
                            try:
                                json_object = json.loads(text_data)
                                if 'operation' in json_object and json_object['operation'] == 'start':
                                    helper.reset()
                                    nearRealTime = bool(json_object.get('nearRealTime', False))
                                    helper.near_real_time=nearRealTime
                                    print(f"Start operation, nearRealTime: {nearRealTime}",helper.near_real_time,json_object)
                                    # is_operating = True
                            except ValueError as e:
                                print(text_data)
                                if text_data == 'start':
                                    print(f"Start operation, no json:")
                                    helper.reset()
                                    # is_operating = True
                                elif text_data == 'stop':
                                    print("Stop operation")
                                    await helper.finalize_data()
                                    print("Stop operation")
                                    # is_operating = False
                                else:
                                    print("other",data)
                                    print("replay sent")
                                    await websocket.send_text("not supported yet!") # Send number of bytes as 8-byte big-endian integer 
                        
                                  
        except (WebSocketDisconnect) as e:
            print("connection closed 2",e)


        
    async def wait_text(helper:socket_helper_webm):
        # data = await websocket.receive_bytes()
        # data= helper.get_next_audio()
        print("wait_text")
        # if not helper.finish_event.is_set():
        the_text=await helper.get_current_text()
        print("the_text",the_text)
        return the_text
        # else:
        #     await asyncio.wait(1)
        #     raise RuntimeError("We finished trancribing")
        # data = await audio_data.get()
        

        
        
    async def _perform_data(websocket: WebSocket,helper:socket_helper_webm):
    # global is_operating
        try:
            while True:
                
                
                the_text = await wait_text(helper)
                # if helper.finalized:
                #     break
                print("recived data:",the_text)
                # if is_operating :
                
                await websocket.send_text(str(the_text))
                # else:
                #     await asyncio.sleep(1) # Sleep for a short period if not operating
        except (WebSocketDisconnect,RuntimeError) as e:
            print("connection closed,error",e)
        

        
    @app.websocket("/listen")
    async def websocket_endpoint(websocket: WebSocket):
        # global pool
        await websocket.accept()
        
        
        loop = asyncio.get_running_loop()
        helper=socket_helper_webm(recognizer= recognizer,pool=pool)
        alive_task = loop.create_task(_socket_hndle_task(websocket,helper))
        send_task = loop.create_task(_perform_data(websocket,helper))

        try:
            # await asyncio.gather(alive_task, send_task)
            done, pending = await asyncio.wait([alive_task,send_task])

            for task in pending:
                print("cancel task",task)
                task.cancel()
        except (asyncio.exceptions.CancelledError) as e:
            print("task cancel ,error",e)
        
        # try:
        #     done, pending = await asyncio.wait([alive_task, send_task])
        #     # done, pending = await asyncio.wait([alive_task])

        #     for task in pending:
        #         print("cancel task",task)
        #         task.cancel()
        # except (asyncio.exceptions.CancelledError) as e:
        #     print("task cancel ,error",e)
                
            
        # if websocket.application_state.active:
        #     await websocket.close()
            
    class Sayhandler(BaseModel):
        say: str
        # language: Optional[str] = None
        
        
    @app.post('/say/')
    async def say_handler(sah_handler:Sayhandler):
        print("say")
        try:

            answer = sah_handler.say
            speech_file_path = Path(__file__).parent / "speech.mp3"
        
            vg=voice.VoiceGenerator()
            vg.speak_to_file_openai(answer,speech_file_path)
            # vg.speak_to_file_google(answer,speech_file_path)

            return FileResponse(speech_file_path, media_type="audio/wav")
            # # Return the wav file as a response
            # return web.FileResponse(speech_file_path)

            
        except Exception as error:
                    result = "Error"
                    print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                    traceback.print_tb(error.__traceback__)


    async def parse_body(request: Request):
        data: bytes = await request.body()
        return data

    @app.post("/predict")
    async def predict(wav_data: bytes = Depends(parse_body)):
        print("in post",len(wav_data))
        
        try:  
            helper_cur=audio_handler_helper(recognizer= recognizer,pool=pool,method_name='recognize_google')
            await helper_cur.add_chank(wav_data)
            await helper_cur.finalize_data()
            
            return {"prediction":str(await helper_cur.get_current_text())}
            
        except Exception as error:
                    result = "Error"
                    print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                    traceback.print_tb(error.__traceback__)

    class Method(str, Enum):
        recognize_google = "recognize_google"
        recognize_openAI = "recognize_openAI"
        recognize_whisper = "recognize_whisper"
        recognize_Transformer = "recognize_Transformer"
        recognize_asr = "recognize_asr"
        recognize_whisper_full = "recognize_whisper_full"
        recognize_azure = "recognize_azure"
        # recognize_tensorflow = "recognize_tensorflow"
        
    
    @app.post('/transcribe')
    async def transcribe_audio(
        method: Method = Method.recognize_google,
        file: UploadFile = File(...),
        language: str = Query("he", description="Language of the audio"),
        options: Any = None):
        try:
            # Save the uploaded file temporarily
            with open(file.filename, 'wb+') as f:
                f.write(await file.read())

            # Transcribe the audio file
            # This part depends on the transcription service you're using
            # Here's a placeholder for where you'd call the transcription service
            file_audio_data = load_audioSource_from_file(file_path=file.filename)
            mpthelper = MultiprocessingTranscriberHelper(recognizer=recognizer,method_name=method,pool=pool)
            if not options is  None:
                options_dict = json.loads(options)
                print(type(options_dict) ,options_dict)
                the_text=await mpthelper.perform_multiprocess(audio_data=file_audio_data,language=language,**options_dict)
            else:
                the_text=await mpthelper.perform_multiprocess(audio_data=file_audio_data,language=language)
            # os.remove(file.filename)
            # print("the_text",the_text)
        
            return {"prediction":str( the_text)}
        except Exception as error:
                    result = "Error"
                    print("An error occurred while performing recognition:", type(error).__name__, "–", error) # An error occurred: NameError – name 'x' is not defined
                    traceback.print_tb(error.__traceback__)
        

    @app.get('/test')
    async def test(request):
        return templates.TemplateResponse("index_test.html", {"request": request})


    #index
    @app.get("/")
    def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})


    # pool = multiprocessing.Pool(initializer=init_worker,processes=5) # limit to 10 processes
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
if __name__ == "__main__":
    pool = multiprocessing.Pool(initializer=init_worker,processes=5) # limit to 10 processes
    asyncio.run(main())
    