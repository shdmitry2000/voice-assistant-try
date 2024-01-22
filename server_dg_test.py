from typing import Any, Union, Tuple, List, Dict, Awaitable, cast
import json

from deepgram import Deepgram
import asyncio
import aiohttp
from enum import Enum

# Your Deepgram API Key
DEEPGRAM_API_KEY = 'YOUR_DEEPGRAM_API_KEY'

# URL for the realtime streaming audio you would like to transcribe
URL = 'http://stream.live.vc.bbcmedia.co.uk/bbc_world_service'


class LiveTranscriptionEvent(Enum):
    OPEN = 'open'
    CLOSE = 'close'
    TRANSCRIPT_RECEIVED = 'transcript_received'
    ERROR = 'error'

class Caption(Enum):
    SRT = 'srt'
    WEBVTT = 'webvtt'


class LiveTranscription:
    """
    This class allows you to perform live transcription by connecting to Deepgram's Transcribe Streaming API.
    It takes in options for the transcription job, and a callback function to handle events.

    """

    _root = "/listen"
    MESSAGE_TIMEOUT = 1.0

    def __init__(self, options: Options,
                 transcription_options: LiveOptions, endpoint) -> None:
        """
        The __init__ function is called when an instance of the class is created.
        It initializes all of the attributes that are part of the object, and can be
        accessed using "self." notation. In this case, it sets up a list to store any
        messages received from Transcribe Streaming.
        
        :param options:Options: Used to Pass the options for the transcription job.
        :param transcription_options:LiveOptions: Used to Pass in the configuration for the transcription job.
        :return: None.
        
        """

        self.options = options
        if endpoint is not None:
            self._root = endpoint
        self.transcription_options = transcription_options
        self.handlers: List[Tuple[LiveTranscriptionEvent, EventHandler]] = []
        # all received messages
        self.received: List[Union[LiveTranscriptionResponse, Metadata]] = []
        # is the transcription job done?
        self.done = False
        self._socket = cast(websockets.client.WebSocketClientProtocol, None)
        self._queue: asyncio.Queue[Tuple[bool, Any]] = asyncio.Queue()

    async def __call__(self) -> 'LiveTranscription':
        """
        The __call__ function is a special method that allows the object to be called
        as a function. In this case, it is used to connect the client and start the
        transcription process. It returns itself after starting so that operations can
        be chained.
        
        :return: The object itself.
        
        """
        self._socket = await _socket_connect(
            f'{self._root}{_make_query_string(self.transcription_options)}',
            self.options
        )
        asyncio.create_task(self._start())
        return self

    async def _start(self) -> None:
        """
        The _start function is the main function of the LiveTranscription class.
        It is responsible for creating a websocket connection to Deepgram Transcribe,
        and then listening for incoming messages from that socket. It also sends any 
        messages that are in its queue (which is populated by other functions). The 
        _start function will run until it receives a message with an empty transcription, 
        at which point it will close the socket and return.
        
        :return: None.

        """

        asyncio.create_task(self._receiver())
        self._ping_handlers(LiveTranscriptionEvent.OPEN, self)

        while not self.done:
            try:
                incoming, body = await asyncio.wait_for(self._queue.get(), self.MESSAGE_TIMEOUT)
            except asyncio.TimeoutError:
                if self._socket.closed:
                    self.done = True
                    break
                continue

            if incoming:
                try:
                    parsed: Union[
                        LiveTranscriptionResponse, Metadata
                    ] = json.loads(body)
                    # Stream-ending response is only a metadata object
                    self._ping_handlers(
                        LiveTranscriptionEvent.TRANSCRIPT_RECEIVED,
                        parsed
                    )
                    self.received.append(parsed)
                    if 'sha256' in parsed: 
                        self.done = True
                except json.decoder.JSONDecodeError:
                    self._ping_handlers(
                        LiveTranscriptionEvent.ERROR,
                        f'Couldn\'t parse response JSON: {body}'
                    )
            else:
                await self._socket.send(body)
        self._ping_handlers(
            LiveTranscriptionEvent.CLOSE,
            self._socket.close_code
        )

    async def _receiver(self) -> None:
        """
        The _receiver function is a coroutine that receives messages from the socket and puts them in a queue.
        It is started by calling start_receiver() on an instance of AsyncSocket. It runs until the socket is closed,
        or until an exception occurs.
        
        :return: None.

        """

        while not self.done:
            try:
                body = await self._socket.recv()
                self._queue.put_nowait((True, body))
            except websockets.exceptions.ConnectionClosedOK:
                await self._queue.join()
                self.done = True # socket closed, will terminate on next loop

    def _ping_handlers(self, event_type: LiveTranscriptionEvent,
                       body: Any) -> None:
        """
        The _ping_handlers function is a callback that is called when the
        transcription service sends a ping event.  It calls all of the functions
        in self.handlers, which are registered by calling add_ping_handler().
        
        :param event_type:LiveTranscriptionEvent: Used to Determine if the function should be called.
        :param body:Any: Used to Pass the event data to the handler function.
        :return: The list of handlers for the event type.

        """
        
        for handled_type, func in self.handlers:
            if handled_type is event_type:
                if inspect.iscoroutinefunction(func):
                    asyncio.create_task(cast(Awaitable[None], func(body)))
                else:
                    func(body)

    # Public

    def register_handler(self, event_type: LiveTranscriptionEvent,
                         handler: EventHandler) -> None:
        """Adds an event handler to the transcription client."""

        self.handlers.append((event_type, handler))

    # alias for incorrect method name in v0.1.x
    def registerHandler(self, *args, **kwargs):
        warn(
            (
                "This method name is deprecated, "
                "and will be removed in the future - "
                "use `register_handler`."
            ),
            DeprecationWarning
        )
        return self.register_handler(*args, **kwargs)

    def deregister_handler(self, event_type: LiveTranscriptionEvent,
                           handler: EventHandler) -> None:
        """Removes an event handler from the transcription client."""

        self.handlers.remove((event_type, handler))

    # alias for incorrect method name in v0.1.x
    def deregisterHandler(self, *args, **kwargs):
        warn(
            (
                "This method name is deprecated, "
                "and will be removed in the future - "
                "use `deregister_handler`."
            ),
            DeprecationWarning
        )
        return self.deregister_handler(*args, **kwargs)

    def send(self, data: Union[bytes, str]) -> None:
        """Sends data to the Deepgram endpoint."""

        self._queue.put_nowait((False, data))

    def configure(self, config: ToggleConfigOptions) -> None:
        """Sends messages to configure transcription parameters mid-stream."""
        self._queue.put_nowait((False, json.dumps({
            "type": "Configure",
            "processors": config
        })))

    def keep_alive(self) -> None:
        """Keeps the connection open when no audio data is being sent."""
        self._queue.put_nowait((False, json.dumps({"type": "KeepAlive"})))

    async def finish(self) -> None:
        """Closes the connection to the Deepgram endpoint,
        waiting until ASR is complete on all submitted data."""

        self.send(json.dumps({"type": "CloseStream"}))  # Set message for "data is finished sending"
        while not self.done:
            await asyncio.sleep(0.1)

    @property
    def event(self) -> Enum:
        """An enum representing different possible transcription events
        that handlers can be registered against."""

        return cast(Enum, LiveTranscriptionEvent)






async def main():
  # Initialize the Deepgram SDK
  deepgram = Deepgram(DEEPGRAM_API_KEY)

  # Create a websocket connection to Deepgram
  # In this example, punctuation is turned on, interim results are turned off, and language is set to UK English.
  try:
    deepgramLive = await deepgram.transcription.live({
      'smart_format': True,
      'interim_results': False,
      'language': 'en-US',
      'model': 'nova-2',
    })
  except Exception as e:
    print(f'Could not open socket: {e}')
    return

  # Listen for the connection to close
  deepgramLive.register_handler(deepgramLive.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))

  # Listen for any transcripts received from Deepgram and write them to the console
  deepgramLive.register_handler(deepgramLive.event.TRANSCRIPT_RECEIVED, print)

  # Listen for the connection to open and send streaming audio from the URL to Deepgram
  async with aiohttp.ClientSession() as session:
    async with session.get(URL) as audio:
      while True:
        data = await audio.content.readany()
        deepgramLive.send(data)

        # If no data is being sent from the live stream, then break out of the loop.
        if not data:
            break

  # Indicate that we've finished sending data by sending the customary zero-byte message to the Deepgram streaming endpoint, and wait until we get back the final summary metadata object
  await deepgramLive.finish()

# If running in a Jupyter notebook, Jupyter is already running an event loop, so run main with this line instead:
#await main()
asyncio.run(main())
















from flask import Flask, render_template
from deepgram import Deepgram
from dotenv import load_dotenv
import os
import asyncio
from aiohttp import web
from aiohttp_wsgi import WSGIHandler

from typing import Dict, Callable


load_dotenv()

app = Flask('aioflask')

dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))

async def process_audio(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
        
            if transcript:
                await fast_socket.send_str(transcript)

    deepgram_socket = await connect_to_deepgram(get_transcript)

    return deepgram_socket

async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]) -> str:
    try:
        socket = await dg_client.transcription.live({'punctuate': True, 'interim_results': False})
        socket.registerHandler(socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
        socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler)

        return socket
    except Exception as e:
        raise Exception(f'Could not open socket: {e}')

@app.route('/')
def index():
    return render_template('index.html')

async def socket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request) 

    deepgram_socket = await process_audio(ws)

    while True:
        data = await ws.receive_bytes()
        deepgram_socket.send(data)

  

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    aio_app = web.Application()
    wsgi = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info: *}', wsgi.handle_request)
    aio_app.router.add_route('GET', '/listen', socket)
    web.run_app(aio_app, port=5555)