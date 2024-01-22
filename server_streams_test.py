import asyncio
from typing import Dict
from typing import Tuple

import argparse
import asyncio
import os
from typing import AsyncIterator

CHUNK_SIZE = 100


async def readlines(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while line := await read_until_eol(reader):
        yield line


async def read_until_eol(reader: asyncio.StreamReader) -> bytes:
    """Returns a line of text or empty bytes object if EOF is received.
    """
    data = b''
    sep = os.linesep.encode()
    while data := data + await reader.read(CHUNK_SIZE):
        if sep in data:
            message, _, data = data.partition(sep)
            return message + sep


async def write(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(data)
    await writer.drain()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8080)
    return parser.parse_args()



users: Dict[Tuple[str, int], asyncio.StreamWriter] = {}


async def handle_connection(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
    addr = writer.get_extra_info('peername')
    users[addr] = writer
    print(f'{addr}: Connection established.')

    async for data in readlines(reader):
        print(f'{addr}: {data.decode()!r}')
        writes = (write(writer, data) for user, writer in users.items()
                  if user != addr)
        await asyncio.gather(*writes)

    del users[addr]
    print(f'{addr}: Connection closed by the remote peer.')


async def start_server() -> None:
    # args = parse_args()
    # server = await asyncio.start_server(handle_connection, args.host,
    #                                     args.port)
    server = await asyncio.start_server(handle_connection, '0.0.0.0',
                                        '8080')
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    try:
        asyncio.run(start_server())
    except Exception as e :
            print(f'Connection closed {e.code}') # This line is never reached unless I send anything to the client
            # await websocket.close