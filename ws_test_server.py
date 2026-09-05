"""
Standalone WebSocket test server — no ROS, no OmniGibson dependencies.
Just receives actions from PEFA and echoes back success.

Usage:
    python ws_test_server.py --port 8765
"""
import asyncio
import websockets
import json
import argparse

async def handle_pefa(websocket):
    print("[WS] PEFA connected!")
    async for message in websocket:
        data = json.loads(message)
        print(f"[WS RECV] {data}")
        response = json.dumps({"success": True, "info": "test_mode"})
        print(f"[WS SEND] {response}")
        await websocket.send(response)

async def main(port):
    print(f"[WS] Server listening on 0.0.0.0:{port}")
    async with websockets.serve(handle_pefa, "0.0.0.0", port):
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main(args.port))
