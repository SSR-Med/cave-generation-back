from fastapi import FastAPI
from fastapi import Body
from dungeon import create_dungeon
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/image")
async def send_image(type: str = Body()):
    create_dungeon(type)
    return FileResponse("static/generation.jpg")
