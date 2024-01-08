from fastapi import FastAPI
from fastapi import Body
from dungeon import create_dungeon
from fastapi.responses import FileResponse
app = FastAPI()


@app.post("/image")
async def send_image(type: str = Body()):
    create_dungeon(type)
    return FileResponse("static/generation.jpg")
