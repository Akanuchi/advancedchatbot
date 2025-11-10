from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from AdvancedChatBot import AdvancedChatBot

app = FastAPI()
bot = AdvancedChatBot()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query_bot(request: Request, file: UploadFile = File(...), question: str = Form(...)):
    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the uploaded PDF
    bot.process_all(file_path)
    answer = bot.query(question)

    # Optionally remove file after processing
    os.remove(file_path)

    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": answer}
    )
