from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from AdvancedChatBot import AdvancedChatBot

app = FastAPI()
bot = AdvancedChatBot()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
def query_bot(request: Request, url: str = Form(...), question: str = Form(...)):
    bot.process_all(url)
    answer = bot.query(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer})
