import os
import shutil
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine
from llama_index.core import SQLDatabase

from AdvancedChatBot import AdvancedChatBot

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Uploads dir
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Env configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")  # postgresql://user:pass@host:5432/dbname

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Configure LLM via env for OpenAI SDKs
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Optional SQL database (only needed for database mode)
sql_database: Optional[SQLDatabase] = None
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    sql_database = SQLDatabase(engine)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    question: str = Form(...),
    retriever: str = Form(...)
):
    try:
        # Database query mode (no file required)
        if retriever == "database":
            if not sql_database:
                raise HTTPException(status_code=400, detail="Database connection not configured.")
            bot = AdvancedChatBot(retriever_type="database", sql_database=sql_database)
            answer = bot.query_database(question)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "question": question,
                "answer": answer,
                "retriever": retriever
            })

        # PDF modes: vector, summary, keyword
        if not file or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a valid PDF for document query modes.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        bot = AdvancedChatBot(pdf_path=file_path, retriever_type=retriever)
        bot.process_pdf()
        answer = bot.query_pdf(question)

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception:
            pass

        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "answer": answer,
            "retriever": retriever
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
