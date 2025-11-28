import os
import shutil
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from llama_index.core import SQLDatabase
from AdvancedChatBot import AdvancedChatBot

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SESSION_SECRET = os.getenv("SESSION_SECRET")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 5432))

# Construct DATABASE_URL
if not DB_USER or not DB_PASSWORD or not DB_NAME:
    raise RuntimeError("Database credentials are not set in environment variables.")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print("DATABASE_URL constructed:", DATABASE_URL)

# Set OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global SQL engine
SQL_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz")
def health_check():
    try:
        with SQL_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True}
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database connection failed.")

@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    question: str = Form(...),
    retriever: str = Form(...)
):
    try:
        if retriever == "database":
            sql_database = SQLDatabase(SQL_ENGINE)
            bot = AdvancedChatBot(retriever_type="database", sql_database=sql_database)
            answer = bot.query_database(question)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "question": question,
                "answer": answer,
                "retriever": retriever
            })

        if not file or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a valid PDF.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        bot = AdvancedChatBot(pdf_path=file_path, retriever_type=retriever)
        bot.process_pdf()
        answer = bot.query_pdf(question)

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
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
