import os
import shutil
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from llama_index.core import SQLDatabase

from AdvancedChatBot import AdvancedChatBot

app = FastAPI(title="AdvancedChatBot API")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load secrets from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SESSION_SECRET = os.getenv("SESSION_SECRET")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

DB_HOST = "127.0.0.1"  # Cloud SQL Proxy runs locally
DB_PORT = 5432

# Construct DATABASE_URL dynamically
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print("DATABASE_URL constructed:", DATABASE_URL)

# Set OPENAI_API_KEY for any library using it
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global SQL engine
SQL_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)

# Startup check: validate DB connection
@app.on_event("startup")
def startup_event():
    try:
        with SQL_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful.")
    except OperationalError as e:
        print(f"❌ Database connection failed: {e}")

# Health check endpoint
@app.get("/healthz")
def health_check():
    try:
        with SQL_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True}
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database connection failed.")

# Root endpoint
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Query endpoint
@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    question: str = Form(...),
    retriever: str = Form(...)
):
    try:
        # Database retrieval
        if retriever == "database":
            if not SQL_ENGINE:
                raise HTTPException(status_code=400, detail="Database not configured.")
            
            sql_database = SQLDatabase(SQL_ENGINE)
            bot = AdvancedChatBot(retriever_type="database", sql_database=sql_database)
            answer = bot.query_database(question)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "question": question,
                "answer": answer,
                "retriever": retriever
            })

        # PDF retrieval
        if not file or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a valid PDF.")

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
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
