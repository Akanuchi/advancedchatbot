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


app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Load environment variables
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Cloud SQL Proxy runs inside container as localhost
DB_HOST = "127.0.0.1"
DB_PORT = 5432

# Validate required env vars early
missing_vars = []
for var_name, var_value in [
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("DB_USER", DB_USER),
    ("DB_PASSWORD", DB_PASSWORD),
    ("DB_NAME", DB_NAME),
]:
    if not var_value:
        missing_vars.append(var_name)

if missing_vars:
    print("❌ Missing environment variables:", missing_vars)

# Construct DB URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print("DATABASE_URL constructed:", DATABASE_URL)

# Required for LLM libraries
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global SQL engine
SQL_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
def health_check():
    """
    Kubernetes liveness & readiness probe.
    Checks whether DB connection is alive.
    """
    try:
        with SQL_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True}
    except Exception as e:
        print("❌ Health check DB error:", str(e))
        raise HTTPException(status_code=503, detail="Database connection failed.")


@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    question: str = Form(...),
    retriever: str = Form(...)
):
    try:
        # ------------------------
        # Database mode
        # ------------------------
        if retriever == "database":
            sql_database = SQLDatabase(SQL_ENGINE)
            bot = AdvancedChatBot(retriever_type="database", sql_database=sql_database)
            answer = bot.query_database(question)

            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "question": question,
                    "answer": answer,
                    "retriever": retriever,
                },
            )

        # ------------------------
        # PDF mode
        # ------------------------
        if not file or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        bot = AdvancedChatBot(pdf_path=file_path, retriever_type=retriever)
        bot.process_pdf()
        answer = bot.query_pdf(question)

        # Clean up
        try:
            os.remove(file_path)
        except:
            pass

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "question": question,
                "answer": answer,
                "retriever": retriever,
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        print("❌ Internal server error:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
