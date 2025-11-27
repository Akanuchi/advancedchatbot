import os
import shutil
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, text # ADDED: text for simple DB query
from sqlalchemy.exc import OperationalError # ADDED: For health check
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
# Database URL is now expected to point to the local proxy: 
# postgresql://user:pass@127.0.0.1:5432/dbname
DATABASE_URL = os.getenv("DATABASE_URL") 

# Debug log for DATABASE_URL
print("DATABASE_URL from env:", DATABASE_URL)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global SQL engine setup
SQL_ENGINE = None
if DATABASE_URL:
    # Use a global engine that may be initialized before the proxy is ready
    SQL_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ADDED: Kubernetes Health Check
@app.get("/healthz")
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    if SQL_ENGINE:
        try:
            # Try a simple connection test to ensure the proxy and DB are alive
            with SQL_ENGINE.connect() as connection:
                connection.execute(text("SELECT 1"))
            return {"status": "ok", "db_connected": True}
        except OperationalError:
            # This will cause the readiness/liveness probe to fail if the proxy/DB isn't ready
            raise HTTPException(status_code=503, detail="Database connection failed (Auth Proxy not ready or DB unavailable).")
    
    # If no database is configured, just return OK
    return {"status": "ok", "db_connected": False}


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
            if not SQL_ENGINE:
                raise HTTPException(status_code=400, detail="Database connection not configured.")
            
            # The SQLDatabase object is created only when needed, ensuring the latest connection state
            sql_database = SQLDatabase(SQL_ENGINE)
            bot = AdvancedChatBot(retriever_type="database", sql_database=sql_database)
            answer = bot.query_database(question)
            
            return templates.TemplateResponse("index.html", {
                "request": request,
                "question": question,
                "answer": answer,
                "retriever": retriever
            })

        # ... (rest of the PDF handling logic remains the same)
        # ... (omitted for brevity, assume PDF logic is unchanged)
        
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
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))