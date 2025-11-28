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
# Load environment variables (Simplified)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Startup Validation ---
if not OPENAI_API_KEY:
    print("❌ Critical: OPENAI_API_KEY not found. Application cannot start.")
    # Raise error to stop the application gracefully (but prevent Uvicorn from starting)
    raise ValueError("OPENAI_API_KEY not set.") 

if not DATABASE_URL:
    print("❌ Critical: DATABASE_URL not found. Database features will fail.")
    # Raise error to stop the application
    raise ValueError("DATABASE_URL not set.") 

# Required for LLM libraries
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("✅ Database URL configured successfully.")

# Global SQL engine - Will fail here if URL or credentials are wrong
try:
    SQL_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    print(f"❌ Error creating SQL engine: {e}")
    # Raise error to stop the application
    raise RuntimeError(f"Failed to initialize SQL database: {e}")


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
            # Use 'SELECT 1' to check connection without querying tables
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True}
    except Exception as e:
        print("❌ Health check DB error:", str(e))
        # This will trigger the Kubernetes probe failure (503 status code)
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
