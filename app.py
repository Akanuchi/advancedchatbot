from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import shutil, os
from AdvancedChatBot import AdvancedChatBot

# ✅ Define the app before using it
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="super-secret-session-key")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(""),
    question: str = Form(...),
    retriever: str = Form("vector"),
    datasource: str = Form("pdf"),
    sf_user: str = Form(""),
    sf_password: str = Form(""),
    sf_account: str = Form(""),
    sf_warehouse: str = Form(""),
    sf_database: str = Form(""),
    sf_schema: str = Form("")
):
    try:
        pdf_path = None
        if file and file.filename and datasource == "pdf":
            pdf_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Store Snowflake creds in session if provided
        if datasource == "snowflake" and sf_user and sf_password:
            request.session["snowflake"] = {
                "sf_user": sf_user,
                "sf_password": sf_password,
                "sf_account": sf_account,
                "sf_warehouse": sf_warehouse,
                "sf_database": sf_database,
                "sf_schema": sf_schema
            }

        creds = request.session.get("snowflake", {})

        bot = AdvancedChatBot(
            pdf_path=pdf_path,
            url=url.strip(),
            retriever_type=retriever,
            datasource=datasource,
            sf_user=creds.get("sf_user"),
            sf_password=creds.get("sf_password"),
            sf_account=creds.get("sf_account"),
            sf_warehouse=creds.get("sf_warehouse"),
            sf_database=creds.get("sf_database"),
            sf_schema=creds.get("sf_schema")
        )
        bot.process_all()
        answer = bot.query(question)

        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "answer": answer,
            "retriever": retriever,
            "datasource": datasource
        })

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
