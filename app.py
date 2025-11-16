@app.post("/query", response_class=HTMLResponse)
async def query_bot(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(""),
    question: str = Form(...),
    retriever: str = Form("vector"),
    datasource: str = Form("pdf")
):
    try:
        pdf_path = None
        if file and file.filename and datasource == "pdf":
            pdf_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        bot = AdvancedChatBot(
            pdf_path=pdf_path,
            url=url.strip(),
            retriever_type=retriever,
            datasource=datasource
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
