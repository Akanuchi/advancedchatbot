import os
import logging
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedChatBot")

class AdvancedChatBot:
    def __init__(self, pdf_path=None, url=None, retriever_type="vector", datasource="pdf"):
        self.pdf_path = pdf_path
        self.url = url
        self.retriever_type = retriever_type
        self.datasource = datasource
        self.source_text = None

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def process_all(self):
        if self.datasource == "pdf":
            self.source_text = self._extract_pdf_text(self.pdf_path)
        elif self.datasource == "url":
            self.source_text = self._scrape_url_text(self.url)

    def _extract_pdf_text(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            doc = fitz.open(path)
            text = "\n".join([page.get_text("text") for page in doc])
            return text[:100000]
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return None

    def _scrape_url_text(self, url):
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            texts = [tag.get_text(" ", strip=True) for tag in soup.find_all(["h1","h2","h3","p","li"])]
            return "\n".join(texts)[:100000]
        except Exception as e:
            logger.error(f"URL scraping failed: {e}")
            return None

    def query(self, question: str) -> str:
        if not self.source_text:
            return "No content available to answer your question."
        if "summarise" in question.lower() or "summarize" in question.lower():
            return self._summarize(self.source_text)
        return self._summarize(f"Answer the question based on:\n{self.source_text}\n\nQuestion: {question}")

    def _summarize(self, text: str) -> str:
        if not self.openai_api_key:
            return text[:1000]  # fallback preview
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":text}],
                max_tokens=600,
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:1000]
