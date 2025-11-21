# AdvancedChatBot.py
import os
import logging
from typing import Optional, List

import requests
from bs4 import BeautifulSoup

# PDF extraction (PyMuPDF is fast and reliable)
import fitz  # PyMuPDF

# Snowflake connector
import snowflake.connector

# Optional LLM summarization
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedChatBot")


class AdvancedChatBot:
    """
    Multi-source chatbot:
      - pdf: extract text from uploaded PDF and summarize/answer
      - url: scrape page text and summarize/answer
      - snowflake: run queries against a fixed, preconfigured Snowflake account

    Credentials for Snowflake are loaded from environment variables:
      SF_USER, SF_PASSWORD, SF_ACCOUNT, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA
    Optionally, OpenAI summarization can be used if OPENAI_API_KEY is set.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        url: Optional[str] = None,
        retriever_type: str = "vector",
        datasource: str = "pdf",
    ):
        self.pdf_path = pdf_path
        self.url = url
        self.retriever_type = retriever_type
        self.datasource = datasource

        # Collected text from PDF or URL (preprocessing step)
        self.source_text: Optional[str] = None

        # Load Snowflake credentials from environment variables
        self.sf_user = os.getenv("SF_USER")
        self.sf_password = os.getenv("SF_PASSWORD")
        self.sf_account = os.getenv("SF_ACCOUNT")
        self.sf_warehouse = os.getenv("SF_WAREHOUSE")
        self.sf_database = os.getenv("SF_DATABASE")
        self.sf_schema = os.getenv("SF_SCHEMA")

        # OpenAI setup (optional)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    # -----------------------------
    # Preprocessing for each source
    # -----------------------------
    def process_all(self) -> None:
        """
        Prepare text or connections based on the selected datasource.
        """
        if self.datasource == "pdf":
            self.source_text = self._extract_pdf_text(self.pdf_path)
        elif self.datasource == "url":
            self.source_text = self._scrape_url_text(self.url)
        elif self.datasource == "snowflake":
            # No text to prepare; connections are handled at query time
            self.source_text = None
        else:
            logger.warning(f"Unsupported datasource: {self.datasource}")
            self.source_text = None

    def _extract_pdf_text(self, path: Optional[str]) -> Optional[str]:
        if not path or not os.path.exists(path):
            logger.error("PDF path is missing or does not exist.")
            return None
        try:
            doc = fitz.open(path)
            parts: List[str] = []
            for page in doc:
                parts.append(page.get_text("text"))
            text = "\n".join(parts).strip()
            if not text:
                logger.warning("No text extracted from PDF.")
                return None
            # Limit to avoid extremely large prompts; adjust as needed
            return text[:100_000]
        except Exception as e:
            logger.exception(f"Failed to read PDF: {e}")
            return None

    def _scrape_url_text(self, url: Optional[str]) -> Optional[str]:
        if not url:
            logger.error("URL is missing.")
            return None
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Collect meaningful text (paragraphs + headings)
            texts = []
            for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
                t = tag.get_text(separator=" ", strip=True)
                if t:
                    texts.append(t)
            combined = "\n".join(texts).strip()
            if not combined:
                logger.warning("No significant text extracted from URL.")
                return None
            return combined[:100_000]
        except Exception as e:
            logger.exception(f"Failed to scrape URL: {e}")
            return None

    # -----------------------------
    # Snowflake connectivity
    # -----------------------------
    def _validate_snowflake_env(self) -> None:
        required = {
            "SF_USER": self.sf_user,
            "SF_PASSWORD": self.sf_password,
            "SF_ACCOUNT": self.sf_account,
            "SF_WAREHOUSE": self.sf_warehouse,
            "SF_DATABASE": self.sf_database,
            "SF_SCHEMA": self.sf_schema,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing Snowflake environment variables ({', '.join(missing)})"
            )

    def _connect_snowflake(self):
        self._validate_snowflake_env()
        conn = snowflake.connector.connect(
            user=self.sf_user,
            password=self.sf_password,
            account=self.sf_account,
            warehouse=self.sf_warehouse,
            database=self.sf_database,
            schema=self.sf_schema,
        )
        return conn

    # Example query: count tables
    def _count_tables(self) -> str:
        """
        Returns the count of tables visible via INFORMATION_SCHEMA.TABLES
        within the configured database/schema context.
        """
        conn = None
        try:
            conn = self._connect_snowflake()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES;")
            row = cur.fetchone()
            count = row[0] if row else 0
            return f"There are {count} tables in your database."
        finally:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    # -----------------------------
    # Summarization / QA
    # -----------------------------
    def _summarize_with_openai(self, text: str, instruction: str = "Summarize the following content:") -> str:
        """
        Summarize text using OpenAI if API key is present; otherwise fallback.
        """
        if not self.openai_api_key:
            return self._fallback_summary(text)

        try:
            # Prefer the Chat Completions API with a capable model
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise, accurate summarization assistant."},
                    {"role": "user", "content": f"{instruction}\n\n{text}"}
                ],
                temperature=0.2,
                max_tokens=600
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception(f"OpenAI summarization failed: {e}")
            return self._fallback_summary(text)

    def _fallback_summary(self, text: str) -> str:
        """
        Naive local summary if LLM is unavailable: return first N sentences or characters.
        """
        if not text:
            return "No content available to summarize."
        # Simple heuristic: first ~1200 characters
        snippet = text[:1200]
        return f"(Local summary preview)\n{snippet}"

    # -----------------------------
    # Main query router
    # -----------------------------
    def query(self, question: str) -> str:
        """
        Route the question based on the selected datasource.
        """
        try:
            if self.datasource == "snowflake":
                # Basic intent handling for demonstration
                q_lower = question.lower()
                if "how many tables" in q_lower or "count tables" in q_lower:
                    return self._count_tables()
                # Example: version info (connectivity check)
                if "version" in q_lower or "current version" in q_lower:
                    return self._snowflake_version()
                # Default response if no intent matched
                return "Snowflake connected. Please specify your query (e.g., 'How many tables are in my database?')."

            elif self.datasource in ("pdf", "url"):
                if not self.source_text:
                    return "No content available. Please provide a valid PDF or URL."
                # Summarize or answer generically using the source_text
                if any(term in question.lower() for term in ["summarise", "summarize", "summary"]):
                    return self._summarize_with_openai(self.source_text, "Summarize the following content:")
                # Generic QA prompt
                prompt = f"Answer the user question based on the content below.\n\nContent:\n{self.source_text}\n\nQuestion:\n{question}"
                return self._summarize_with_openai(prompt, "Provide a concise, direct answer:")

            else:
                return "Unsupported datasource."

        except Exception as e:
            logger.exception(f"Query handling failed: {e}")
            return f"Error: {e}"

    def _snowflake_version(self) -> str:
        conn = None
        try:
            conn = self._connect_snowflake()
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_VERSION();")
            rows = cur.fetchall()
            # Format a readable response
            parts = []
            for r in rows:
                parts.append(f"Account: {r[0]}, Region: {r[1]}, Version: {r[2]}")
            return "Snowflake info: " + " | ".join(parts)
        finally:
            try:
                if cur:
                    cur.close()
            except Exception:
                pass
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
