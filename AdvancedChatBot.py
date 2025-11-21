import os
import snowflake.connector
import requests
from bs4 import BeautifulSoup

class AdvancedChatBot:
    def __init__(self, pdf_path=None, url=None, retriever_type="vector", datasource="pdf"):
        self.pdf_path = pdf_path
        self.url = url
        self.retriever_type = retriever_type
        self.datasource = datasource

        # Load Snowflake credentials from environment variables (set in Kubernetes/Deployment)
        self.sf_user = os.getenv("SF_USER")
        self.sf_password = os.getenv("SF_PASSWORD")
        self.sf_account = os.getenv("SF_ACCOUNT")
        self.sf_warehouse = os.getenv("SF_WAREHOUSE")
        self.sf_database = os.getenv("SF_DATABASE")
        self.sf_schema = os.getenv("SF_SCHEMA")

    def process_all(self):
        # Stub for pre-processing; wire PDF/URL ingestion as needed
        if self.datasource == "url" and self.url:
            try:
                html = requests.get(self.url, timeout=20).text
                soup = BeautifulSoup(html, "html.parser")
                self.scraped_text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])[:5000]
            except Exception as e:
                self.scraped_text = f"Failed to scrape URL: {e}"
        elif self.datasource == "pdf" and self.pdf_path:
            # Implement your PDF parsing/extraction here
            self.scraped_text = f"PDF uploaded: {os.path.basename(self.pdf_path)}"
        else:
            self.scraped_text = None

    def _connect_snowflake(self):
        # Basic validation
        required = [self.sf_user, self.sf_password, self.sf_account, self.sf_warehouse, self.sf_database, self.sf_schema]
        if any(v is None or v == "" for v in required):
            raise RuntimeError("Missing Snowflake environment variables (SF_USER, SF_PASSWORD, SF_ACCOUNT, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA).")
        conn = snowflake.connector.connect(
            user=self.sf_user,
            password=self.sf_password,
            account=self.sf_account,
            warehouse=self.sf_warehouse,
            database=self.sf_database,
            schema=self.sf_schema,
        )
        return conn

    def query(self, question: str) -> str:
        # Route to datasource
        if self.datasource == "snowflake":
            try:
                conn = self._connect_snowflake()
                cur = conn.cursor()
                # Replace this with your NL-to-SQL or fixed queries
                cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_VERSION();")
                rows = cur.fetchall()
                cur.close()
                conn.close()
                return f"Snowflake connection OK. Info: {rows}"
            except Exception as e:
                return f"Snowflake query failed: {e}"

        # URL/text mode
        if self.datasource == "url":
            if self.scraped_text:
                return f"URL scraped content preview: {self.scraped_text[:500]}"
            return "No content scraped from URL."

        # PDF mode
        if self.datasource == "pdf":
            if self.scraped_text:
                return f"PDF processed: {self.scraped_text}"
            return "No PDF provided."

        return "Unsupported datasource."
