import os
import requests
import snowflake.connector
from bs4 import BeautifulSoup
from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    KeywordTableIndex,
    Settings,
    Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    SummaryIndexRetriever,
    KeywordTableSimpleRetriever
)
from llama_index.readers.file import PDFReader

class AdvancedChatBot:
    def __init__(self, pdf_path=None, url=None, retriever_type="vector", datasource="pdf"):
        self.pdf_path = pdf_path
        self.url = url
        self.retriever_type = retriever_type
        self.datasource = datasource

        # Configure LLM and embedding
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.2)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        self.retriever = None

    def load_documents(self):
        docs = []
        if self.pdf_path and self.datasource == "pdf":
            reader = PDFReader()
            docs.extend(reader.load_data(file=self.pdf_path))
        if self.url and self.datasource == "url":
            html = requests.get(self.url, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            print(f"Scraped text from URL: {text[:200]}...")
            docs.append(Document(text=text))
        return docs

    def build_index(self, docs):
        if self.retriever_type == "vector":
            self.index = VectorStoreIndex.from_documents(docs)
            self.retriever = VectorIndexRetriever(index=self.index)
        elif self.retriever_type == "summary":
            self.index = DocumentSummaryIndex.from_documents(docs)
            self.retriever = SummaryIndexRetriever(index=self.index)
        elif self.retriever_type == "keyword":
            self.index = KeywordTableIndex.from_documents(docs)
            self.retriever = KeywordTableSimpleRetriever(index=self.index)

    def query(self, question):
        if self.datasource == "snowflake":
            return self.query_snowflake(question)
        else:
            nodes = self.retriever.retrieve(question)
            context = "\n".join([node.get_content() for node in nodes])
            prompt = f"""Based on the following context, answer the question concisely:

Context:
{context}

Question:
{question}

Answer:"""
            response = self.llm.complete(prompt)
            return response.text.strip()

    def query_snowflake(self, prompt):
        """Convert natural language prompt to SQL and query Snowflake."""
        sql_prompt = f"Convert the following request into a valid Snowflake SQL query:\n\n{prompt}\n\nSQL:"
        sql_query = self.llm.complete(sql_prompt).text.strip()
        print(f"Generated SQL: {sql_query}")

        # Connect to Snowflake (credentials should be set via env vars or secrets)
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return f"SQL: {sql_query}\nResults: {results}"

    def process_all(self):
        if self.datasource in ["pdf", "url"]:
            docs = self.load_documents()
            if not docs:
                raise ValueError("No valid documents found from PDF or URL.")
            self.build_index(docs)
