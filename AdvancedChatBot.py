import os
import pdfplumber
from typing import List, Optional

from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    KeywordTableIndex,
    Settings,
    Document,
    SQLDatabase,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    SummaryIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine


class AdvancedChatBot:
    def __init__(self, pdf_path: Optional[str] = None, retriever_type: str = "vector", sql_database: Optional[SQLDatabase] = None):
        self.pdf_path = pdf_path
        self.retriever_type = retriever_type
        self.sql_database = sql_database

        # Models
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.2)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Global settings for LlamaIndex
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        self.retriever = None

    def load_pdf(self) -> List[Document]:
        """Extract text from PDF using pdfplumber and wrap into LlamaIndex Document."""
        if not self.pdf_path:
            raise ValueError("PDF path not provided for PDF retrieval mode.")

        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise ValueError("No readable text extracted from PDF. If the PDF is scanned, enable OCR.")

        return [Document(text=text)]

    def build_index(self, docs: List[Document]) -> None:
        """Build index and retriever based on selected strategy."""
        if self.retriever_type == "vector":
            self.index = VectorStoreIndex.from_documents(docs)
            self.retriever = VectorIndexRetriever(index=self.index)
        elif self.retriever_type == "summary":
            self.index = DocumentSummaryIndex.from_documents(docs)
            self.retriever = SummaryIndexRetriever(index=self.index)
        elif self.retriever_type == "keyword":
            self.index = KeywordTableIndex.from_documents(docs)
            self.retriever = KeywordTableSimpleRetriever(index=self.index)
        else:
            raise ValueError(f"Invalid retriever type: {self.retriever_type}")

    def query_pdf(self, question: str) -> str:
        """Query the selected PDF retriever and generate a response."""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized. Call process_pdf() first.")

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

    def process_pdf(self) -> None:
        """Full PDF pipeline: load, index, and prepare retriever."""
        docs = self.load_pdf()
        self.build_index(docs)

    def query_database(self, question: str) -> str:
        """Translate natural language into SQL and execute against Postgres."""
        if not self.sql_database:
            raise ValueError("SQL database not provided for database query mode.")

        query_engine = NLSQLTableQueryEngine(self.sql_database)
        response = query_engine.query(question)
        return str(response)
