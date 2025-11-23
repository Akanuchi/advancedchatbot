import os
import pdfplumber
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


class AdvancedChatBot:
    def __init__(self, pdf_path=None, retriever_type="vector"):
        self.pdf_path = pdf_path
        self.retriever_type = retriever_type
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.2)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Apply global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        self.retriever = None

    def load_pdf(self):
        """Extract text from PDF using pdfplumber and wrap into LlamaIndex Document."""
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise ValueError("No readable text extracted from PDF. Consider using OCR for scanned PDFs.")

        # Correct usage: keyword argument for text
        return [Document(text=text)]

    def build_index(self, docs):
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

    def query(self, question):
        """Query the selected retriever and generate a response."""
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

    def process_all(self):
        """Full pipeline: load, index, and prepare retriever."""
        docs = self.load_pdf()
        self.build_index(docs)
