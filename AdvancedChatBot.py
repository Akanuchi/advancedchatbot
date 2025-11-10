from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os


class AdvancedChatBot:
    def __init__(self, pdf_path=None):
        self.text = None
        self.chunks = None
        self.embeddings = None
        self.embedded_documents = None
        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
        self.docs = None
        self.inputs = None
        self.pdf_path = pdf_path

    def load_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        self.text = "\n".join([page.page_content for page in pages])
        return self.text

    def split_text(self):
        """Split text into chunks for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", ".", "?", "!"]
        )
        self.chunks = splitter.split_text(self.text)
        return self.chunks
    
    def embed_text(self):
        """Embed text chunks using OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embedded_documents = self.embeddings.embed_documents(self.chunks)
        return self.embeddings, self.embedded_documents

    def build_vector_db(self):
        """Create a FAISS vector database from embedded documents."""
        docs = [Document(page_content=chunk) for chunk in self.chunks]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return self.vectorstore
    
    def query(self, question, k=3):
        """Query the vector store and generate an LLM answer."""
        if self.vectorstore is None:
            self.build_vector_db()
    
        # Retrieve most relevant text chunks
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n".join(doc.page_content for doc in docs)
    
        # Construct the prompt
        prompt = f"""Based on the following context, answer the question concisely:

Context:
{context}

Question:
{question}

Answer:"""
    
        # Generate answer with LLM
        response = self.llm.invoke(prompt)
        return response.content

    def process_all(self, pdf_path):
        """Full pipeline for a PDF file."""
        self.load_pdf(pdf_path)
        self.split_text()
        self.embed_text()
        self.build_vector_db()
        return True
