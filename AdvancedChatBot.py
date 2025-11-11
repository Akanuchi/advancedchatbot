from llama_index.core import VectorStoreIndex, DocumentSummaryIndex, KeywordTableIndex, SimpleDirectoryReader
from llama_index.core import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.schema import Document
import os

class AdvancedChatBot:
    def __init__(self, pdf_path=None, retriever_type="vector"):
        self.pdf_path = pdf_path
        self.retriever_type = retriever_type
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.2)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)
        self.index = None
        self.retriever = None

    def load_pdf(self):
        loader = SimpleDirectoryReader(input_files=[self.pdf_path])
        docs = loader.load_data()
        return docs

    def build_index(self, docs):
        if self.retriever_type == "vector":
            self.index = VectorStoreIndex.from_documents(docs, service_context=self.service_context)
            self.retriever = VectorIndexRetriever(index=self.index)
        elif self.retriever_type == "summary":
            self.index = DocumentSummaryIndex.from_documents(docs, service_context=self.service_context)
            self.retriever = SummaryIndexRetriever(index=self.index)
        elif self.retriever_type == "keyword":
            self.index = KeywordTableIndex.from_documents(docs, service_context=self.service_context)
            self.retriever = KeywordTableSimpleRetriever(index=self.index)
        else:
            raise ValueError("Invalid retriever type")

    def query(self, question):
        nodes = self.retriever.retrieve(question)
        context = "\n".join([n.get_content() for n in nodes])
        prompt = f"""Based on the following context, answer the question concisely:

Context:
{context}

Question:
{question}

Answer:"""
        return self.llm.complete(prompt).text.strip()

    def process_all(self):
        docs = self.load_pdf()
        self.build_index(docs)
