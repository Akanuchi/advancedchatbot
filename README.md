DETAILED PROJECT OVERVIEW

The Advanced Chatbot is a secure, multi-modal AI application built on Python/FastAPI and deployed to Google Kubernetes Engine (GKE) via a GitHub Actions CI/CD Pipeline. It provides users with a unified interface to query both unstructured data (uploaded PDF documents) and structured data (PostgreSQL database hosted on Google Cloud SQL) using powerful Large Language Models (LLMs) and the LlamaIndex framework.

KEY FEATURES

Multi-Modal Querying: Seamlessly switches between querying uploaded PDF documents and translating natural language into SQL queries.

Production CI/CD Pipeline: Automated Build → Push → Deploy → Rollout workflow managed entirely by GitHub Actions.

Secure Deployment: Deployed on GKE for automated orchestration, scaling, and high availability.

Zero-Trust Authentication (Workload Identity): Uses GKE Workload Identity to securely connect pods to the Cloud SQL instance, enforcing the principle of least privilege.

Advanced RAG Pipelines: Implements three distinct PDF retrieval strategies: Vector Store, Document Summary, and Keyword Table indexing, showcasing flexible RAG techniques powered by LlamaIndex.

FastAPI Backend: Provides a high-performance, asynchronous Python web server handling file uploads, API key validation, and health checks.

APPLICATION ARCHITECTURE AND CORE TECHNOLOGIES

The application is built on a modern, high-performance stack starting with Python 3.10. The entire AI and API logic is served by the FastAPI framework running on Uvicorn, providing an asynchronous and rapid web server.

For the core intelligence, the project relies heavily on the LlamaIndex framework, specifically using its Core, LLMs, and Embeddings modules for data indexing, Retrieval-Augmented Generation (RAG), and Natural Language to SQL translation.

The deployment infrastructure is entirely cloud-native, hosted on Google Cloud Platform (GCP), utilizing Google Kubernetes Engine (GKE) for managed container orchestration and scaling. The persistence layer is provided by Google Cloud SQL (PostgreSQL), and the entire deployment lifecycle—from code commit to active service—is managed by a robust GitHub Actions CI/CD Pipeline. For secure database access, the system implements Workload Identity and the Cloud SQL Auth Proxy.

PDF RETRIEVAL STRATEGIES (RAG Pipelines)

The core logic, encapsulated in the AdvancedChatBot class, offers three distinct and flexible strategies for querying PDF documents:

Vector Store Index: This is the standard RAG method. It uses the VectorStoreIndex from LlamaIndex and OpenAI Embedding models to find semantic matches for a user's question within the document text.

Document Summary Index: This strategy utilizes the DocumentSummaryIndex, which pre-processes the PDF into high-level summaries. This allows for faster responses to broad questions and is ideal for quick, high-level analysis.

Keyword Table Index: The KeywordTableIndex creates a simple keyword lookup table. This method is best suited for finding direct, specific phrase matches and targeted information within the document quickly.


DEPLOTMENT AND CI/CD PIPELINE WORKFLOW
The deployment process is entirely automated through the GitHub Actions pipeline, ensuring consistent, one-click deployments from the main branch.

The pipeline executes a sequence of six main stages:

Authentication: The pipeline first authenticates the GitHub Runner to Google Cloud using the google-github-actions/auth action, leveraging a secured GCP Service Account Key stored in GitHub Secrets.

Build & Push: It then builds the Docker image locally based on the Dockerfile and uses docker push to upload the resulting container image to Artifact Registry (us-east4-docker.pkg.dev/...).

GKE Connection: Credentials for the target GKE cluster (advancedchatbot-cluster) are retrieved and configured locally on the runner using the google-github-actions/get-gke-credentials action.

Secrets & Deploy: The workflow dynamically creates the Kubernetes app-secret using sensitive values (like API keys and constructed DATABASE_URL) passed from GitHub Secrets. It then applies the deployment.yaml and service.yaml manifests to the cluster using kubectl apply.

Rollout: Finally, a kubectl rollout restart command is issued to the deployment to ensure the running pods are immediately replaced with new versions pulling the latest Docker image from Artifact Registry.


