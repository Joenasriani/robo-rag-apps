# rag_as_a_service

Apps in this tier expose the RAG pipeline as a REST API so any client can index documents and query them over HTTP.

## Apps

| App | Description | Status |
|-----|-------------|--------|
| [api_service](api_service/) | FastAPI service with `/index` and `/query` endpoints, OpenAI embeddings + Chroma | ✅ Ready |
