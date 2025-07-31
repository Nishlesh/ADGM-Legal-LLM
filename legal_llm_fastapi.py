import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationConfig
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- Configuration ---
# Set up authentication - try to get project_id from client_secret.json first
try:
    with open("./client_secret.json", "r") as f:
        client_cred = json.load(f)
    PROJECT_ID = client_cred.get('installed', {}).get('project_id')
    if not PROJECT_ID:
        raise ValueError("Could not find project_id in client_secret.json")
except Exception as e:
    print(f"Warning: Could not read client_secret.json: {e}")
    # Fallback to application_default_credentials.json
    try:
        with open("./application_default_credentials.json", "r") as f:
            app_cred = json.load(f)
        # For user credentials, we need to get project_id from environment or set it manually
        PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT') or "finllm-463312"
        print(f"Using project_id: {PROJECT_ID}")
    except Exception as e2:
        print(f"Error reading credentials: {e2}")
        PROJECT_ID = "finllm-463312"  # Fallback to hardcoded project_id
        print(f"Using fallback project_id: {PROJECT_ID}")

# Set up credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./application_default_credentials.json"

LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Corpus resource names (replace with your actual corpus IDs)
corpus_resource_names = [
    "projects/finllm-463312/locations/us-central1/ragCorpora/576460752303423488",
    "projects/finllm-463312/locations/us-central1/ragCorpora/6917529027641081856",
    "projects/finllm-463312/locations/us-central1/ragCorpora/3458764513820540928",
    "projects/finllm-463312/locations/us-central1/ragCorpora/5188146770730811392",
    "projects/finllm-463312/locations/us-central1/ragCorpora/1729382256910270464"
]

# Verify corpora exist with better error handling
available_corpora = []
for corpus_name in corpus_resource_names:
    try:
        corpus = rag.get_corpus(corpus_name)
        print(f"Successfully connected to corpus: {corpus.name}")
        available_corpora.append(corpus_name)
    except Exception as e:
        print(f"Warning: Could not connect to corpus {corpus_name}: {e}")
        print("This corpus will be skipped. Please verify the corpus exists and you have proper permissions.")

if not available_corpora:
    print("ERROR: No corpora are accessible. Please check your permissions and corpus IDs.")
    print("Available permissions needed:")
    print("- aiplatform.ragCorpora.get")
    print("- aiplatform.ragCorpora.list")
    print("- aiplatform.ragCorpora.search")
    exit(1)

# --- Set up Retrieval Configuration ---
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,
    filter=rag.Filter(vector_distance_threshold=0.5)
)

# Create retrieval tools for available corpora only
rag_tools = []
if available_corpora:
    rag_tools = [
        Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                    rag_retrieval_config=rag_retrieval_config,
                ),
            )
        ) for corpus_name in available_corpora
    ]
else:
    print("WARNING: No RAG tools created - no accessible corpora found")

# --- Initialize the Generative Model ---
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001",  # Updated to working model
    tools=rag_tools
)

# FastAPI app
app = FastAPI(
    title="Legal LLM API",
    description="A RAG-powered legal document query API using Vertex AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    query: str

def generate_rag_response(user_query: str) -> str:
    """
    Generates a response using the RAG-enabled Generative Model.
    """
    if not user_query:
        return "Please enter a query."

    try:
        response = rag_model.generate_content(
            user_query,
            generation_config=GenerationConfig(
                max_output_tokens=2048,  # Increased for better responses
                temperature=0.3,        # Balanced creativity
                top_p=0.9
            )
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal LLM API is running",
        "endpoints": {
            "/query": "POST - Submit a query to the RAG system",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "gemini-2.0-flash-001"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Submit a query to the RAG system and get a response.
    """
    try:
        response_text = generate_rag_response(request.query)
        return QueryResponse(response=response_text, query=request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)