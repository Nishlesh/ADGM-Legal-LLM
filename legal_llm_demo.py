import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationConfig
import json
import os
import gradio as gr

# --- Configuration ---
with open("./client_secret.json", "r") as f:
    cred = json.load(f)

PROJECT_ID = cred['installed']['project_id']
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./application_default_credentials.json"

# Corpus resource names (replace with your actual corpus IDs)
corpus_resource_names = [
    "projects/finllm-463312/locations/us-central1/ragCorpora/576460752303423488",
    "projects/finllm-463312/locations/us-central1/ragCorpora/6917529027641081856",
    "projects/finllm-463312/locations/us-central1/ragCorpora/3458764513820540928",
    "projects/finllm-463312/locations/us-central1/ragCorpora/5188146770730811392",
    "projects/finllm-463312/locations/us-central1/ragCorpora/1729382256910270464"
]

# Verify corpora exist
for corpus_name in corpus_resource_names:
    try:
        corpus = rag.get_corpus(corpus_name)
        print(f"Successfully connected to corpus: {corpus.name}")
    except Exception as e:
        print(f"Error connecting to corpus {corpus_name}: {e}")
        exit()

# --- Set up Retrieval Configuration ---
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,
    filter=rag.Filter(vector_distance_threshold=0.5)
)

# Create retrieval tools for all corpora
rag_tools = [
    Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    ) for corpus_name in corpus_resource_names
]

# --- Initialize the Generative Model ---
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001",  # Updated to working model
    tools=rag_tools
)

# --- Gradio Interface ---
def generate_rag_response(user_query: str) -> str:
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

iface = gr.Interface(
    fn=generate_rag_response,
    inputs=gr.Textbox(lines=3, placeholder="Ask about legal documents...", label="Your Question"),
    outputs=gr.Textbox(label="Generated Response"),
    title="Legal Document Assistant (RAG-Powered)",
    description="Ask questions about legal documents. Powered by Vertex AI Gemini with RAG.",
    examples=[
        ["What are the requirements for business registration in ADGM?"],
        ["Explain the data protection laws in this jurisdiction"],
        ["Summarize the key points about intellectual property rights"]
    ]
)

# Launch with sharing enabled
if __name__ == "__main__":
    iface.launch(share=True, server_port=7860)