import vertexai
from vertexai import rag
import json
import os

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

print(f"Project ID: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print("\nListing available corpora...")

try:
    # List all corpora in the project
    corpora = rag.list_corpora()
    print(f"\nFound {len(corpora)} corpora:")
    for corpus in corpora:
        print(f"- {corpus.name}")
        print(f"  Display name: {corpus.display_name}")
        print(f"  Description: {corpus.description}")
        print()
        
except Exception as e:
    print(f"Error listing corpora: {e}")
    print("\nThis might indicate permission issues. Please check:")
    print("1. Your service account has the 'Vertex AI User' role")
    print("2. The Vertex AI API is enabled in your project")
    print("3. Your credentials file is correct") 