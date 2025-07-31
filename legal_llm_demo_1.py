import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import json
import os
import gradio as gr
# --- Configuration (from your original code) ---
with open ("/home/user/legal_llm/client_secret.json", "r") as f:
    cred = json.load(f)

PROJECT_ID = cred['installed']['project_id']
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Ensure GOOGLE_APPLICATION_CREDENTIALS is set for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="/home/user/legal_llm/application_default_credentials.json"

# Define the display name of your existing corpus
# IMPORTANT: This assumes 'my_corpus_4' already exists and is populated.
# If you need to create and populate it, you'd run the `create_corpus` and `import_files` sections from your original script first.
display_name = "my_corpus"
display_name_1 = "my_corpus_1"
display_name_2 = "my_corpus_2"
display_name_3 = "my_corpus_3"
display_name_4 = "my_corpus_4"

# --- Retrieve your existing corpus ---
# You can get the corpus by listing and filtering, or if you know its exact name (which is recommended for consistency)
# you can construct it directly if it's already created.
# For simplicity and direct use with an *existing* corpus, we'll assume the resource name structure.
# In a real application, you might use rag.list_corpora() and find it by display_name.
try:
    # Construct the full resource name for the corpus
    corpus_resource_name = "projects/finllm-463312/locations/us-central1/ragCorpora/576460752303423488"#f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{display_name}"
    corpus_resource_name_1 = "projects/finllm-463312/locations/us-central1/ragCorpora/6917529027641081856" #f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{display_name_1}"
    corpus_resource_name_2 = f"projects/finllm-463312/locations/us-central1/ragCorpora/3458764513820540928"
    corpus_resource_name_3 = f"projects/finllm-463312/locations/us-central1/ragCorpora/5188146770730811392"
    corpus_resource_name_4 = f"projects/finllm-463312/locations/us-central1/ragCorpora/1729382256910270464"
    #corpus_resource_name_5 = f"projects/finllm-463312/locations/us-central1/ragCorpora/2594073385365405696"
    # Verify the corpus exists (optional, but good for robust code)
    existing_corpus = rag.get_corpus(corpus_resource_name)
    #print(f"Successfully connected to existing corpus: {existing_corpus.name}")
    existing_corpus_1 = rag.get_corpus(corpus_resource_name_1)
    #print(f"Successfully connected to existing corpus: {existing_corpus_1.name}")
    existing_corpus_2 = rag.get_corpus(corpus_resource_name_2)
    #print(f"Successfully connected to existing corpus: {existing_corpus_2.name}")
    existing_corpus_3 = rag.get_corpus(corpus_resource_name_3)
    #print(f"Successfully connected to existing corpus: {existing_corpus_3.name}")
    existing_corpus_4 = rag.get_corpus(corpus_resource_name_4)
    #print(f"Successfully connected to existing corpus: {existing_corpus_4.name}")
except Exception as e:
    print(f"Error connecting to corpus {display_name}. Make sure it exists and is in {LOCATION}. Error: {e}")
    # If the corpus doesn't exist, you'd need to run the creation and import logic first.
    exit() # Exit or handle the error appropriately if corpus is not found.


# --- Set up Retrieval Configuration ---
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=5,
    #filter=rag.Filter(vector_similarity_threshold=0)
)

# --- Define the Retrieval Tool for the Generative Model ---
# This tool tells the GenerativeModel how to access your RAG corpus
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_retrieval_tool_1 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_1)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_retrieval_tool_2 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_2)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_retrieval_tool_3 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_3)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_retrieval_tool_4 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_4)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)
corpus_resource_name_5 = f"projects/finllm-463312/locations/us-central1/ragCorpora/8646911284551352320"
existing_corpus_5 = rag.get_corpus(corpus_resource_name_5)
#print(f"Successfully connected to existing corpus: {existing_corpus_5.name}")
rag_retrieval_tool_5 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_5)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_6 = f"projects/finllm-463312/locations/us-central1/ragCorpora/6052837899185946624"
existing_corpus_6 = rag.get_corpus(corpus_resource_name_6)
#print(f"Successfully connected to existing corpus: {existing_corpus_6.name}")
rag_retrieval_tool_6 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_6)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_7 = f"projects/finllm-463312/locations/us-central1/ragCorpora/864691128455135232"
existing_corpus_7 = rag.get_corpus(corpus_resource_name_7)
#print(f"Successfully connected to existing corpus: {existing_corpus_7.name}")
rag_retrieval_tool_7 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_7)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_8 = f"projects/finllm-463312/locations/us-central1/ragCorpora/7782220156096217088"
existing_corpus_8 = rag.get_corpus(corpus_resource_name_8)
#print(f"Successfully connected to existing corpus: {existing_corpus_8.name}")
rag_retrieval_tool_8 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_8)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_9 = f"projects/finllm-463312/locations/us-central1/ragCorpora/4323455642275676160"
existing_corpus_9 = rag.get_corpus(corpus_resource_name_9)
#print(f"Successfully connected to existing corpus: {existing_corpus_9.name}")
rag_retrieval_tool_9 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_9)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_10 = f"projects/finllm-463312/locations/us-central1/ragCorpora/4755801206503243776"
existing_corpus_10 = rag.get_corpus(corpus_resource_name_10)
#print(f"Successfully connected to existing corpus: {existing_corpus_10.name}")
rag_retrieval_tool_10 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_10)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_11 = f"projects/finllm-463312/locations/us-central1/ragCorpora/1297036692682702848"
existing_corpus_11 = rag.get_corpus(corpus_resource_name_11)
#print(f"Successfully connected to existing corpus: {existing_corpus_11.name}")
rag_retrieval_tool_11 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_11)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_12 = f"projects/finllm-463312/locations/us-central1/ragCorpora/8214565720323784704"
existing_corpus_12 = rag.get_corpus(corpus_resource_name_12)
#print(f"Successfully connected to existing corpus: {existing_corpus_12.name}")
rag_retrieval_tool_12 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_12)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_13 = f"projects/finllm-463312/locations/us-central1/ragCorpora/3026418949592973312"

existing_corpus_13 = rag.get_corpus(corpus_resource_name_13)
#print(f"Successfully connected to existing corpus: {existing_corpus_13.name}")
rag_retrieval_tool_13 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_13)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_14 = f"projects/finllm-463312/locations/us-central1/ragCorpora/6485183463413514240"
existing_corpus_14 = rag.get_corpus(corpus_resource_name_14)
#print(f"Successfully connected to existing corpus: {existing_corpus_14.name}")
rag_retrieval_tool_14 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_14)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_15 = f"projects/finllm-463312/locations/us-central1/ragCorpora/432345564227567616"
existing_corpus_15 = rag.get_corpus(corpus_resource_name_15)
#print(f"Successfully connected to existing corpus: {existing_corpus_15.name}")
rag_retrieval_tool_15 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_15)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_16 = f"projects/finllm-463312/locations/us-central1/ragCorpora/7349874591868649472"
existing_corpus_16 = rag.get_corpus(corpus_resource_name_16)
#print(f"Successfully connected to existing corpus: {existing_corpus_16.name}")
rag_retrieval_tool_16 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_16)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_17 = f"projects/finllm-463312/locations/us-central1/ragCorpora/3891110078048108544"
existing_corpus_17 = rag.get_corpus(corpus_resource_name_17)
#print(f"Successfully connected to existing corpus: {existing_corpus_17.name}")
rag_retrieval_tool_17 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_17)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_18 = f"projects/finllm-463312/locations/us-central1/ragCorpora/5620492334958379008"
existing_corpus_18 = rag.get_corpus(corpus_resource_name_18)
#print(f"Successfully connected to existing corpus: {existing_corpus_18.name}")
rag_retrieval_tool_18 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_18)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

corpus_resource_name_19 = f"projects/finllm-463312/locations/us-central1/ragCorpora/2161727821137838080"
existing_corpus_19 = rag.get_corpus(corpus_resource_name_19)
#print(f"Successfully connected to existing corpus: {existing_corpus_19.name}")
rag_retrieval_tool_19 = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            # Use the resource name of your existing corpus
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name_19)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)
# --- Initialize the Generative Model with the Retrieval Tool ---
# --- Initialize the Generative Model with a low temperature ---
rag_model = GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[rag_retrieval_tool, rag_retrieval_tool_1, rag_retrieval_tool_2, rag_retrieval_tool_3, 
           rag_retrieval_tool_4, rag_retrieval_tool_5, rag_retrieval_tool_6, rag_retrieval_tool_7, 
           rag_retrieval_tool_8, rag_retrieval_tool_9, rag_retrieval_tool_10, rag_retrieval_tool_11, 
           rag_retrieval_tool_12, rag_retrieval_tool_13, rag_retrieval_tool_14, rag_retrieval_tool_15, 
           rag_retrieval_tool_16, rag_retrieval_tool_17, rag_retrieval_tool_18, rag_retrieval_tool_19],
    #temperature=0.0, # Crucial for strict format adherence
    # max_output_tokens=2048, # Adjust based on the expected length of your detailed answers
)

rag_model_1 = GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[],  # No retrieval tool for this model
)
# # --- Construct your prompt with the template ---
# question_for_model = "What are the data protection requirements for handling personal data in the DIFC?"
# # You would likely embed your retrieved context here as well.
# # For example: context_from_retrieval = "..."

# full_prompt = f"""
# You are an expert legal assistant. Provide a comprehensive answer to the following legal question, strictly adhering to the specified template. Ensure all sections are present and clearly labeled.

# ---
# Legal Question: {question_for_model}
# ---

# Response Template for Each Answer:

# Topic Title: [Your concise subject line here]
# Jurisdiction: [DIFC / ADGM / Both, as applicable]
# Summary / Overview: [Plain-English explanation of the legal position.]
# Relevant Law / Statutes: [Exact legal citations with links, e.g., 'DIFC Law No. 5 of 2020, Article 10 (https://www.difc.ae/laws-regulations/data-protection-law)']
# Analysis / Reasoning: [Step-by-step legal reasoning, definitions, tests, precedent comparisons.]
# Risks / Considerations: [Grey areas, exceptions, procedural traps.]
# Actionable Steps / Recommendations: [Clear practical guidance for the client or user.]
# References / Citations: [Hyperlinked statutes, regulations, rules, judgments. Format: [Title](URL)]

# ---
# Please generate the answer based on the information you have, following this exact template. If a section is not applicable, state 'N/A' or 'Not Applicable' but do not omit the section header.
# """

# # --- Generate the content ---
# response = rag_model.generate_content(full_prompt)

# # --- Extract and save the text ---
# output_text = response.text

# # --- Save to a .txt file ---
# file_name = "legal_answer.txt"
# with open(file_name, "w", encoding="utf-8") as f:
#     f.write(output_text)

# print(f"Response saved to {file_name}")
# # --- Example Usage: Generate Content with RAG ---

print("\n--- RAG Engine Ready. Asking Questions ---")


query1 = "Tell me about the takeover regulations in the ADGM"
#question_for_model = "What are the data protection requirements for handling personal data in the DIFC?"
# You would likely embed your retrieved context here as well.
# For example: context_from_retrieval = "..."

full_prompt = f"""

You are an expert legal assistant specializing in financial and business law. Your task is to provide a comprehensive and accurate answer to the following legal question, drawing primarily from your knowledge and, crucially, **integrating relevant information retrieved from the provided RAG corpus.**

**Strictly adhere to the specified template.** Ensure all sections are present and clearly labeled. If the jurisdiction of the query falls outside of ADGM, you *must* state "I don't know" in the "Jurisdiction" section and then continue to fill out the remaining sections with "N/A" where applicable, *except* for "Information from the corpus," which should reflect if any *irrelevant* information was retrieved.

---
Legal Question: {query1}
---

**Response Template for Each Answer:**

Topic Title:
Jurisdiction:
Summary / Overview:
Relevant Law / Statutes: 
Analysis / Reasoning: 
Risks / Considerations: 
Actionable Steps / Recommendations: 
References / Citations: 
Information from the corpus:

"""

full_prompt_1 = f"""

You are an expert legal assistant specializing in financial and business law. Your task is to provide a comprehensive and accurate answer to the following legal question, drawing primarily from your knowledge and, crucially, **integrating relevant information retrieved from the provided RAG corpus.**

**Strictly adhere to the specified template.** Ensure all sections are present and clearly labeled. If the jurisdiction of the query falls outside of ADGM, you *must* state "I don't know" in the "Jurisdiction" section and then continue to fill out the remaining sections with "N/A" where applicable, *except* for "Information from the corpus," which should reflect if any *irrelevant* information was retrieved.

---
Legal Question: {query1}
---

**Response Template for Each Answer:**

Topic Title: 
Jurisdiction: 
Summary / Overview:
Relevant Law / Statutes: 
Analysis / Reasoning: 
Risks / Considerations: 
Actionable Steps / Recommendations: 
References / Citations: 
"""

print(f"\nQuery: {query1}")
response1 = rag_model.generate_content(full_prompt)
print("\n\n\n Response with RAG:")
print(response1.text)


response2 = rag_model_1.generate_content(full_prompt_1)
print("\n\n\n\n Response without RAG:")
print(response2.text)
# You can also inspect the retrieval results if needed
# if response1.grounding_metadata and response1.grounding_metadata.retrieval_queries:
#     for retrieval_query in response1.grounding_metadata.retrieval_queries:
#         print(f"Retrieved passages for query: {retrieval_query.query}")
#         for passage in retrieval_query.passages:
#             print(f"  Source: {passage.uri}, Text: {passage.text[:100]}...") # Print first 100 chars

# # Question 2
# query2 = "What are the key differences between DIFC and ADGM in terms of corporate governance?"

# full_prompt = f"""
# You are an expert legal assistant. Provide a comprehensive answer to the following legal question, strictly adhering to the specified template. Ensure all sections are present and clearly labeled.

# ---
# Legal Question: {query2}
# ---

# Response Template for Each Answer:

# Topic Title: [Your concise subject line here]
# Jurisdiction: [DIFC / ADGM / Both, as applicable]
# Summary / Overview: [Plain-English explanation of the legal position.]
# Relevant Law / Statutes: [Exact legal citations with links, e.g., 'DIFC Law No. 5 of 2020, Article 10 (https://www.difc.ae/laws-regulations/data-protection-law)']
# Analysis / Reasoning: [Step-by-step legal reasoning, definitions, tests, precedent comparisons.]
# Risks / Considerations: [Grey areas, exceptions, procedural traps.]
# Actionable Steps / Recommendations: [Clear practical guidance for the client or user.]
# References / Citations: [Hyperlinked statutes, regulations, rules, judgments. Format: [Title](URL)]

# ---
# Please generate the answer based on the information you have, following this exact template. If a section is not applicable, state 'N/A' or 'Not Applicable' but do not omit the section header.
# """
# print(f"\nQuery: {query2}")
# response2 = rag_model.generate_content(full_prompt)
# print("Response:")
# print(response2.text)
# # Question 3 - Another example from your original prompt
# query3 = "Help in establishing a restaurant business in Abu Dhabi?"

# full_prompt = f"""
# You are an expert legal assistant. Provide a comprehensive answer to the following legal question, strictly adhering to the specified template. Ensure all sections are present and clearly labeled.

# ---
# Legal Question: {query3}
# ---

# Response Template for Each Answer:

# Topic Title: [Your concise subject line here]
# Jurisdiction: [DIFC / ADGM / Both, as applicable]
# Summary / Overview: [Plain-English explanation of the legal position.]
# Relevant Law / Statutes: [Exact legal citations with links, e.g., 'DIFC Law No. 5 of 2020, Article 10 (https://www.difc.ae/laws-regulations/data-protection-law)']
# Analysis / Reasoning: [Step-by-step legal reasoning, definitions, tests, precedent comparisons.]
# Risks / Considerations: [Grey areas, exceptions, procedural traps.]
# Actionable Steps / Recommendations: [Clear practical guidance for the client or user.]
# References / Citations: [Hyperlinked statutes, regulations, rules, judgments. Format: [Title](URL)]

# ---
# Please generate the answer based on the information you have, following this exact template. If a section is not applicable, state 'N/A' or 'Not Applicable' but do not omit the section header.
# """
# print(f"\nQuery: {query3}")
# response3 = rag_model.generate_content(full_prompt)
# print("Response:")
# print(response3.text)
# def generate_rag_response(user_query: str) -> str:
#     """
#     Generates a response using the RAG-enabled Generative Model.
#     """
#     full_prompt = f"""
#     You are an expert legal assistant. Provide a comprehensive answer to the following legal question, strictly adhering to the specified template. Ensure all sections are present and clearly labeled.

#     ---
#     Legal Question: {user_query}
#     ---

#     Response Template for Each Answer:

#     Topic Title: [Your concise subject line here]
#     Jurisdiction: [DIFC / ADGM / Both, as applicable]
#     Summary / Overview: [Plain-English explanation of the legal position.]
#     Relevant Law / Statutes: [Exact legal citations with links, e.g., 'DIFC Law No. 5 of 2020, Article 10 (https://www.difc.ae/laws-regulations/data-protection-law)']
#     Analysis / Reasoning: [Step-by-step legal reasoning, definitions, tests, precedent comparisons.]
#     Risks / Considerations: [Grey areas, exceptions, procedural traps.]
#     Actionable Steps / Recommendations: [Clear practical guidance for the client or user.]
#     References / Citations: [Hyperlinked statutes, regulations, rules, judgments. Format: [Title](URL)]

#     ---
#     Please generate the answer based on the information you have, following this exact template. If a section is not applicable, state 'N/A' or 'Not Applicable' but do not omit the section header.
#     """
#     if not user_query:
#         return "Please enter a query."
#     try:
#         response = rag_model.generate_content(user_query)
#         return response.text
#     except Exception as e:
#         return f"An error occurred during response generation: {e}"

# # Create the Gradio interface
# iface = gr.Interface(
#     fn=generate_rag_response,
#     inputs=gr.Textbox(lines=2, placeholder="Ask a question related to your documents..."),
#     outputs="textbox",
#     title="Vertex AI RAG-Powered Chatbot",
#     description="Ask questions about your documents, and the model will use Retrieval Augmented Generation to provide answers."
# )

# # # Launch the Gradio app
# print("\nLaunching Gradio interface...")
# iface.launch(share=True) # Set share=True to get a public URL for sharing (might be slow)

