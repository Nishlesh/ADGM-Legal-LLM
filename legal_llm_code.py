from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import json
from tqdm import tqdm
with open ("./client_secret.json", "r") as f:
    cred = json.load(f)
cred

PROJECT_ID = cred['installed']['project_id']
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define where your source files are stored
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="./application_default_credentials.json"
paths = [
    # "https://drive.google.com/file/d/1-EV7_3KSWjYVhsvipoZt2lminE6KK-Y1/view?usp=drive_link",
    # "https://drive.google.com/file/d/18BPuD1YnfIz69u_NfZLsmTy8RJSMEdd-/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1ql4YMkjSZkryG_f9JuiD5tKZZuRMo5h-/view?usp=drive_link", 
    # "https://drive.google.com/file/d/15UcEQ7DAc9dPMWChw0QAd6Zes67OA6KU/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1VgrrA5Pd9sKuPQJjrVjSxmps4bpNV6Wj/view?usp=drive_link", 
    # "https://drive.google.com/file/d/14Y1MA12Pyd4C7bC6s0R07evJZKor8h8Y/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1WSpDh7A16Ghv36sP39dXowzinK9IdUC6/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1B3SxyaEQyoLPM6NcXZwxJ83CJtBZbYEt/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1jfyxHheEl3lz0TrpV6UfxrK-4_IY4F-b/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1yco8a5dlRKYf0qnCApmeP6DL0jfkmkjg/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1diHCTyibMA1oFvsBUmhpsXYjxIzH7r5P/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1GgHOKjupAYbHcUTJYSa7K6-ucw_ei2k7/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1iaVwn30Kmw0ezviuZKYkYzAHAfJcy3FI/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1ZDkg-pBeC7_CVNVcVcdJSiXJ9qkVvHfo/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1eTtv7FhuMFiGaUsO1CRhCN6HGvsmwmJJ/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1Sb5Ka3zen9CK3IFdFl4NQQ8ktBwFk_Z6/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1grx8qrJZopCFxlL2FR_2_nKkyjjoh_qy/view?usp=drive_link",
    # "https://drive.google.com/file/d/1kNrWL0RZVDvSUl9ebasqjmjzBUTuw-Qs/view?usp=drive_link",
    # "https://drive.google.com/file/d/1JtpIkN-yPxSHhZ6tp8tb4hKWP_XAD_qF/view?usp=drive_link",
    # "https://drive.google.com/file/d/1VkxdY2WYlZqHM9GSaAZQHMobG8IaHS6p/view?usp=drive_link",
    # "https://drive.google.com/file/d/1GjhE5xAeXTW63RO__AL1VRXRxH7_O-nn/view?usp=drive_link", 
    # "https://drive.google.com/file/d/18pb_8MNf6uM1ZCguvU-EVCJuGUSC9aPS/view?usp=drive_link", 
    # "https://drive.google.com/file/d/18lIUrcP8yQO4JOivhom43uvymEvWqzIC/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1T2iQC9gvygH8AYlu78yW-ZjQSSIJtzsy/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1O6Fl_j-RMBf4gg_Xr6Xym2MiS7VDr9C6/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1lPXQneTErCqw_4HX1MS3i8iLcX3p_3Bt/view?usp=drive_link", 
    # "https://drive.google.com/file/d/12Nrv8tv6YYOo1ZtMfnzkhYSXCjsk8wBI/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1CWCWIQ3NF_fDaIcdeY0NETzp1cj5xk4_/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1inbWA_UoZVPgGJwnvzUIZPbEcgTDntXm/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1bJnB97WQcAs8EjcKRknzMT9iu3x6H2bb/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1HT5rXYsdiEwMWCjTYdf-R-lsK9swu22A/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1YPLUgYxID-1EeEokOSUXOVqxqp4jOSfa/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1i9jgGfND5tjLpZGqMZBTNc9fN-GgZL6Z/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1aN7cI-yo58zz0SbHf-u821g9EX_mI3ft/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1mSW1BuFHs6Fv-gpafxh1_6hH-pVTMIJF/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1Cl7RfMM9LSShkBlwjX6Iwvu8xpaD1rs4/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1R4M8tn37eYgJ_fWokKNVI8-JeIcvRA_j/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1ABmc7ySBl1t61voNAULUXZ1G1_Mx6RlN/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1hub6h1tdVZYppvULHIf_rnKRb4dtIlPu/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1_PrtqVXI1b7F2VFsf2v9ptAvh2dKJ163/view?usp=drive_link", 
    # "https://drive.google.com/file/d/18nZy5ue5iOfpfw2gOyH3xRn7p0qhzIit/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1jEgkfnouWwmawQiQUp30MwjABQxF1FV7/view?usp=drive_link", 
    # "https://drive.google.com/file/d/17H_C7yLzig1lg4P8KEsEXtfDhMzmsaxt/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1f-PRRRmyF2pdH_EOSWwpOyy_d1TOBw5r/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1HxWqbvtsANcjV7lLINHSYJNeq060u-te/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1U1HfEm6ewpIKUj4Qe_Z4VZkd-64zr9NR/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1AID1kG4e98E8q94ZJCFKIn_8L43cCIQc/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1yzoO2uiAC6szrf5wXXLZKKVUhEteCKTq/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1bBYi3PsdZIFjnxCeWQkzqeSJJbuvnXVR/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1F4VKWp-vVMnx2o6sjKi04YPPF0X_Jlpk/view?usp=drive_link",
    # "https://drive.google.com/file/d/1yDsvgAtBgF-8wmm98MPWCsihLYU66Rhk/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1reulgbzfTat7YphryJ70grrDCuzs9kJC/view?usp=drive_link", 
    # "https://drive.google.com/file/d/18541_ZLjJKlI0YdK37ftnCQQ5dgdR2AZ/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1QevhmVsmcINjmI01KeLn1Vz9p1Rl0YqU/view?usp=drive_link", 
    # "https://drive.google.com/file/d/17vKcBwwKz1DPwX-iBNggoBJhIVoZ0S8e/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1qqL2NE7SC2EpXELQ3fL-LOuIwKZ9Dxll/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1i0DXUX9yEaVoMCYg3XayGsXqcpxC6YAc/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1qBwCnL01kpD4IrobnL4PUbxr2wxGQec9/view?usp=drive_link", "https://drive.google.com/file/d/1gF_o4W3Yg5slyVlQ8p7kClmKToIhiuhX/view?usp=drive_link", "https://drive.google.com/file/d/1pebQGq5FwWBtcuW4wyFJyNWTDhSk_NiC/view?usp=drive_link", "https://drive.google.com/file/d/1CsXWsotNjqDL4NhV81K7dAlHOuxAjIjH/view?usp=drive_link",
    # "https://drive.google.com/file/d/15mXVkyLdRgw5w9r25BX1SjeEZ0Wc27Ct/view?usp=drive_link", "https://drive.google.com/file/d/1eQTD_93mPeNB-TR9CFS1Nk-26JSOL4it/view?usp=drive_link", "https://drive.google.com/file/d/1r4KFifLvAk83HCeFGeG5XvAfA6uDOart/view?usp=drive_link", "https://drive.google.com/file/d/1_bYsYQCmiCAPGumuTug7A2nu3vprkBYM/view?usp=drive_link", "https://drive.google.com/file/d/1htHJaQfKBCfEaPk7zFtpT3jbHh4Vxzjt/view?usp=drive_link", "https://drive.google.com/file/d/1BdkmTjxYOUC1yXnPdzTNDC5Cxw7040XP/view?usp=drive_link", "https://drive.google.com/file/d/1o9j2EtA-I90H2R9jOvpvK-V1wBbuoiXV/view?usp=drive_link", "https://drive.google.com/file/d/1_zJ1W0Vi7zhSjgqM_DA7X--lGRqA3c6v/view?usp=drive_link", "https://drive.google.com/file/d/1xOjkx_xhDTV6pd-5LZa5zZORbjy9Zrw9/view?usp=drive_link",
    # "https://drive.google.com/file/d/1WBDtrExQPWTGafktLf0Un3I5igwN7i1X/view?usp=drive_link", "https://drive.google.com/file/d/1J-JgYEpvNLetk1Y44eElMMlz7728lkwA/view?usp=drive_link", "https://drive.google.com/file/d/10SZqgpTilvFVTOh5OnF435w9qe8HB801/view?usp=drive_link", "https://drive.google.com/file/d/1qqM-VzrEN2rLJzXxgQMbbL6qLGY7aPPe/view?usp=drive_link", "https://drive.google.com/file/d/1KlVkQ--s30Xp01COILkWW3uy1etWLQCe/view?usp=drive_link", "https://drive.google.com/file/d/1JJYipe8-euR1nIuGaI1_cVrtuBrxGezz/view?usp=drive_link", 
    # "https://drive.google.com/file/d/19X77z5oRCcwXS0Cre3tfzX07f8LN4opV/view?usp=drive_link",
    # "https://drive.google.com/file/d/1XxZf_3njxfYFI3UWPGb9qVEJ7gSk--wA/view?usp=drive_link", "https://drive.google.com/file/d/1UNKOMbDr_EPAZPGokLKoesr5-jA56LvT/view?usp=drive_link", "https://drive.google.com/file/d/1CC0U1auBFp9QesjwPIDjJVBODwsfqJ2h/view?usp=drive_link", "https://drive.google.com/file/d/1vZ-Fo-MSltu96vzur7EN5Xi9wdZArjpb/view?usp=drive_link", "https://drive.google.com/file/d/1UHfZjXdIQcSgGjTJmLFtBCsM9k9zpvae/view?usp=drive_link", "https://drive.google.com/file/d/1pMtxikB_rCHS8Ba03uA03CCPMKdBzsxr/view?usp=drive_link", "https://drive.google.com/file/d/1kR-X5DBiF9-Wua0WZ38GCVWkZz7VB3P6/view?usp=drive_link", "https://drive.google.com/file/d/1HEdk4PDbZ7pY_VoIqA6DmfCoYvVX__Yq/view?usp=drive_link", "https://drive.google.com/file/d/149YHT_JUrlE738NGYFlpkNW4pJp-qDj4/view?usp=drive_link", "https://drive.google.com/file/d/16IBpIMTohjZ7kaqjdyLFvVgfhzMascr1/view?usp=drive_link", "https://drive.google.com/file/d/10AMTfCFYT339ZGq1EWeaJ60obDKY4-mf/view?usp=drive_link", "https://drive.google.com/file/d/1PhjaFG3No33lQCCYtaKNnFDnDyFxYd29/view?usp=drive_link", "https://drive.google.com/file/d/1ArZh1s7-3x-5ETy6MASgYhI_ZDaLInK8/view?usp=drive_link", "https://drive.google.com/file/d/1eVQmlGJemSl5LZmVMVC3NZnXCO_c6RPp/view?usp=drive_link", "https://drive.google.com/file/d/1nXwfTGHmR8zZiqKhqnUs2hQwiaCDpQno/view?usp=drive_link", "https://drive.google.com/file/d/1J5iTY-DzJt8nYHjtzLweGYvOG1o5AZKG/view?usp=drive_link", "https://drive.google.com/file/d/1de-4CPq1_HyeCiaCXKjS-JrcNXLUrhtc/view?usp=drive_link", "https://drive.google.com/file/d/1MfZ4Vy0JAdMsIzaLtJztkGHUQCd8lH5y/view?usp=drive_link", "https://drive.google.com/file/d/1jdJejHUv_bGeF6riJY7OXR3SxzD2jPFV/view?usp=drive_link", "https://drive.google.com/file/d/1SEQU5FPTdUZ3j-MLtse2qEriQ9r_0S3D/view?usp=drive_link", "https://drive.google.com/file/d/1Tp2bKixRDRpceyOZqP1G-MeZpwpDC2nq/view?usp=drive_link", "https://drive.google.com/file/d/1WKtwAimbGwQInWBg6MsOkIXej6I1tpjJ/view?usp=drive_link",
    # "https://drive.google.com/file/d/1N48PohPHscQWOhCtGMAvMtJKbHliQy46/view?usp=drive_link",
    # "https://drive.google.com/file/d/17X9AedoZEaK2cFWHUqj3iM-DbYh_ktCQ/view?usp=drive_link", 
    # "https://drive.google.com/file/d/17WhMmyk2-UtSSQwah-VkPAGwSvdv9CUN/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1ul9JGBgoTG098moFyXRpn3Y6wIriDNfC/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1nkgQsKYL2D3848z-uveHiRbn-Z9hgniu/view?usp=drive_link",
    # "https://drive.google.com/file/d/1B6Chsy0PjZfu9mFSDuL_ZE8eQ2UCBupE/view?usp=drive_link",
    # "https://drive.google.com/file/d/1A8Qe-H2SeVpnnHDIZMo3ec-jPHNQ-Ipl/view?usp=drive_link",
    # "https://drive.google.com/file/d/14wKRg2BCUx_8hNOtgxNCklYyng46pKEd/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1B7f1Ptj9ujhm5DpGXwOBlD8rGhu8YQDs/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1qFyklv1stRHfxPL3bm_TGNXoXaRo0f_I/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1zGTyl5uGikW_eaqip9T_dyHkS-Hgs3hv/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1TaeI4Kqa74W7UoJ_OBo2Jk6_BFdTtpW1/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1jsQZ2XMmq1i9S60hFIe7hrDPQqOndKVZ/view?usp=drive_link", 
    # "https://drive.google.com/file/d/1lEXcSlfSeyVJVAsa8QxEiPIhXrIn1-bH/view?usp=drive_link"
    #"gs://ADGM procedures"
    "https://drive.google.com/file/d/18AZV-fbvh2CvDpOGEHv3vErCbb1UJ2Fi/view?usp=drive_link", 
    "https://drive.google.com/file/d/1EZpJBwJu9FaNbpiilxF-4fPu3vWyIp7u/view?usp=drive_link", 
    "https://drive.google.com/file/d/1Z3Y_-f65PTvOaw4mi9msO2UE8SPeBfiZ/view?usp=drive_link", 
    "https://drive.google.com/file/d/1JCkZi0aPKuHn4OlCOnY3u-T8lj-uC3Nc/view?usp=drive_link", 
    "https://drive.google.com/file/d/1FCymbPK-87Fwct3X24BDbecbkE958g2t/view?usp=drive_link", 
    "https://drive.google.com/file/d/1o9JmX0QLoNdh5PElmawc7a1dxZbC0a8o/view?usp=drive_link", 
    "https://drive.google.com/file/d/1eC8Y8hxds9LU--9lvOAMDwuLDq7EzbZs/view?usp=drive_link", 
    "https://drive.google.com/file/d/10dDpv7T9lLTS7BxkrAZhqLyPjpVc7FmO/view?usp=drive_link"   
]

# Optional: change display_name and embedding model if needed
display_name = "my_corpus_5"

embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# Create the corpus for your RAG project
rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    )
)

paths1 = paths
# Import files into the corpus, configure chunking as needed
for path in tqdm(paths1, desc="Importing files"):
    rag.import_files(
        rag_corpus.name,
        [path],  # Wrap in list to keep parameter type expected
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000
    )

# Set up retrieval configuration
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,
    filter=rag.Filter(vector_distance_threshold=0.5)
)

# Retrieve relevant context for a specific query
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(rag_corpus=rag_corpus.name),
    ],
    text="What is RAG and why is it helpful?",
    rag_retrieval_config=rag_retrieval_config,
)

print(response)

# Use Gemini model for full generation
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[rag_retrieval_tool]
)

response = rag_model.generate_content("Help in establishing a restaurant business in Abu Dhabi?")
print(response.text)