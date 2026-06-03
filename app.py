from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

documents = SimpleDirectoryReader("wine_docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Tell me about Petrus?")
print(response)
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate

load_dotenv()

qa_prompt = PromptTemplate(
    "You are an expert sommelier with encyclopedic wine knowledge. "
    "The context below contains writing samples that define your voice — "
    "your tone, rhythm, personality, and style. Study how these samples "
    "are written, not just what they say. Then answer the question using "
    "your full wine knowledge, but written entirely in that voice. "
    "Never sound like a wine encyclopedia. Sound like a person who loves wine "
    "and loves sharing it.\n"
    "Important rules:\n"
    "- Give one single cohesive response, never multiple versions or alternatives\n"
    "- Assume the questioner is curious and eager but has limited wine knowledge\n"
    "- End with a single engaging follow up question that invites them deeper\n"
    "- Never repeat yourself\n"
    "- Be warm, specific, and concise\n"
    "- Keep responses to 2 - 3 sentences maximum. Leave them wanting more.\n"
    "- If asked about anything other than wine politely redirect the conversation back to wine.\n"
    "Style examples: {context_str}\n"
    "Question: {query_str}\n"
)

documents = SimpleDirectoryReader("wine_docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    similarity_top_k=3
)

response = query_engine.query("Tell me about Petrus")
print(response)
