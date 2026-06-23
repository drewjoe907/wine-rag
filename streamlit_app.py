import streamlit as st
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KY"]
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate


st.title("🍷 Wine Assistant")
st.caption("Ask me anything about wine")
st.markdown("""<style>
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
header [data-testid="stBaseButton-headerNoPadding"] {display: none;}
button[kind="headerNoPadding"] {display: none;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_index():
    documents = SimpleDirectoryReader("wine_docs").load_data()
    return VectorStoreIndex.from_documents(documents)

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
    "- Keep responses to 3-4 sentences maximum. Leave them wanting more.\n"
    "- You are knowledgeable about all beverages including wine, spirits, cocktails, beer, coffee, tea, and soft drinks. Answer questions about any beverage with the same warmth and expertise.\n"
    "- Always respond in the same language the user writes in.\n"
    "Style examples: {context_str}\n"
    "Question: {query_str}\n"
)

index = load_index()
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    similarity_top_k=5
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a wine..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(str(response))
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
