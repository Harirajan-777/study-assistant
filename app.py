from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

# Load Groq key from Streamlit secrets (Cloud)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


VECTOR_DB_PATH = "./anat_vector_db"


# -------------------------------
# Chatbot Class
# -------------------------------
class OrthoChatbot:
    def __init__(self):

        # Embeddings (must match what was used during ingestion)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Groq LLaMA 3.3
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    def get_rag_chain(self):

        if not os.path.exists(VECTOR_DB_PATH):
            return None

        vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_template("""
You are a Senior Orthopedic Surgery Consultant.

Use ONLY the provided textbook context to answer the question.
If the answer is not present, say clearly:
"Not found in the orthopedic textbook database."

--------------------
TEXTBOOK CONTEXT:
{context}
--------------------

QUESTION:
{input}

SURGICAL CONSULTATION ANSWER:
""")

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        return rag_chain


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Anatomy Study Assistant", page_icon="🩺")
st.title("Shru's Study Assistant 🩺 📕 🌸")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Ask an anatomy question..."):

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        bot = OrthoChatbot()
        rag_chain = bot.get_rag_chain()

        if rag_chain is None:
            st.error("❌ Vector database not found in deployment.")
        else:
            with st.spinner("📚 Consulting Bd Chaurasia..."):
                try:
                    response = rag_chain.invoke({"input": query})

                    answer = response["answer"]
                    context_docs = response["context"]

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    with st.expander("📝 Evidence (Retrieved Textbook Chunks)"):
                        for doc in context_docs:
                            st.caption(
                                f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                            )
                            st.write(doc.page_content)

                except Exception as e:
                    st.error(f"⚠️ Error during retrieval: {e}")
