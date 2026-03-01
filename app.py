from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ✅ Modern HuggingFace embeddings (no quota issues)
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Modern LangChain RAG
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate


VECTOR_DB_PATH = "./anat_vector_db"


# -------------------------------
# Ortho Chatbot Class
# -------------------------------
class OrthoChatbot:
    def __init__(self):
        # ✅ Local embeddings (fast + unlimited)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Gemini LLM for answering
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

    def get_rag_chain(self):
        """Creates and returns the Retrieval-Augmented Generation chain."""

        if not os.path.exists(VECTOR_DB_PATH):
            return None

        # Load Chroma vector DB
        vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        # ✅ Prompt Template (modern format)
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

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        rag_chain = create_retrieval_chain(retriever, document_chain)

        return rag_chain


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Anatomy Study Assistant", page_icon="🩺")
st.title("Shru's study Assistant 🩺 📕 🌸")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if query := st.chat_input("Ask an anatomy question..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):

        bot = OrthoChatbot()
        rag_chain = bot.get_rag_chain()

        if rag_chain is None:
            st.error("❌ Vector database not found. Please run `python ingest.py` first.")
        else:
            with st.spinner("📚 Consulting Bd chaurasia..."):
                try:
                    response = rag_chain.invoke({"input": query})

                    answer = response["answer"]
                    context_docs = response["context"]

                    # Show answer
                    st.markdown(answer)

                    # Save assistant reply
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Evidence expander
                    with st.expander("📝 Evidence (Retrieved Textbook Chunks)"):
                        for doc in context_docs:
                            st.caption(
                                f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                            )
                            st.write(doc.page_content)

                except Exception as e:
                    st.error(f"⚠️ Error during retrieval: {e}")
