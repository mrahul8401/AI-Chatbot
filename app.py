import os
import traceback
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint

from streamlit_lottie import st_lottie  # 👈 new import

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,  
        huggingfacehub_api_token=hf_token
    )

# 🔹 Function to load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    # 🔹 Load and show the robot animation
    robot_lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    if robot_lottie:
        st_lottie(robot_lottie, speed=1, reverse=False, loop=True, quality="high", height=200, width=200)
    
    st.title("AI Assistant Chatbot! 🤖")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here 🖕")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say you don't know — don't make it up.
        Only use the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("Hugging Face token not found. Please check your .env file.")
            return

        try:
            vectorstore = get_vectorstore()
            if not vectorstore:
                st.error("Failed to load vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})

            result = response.get("result", "❗Sorry, I couldn’t find a relevant answer.")
            source_docs = response.get("source_documents", [])
            sources = "\n".join(
                f"- {os.path.basename(doc.metadata.get('source', 'N/A'))}, page {doc.metadata.get('page', 'N/A')}"
                for doc in source_docs
            )

            final_output = f"{result}\n\n**Source Documents:**\n{sources}"
            st.chat_message('assistant').markdown(final_output)
            st.session_state.messages.append({'role': 'assistant', 'content': final_output})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.text(traceback.format_exc())  # Shows full error trace

if __name__ == "__main__":
    main()

