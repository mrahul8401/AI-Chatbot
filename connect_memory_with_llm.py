import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1 - setup LLM (Mistral-7b-0.3 with huggingface)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens = 512
    )
    return llm

# Step 2 - Connect LLM with FAISS and create chain

CUSTOM_PROMPT_TAMPLATE = """\
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside of the given context.

context: {context}
question: {question}

Start the answer directly. No small talk, please, and provide precise answers.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load data

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGING_FACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),  
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TAMPLATE)}
)

# Now invoke with a single query

user_query = input("write query here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENT:", response["source_documents"])  

