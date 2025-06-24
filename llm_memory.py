
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/"    # note this the shortcut for datapath, if are in the same folder and / represents that , data is a folder 

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob = '*.pdf',
                             loader_cls = PyPDFLoader)
    
    documents = loader.load()
    return documents

docs = load_pdf_files(data = DATA_PATH)
#print('length of pdf pages: ', len(docs))


# Creating chunks- step 2 

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data = docs)
#print("length of the extracted chunks:", len(text_chunks))



#creating vector embedding - step 3 

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-mpnet-base-v2") # sentence transformer model, it helps in symantic search
    return embedding_model

embedding_model = get_embedding_model()


# storing vector embeddings in FAISS - step 4 


from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)






