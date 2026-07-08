from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def upload_htmls():
    """
    This function: htmls --> chunking --> embedding --> FAISS
    - Read recursively through the given folder hr-policies
    - Load the pages (documents)
    - Loaded docs are split into chunks using Splitter
    - These chunks are converted into Embeddings and loaded as vectors into a local FAISS Vectors
    """
    
    loader = DirectoryLoader(path="hr-policies")
    documents = loader.load()
    print(f"{len(documents)} pages loaded")

    # split loaded docs into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n", "", " ", "\n\n"]
    )

    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(documents)} documents...")

    print(split_documents[0].metadata)

    # upload chunks as vector embeddings into FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_documents, embeddings)

    # save the FAISS DB locallu
    db.save_local("faiss_index")

def faiss_query():
    """
    This function: (RETRIEVAL in RAG): 
    user query --> embedding query --> FAISS similarity Search --> Top-k relevant chunks 
    
    - Load the local FAISS database
    - trigger a Semantic Similarity Search using Query
    - this retrieves semantically matching Vectors from the DB
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    query = "How many paid leave days do employees receive?"
    docs = new_db.similarity_search(query, k=3)

    for i, doc in enumerate(docs, 1):
        print("=" * 60)
        print(f"RESULT {i}")
        print("SOURCE:", doc.metadata['source'])
        print("-" * 60)
        print(doc.page_content)

if __name__ == "__main__":
    # upload_htmls() # exectue only once and then comment as vector DB is
    faiss_query()







