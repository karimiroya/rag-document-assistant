from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


VECTOR_PATH = "vectorstore/faiss_index"


# -------------------------------
# Load PDF
# -------------------------------

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# -------------------------------
# Split PDF into chunks
# -------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# -------------------------------
# Create embeddings
# -------------------------------

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# -------------------------------
# Create FAISS vector database
# -------------------------------

def create_vector_store(chunks):

    embeddings = create_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# -------------------------------
# Save FAISS
# -------------------------------

def save_vector_store(vector_store):

    vector_store.save_local(
        VECTOR_PATH
    )


# -------------------------------
# Load FAISS
# -------------------------------

def load_vector_store():

    embeddings = create_embeddings()

    vector_store = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# -------------------------------
# Full PDF pipeline
# -------------------------------

def process_pdf(pdf_path):

    documents = load_pdf(
        pdf_path
    )

    chunks = split_documents(
        documents
    )

    vector_store = create_vector_store(
        chunks
    )

    save_vector_store(
        vector_store
    )

    return vector_store, len(chunks)


# -------------------------------
# Retrieve relevant chunks
# -------------------------------

def retrieve_chunks(
    vector_store,
    question,
    k=3
):

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results


# -------------------------------
# Load LLM
# -------------------------------

def load_generator():

    generator = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base"
    )

    return generator


# -------------------------------
# Build context
# -------------------------------

def build_context(retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:

        page = chunk.metadata.get(
            "page",
            0
        ) + 1

        context_parts.append(
            f"""
Page {page}:
{chunk.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    return context


