from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


VECTOR_PATH = "vectorstore/faiss_index"


# -------------------------------
# Load PDF
# -------------------------------

def load_pdf(pdf_path):

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    return documents


# -------------------------------
# Split PDF into chunks
# -------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
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
# Create FAISS vector store
# -------------------------------

def create_vector_store(chunks):

    embeddings = create_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# -------------------------------
# Save FAISS vector store
# -------------------------------

def save_vector_store(vector_store):

    vector_store.save_local(
        VECTOR_PATH
    )


# -------------------------------
# Load FAISS vector store
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
# Detect broad questions
# -------------------------------

def is_broad_question(question):

    question_lower = question.lower()

    keywords = [
        "purpose",
        "objective",
        "goal",
        "main idea",
        "main point",
        "about",
        "summary",
        "summarize",
        "contribution",
        "motivation"
    ]

    return any(
        keyword in question_lower
        for keyword in keywords
    )


# -------------------------------
# Improve query
# -------------------------------

def expand_question(question):

    if is_broad_question(question):

        return (
            f"{question} "
            "abstract introduction objective purpose "
            "motivation contribution main idea"
        )

    return question


# -------------------------------
# Detect low-quality chunks
# -------------------------------

def is_reference_chunk(text):

    text_lower = text.lower()

    signals = [
        "references",
        "bibliography",
        "proceedings of",
        "doi:",
        "arxiv:",
        "et al."
    ]

    count = sum(
        signal in text_lower
        for signal in signals
    )

    return count >= 2


# -------------------------------
# Detect useful introductory text
# -------------------------------

def introduction_bonus(text):

    text_lower = text.lower()

    useful_terms = [
        "abstract",
        "introduction",
        "we propose",
        "we present",
        "we introduce",
        "in this paper",
        "our model",
        "our approach",
        "our work"
    ]

    matches = sum(
        term in text_lower
        for term in useful_terms
    )

    return matches


# -------------------------------
# Retrieve + rerank chunks
# -------------------------------

def retrieve_chunks(
    vector_store,
    question,
    k=4
):

    search_question = expand_question(
        question
    )

    # Retrieve more candidates first
    candidates = (
        vector_store
        .similarity_search_with_score(
            search_question,
            k=12
        )
    )

    reranked = []

    broad_question = is_broad_question(
        question
    )

    for chunk, distance in candidates:

        text = chunk.page_content

        # Ignore bibliography/reference-heavy chunks
        if is_reference_chunk(text):
            continue

        page = (
            chunk.metadata.get(
                "page",
                0
            )
            + 1
        )

        adjusted_score = float(
            distance
        )

        # In FAISS here:
        # lower distance = better result

        # For broad questions, slightly prefer
        # early pages where abstract/introduction
        # usually appear.
        if broad_question:

            if page <= 2:
                adjusted_score -= 0.30

            elif page <= 4:
                adjusted_score -= 0.15

        # Reward chunks containing phrases
        # typical of abstracts/intros
        intro_matches = introduction_bonus(
            text
        )

        adjusted_score -= (
            intro_matches * 0.10
        )

        reranked.append(
            (
                adjusted_score,
                chunk
            )
        )

    # Best score first
    reranked.sort(
        key=lambda item: item[0]
    )

    results = [
        chunk
        for score, chunk
        in reranked[:k]
    ]

    # Fallback if filtering removed too much
    if len(results) < k:

        fallback = vector_store.similarity_search(
            search_question,
            k=k
        )

        for chunk in fallback:

            if chunk not in results:
                results.append(
                    chunk
                )

            if len(results) >= k:
                break

    return results


# -------------------------------
# Load tokenizer and model
# -------------------------------

def load_generator():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


# -------------------------------
# Build context
# -------------------------------

def build_context(
    retrieved_chunks
):

    context_parts = []

    for chunk in retrieved_chunks:

        page = (
            chunk.metadata.get(
                "page",
                0
            )
            + 1
        )

        context_parts.append(
            f"""
SOURCE PAGE {page}

{chunk.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    return context


# -------------------------------
# Generate answer
# -------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    generator
):

    tokenizer, model = generator

    # Use the strongest retrieved chunks
    best_chunks = retrieved_chunks[:3]

    context = build_context(
        best_chunks
    )

    prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Give a direct answer in 1 or 2 sentences.
If the answer is not in the context, say:
I could not find this information in the document.

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()