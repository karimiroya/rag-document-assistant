import streamlit as st

from rag import (
    process_pdf,
    retrieve_chunks,
    generate_answer,
    load_generator
)


# -------------------------------
# Page setup
# -------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚"
)

st.title(
    "📚 RAG Document Assistant"
)

st.write(
    """
    Upload a PDF and ask questions about its content.
    The assistant retrieves relevant passages using
    semantic search and generates an answer using an LLM.
    """
)


# -------------------------------
# Cache LLM
# -------------------------------

@st.cache_resource
def get_generator():

    return load_generator()


generator = get_generator()


# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf"
)


if uploaded_file:

    # ---------------------------
    # Save uploaded PDF
    # ---------------------------

    with open(
        "uploaded.pdf",
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # ---------------------------
    # Process PDF
    # ---------------------------

    if (
        "vector_store"
        not in st.session_state
        or
        st.session_state.get(
            "file_name"
        )
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            (
                vector_store,
                chunk_count
            ) = process_pdf(
                "uploaded.pdf"
            )

        st.session_state.vector_store = (
            vector_store
        )

        st.session_state.file_name = (
            uploaded_file.name
        )

        st.success(
            f"""
            PDF processed successfully:
            {chunk_count} chunks created.
            """
        )


    # ---------------------------
    # Ask question
    # ---------------------------

    st.divider()

    question = st.text_input(
        "Ask a question about your PDF"
    )


    if question:

        # -----------------------
        # Retrieve context
        # -----------------------

        with st.spinner(
            "Searching document..."
        ):

            results = retrieve_chunks(
                st.session_state.vector_store,
                question,
                k=3
            )


        # -----------------------
        # Generate answer
        # -----------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                results,
                generator
            )


        # -----------------------
        # Show final answer
        # -----------------------

        st.subheader(
            "🤖 Answer"
        )

        st.write(
            answer
        )


        # -----------------------
        # Show sources
        # -----------------------

        st.subheader(
            "📄 Sources"
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            page = (
                result.metadata.get(
                    "page",
                    0
                )
                + 1
            )


            with st.expander(
                f"""
                Source {i}
                — Page {page}
                """
            ):

                st.write(
                    result.page_content
                )


else:

    st.info(
        "👆 Upload a PDF to start"
    )