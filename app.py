import streamlit as st

from rag import (
    process_pdf,
    retrieve_chunks
)


# -------------------------------
# Page setup
# -------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚"
)

st.title("📚 RAG Document Assistant")


# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf"
)


if uploaded_file:

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())


    # ---------------------------
    # Process PDF
    # ---------------------------

    if (
        "vector_store" not in st.session_state
        or st.session_state.get("file_name") != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            vector_store, chunk_count = process_pdf(
                "uploaded.pdf"
            )

        st.session_state.vector_store = vector_store
        st.session_state.file_name = uploaded_file.name

        st.success(
            f"PDF processed successfully: {chunk_count} chunks"
        )


    # ---------------------------
    # Question
    # ---------------------------

    st.divider()

    question = st.text_input(
        "Ask a question about your PDF"
    )


    if question:

        results = retrieve_chunks(
            st.session_state.vector_store,
            question
        )


        st.subheader("Relevant Passages")


        for i, result in enumerate(
            results,
            start=1
        ):

            page = result.metadata.get(
                "page",
                0
            ) + 1


            st.markdown(
                f"### Result {i}"
            )


            st.write(
                result.page_content
            )


            st.caption(
                f"Source: Page {page}"
            )


            st.divider()


else:

    st.info(
        "👆 Upload a PDF to start"
    )