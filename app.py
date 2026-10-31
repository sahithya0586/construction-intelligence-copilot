import os
import sys
from pathlib import Path

import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAG_PATH = PROJECT_ROOT / "app" / "rag"

if str(RAG_PATH) not in sys.path:
    sys.path.insert(0, str(RAG_PATH))

# Load Gemini API key from Streamlit Cloud secrets
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# ============================================================
# RAG IMPORTS
# ============================================================

from rag_pipeline import (
    retrieve_documents,
    build_context,
    build_prompt,
    generate_answer,
    get_source_type,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Construction Intelligence Copilot",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

/* Main hero */
.hero-box {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 55%,
        #2563eb 100%
    );
    color: white;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1rem;
    opacity: 0.9;
}

/* Status cards */
.status-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    background: white;
    min-height: 100px;
}

.status-title {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 8px;
}

.status-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827;
}

/* Chat answer */
.answer-box {
    padding: 18px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 15px;
}

/* Evidence */
.evidence-box {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: white;
    margin-bottom: 10px;
}

.evidence-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.evidence-title {
    font-weight: 650;
    margin-top: 4px;
}

.evidence-meta {
    font-size: 0.82rem;
    color: #6b7280;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    padding: 30px 0 15px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">🏗️ Construction Intelligence Copilot</div>
    <div class="hero-subtitle">
        Multimodal RAG for construction documents,
        site images, safety analysis and evidence-backed answers
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏗️ Project")

    st.markdown(
        """
**Construction Intelligence Copilot**

An AI-powered construction intelligence system combining:

- 📄 Construction documents
- 📷 Construction site images
- 🧠 Vision-Language Model
- 🔎 Semantic retrieval
- 🗄️ ChromaDB
- 🤖 Gemini
"""
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown(
        """
**AI**
- Gemini 3.8 Flash
- Vision-Language Model

**RAG**
- Sentence Transformers
- Semantic Embeddings
- ChromaDB

**Documents**
- PDF extraction
- Chunking
- Metadata

**Interface**
- Streamlit
"""
    )

    st.divider()

    st.markdown("### Example Questions")

    examples = [
        "What safety hazards are visible in the construction site image?",
        "What workers are visible in the image?",
        "What equipment is visible in the construction site?",
        "What safety procedures are recommended by the document?",
        "What hazards are visible and how can they be prevented?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state["selected_question"] = example
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state["messages"] = []
        st.session_state["selected_question"] = ""
        st.rerun()


# ============================================================
# CHROMADB STATUS
# ============================================================

try:

    import chromadb

    chroma_path = PROJECT_ROOT / "data" / "chroma"

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    collection = chroma_client.get_collection(
        name="construction_documents_v2"
    )

    source_count = collection.count()

except Exception:

    collection = None
    source_count = 0


# ============================================================
# STATUS CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
<div class="status-card">
<div class="status-title">SYSTEM STATUS</div>
<div class="status-value">🟢 Online</div>
</div>
""",
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
<div class="status-card">
<div class="status-title">KNOWLEDGE BASE</div>
<div class="status-value">{source_count} Sources</div>
</div>
""",
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
<div class="status-card">
<div class="status-title">DATA TYPES</div>
<div class="status-value">📄 + 📷</div>
</div>
""",
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        """
<div class="status-card">
<div class="status-title">AI ENGINE</div>
<div class="status-value">Gemini 3.8</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = ""


# ============================================================
# CHAT TITLE
# ============================================================

st.markdown("## 🔎 Construction Copilot")

st.caption(
    "Ask questions about construction documents, site images, "
    "safety conditions and construction activities."
)


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state["messages"]:

    role = message["role"]

    if role == "user":

        with st.chat_message("user"):

            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(message["content"])

            # --------------------------------------------
            # Evidence
            # --------------------------------------------

            documents = message.get("documents", [])
            metadatas = message.get("metadatas", [])
            distances = message.get("distances", [])

            if documents:

                with st.expander(
                    "📚 View Evidence Used"
                ):

                    image_results = []
                    pdf_results = []

                    for document, metadata, distance in zip(
                        documents,
                        metadatas,
                        distances,
                    ):

                        source_type = get_source_type(
                            metadata
                        )

                        if source_type == "image":

                            image_results.append(
                                (
                                    document,
                                    metadata,
                                    distance,
                                )
                            )

                        else:

                            pdf_results.append(
                                (
                                    document,
                                    metadata,
                                    distance,
                                )
                            )

                    # ----------------------------------------
                    # Image Evidence
                    # ----------------------------------------

                    if image_results:

                        st.markdown(
                            "### 📷 Site Image Evidence"
                        )

                        image_columns = st.columns(
                            min(len(image_results), 3)
                        )

                        for index, (
                            document,
                            metadata,
                            distance,
                        ) in enumerate(image_results):

                            with image_columns[
                                index % len(image_columns)
                            ]:

                                image_filename = metadata.get(
                                    "image_filename",
                                    "",
                                )

                                image_path = (
                                    PROJECT_ROOT
                                    / "data"
                                    / "images"
                                    / image_filename
                                )

                                if image_path.exists():

                                    st.image(
                                        str(image_path),
                                        use_container_width=True,
                                    )

                                st.markdown(
                                    f"""
<div class="evidence-box">
<div class="evidence-label">
IMAGE EVIDENCE
</div>

<div class="evidence-title">
{image_filename}
</div>

<div class="evidence-meta">
Phase:
{metadata.get(
    "construction_phase",
    "Unknown"
)}

<br>

Retrieval distance:
{distance:.4f}
</div>
</div>
""",
                                    unsafe_allow_html=True,
                                )

                                with st.expander(
                                    "View VLM Analysis"
                                ):

                                    st.write(document)

                    # ----------------------------------------
                    # PDF Evidence
                    # ----------------------------------------

                    if pdf_results:

                        st.markdown(
                            "### 📄 Document Evidence"
                        )

                        for (
                            document,
                            metadata,
                            distance,
                        ) in pdf_results:

                            source = metadata.get(
                                "source",
                                "Unknown PDF",
                            )

                            page = metadata.get(
                                "page",
                                "Unknown",
                            )

                            chunk = metadata.get(
                                "chunk",
                                "Unknown",
                            )

                            with st.expander(
                                f"📄 {source} — Page {page}"
                            ):

                                st.caption(
                                    f"Chunk: {chunk} | "
                                    f"Retrieval distance: "
                                    f"{distance:.4f}"
                                )

                                st.write(document)


# ============================================================
# SELECTED QUESTION FROM SIDEBAR
# ============================================================

selected_question = st.session_state.get(
    "selected_question",
    "",
)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a construction question..."
)


# If user clicked an example question
if selected_question and not question:

    question = selected_question

    st.session_state["selected_question"] = ""


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # ----------------------------------------
    # Show user question
    # ----------------------------------------

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # ----------------------------------------
    # Retrieve evidence
    # ----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching construction evidence..."
        ):

            results = retrieve_documents(
                question
            )

        question_type = results["question_type"]

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]


        # ----------------------------------------
        # Query classification
        # ----------------------------------------

        if question_type == "mixed":

            st.info(
                "🔄 Mixed query detected — "
                "retrieving image and document evidence."
            )

        elif question_type == "image":

            st.info(
                "📷 Image query detected — "
                "retrieving site-image evidence."
            )

        elif question_type == "pdf":

            st.info(
                "📄 Document query detected — "
                "retrieving PDF evidence."
            )

        else:

            st.info(
                "🔎 General construction query detected."
            )


        # ----------------------------------------
        # Build RAG context
        # ----------------------------------------

        context = build_context(
            results
        )

        prompt = build_prompt(
            question,
            context,
            question_type,
        )


        # ----------------------------------------
        # Gemini answer
        # ----------------------------------------

        with st.spinner(
            "🤖 Gemini is analyzing the evidence..."
        ):

            answer = generate_answer(
                prompt
            )


        # ----------------------------------------
        # Display answer
        # ----------------------------------------

        st.markdown(answer)


        # ----------------------------------------
        # Evidence
        # ----------------------------------------

        if documents:

            with st.expander(
                "📚 View Evidence Used"
            ):

                image_results = []
                pdf_results = []

                for document, metadata, distance in zip(
                    documents,
                    metadatas,
                    distances,
                ):

                    source_type = get_source_type(
                        metadata
                    )

                    if source_type == "image":

                        image_results.append(
                            (
                                document,
                                metadata,
                                distance,
                            )
                        )

                    else:

                        pdf_results.append(
                            (
                                document,
                                metadata,
                                distance,
                            )
                        )


                # ------------------------------------
                # Images
                # ------------------------------------

                if image_results:

                    st.markdown(
                        "### 📷 Site Image Evidence"
                    )

                    image_columns = st.columns(
                        min(len(image_results), 3)
                    )

                    for index, (
                        document,
                        metadata,
                        distance,
                    ) in enumerate(image_results):

                        with image_columns[
                            index % len(image_columns)
                        ]:

                            image_filename = metadata.get(
                                "image_filename",
                                "",
                            )

                            image_path = (
                                PROJECT_ROOT
                                / "data"
                                / "images"
                                / image_filename
                            )

                            if image_path.exists():

                                st.image(
                                    str(image_path),
                                    use_container_width=True,
                                )

                            st.markdown(
                                f"""
<div class="evidence-box">

<div class="evidence-label">
IMAGE EVIDENCE
</div>

<div class="evidence-title">
{image_filename}
</div>

<div class="evidence-meta">
Phase:
{metadata.get(
    "construction_phase",
    "Unknown"
)}

<br>

Retrieval distance:
{distance:.4f}
</div>

</div>
""",
                                unsafe_allow_html=True,
                            )

                            with st.expander(
                                "View VLM Analysis"
                            ):

                                st.write(document)


                # ------------------------------------
                # PDFs
                # ------------------------------------

                if pdf_results:

                    st.markdown(
                        "### 📄 Document Evidence"
                    )

                    for (
                        document,
                        metadata,
                        distance,
                    ) in pdf_results:

                        source = metadata.get(
                            "source",
                            "Unknown PDF",
                        )

                        page = metadata.get(
                            "page",
                            "Unknown",
                        )

                        chunk = metadata.get(
                            "chunk",
                            "Unknown",
                        )

                        with st.expander(
                            f"📄 {source} — Page {page}"
                        ):

                            st.caption(
                                f"Chunk: {chunk} | "
                                f"Retrieval distance: "
                                f"{distance:.4f}"
                            )

                            st.write(document)

        else:

            st.warning(
                "No sufficiently relevant evidence was found."
            )


    # ----------------------------------------
    # Save assistant message
    # ----------------------------------------

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
    )


    # ----------------------------------------
    # Rerun so chat appears cleanly
    # ----------------------------------------

    st.rerun()


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.divider()

st.markdown("## 🧠 How the System Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
### 1️⃣ Ingestion

📄 Construction PDFs  
📷 Site Images  

↓

PDF text extraction +  
VLM image analysis
"""
    )


with col2:

    st.markdown(
        """
### 2️⃣ Knowledge Base

🧠 VLM descriptions  
🔢 Embeddings  
🗄️ ChromaDB  

↓

Semantic retrieval
"""
    )


with col3:

    st.markdown(
        """
### 3️⃣ Intelligence

🔎 Query classification  
📚 Evidence retrieval  
🤖 Gemini RAG  

↓

Evidence-backed answer
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
Construction Intelligence Copilot ·
Multimodal Retrieval-Augmented Generation
</div>
""",
    unsafe_allow_html=True,
)
