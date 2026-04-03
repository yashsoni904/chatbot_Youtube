import streamlit as st
import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# -----------------------------
# Extract Video ID
# -----------------------------

def extract_video_id(url):

    if not url:
        return None

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    if "v" in query_params:
        return query_params["v"][0]

    patterns = [
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:embed/)([0-9A-Za-z_-]{11})",
        r"(?:shorts/)([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎥 YouTube Video Q&A using RAG")
st.write("Ask questions about any YouTube video")

video_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://youtube.com/watch?v=..."
)

question = st.text_input(
    "Ask a question about the video",
    placeholder="What is DeepMind?"
)

if st.button("Process Video"):

    if not video_url:

        st.warning("Please enter a YouTube URL")

    else:

        video_id = extract_video_id(video_url)

        if not video_id:

            st.error("Invalid YouTube URL")

        else:

            try:

                with st.spinner("Fetching transcript..."):

                    api = YouTubeTranscriptApi()
                    transcript_list = api.fetch(video_id)

                    transcript = " ".join(
                        chunk.text for chunk in transcript_list
                    )

                st.success("Transcript fetched")

            except TranscriptsDisabled:

                st.error("No captions available")

            else:

                # -----------------------------
                # Text Splitting
                # -----------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.create_documents([transcript])

                st.write("Chunks created:", len(chunks))

                # -----------------------------
                # Embeddings
                # -----------------------------

                with st.spinner("Creating embeddings..."):

                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001",
                        batch_size=10
                    )

                    vector_store = FAISS.from_documents(
                        chunks,
                        embeddings
                    )

                st.success("Vector store ready")

                # -----------------------------
                # Retriever
                # -----------------------------

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                # -----------------------------
                # Prompt
                # -----------------------------

                prompt = PromptTemplate(
                    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient,
just say you don't know.

{context}

Question: {question}
""",
                    input_variables=[
                        "context",
                        "question"
                    ]
                )

                # -----------------------------
                # Chain
                # -----------------------------

                def format_docs(retrieved_docs):

                    context_text = "\n\n".join(
                        doc.page_content
                        for doc in retrieved_docs
                    )

                    return context_text


                parallel_chain = RunnableParallel({

                    "context":
                        retriever
                        | RunnableLambda(format_docs),

                    "question":
                        RunnablePassthrough()

                })

                parser = StrOutputParser()

                main_chain = (
                    parallel_chain
                    | prompt
                    | llm
                    | parser
                )

                # -----------------------------
                # Question Answering
                # -----------------------------

                if question:

                    with st.spinner("Generating answer..."):

                        answer = main_chain.invoke(question)

                    st.success("Answer generated")

                    st.write("### Answer")

                    st.write(answer)

                else:

                    st.info("Enter a question to get answer")