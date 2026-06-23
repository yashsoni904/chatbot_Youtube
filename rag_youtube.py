import streamlit as st
import re
import os
import tempfile
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, RequestBlocked, NoTranscriptFound
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# -----------------------------
# API Key Setup
# -----------------------------

load_dotenv()
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

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
# Fetch Transcript (with cookie bypass for cloud)
# -----------------------------

def fetch_transcript(video_id):
    """
    Fetches transcript. On Streamlit Cloud, YouTube blocks direct requests.
    We bypass this by passing browser cookies stored in st.secrets.
    """
    cookie_file_path = None

    try:
        # Write YouTube cookies to a temp file if available in secrets
        if "YOUTUBE_COOKIES" in st.secrets:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            tmp.write(st.secrets["YOUTUBE_COOKIES"])
            tmp.close()
            cookie_file_path = tmp.name

        if cookie_file_path:
            api = YouTubeTranscriptApi(cookies=cookie_file_path)
        else:
            api = YouTubeTranscriptApi()

        transcript_list = api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript, None

    except RequestBlocked:
        return None, (
            "⛔ **YouTube blocked this request.**\n\n"
            "Streamlit Cloud's servers are recognized as bots by YouTube. "
            "To fix this, add your YouTube browser cookies as a Streamlit secret "
            "named `YOUTUBE_COOKIES`. See the sidebar for instructions."
        )
    except TranscriptsDisabled:
        return None, "❌ This video has **captions disabled**. Try a different video."
    except NoTranscriptFound:
        return None, "❌ **No transcript found** for this video. It may not have captions."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"

    finally:
        if cookie_file_path and os.path.exists(cookie_file_path):
            os.unlink(cookie_file_path)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎥 YouTube Video Q&A using RAG")
st.write("Ask questions about any YouTube video using AI")

# Sidebar instructions
with st.sidebar:
    st.header("⚙️ Setup Instructions")
    st.markdown("""
    ### Fix YouTube Blocking
    If you see a **RequestBlocked** error, follow these steps:

    **Step 1:** Install the browser extension:
    [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)

    **Step 2:** Go to [youtube.com](https://youtube.com), log in, then click the extension → **Export** → select `youtube.com`

    **Step 3:** Copy the entire file content

    **Step 4:** In Streamlit Cloud → **Manage App** → **Secrets**, add:
    ```toml
    YOUTUBE_COOKIES = \"\"\"
    # Netscape HTTP Cookie File
    .youtube.com  TRUE  /  ...paste full cookies here...
    \"\"\"
    ```
    """)

    st.divider()
    st.markdown("**Model:** `gemini-2.5-flash`")
    st.markdown("**Embeddings:** `gemini-embedding-001`")

video_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://youtube.com/watch?v=..."
)

question = st.text_input(
    "Ask a question about the video",
    placeholder="What is DeepMind?"
)

if st.button("🚀 Process Video", type="primary"):

    if not video_url:
        st.warning("Please enter a YouTube URL")

    else:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the URL and try again.")

        else:
            # -----------------------------
            # Fetch Transcript
            # -----------------------------

            with st.spinner("📄 Fetching transcript..."):
                transcript, error = fetch_transcript(video_id)

            if error:
                st.error(error)
                st.stop()

            st.success(f"✅ Transcript fetched ({len(transcript.split())} words)")

            # -----------------------------
            # Text Splitting
            # -----------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.create_documents([transcript])
            st.write(f"📦 Chunks created: **{len(chunks)}**")

            # -----------------------------
            # Embeddings & Vector Store
            # -----------------------------

            with st.spinner("🔢 Creating embeddings..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        batch_size=10
                    )
                    vector_store = FAISS.from_documents(chunks, embeddings)
                except Exception as e:
                    st.error(f"❌ Embedding failed: {str(e)}")
                    st.stop()

            st.success("✅ Vector store ready")

            # -----------------------------
            # Retriever & Chain
            # -----------------------------

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            prompt = PromptTemplate(
                template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
                input_variables=["context", "question"]
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            # -----------------------------
            # Question Answering
            # -----------------------------

            if question:
                with st.spinner("🤖 Generating answer..."):
                    try:
                        answer = main_chain.invoke(question)
                    except Exception as e:
                        st.error(f"❌ LLM error: {str(e)}")
                        st.stop()

                st.success("✅ Answer generated")
                st.markdown("### 💡 Answer")
                st.write(answer)

            else:
                st.info("💬 Enter a question above to get an answer about the video.")