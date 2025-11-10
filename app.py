import streamlit as st
from google import genai
from google.genai import types
import time
import os
import tempfile
import pandas as pd
import random # Used for dummy analytics data

# --- Page Configuration ---
st.set_page_config(
    page_title="UN Good Offices - AI Analyst",
    page_icon="🇺🇳",
    layout="wide"
)

# --- 1. Helper Functions (Gemini API) ---

@st.cache_resource(show_spinner="Connecting to Google AI...")
def get_genai_client():
    """Configures and returns the Generative AI client from Streamlit secrets."""
    try:
        # Check if the secret exists
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("🚨 **Error:** `GOOGLE_API_KEY` not found in Streamlit secrets.")
            st.help("Please add your API key to your Streamlit Cloud 'Secrets' to run this app.")
            st.stop()
        
        # The Client() automatically finds the GOOGLE_API_KEY from secrets
        client = genai.Client()
        return client
        
    except Exception as e:
        st.error(f"An error occurred during client initialization: {e}")
        st.stop()

@st.cache_resource(show_spinner="Creating new file store for this session...")
def create_file_search_store(_client):
    """Creates a new FileSearchStore and caches it for the session."""
    try:
        file_search_store = _client.file_search_stores.create(
            config={'display_name': 'un-good-offices-demo-store'}
        )
        return file_search_store
    except Exception as e:
        st.error(f"Failed to create file store: {e}")
        st.stop()

def upload_to_file_search_store(client, file_path, file_search_store_name, display_name):
    """Uploads and imports a file into the store."""
    with st.spinner(f"Indexing '{display_name}'..."):
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=file_search_store_name,
                config={'display_name': display_name},
            )

            # Wait for import to complete with a timeout
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > 600: # 10-minute timeout
                    st.error(f"Timeout: Indexing '{display_name}' took too long.")
                    return None
                st.toast(f"Processing '{display_name}'... please wait.")
                time.sleep(5)
                operation = client.operations.get(operation)
            
            st.success(f"✅ Successfully indexed '{display_name}'!")
            return operation
        except Exception as e:
            st.error(f"Error indexing '{display_name}': {e}")
            return None

def query_file_search_store(client, store_name, query):
    """Asks a question about the file store and returns the response."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Use a supported model
            contents=[query],
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ]
            )
        )
        return response
    except Exception as e:
        st.error(f"Error during query: {e}")
        return None

# --- 2. Dummy Analytics Function (for the Dashboard) ---
def get_analytics_data(file_names):
    """Generates dummy analytics data for the dashboard."""
    if not file_names:
        return None, None
    
    # Dummy Keywords
    keywords = ["Ceasefire", "Humanitarian Aid", "Political Dialogue", "Security Council", "Refugees", "Resolution 2254", "Peace Process"]
    data = {
        "Keyword": random.sample(keywords, min(5, len(keywords))),
        "Frequency": [random.randint(5, 50) for _ in range(5)]
    }
    keyword_df = pd.DataFrame(data).set_index("Keyword")
    
    # Dummy Stakeholders
    stakeholders = ["Party A", "Party B", "UN Envoy", "Civil Society Org.", "Neighboring State C"]
    stakeholder_data = {
        "Stakeholder": random.sample(stakeholders, min(3, len(stakeholders))),
        "Mentions": [random.randint(10, 40) for _ in range(3)],
        "Sentiment": [random.choice(["Positive", "Negative", "Neutral"]) for _ in range(3)]
    }
    stakeholder_df = pd.DataFrame(stakeholder_data).set_index("Stakeholder")
    
    return keyword_df, stakeholder_df

# --- 3. Streamlit App UI ---

# 
st.title("🇺🇳 Good Offices AI Analyst")
st.markdown("A 'Diplomatic Pulse' dashboard powered by Gemini File Search.")

# --- Initialize Client and File Store ---
client = get_genai_client()

if "file_search_store" not in st.session_state:
    st.session_state.file_search_store = create_file_search_store(client)
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

file_search_store = st.session_state.file_search_store

# --- Sidebar: Document Hub ---
with st.sidebar:
    st.header("Document Hub")
    st.markdown("Upload your corpus of agreements, reports, and emails.")
    
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files", 
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                # Save to a temp file to get a persistent path
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{uploaded_file.name}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                # Upload and index the file
                if upload_to_file_search_store(client, temp_file_path, file_search_store.name, uploaded_file.name):
                    st.session_state.uploaded_files.append(uploaded_file.name)
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
    
    st.divider()
    st.subheader("Indexed Document Corpus")
    if not st.session_state.uploaded_files:
        st.info("No documents indexed for this session.")
    else:
        for name in st.session_state.uploaded_files:
            st.write(f"📁 `{name}`")

    st.divider()
    if st.button("Clear Session & Delete Store", type="primary", use_container_width=True):
        with st.spinner("Deleting file store..."):
            try:
                client.file_search_stores.delete(name=file_search_store.name, config={'force': True})
            except Exception as e:
                st.error(f"Could not delete store: {e}")
        st.session_state.clear()
        st.rerun()

# --- Main Page: Tabs ---
tab1, tab2 = st.tabs(["💬 Document Chat (RAG)", "📊 Analytics Dashboard"])

# --- Tab 1: Document Chat (RAG) ---
with tab1:
    st.subheader("Query Your Document Corpus")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                with st.expander("Show Sources (from your documents)"):
                    for citation in message["citations"]:
                        st.info(f"**From '{citation['file_name']}' (Snippet):**\n\n> {citation['snippet']}")
            # Add this check for older messages that might not have citations
            elif message["role"] == "assistant" and "citations" not in message:
                st.caption("ℹ️ This response was based on the model's general knowledge.")


    # The chat input bar
    if prompt := st.chat_input("Ask a question based *only* on the uploaded documents..."):
        if not st.session_state.uploaded_files:
            st.info("Please upload at least one document in the sidebar to start the chat.")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get model response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and formulating answer..."):
                    response = query_file_search_store(client, file_search_store.name, prompt)
                    
                    if response:
                        response_text = response.text
                        citations = []
                        
                        # --- 1. Check for and extract citations ---
                        try:
                            grounding_meta = response.candidates[0].grounding_metadata
                            if grounding_meta.search_quotes:
                                for quote in grounding_meta.search_quotes:
                                    citations.append({
                                        "file_name": quote.file_display_name,
                                        "snippet": quote.text_snippet
                                    })
                        except (AttributeError, IndexError):
                            pass # No citations found

                        # --- 2. Display the response ---
                        st.markdown(response_text)
                        
                        # --- 3. Display sources OR a general knowledge notice ---
                        if citations:
                            with st.expander("Show Sources (from your documents)"):
                                for citation in citations:
                                    st.info(f"**From '{citation['file_name']}' (Snippet):**\n\n> {citation['snippet']}")
                            # Add to session state
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text, 
                                "citations": citations
                            })
                        else:
                            st.caption("ℹ️ This response is based on the model's general knowledge.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text
                            })
                    else:
                        st.error("Failed to get a response from the model.")

# --- Tab 2: Analytics Dashboard ---
with tab2:
    st.subheader("Automated 'Diplomatic Pulse' Analytics")

    if not st.session_state.uploaded_files:
        st.info("Upload documents in the sidebar to generate analytics.")
    else:
        st.markdown("This dashboard provides an automated summary of your document corpus. *(This is a demo concept; a real app would use another Gemini call to extract entities)*.")
        
        keyword_df, stakeholder_df = get_analytics_data(st.session_state.uploaded_files)
        
        if keyword_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Topic Frequency")
                st.bar_chart(keyword_df["Frequency"])
            
            with col2:
                st.subheader("Stakeholder Mentions")
                st.dataframe(stakeholder_df, use_container_width=True)
                
            st.divider()
            st.subheader("Conceptual: Stakeholder Relationship Map")
            st.graphviz_chart('''
                digraph {
                    rankdir=LR;
                    "UN Envoy" -> "Party A" [label = " negotiates"];
                    "UN Envoy" -> "Party B" [label = " negotiates"];
                    "Party A" -> "Party B" [label = " ceasefire"];
                    "Neighboring State C" -> "Party A" [label = " supports"];
                    "Civil Society Org." -> "UN Envoy" [label = " advises"];
                }
            ''')
            st.caption("This graph is a static demo. A full implementation would use AI to extract relationships from your text.")
