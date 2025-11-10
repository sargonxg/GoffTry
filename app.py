import streamlit as st
from google import genai
from google.genai import types
import time
import os
import tempfile
import pandas as pd
import json # For parsing the model's analytics response
import plotly.express as px # For the map
import country_converter as coco # To get country codes for the map

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
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("🚨 **Error:** `GOOGLE_API_KEY` not found in Streamlit secrets.")
            st.help("Please add your API key to your Streamlit Cloud 'Secrets' to run this app.")
            st.stop()
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
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > 600:
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
            model="gemini-2.5-flash",
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

# --- 2. NEW: Analytics Function ---

def get_document_analytics(client, store_name):
    """
    Prompts the model to act as an analyst and extract structured data
    (countries and UN entities) from the entire document corpus.
    """
    
    # This is a "meta-prompt" asking the AI to analyze its own RAG source
    analytics_prompt = """
    Analyze all of the provided documents in their entirety.
    Your task is to act as a UN political analyst.
    
    1.  List every country mentioned in the documents.
    2.  List every United Nations (UN) entity, program, or agency mentioned
        (e.g., "UNDP", "Security Council", "UNICEF", "DPPA").
        
    Return your findings as a single, clean JSON object.
    The JSON object must have two keys:
    - "countries": a list of country name strings.
    - "un_entities": a list of UN entity name strings.
    
    Example response:
    {
      "countries": ["Syria", "Yemen", "Colombia", "Syria"],
      "un_entities": ["Security Council", "UNDP", "Security Council"]
    }
    
    Do not include any text outside of the JSON object.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[analytics_prompt],
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
        
        # Clean the response to extract only the JSON
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_str)
        return data
    
    except json.JSONDecodeError:
        st.error("Error: Failed to decode analytics data from the model. The model may have returned non-JSON text.")
        st.code(response.text) # Show what the model returned
        return None
    except Exception as e:
        st.error(f"Error during analytics query: {e}")
        return None

# --- 3. Streamlit App UI ---

st.title("🇺🇳 Good Offices AI Analyst")
st.markdown("A 'Diplomatic Pulse' dashboard powered by Gemini File Search.")

client = get_genai_client()

if "file_search_store" not in st.session_state:
    st.session_state.file_search_store = create_file_search_store(client)
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analytics_results" not in st.session_state:
    st.session_state.analytics_results = None

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
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{uploaded_file.name}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                if upload_to_file_search_store(client, temp_file_path, file_search_store.name, uploaded_file.name):
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.session_state.analytics_results = None # Invalidate old analytics
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("Show Sources (from your documents)"):
                    for citation in message["citations"]:
                        st.info(f"**From '{citation['file_name']}' (Snippet):**\n\n> {citation['snippet']}")
            elif message["role"] == "assistant" and ("citations" not in message or not message["citations"]):
                st.caption("ℹ️ This response was grounded in your documents, but no specific snippet was cited.")

    if prompt := st.chat_input("Ask a question based *only* on the uploaded documents..."):
        if not st.session_state.uploaded_files:
            st.info("Please upload at least one document in the sidebar to start the chat.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents and formulating answer..."):
                    response = query_file_search_store(client, file_search_store.name, prompt)
                    
                    if response:
                        response_text = response.text
                        citations = []
                        try:
                            grounding_meta = response.candidates[0].grounding_metadata
                            if grounding_meta.search_quotes:
                                for quote in grounding_meta.search_quotes:
                                    citations.append({
                                        "file_name": quote.file_display_name,
                                        "snippet": quote.text_snippet
                                    })
                        except (AttributeError, IndexError):
                            pass
                        
                        st.markdown(response_text)
                        
                        if citations:
                            with st.expander("Show Sources (from your documents)"):
                                for citation in citations:
                                    st.info(f"**From '{citation['file_name']}' (Snippet):**\n\n> {citation['snippet']}")
                        else:
                            st.caption("ℹ️ This response was grounded in your documents, but no specific snippet was cited.")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text, 
                            "citations": citations
                        })
                    else:
                        st.error("Failed to get a response from the model.")

# --- Tab 2: Analytics Dashboard (NOW FUNCTIONAL) ---
with tab2:
    st.subheader("Automated 'Diplomatic Pulse' Analytics")

    if not st.session_state.uploaded_files:
        st.info("Upload documents in the sidebar to generate analytics.")
    else:
        st.markdown("Click the button to run a full analysis of the document corpus. The AI will read all documents to extract countries and UN entities.")
        
        if st.button("Generate Corpus Analytics", type="primary", use_container_width=True):
            with st.spinner("AI Analyst is reading all documents... this may take a moment."):
                st.session_state.analytics_results = get_document_analytics(client, file_search_store.name)
        
        st.divider()

        # Display the results once they are in the session state
        if st.session_state.analytics_results:
            data = st.session_state.analytics_results
            
            try:
                # --- 1. UN Entity Frequency Chart ---
                st.subheader("UN Entity Mentions")
                un_entities = data.get("un_entities", [])
                if un_entities:
                    entity_df = pd.Series(un_entities).value_counts().to_frame('Mentions')
                    st.bar_chart(entity_df)
                else:
                    st.info("No UN entities found in the documents.")
                
                # --- 2. Country Mentions Map ---
                st.subheader("Country Mentions (Geographic Map)")
                countries = data.get("countries", [])
                if countries:
                    # Use pandas to count frequencies
                    country_df = pd.Series(countries).value_counts().to_frame('Mentions').reset_index()
                    country_df.columns = ['Country', 'Mentions']
                    
                    # Convert country names to ISO-3 codes for Plotly
                    cc = coco.CountryConverter()
                    country_df['iso_alpha'] = country_df['Country'].apply(
                        lambda x: cc.convert(names=x, to='ISO3')
                    )
                    
                    # Create the choropleth map
                    fig = px.choropleth(
                        country_df,
                        locations="iso_alpha",
                        color="Mentions",
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Geographic Distribution of Country Mentions"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No countries found in the documents.")

            except Exception as e:
                st.error("An error occurred while trying to display the analytics.")
                st.write(e)
                st.json(st.session_state.analytics_results) # Show the raw data for debugging

        elif st.session_state.analytics_results is None and not st.session_state.uploaded_files:
             st.info("Please upload documents and click 'Generate Corpus Analytics'.")
