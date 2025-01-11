import streamlit as st
import requests
import time
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 10
RETRY_DELAY = 1  # seconds

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def _make_request_with_retry(self, method: str, endpoint: str, **kwargs) -> Optional[dict]:
        url = f"{self.base_url}{endpoint}"
        retries = 0
        
        while retries < MAX_RETRIES:
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                retries += 1
                logger.warning(f"Request failed (attempt {retries}/{MAX_RETRIES}): {str(e)}")
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    st.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                    return None
                
    def submit_query(self, query: str) -> Optional[dict]:
        return self._make_request_with_retry(
            'POST', 
            '/query/submit',
            json={"query": query}
        )
    
    def retrieve_results(self) -> Optional[dict]:
        return self._make_request_with_retry('POST', '/query/retrieve')
    
    def configure_search(self, config: dict) -> Optional[dict]:
        return self._make_request_with_retry(
            'POST',
            '/vector-search/configure',
            params=config
        )
    
    def get_similar_queries(self, query: str, max_results: int = 5) -> Optional[dict]:
        return self._make_request_with_retry(
            'POST',
            '/query/similarity',
            json={"query": query, "max_results": max_results}
        )

def main():
    st.set_page_config(page_title="Vector Search", layout="wide")
    
    # Initialize session state
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient('https://smooth-dryers-jump.loca.lt')  # Replace with your localtunnel URL
    
    # Main layout
    st.title("Vector Search")
    
    # Left column - Search and Configuration
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Query input
        query = st.text_area(
            "Enter complete user question to view similarity Correlations of knowledge slices",
            height=100
        )
        
        if st.button("Retrieval", type="primary"):
            if query:
                # Submit query
                submit_response = st.session_state.api_client.submit_query(query)
                if submit_response:
                    # Retrieve results
                    results = st.session_state.api_client.retrieve_results()
                    if results:
                        st.session_state.search_results = results
        
        # Vector Search Configuration
        with st.expander("Vector Search Configuration", expanded=True):
            config = {}
            
            # Doc Correlation
            config['doc_correlation'] = st.slider(
                "Doc Correlation (optional)",
                min_value=0.0,
                max_value=0.95,
                value=0.85,
                step=0.01
            )
            
            # Recall Number
            config['recall_number'] = st.slider(
                "Recall Number",
                min_value=1,
                max_value=50,
                value=10
            )
            
            # Knowledge retrieval weight
            retrieval_weight = st.radio(
                "Knowledge retrieval weight",
                options=["Mixed", "Semantic", "Keyword"],
                horizontal=True
            )
            config['retrieval_weight'] = retrieval_weight
            
            if retrieval_weight == "Mixed":
                config['mixed_percentage'] = st.slider(
                    "",
                    min_value=0,
                    max_value=100,
                    value=50
                )
            
            # Rerank Model
            config['rerank_enabled'] = st.radio(
                "Rerank Model",
                options=["Enable", "Disable"],
                horizontal=True
            ) == "Enable"
            
            if st.button("Apply Configuration"):
                response = st.session_state.api_client.configure_search(config)
                if response:
                    st.success("Configuration updated successfully")
    
    # Right column - Search Results
    with right_col:
        st.subheader("Search Results")
        
        if 'search_results' in st.session_state and st.session_state.search_results:
            results = st.session_state.search_results.get('results', [])
            
            for result in results:
                with st.container():
                    col1, col2 = st.columns([0.8, 0.2])
                    
                    with col1:
                        st.markdown(f"🔄 {result['id']}")
                    
                    with col2:
                        st.markdown(f"{result.get('tokens', 0)} tokens")
                    
                    st.markdown(result['content'])
                    
                    # Display keywords as tags
                    if 'metadata' in result and 'keywords' in result['metadata']:
                        keywords = result['metadata']['keywords']
                        if isinstance(keywords, str):
                            keywords = keywords.split(',')
                        
                        cols = st.columns(len(keywords))
                        for idx, keyword in enumerate(keywords):
                            with cols[idx]:
                                st.markdown(f"<span style='background-color: #f0f2f6; padding: 2px 8px; border-radius: 10px;'>{keyword.strip()}</span>", unsafe_allow_html=True)
                    
                    st.markdown(f"Correlation: {result['correlation']}")
                    st.divider()

if __name__ == "__main__":
    main()
