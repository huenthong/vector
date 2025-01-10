import streamlit as st
import requests
import time
from typing import Dict, Any
import pandas as pd

# Constants
BASE_URL = "http://localhost:8000"
MAX_RETRIES = 10
RETRY_DELAY = 1  # seconds

def make_api_request(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None, params: Dict[str, Any] = None) -> Dict:
    """Make API request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", params=params)
            else:  # POST
                response = requests.post(f"{BASE_URL}{endpoint}", json=data, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"Failed to connect to the backend after {MAX_RETRIES} attempts: {str(e)}")
                raise
            time.sleep(RETRY_DELAY)
            continue

def config_section():
    """Create the Vector Search Configuration section"""
    with st.expander("Vector Search Configuration", expanded=True):
        # Create a container for consistent padding and styling
        with st.container():
            # Doc Correlation with input field
            col1, col2, col3 = st.columns([6, 2, 1])
            with col1:
                doc_correlation = st.slider(
                    "Doc Correlation (optional)",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.85,
                    step=0.01,
                )
            with col2:
                doc_correlation = st.number_input(
                    "",
                    value=doc_correlation,
                    min_value=0.0,
                    max_value=0.95,
                    step=0.01,
                    key="doc_correlation_input",
                    label_visibility="collapsed"
                )
            with col3:
                st.write("/ 0.95")

            # Recall Number with input field
            col1, col2, col3 = st.columns([6, 2, 1])
            with col1:
                recall_number = st.slider(
                    "Recall Number",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                )
            with col2:
                recall_number = st.number_input(
                    "",
                    value=recall_number,
                    min_value=1,
                    max_value=50,
                    step=1,
                    key="recall_number_input",
                    label_visibility="collapsed"
                )
            with col3:
                st.write("/ 50")

            # Knowledge retrieval weight with radio buttons
            st.write("Knowledge retrieval weight ℹ️")
            retrieval_weight = st.radio(
                "",
                options=["Mixed", "Semantic", "Keyword"],
                horizontal=True,
                key="retrieval_weight",
                label_visibility="collapsed"
            )

            # Mixed percentage slider (only show if Mixed is selected)
            if retrieval_weight == "Mixed":
                col1, col2, col3 = st.columns([6, 2, 1])
                with col1:
                    mixed_percentage = st.slider(
                        "",
                        min_value=0,
                        max_value=100,
                        value=50,
                        step=1,
                        key="mixed_percentage_slider",
                        label_visibility="collapsed"
                    )
                with col2:
                    mixed_percentage = st.number_input(
                        "",
                        value=mixed_percentage,
                        min_value=0,
                        max_value=100,
                        step=1,
                        key="mixed_percentage_input",
                        label_visibility="collapsed"
                    )
                with col3:
                    st.write("/ 100%")

            # Rerank Model with radio buttons
            st.write("Rerank Model")
            rerank_enabled = not st.radio(
                "",
                options=["Enable", "Disable"],
                index=1,  # Default to Disable
                horizontal=True,
                key="rerank_model",
                label_visibility="collapsed"
            ) == "Disable"

            # Return configuration values
            return {
                "doc_correlation": doc_correlation,
                "recall_number": recall_number,
                "retrieval_weight": retrieval_weight,
                "mixed_percentage": mixed_percentage if retrieval_weight == "Mixed" else None,
                "rerank_enabled": rerank_enabled
            }

def main():
    st.title("Vector Search")
    
    # Create two columns for the main layout
    left_col, right_col = st.columns([3, 4])
    
    with left_col:
        # Query input
        query = st.text_area(
            "Enter complete user question to view similarity Correlations of knowledge slices",
            height=100
        )
        
        # Search button
        if st.button("Retrieval", type="primary"):
            if query:
                # Submit query
                make_api_request(
                    "/query/submit",
                    method="POST",
                    data={"query": query}
                )
                
                # Retrieve results
                results = make_api_request("/query/retrieve", method="POST")
                
                if results.get("results"):
                    st.session_state.search_results = results["results"]
                else:
                    st.warning("No results found.")
        
        # Vector Search Configuration
        config = config_section()
        
        # Apply configuration button
        if st.button("Apply Configuration"):
            config_response = make_api_request(
                "/vector-search/configure",
                method="POST",
                params=config
            )
            st.success("Configuration updated successfully!")

    # Display results in right column
    with right_col:
        st.subheader("Search Results")
        
        if "search_results" in st.session_state:
            for idx, result in enumerate(st.session_state.search_results, 1):
                with st.container():
                    st.markdown(f"""
                    ##### {idx:03d} {result['tokens']} tokens
                    
                    **Chunk ID:** {result['id']}
                    
                    **Content:** {result['content']}
                    
                    **Correlation:** {result['correlation']}
                    """)
                    
                    # Display metadata as tags
                    if result.get('metadata'):
                        metadata_tags = []
                        for key, value in result['metadata'].items():
                            if isinstance(value, bool):
                                value = str(value).lower()
                            metadata_tags.append(f"*{key}*: {value}")
                        st.markdown(" • ".join(metadata_tags))
                    
                    st.divider()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Vector Search",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS to match the design more closely
    st.markdown("""
        <style>
        .stRadio > label {
            display: none !important;
        }
        .stExpander {
            border: 1px solid #e6e6e6;
            border-radius: 6px;
            padding: 10px;
        }
        .stSlider {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    main()
