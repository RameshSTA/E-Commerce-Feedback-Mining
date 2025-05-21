# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
# Plotly should be imported in ui_sections files where used, or here if app.py makes direct Plotly calls.
# For modularity, better to keep them in ui_sections.
# import plotly.express as px
# import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import ast

# Import functions from your ui_sections module
from ui_sections import executive_summary_view, sentiment_view, topic_modeling_view, \
                        network_view, recommendations_view, about_me_view

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# --- Artifact File Paths ---
DATA_FILE_PATH = os.path.join(DATA_DIR, 'reviews_final_for_streamlit.csv')
TFIDF_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.joblib')
LDA_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'lda_model.joblib')
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf_feature_names.joblib')
NETWORK_GRAPH_PATH = os.path.join(ARTIFACTS_DIR, 'topic_network.gexf')
PROJECT_LOGO_FILENAME = "logo.png"  # !!! YOUR LOGO FILENAME IN ASSETS FOLDER !!!
PROJECT_LOGO_PATH = os.path.join(ASSETS_DIR, PROJECT_LOGO_FILENAME)

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="E-Commerce Feedback Mining by Ramesh Shrestha",
    page_icon=PROJECT_LOGO_PATH if os.path.exists(PROJECT_LOGO_PATH) else "📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/RameshSTA/E-Commerce-Feedback-Mining/issues',  # REPLACE
        'Report a bug': "https://github.com/RameshSTA/E-Commerce-Feedback-Mining/issues",  # REPLACE
        'About': f"""
        ### E-Commerce Feedback Mining: Sentiment, Topics & Co-occurrence Networks
        This app showcases an NLP analysis of women's e-commerce clothing reviews.
        Developed by: Ramesh Shrestha (shrestha.ramesh000@gmail.com)
        Version: 2.8 (Corrected Subtitle)
        """
    }
)

# --- Inject Custom CSS for "Google-like" Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="st-"], .main, .stApp { /* Apply to main containers too */
        font-family: 'Roboto', sans-serif !important; 
        color: #202124 !important; 
    }

    /* Main app container background */
    .stApp { /* This targets the main app container */
        background-color: #f8f9fa !important; 
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Open Sans', sans-serif !important; 
        color: #202124 !important; 
    }
    
    /* Specific styling for Streamlit buttons in the horizontal nav to make them more uniform */
    div[data-testid="stHorizontalBlock"] button { 
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: .45em .65em !important;
        margin: 0.1em 0.2em !important; 
        height: 48px !important; 
        font-size: 0.88rem !important; 
        line-height: 1.3 !important;
        border-radius: 0.3rem !important;
        border: 1px solid #dadce0 !important; 
        width: 100% !important; 
        font-weight: 500 !important;
        transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out; 
    }

    div[data-testid="stHorizontalBlock"] button[kind="primary"] { /* Active button */
        border: 1.5px solid #1a73e8 !important; 
        background-color: #e8f0fe !important; 
        color: #1967d2 !important; 
        box-shadow: 0 1px 2px 0 rgba(26,115,232,.2) !important;
    }
    
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] { /* Inactive buttons */
        background-color: #ffffff !important; 
        color: #3c4043 !important; 
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background-color: #f1f3f4 !important; 
        border-color: #c6c6c6 !important;
    }

    /* Styling for expanders to have a cleaner look */
    [data-testid="stExpander"] summary {
        font-size: 1.1em !important; 
        font-weight: 600 !important; 
        font-family: 'Open Sans', sans-serif !important;
        padding: 0.5rem 0rem !important; 
    }
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0 !important;
        border-radius: 0.3rem !important;
        padding: 0.5rem !important;
    }
    
    /* Style for st.info, st.success etc. to align with Google feel */
    div[data-testid="stInfo"], div[data-testid="stSuccess"], 
    div[data-testid="stWarning"], div[data-testid="stError"] {
        padding: 1em !important;
        border-radius: 4px !important;
        font-size: 0.95em !important;
    }
    div[data-testid="stInfo"] {
        background-color: #e8f0fe !important; border-left: 5px solid #4285F4 !important; color: #202124 !important;
    }
    div[data-testid="stSuccess"] {
        background-color: #e6f4ea !important; border-left: 5px solid #34a853 !important; color: #202124 !important;
    }
    div[data-testid="stWarning"] {
        background-color: #fef7e0 !important; border-left: 5px solid #fbbc05 !important; color: #202124 !important;
    }
    div[data-testid="stError"] { 
        background-color: #fce8e6 !important; border-left: 5px solid #ea4335 !important; color: #202124 !important;
    }

</style>
""", unsafe_allow_html=True)


# --- Caching Functions (Definitions moved to the top) ---
@st.cache_data
def load_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        list_like_cols = ['processed_tokens', 'active_lda_topics_above_threshold']
        for col in list_like_cols:
            if col in df.columns and not df[col].empty and df[col].dtype == 'object':
                first_valid_entry = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid_entry, str) and first_valid_entry.strip().startswith('[') and first_valid_entry.strip().endswith(']'):
                    try: df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) else x)
                    except (ValueError, SyntaxError): pass
        if 'dominant_lda_topic' in df.columns: df['dominant_lda_topic'] = df['dominant_lda_topic'].astype(int)
        if 'Review Text' in df.columns: df['Review Text'] = df['Review Text'].astype(str).fillna('')
        if 'processed_text_joined' in df.columns: df['processed_text_joined'] = df['processed_text_joined'].astype(str).fillna('')
        return df
    except FileNotFoundError: st.error(f"FATAL ERROR: Main data file ('{os.path.basename(DATA_FILE_PATH)}') missing from '{DATA_DIR}'. App cannot function."); return None
    except Exception as e: st.error(f"Fatal Error loading data from '{file_path}': {e}"); return None

@st.cache_resource
def load_sklearn_model(file_path, model_name="Model"):
    try: return joblib.load(file_path)
    except FileNotFoundError: st.error(f"FATAL ERROR: {model_name} file ('{os.path.basename(file_path)}') missing from '{ARTIFACTS_DIR}'. App cannot function."); return None
    except Exception as e: st.error(f"Fatal Error loading {model_name} from '{file_path}': {e}"); return None

@st.cache_resource
def load_networkx_graph(file_path):
    try:
        if file_path.endswith('.gexf'):
            graph = nx.read_gexf(file_path)
            relabel_mapping = {node_id: int(node_id) for node_id in graph.nodes() if isinstance(node_id, str) and node_id.isdigit()}
            if relabel_mapping: graph = nx.relabel_nodes(graph, relabel_mapping, copy=True)
        elif file_path.endswith('.joblib'): graph = joblib.load(file_path)
        else: st.error(f"Unsupported graph file format: {file_path}."); return None
        return graph
    except FileNotFoundError: st.error(f"FATAL ERROR: Network graph ('{os.path.basename(NETWORK_GRAPH_PATH)}') missing from '{ARTIFACTS_DIR}'. App cannot function."); return None
    except Exception as e: st.error(f"Fatal Error loading graph from '{NETWORK_GRAPH_PATH}': {e}"); return None

# --- Load Data and Models ---
df_processed = load_dataframe(DATA_FILE_PATH)
tfidf_vectorizer = load_sklearn_model(TFIDF_VECTORIZER_PATH, "TF-IDF Vectorizer")
lda_model = load_sklearn_model(LDA_MODEL_PATH, "LDA Model")
feature_names = load_sklearn_model(FEATURE_NAMES_PATH, "TF-IDF Feature Names")
topic_network_graph = load_networkx_graph(NETWORK_GRAPH_PATH)
analyzer = None
try: analyzer = SentimentIntensityAnalyzer()
except LookupError:
    with st.spinner("Downloading VADER lexicon..."): import nltk; nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
except Exception as e: st.error(f"VADER Analyzer init error: {e}")

# --- Project Specific Configurations ---
NUM_TOPICS = 7 
topic_labels_dict = {
    1: "👚 Sizing & Fit", 2: "💖 Style & Appearance", 3: "🧵 Fabric & Material",
    4: "😌 Comfort & Wear", 5: "💰 Value & Returns", 6: "🎨 Color & Print",
    7: "💯 Overall Satisfaction"
}
if df_processed is not None and topic_labels_dict is not None and len(topic_labels_dict) != NUM_TOPICS :
    st.sidebar.warning(f"Config: NUM_TOPICS ({NUM_TOPICS}) vs topic_labels ({len(topic_labels_dict)}). Update `topic_labels_dict`.")


# --- Navigation State & Logic ---
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Executive Summary & Business Value" # Default section

nav_options_horizontal = {
    "🚀 Summary": "Executive Summary & Business Value",
    "🎭 Sentiment": "Sentiment Deep Dive",
    "🔑 Topics": "Topic Modeling Explorer",
    "🕸️ Network": "Topic Interconnections (Network)",
    "💡 Recommendations": "Actionable Recommendations",
    "👨‍💻 About Me": "About Me & My Aspirations"
}

# --- App Header with Title & Logo ---
col_logo_header, col_title_header = st.columns([1, 7]) 
with col_logo_header:
    if os.path.exists(PROJECT_LOGO_PATH):
        st.image(PROJECT_LOGO_PATH, width=70) 
    else:
        st.markdown("📊", help="Project Logo Placeholder") 
with col_title_header:
    st.markdown(f"<h1 style='margin-bottom: -10px; margin-top: 0px; font-family: Open Sans, sans-serif; color: #202124;'>E-Commerce Feedback Mining</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.1em; color: #5f6368; margin-top: 0px; font-family: Roboto, sans-serif;'>Sentiment, Topics & Co-occurrence Networks</p>", unsafe_allow_html=True)

st.markdown(
    f"""
    <p style='font-family: Roboto, sans-serif; font-size: 0.95em; color: #5f6368; margin-top: 5px;'>
        Developed by <strong>Ramesh Shrestha</strong> | 
        <a href='https://github.com/RameshSTA/E-Commerce-Feedback-Mining' target='_blank' style='color: #4285F4; text-decoration: none; font-weight: 500;'>
            GitHub
        </a>
    </p>
    """, 
    unsafe_allow_html=True
)



# --- Horizontal Navigation Bar ---
nav_cols = st.columns(len(nav_options_horizontal))
for i, (button_label, section_name_val) in enumerate(nav_options_horizontal.items()):
    button_type = "primary" if st.session_state.current_section == section_name_val else "secondary"
    if nav_cols[i].button(button_label, key=f"nav_h_btn_{section_name_val.replace(' ', '_')}_css_final_v3", use_container_width=True, type=button_type): # Changed key for uniqueness
        st.session_state.current_section = section_name_val
st.markdown("<hr style='border:1.5px solid #4285F4; margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# --- Main App Content Rendering ---
essential_artifacts_loaded_app_final_themed_v3 = all(obj is not None for obj in [df_processed, lda_model, tfidf_vectorizer, feature_names, topic_network_graph, analyzer])

if not essential_artifacts_loaded_app_final_themed_v3:
    st.error("CRITICAL ERROR: One or more essential project artifacts could not be loaded. The application cannot display content fully. Please check file paths, artifact integrity, and console for errors.")
else:
    current_section_to_render_app_final_themed_v3 = st.session_state.get('current_section', "Executive Summary & Business Value")
    
    # Call the appropriate rendering function from the ui_sections module
    if current_section_to_render_app_final_themed_v3 == "Executive Summary & Business Value":
        executive_summary_view.render_executive_summary(df_processed, NUM_TOPICS)
    elif current_section_to_render_app_final_themed_v3 == "Sentiment Deep Dive":
        sentiment_view.render_sentiment_analysis(df_processed, analyzer)
    elif current_section_to_render_app_final_themed_v3 == "Topic Modeling Explorer":
        topic_modeling_view.render_topic_modeling(df_processed, lda_model, feature_names, NUM_TOPICS, topic_labels_dict)
    elif current_section_to_render_app_final_themed_v3 == "Topic Interconnections (Network)":
        network_view.render_network_analysis(topic_network_graph, NUM_TOPICS, topic_labels_dict, df_processed)
    elif current_section_to_render_app_final_themed_v3 == "Actionable Recommendations":
        recommendations_view.render_recommendations(NUM_TOPICS, topic_labels_dict)
    elif current_section_to_render_app_final_themed_v3 == "About Me & My Aspirations":
        about_me_view.render_about_me(BASE_DIR, ASSETS_DIR)

# --- Footer ---
st.markdown("<hr style='border:1px solid #ddd; margin-top:40px;'>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align: center; color: #5f6368; font-family: Roboto, sans-serif; font-size: 0.9em;'>
        <p style="margin-bottom: 5px;"> 
            Developed by <strong>Ramesh Shrestha</strong> | 
            <a href='https://github.com/RameshSTA/E-Commerce-Feedback-Mining' target='_blank' style='color: #4285F4; text-decoration: none; font-weight: 500;'>
                View Project on GitHub
            </a>
        </p>
        <p style="font-size: 0.85em; color: #777; margin-top: 5px; margin-bottom: 5px;">
            Contact: <a href='mailto:shrestha.ramesh000@gmail.com' style='color: #4285F4; text-decoration: none;'>shrestha.ramesh000@gmail.com</a>
        </p>
        <p style="font-size: 0.8em; color: #777; margin-top: 10px;">
            &copy; {pd.Timestamp('now').year} Ramesh Shrestha. All Rights Reserved.<br> 
            The content of this project is for demonstration and portfolio purposes. 
            For permissions to use or reproduce any part of this project, please contact the developer.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)