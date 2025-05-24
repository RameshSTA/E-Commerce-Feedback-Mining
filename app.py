# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
from streamlit_option_menu import option_menu # <-- Import option_menu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import ast
import base64 # Needed for potential image embedding if required

# --- Mock ui_sections if they don't exist ---
# (Ideally, these should be in a separate ui_sections.py file)
try:
    from ui_sections import executive_summary_view, sentiment_view, topic_modeling_view, \
                            network_view, recommendations_view, about_me_view
except ImportError:
    st.warning("Could not import `ui_sections`. Using mock functions.")
    class MockView:
        def render(self, *args, **kwargs): st.header(f"Mock View: {self.__class__.__name__}")
        def render_executive_summary(self, *args, **kwargs): st.header("Executive Summary")
        def render_sentiment_analysis(self, *args, **kwargs): st.header("Sentiment Analysis")
        def render_topic_modeling(self, *args, **kwargs): st.header("Topic Modeling")
        def render_network_analysis(self, *args, **kwargs): st.header("Network Analysis")
        def render_recommendations(self, *args, **kwargs): st.header("Recommendations")
        def render_about_me(self, *args, **kwargs): st.header("About Me")
    executive_summary_view = MockView()
    sentiment_view = MockView()
    topic_modeling_view = MockView()
    network_view = MockView()
    recommendations_view = MockView()
    about_me_view = MockView()
# --- End Mock ---


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
PROJECT_LOGO_FILENAME = "logo.png"
PROJECT_LOGO_PATH = os.path.join(ASSETS_DIR, PROJECT_LOGO_FILENAME)

# --- Function to load logo and encode ---
def get_logo_svg(file_path):
    # If it's a PNG, try to encode it. If it's an SVG, read it.
    # For simplicity, let's use a placeholder if the logo isn't an SVG,
    # or you can implement a more robust SVG/PNG handler.
    # Here, we'll use a placeholder SVG if the PNG isn't found,
    # or you can replace this with your actual SVG logo if you have one.
    if os.path.exists(file_path):
         # A simple way to embed PNG might be complex, let's use a placeholder
         # or ideally, you'd have an SVG version or use st.image carefully.
         # Using a placeholder for now to avoid base64 complexity here.
        return """
        <svg width="45" height="45" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="100" rx="15" fill="#E8F0FE"/>
            <path d="M25 75V35C25 31.6863 27.6863 29 31 29H69C72.3137 29 75 31.6863 75 35V75" stroke="#4285F4" stroke-width="6" stroke-linecap="round"/>
            <path d="M40 55H60" stroke="#34A853" stroke-width="6" stroke-linecap="round"/>
            <path d="M50 75V55" stroke="#34A853" stroke-width="6" stroke-linecap="round"/>
            <circle cx="50" cy="42" r="6" fill="#FBBC05"/>
            <path d="M35 65H65" stroke="#EA4335" stroke-width="6" stroke-linecap="round"/>
        </svg>
        """
    else:
        return "📊" # Fallback text/emoji

logo_svg_content = get_logo_svg(PROJECT_LOGO_PATH)


# --- Page Configuration ---
st.set_page_config(
    page_title="E-Commerce Feedback Mining by Ramesh Shrestha",
    page_icon="📊", # Simple icon, logo will be in header
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Project Specific Configurations ---
NUM_TOPICS = 7
topic_labels_dict = {
    1: "👚 Sizing & Fit", 2: "💖 Style & Appearance", 3: "🧵 Fabric & Material",
    4: "😌 Comfort & Wear", 5: "💰 Value & Returns", 6: "🎨 Color & Print",
    7: "💯 Overall Satisfaction"
}

# --- Page Definitions for option_menu ---
PAGES = {
    "Summary": {"func": executive_summary_view.render_executive_summary, "icon": "speedometer2", "args": (NUM_TOPICS,)},
    "Sentiment": {"func": sentiment_view.render_sentiment_analysis, "icon": "emoji-smile", "args": (None,)}, # Analyzer will be passed later
    "Topics": {"func": topic_modeling_view.render_topic_modeling, "icon": "tags", "args": (None, None, NUM_TOPICS, topic_labels_dict)}, # Models passed later
    "Network": {"func": network_view.render_network_analysis, "icon": "diagram-3", "args": (None, NUM_TOPICS, topic_labels_dict, None)}, # Graph/DF passed later
    "Recommendations": {"func": recommendations_view.render_recommendations, "icon": "lightbulb", "args": (NUM_TOPICS, topic_labels_dict)},
    "About Me": {"func": about_me_view.render_about_me, "icon": "person-circle", "args": (BASE_DIR, ASSETS_DIR)},
}
PAGE_NAMES = list(PAGES.keys())
PAGE_ICONS = [PAGES[p]["icon"] for p in PAGE_NAMES]


# --- Caching Functions (Keep as is) ---
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

# --- Update args with loaded data ---
if df_processed is not None:
    PAGES["Summary"]["args"] = (df_processed, NUM_TOPICS)
    PAGES["Sentiment"]["args"] = (df_processed, analyzer)
    PAGES["Topics"]["args"] = (df_processed, lda_model, feature_names, NUM_TOPICS, topic_labels_dict)
    PAGES["Network"]["args"] = (topic_network_graph, NUM_TOPICS, topic_labels_dict, df_processed)

essential_artifacts_loaded = all(obj is not None for obj in [df_processed, lda_model, tfidf_vectorizer, feature_names, topic_network_graph, analyzer])


# --- Global CSS Styling (MERGED & ADAPTED) ---
HEADER_TOP_ROW_HEIGHT_PX = 85 # Increased height for more content
HEADER_BOTTOM_ROW_HEIGHT_PX = 60
HEADER_TOTAL_HEIGHT_PX = HEADER_TOP_ROW_HEIGHT_PX + HEADER_BOTTOM_ROW_HEIGHT_PX
FOOTER_HEIGHT_PX = 80 # Increased for multi-line footer

# --- Color Palette (Adapted from Google-like theme) ---
HEADER_BG = "#FFFFFF"
FOOTER_BG = "#f8f9fa" # Use app BG for footer
TEXT_DARK = "#202124"
TEXT_MEDIUM = "#5f6368"
PRIMARY_BLUE = "#4285F4" # Google Blue
BORDER_COLOR = "#dadce0" # Google Grey Border
ACTIVE_BG = "#e8f0fe" # Google Active Blue BG
ACTIVE_TEXT = "#1967d2" # Google Active Blue Text

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Open+Sans:wght@400;600;700&display=swap');
        @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css");

        /* --- Global & Body --- */
        html, body {{ height: 100%; overflow: hidden; }}
        body {{ font-family: 'Roboto', sans-serif !important; background-color: #f8f9fa !important; }}

        /* --- Hide Streamlit Defaults --- */
        header[data-testid="stHeader"],
        button[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] {{ display: none !important; }}

        /* --- Main Content Padding --- */
        .main .block-container {{ padding: 0 !important; }}

        /* --- Main App Wrapper --- */
        .stApp {{ background-color: #f8f9fa !important; }}
        .stApp > div:first-child > div:first-child > div:first-child {{
            display: flex !important; flex-direction: column !important;
            height: 100vh !important; overflow: hidden !important;
        }}

        /* --- Header Container (Fixed) --- */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(1) {{
            position: fixed !important; top: 0 !important; left: 0 !important; right: 0 !important;
            height: {HEADER_TOTAL_HEIGHT_PX}px !important; background-color: {HEADER_BG} !important;
            z-index: 999998 !important; padding: 0 !important;
            border-bottom: 1px solid {BORDER_COLOR} !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            display: flex !important; flex-direction: column !important;
            flex-shrink: 0 !important;
        }}

        /* --- Content Container (Padded & Scrollable) --- */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(2) {{
            margin-top: {HEADER_TOTAL_HEIGHT_PX}px !important;
            margin-bottom: {FOOTER_HEIGHT_PX}px !important;
            padding: 25px 2rem 30px 2rem !important;
            width: 100% !important; overflow-y: auto !important; overflow-x: hidden !important;
            flex-grow: 1 !important;
            background-color: #f8f9fa !important; /* Ensure content BG matches body */
            color: {TEXT_DARK} !important;
        }}

        /* --- Footer Container (Fixed & Centered) --- */
        .stApp > div:first-child > div:first-child > div:first-child > div:nth-child(3) {{
            position: fixed !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
            height: {FOOTER_HEIGHT_PX}px !important; background-color: {FOOTER_BG} !important;
            border-top: 1px solid {BORDER_COLOR} !important; z-index: 999999 !important;
            display: flex !important; justify-content: center !important; align-items: center !important;
            padding: 0 1rem !important; flex-shrink: 0 !important; width: 100% !important;
        }}

        /* --- E-Commerce Theme Specifics --- */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Open Sans', sans-serif !important; color: {TEXT_DARK} !important;
        }}
        [data-testid="stExpander"] summary {{ font-size: 1.1em !important; font-weight: 600 !important; font-family: 'Open Sans', sans-serif !important; padding: 0.5rem 0rem !important; }}
        [data-testid="stExpander"] {{ border: 1px solid #e0e0e0 !important; border-radius: 0.3rem !important; padding: 0.5rem !important; background-color: #fff; }}
        div[data-testid="stInfo"] {{ background-color: #e8f0fe !important; border-left: 5px solid {PRIMARY_BLUE} !important; color: {TEXT_DARK} !important; padding: 1em !important; border-radius: 4px !important; font-size: 0.95em !important; }}
        div[data-testid="stSuccess"] {{ background-color: #e6f4ea !important; border-left: 5px solid #34a853 !important; color: {TEXT_DARK} !important; padding: 1em !important; border-radius: 4px !important; font-size: 0.95em !important; }}
        div[data-testid="stWarning"] {{ background-color: #fef7e0 !important; border-left: 5px solid #fbbc05 !important; color: {TEXT_DARK} !important; padding: 1em !important; border-radius: 4px !important; font-size: 0.95em !important; }}
        div[data-testid="stError"] {{ background-color: #fce8e6 !important; border-left: 5px solid #ea4335 !important; color: {TEXT_DARK} !important; padding: 1em !important; border-radius: 4px !important; font-size: 0.95em !important; }}

        /* --- Header Row 1: Logo, Title, Subtitle --- */
        .header-row-1 {{
            display: flex !important; align-items: center !important; justify-content: space-between !important;
            height: {HEADER_TOP_ROW_HEIGHT_PX}px !important; width: 100% !important;
            padding: 0 1.5rem !important;
        }}
        .title-section {{ display: flex; align-items: center; }}
        .title-section .project-logo {{ margin-right: 1rem; }}
        .title-section .text-content h1 {{
            font-family: 'Open Sans', sans-serif !important; font-size: 1.8rem; font-weight: 700;
            color: {TEXT_DARK} !important; line-height: 1.1; white-space: nowrap; margin: 0;
            margin-bottom: 2px;
        }}
        .title-section .text-content p {{
            font-family: 'Roboto', sans-serif !important; font-size: 1.05em; color: {TEXT_MEDIUM};
            margin: 0; line-height: 1.2;
        }}
        .developer-link p {{
            font-family: 'Roboto', sans-serif; font-size: 0.9rem; color: {TEXT_MEDIUM};
            margin: 0; text-align: right;
        }}
        .developer-link a {{ color: {PRIMARY_BLUE}; text-decoration: none; font-weight: 500; }}
        .developer-link a:hover {{ text-decoration: underline; }}


        /* --- Header Row 2: Navigation --- */
        .header-row-2 {{
            display: flex !important; align-items: center !important; justify-content: center !important;
            height: {HEADER_BOTTOM_ROW_HEIGHT_PX}px !important; width: 100% !important;
            padding: 0 1rem !important; overflow: hidden; background-color: {HEADER_BG} !important;
            border-top: 1px solid {BORDER_COLOR};
        }}

        /* --- Styling streamlit_option_menu (Google Theme) --- */
        .header-row-2 nav.menu {{ width: 100%; }}
        .header-row-2 nav.menu > ul {{
            display: flex !important; flex-direction: row !important; align-items: center !important;
            justify-content: center !important; height: 100% !important; width: 100% !important;
            margin: 0 !important; padding: 0 !important; background-color: {HEADER_BG} !important;
            flex-wrap: nowrap !important;
        }}
        .header-row-2 nav.menu > ul > li {{ list-style: none !important; padding: 0 !important; margin: 0 4px !important; background-color: {HEADER_BG} !important; flex-shrink: 0 !important; }}
        .header-row-2 nav.menu > ul > li > a {{
            display: flex !important; flex-direction: row !important; align-items: center !important;
            justify-content: center !important; text-decoration: none !important;
            color: {TEXT_MEDIUM} !important; font-weight: 500 !important; font-size: 0.9rem !important;
            padding: 8px 16px !important; border-radius: 5px !important;
            transition: color 0.2s ease, background-color 0.2s ease, border-bottom 0.2s ease !important;
            border-bottom: 3px solid transparent !important; line-height: 1.3 !important;
            background-color: {HEADER_BG} !important; white-space: nowrap !important;
            text-align: center !important;
        }}
        .header-row-2 nav.menu > ul > li > a > i {{ margin-right: 8px !important; font-size: 1.1rem !important; color: {TEXT_MEDIUM} !important; line-height: 1 !important; }}
        .header-row-2 nav.menu > ul > li > a:hover {{ color: {PRIMARY_BLUE} !important; background-color: #f1f3f4 !important; }}
        .header-row-2 nav.menu > ul > li > a:hover > i {{ color: {PRIMARY_BLUE} !important; }}
        .header-row-2 nav.menu > ul > li > a[aria-selected="true"] {{
            color: {ACTIVE_TEXT} !important; background-color: {ACTIVE_BG} !important;
            border-bottom: 3px solid {PRIMARY_BLUE} !important; font-weight: 700 !important;
        }}
        .header-row-2 nav.menu > ul > li a[aria-selected="true"] > i {{ color: {ACTIVE_TEXT} !important; }}

        /* --- Footer Styling (E-Commerce Content) --- */
        .footer-wrapper {{
            text-align: center; color: {TEXT_MEDIUM}; font-family: Roboto, sans-serif; font-size: 0.85em;
            line-height: 1.5;
        }}
        .footer-wrapper p {{ margin: 3px 0 !important; }}
        .footer-wrapper strong {{ color: {TEXT_DARK}; }}
        .footer-wrapper a {{ color: {PRIMARY_BLUE}; text-decoration: none; font-weight: 500; }}
        .footer-wrapper a:hover {{ text-decoration: underline; }}
        .footer-wrapper .copyright {{ font-size: 0.9em; color: #888; margin-top: 8px !important;}}


    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Summary" # Default section

# --- Define Containers ---
header = st.container()
content = st.container()
footer = st.container()

# --- Build Header ---
with header:
    # Row 1: Logo and Title (Adapted from E-Commerce App)
    st.markdown(
        f"""
        <div class="header-row-1">
            <div class="title-section">
                <div class="project-logo">{logo_svg_content}</div>
                <div class="text-content">
                    <h1>E-Commerce Feedback Mining</h1>
                    <p>Sentiment, Topics & Co-occurrence Networks</p>
                </div>
            </div>
            <div class="developer-link">
                <p>Developed by <strong>Ramesh Shrestha</strong> |
                    <a href='https://github.com/RameshSTA/E-Commerce-Feedback-Mining' target='_blank'>
                        GitHub 
                    </a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Row 2: Navigation Bar
    st.markdown('<div class="header-row-2">', unsafe_allow_html=True)
    selected_page = option_menu(
        menu_title=None,
        options=PAGE_NAMES,
        icons=PAGE_ICONS,
        menu_icon=None,
        default_index=PAGE_NAMES.index(st.session_state.current_page),
        orientation="horizontal",
        styles={ "container": {"padding": "0!important", "background-color": "transparent"}, }
    )
    st.markdown('</div>', unsafe_allow_html=True)


# --- Update session state and rerun ---
if selected_page and selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

# --- Build Content ---
with content:
    if not essential_artifacts_loaded:
        st.error("CRITICAL ERROR: One or more essential project artifacts could not be loaded. Please check paths and files.")
    else:
        page_config = PAGES.get(st.session_state.current_page)
        if page_config:
            func = page_config["func"]
            args = page_config["args"]
            
            # Create a list of actual arguments to pass
            actual_args = []
            for arg in args:
                if arg is None: # Placeholder, needs to be replaced with loaded data
                    if func == sentiment_view.render_sentiment_analysis:
                        actual_args = [df_processed, analyzer]
                    elif func == topic_modeling_view.render_topic_modeling:
                         actual_args = [df_processed, lda_model, feature_names, NUM_TOPICS, topic_labels_dict]
                    elif func == network_view.render_network_analysis:
                        actual_args = [topic_network_graph, NUM_TOPICS, topic_labels_dict, df_processed]
                    break # Assume all Nones need replacement based on function
                else:
                    actual_args.append(arg)
            
            # If no Nones were found, use original args
            if not any(a is None for a in args):
                actual_args = list(args)

            # Call the function with potentially updated args
            func(*actual_args)

        else:
            st.warning(f"Page '{st.session_state.current_page}' not found. Returning to Summary.")
            st.session_state.current_page = "Summary"
            PAGES["Summary"]["func"](df_processed, NUM_TOPICS) # Call default page func
            st.rerun()

# --- Build Footer ---
with footer:
    st.markdown(
        f"""
        <div class="footer-wrapper">
            <p>
                Developed by <strong>Ramesh Shrestha</strong> |
                <a href='https://github.com/RameshSTA/E-Commerce-Feedback-Mining' target='_blank'>
                    View Project on GitHub
                </a> |
                <a href='mailto:shrestha.ramesh000@gmail.com'>
                    Contact via Email
                </a>
            </p>
            <p class="copyright">
                &copy; {pd.Timestamp('now').year} Ramesh Shrestha. All Rights Reserved. For demonstration purposes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Add a note about troubleshooting ---
st.sidebar.info(
    """
    **E-Commerce Feedback Mining**
    Navigate through different analysis sections using the top bar.
    - **Summary**: High-level overview.
    - **Sentiment**: Review positivity/negativity.
    - **Topics**: Key themes discussed.
    - **Network**: How topics relate.
    - **Recommendations**: Business actions.
    - **About Me**: Developer info.
    """
)
st.sidebar.warning(
    """
    **Layout Note:**
    This app uses CSS for a fixed layout. If you experience issues, try a hard refresh (Ctrl+Shift+R) or check your browser's dev tools for CSS selector mismatches.
    """
)