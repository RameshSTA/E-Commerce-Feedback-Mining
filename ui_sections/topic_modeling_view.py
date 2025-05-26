
import streamlit as st
import pandas as pd
import plotly.express as px

# Define consistent colors (can be imported from a central config if you have one)
# For now, defining them here to match potential global theme colors or local needs.
PRIMARY_BLUE = "#007bff"
SUCCESS_GREEN = "#28a745"
NEUTRAL_GREY = "#6c757d"
POSITIVE_SENTIMENT_COLOR = '#2ECC71'
NEUTRAL_SENTIMENT_COLOR = '#BDC3C7'
NEGATIVE_SENTIMENT_COLOR = '#E74C3C'
BOX_BACKGROUND_COLOR = "#f8f9f9"  # <<< DEFINITION ADDED HERE (matches the hardcoded value used previously)
                                 # You could also use "#FFFFFF" if you want a white background like other content blocks

def display_lda_topics_for_view(
    lda_model_obj, 
    features_list: list[str] | None, 
    num_top_words: int,
    num_topics_config_view: int, 
    topic_labels_config_view: dict
    ):
    """
    Displays the top keywords for each discovered LDA topic in a structured layout.

    This function iterates through the topics identified by the LDA model,
    extracts the most significant words for each topic based on their weights,
    and presents them under their manually assigned (or default) labels.
    The display is arranged in columns for better readability.

    Args:
        lda_model_obj: The trained LDA model object (e.g., from scikit-learn).
        features_list (list[str] | None): A list of feature names (words) corresponding
                                     to the columns in the document-term matrix
                                     that the LDA model was trained on.
        num_top_words (int): The number of top keywords to display for each topic.
        num_topics_config_view (int): The total number of topics configured for the LDA model.
        topic_labels_config_view (dict): A dictionary mapping topic indices (1-based)
                                         to human-interpretable labels.
    """
    if lda_model_obj is not None and features_list is not None and len(features_list) > 0:
        st.markdown("##### Discovered Themes & Their Most Representative Keywords:")
        st.caption(
            "Each theme (topic) identified by the LDA model is characterized by a set of keywords. "
            "These keywords are the terms most strongly associated with that particular theme. "
            "The provided labels (e.g., 'Sizing & Fit') are manual interpretations based on these keywords."
        )

        num_display_cols = min(num_topics_config_view, 3) if num_topics_config_view > 0 else 1
        topic_cols_display = st.columns(num_display_cols)
        
        col_idx_current = 0
        for topic_idx, topic_weights in enumerate(lda_model_obj.components_):
            current_col_display = topic_cols_display[col_idx_current % num_display_cols]
            topic_num_for_user = topic_idx + 1
            label_for_topic = topic_labels_config_view.get(topic_num_for_user, f"Topic {topic_num_for_user} (Unlabeled)")
            
            with current_col_display:
                with st.expander(f"**{label_for_topic}**", expanded=True): 
                    top_word_indices = topic_weights.argsort()[:-num_top_words - 1:-1]
                    top_words = [features_list[i] for i in top_word_indices if i < len(features_list)]
                    
                    if top_words:
                        keywords_md = "<ul>" + "".join([f"<li style='font-size:0.9em;'>{word}</li>" for word in top_words]) + "</ul>"
                        st.markdown(keywords_md, unsafe_allow_html=True)
                    else:
                        st.markdown("_(No distinct top words found for this topic)_")
            col_idx_current +=1
    else:
        st.warning(
            "LDA model components or feature names are not available (or feature list is empty). "
            "Cannot display topic keywords. Please ensure the LDA model was trained and artifacts loaded correctly."
        )

def render_topic_modeling(
    df_processed: pd.DataFrame | None, 
    lda_model: object | None, 
    feature_names: list[str] | None, 
    num_topics_config: int, 
    topic_labels_config: dict
    ):
    """
    Renders the Topic Modeling Insights page for the E-Commerce Feedback Mining dashboard.

    This page explains TF-IDF and Latent Dirichlet Allocation (LDA), showcases
    keywords for discovered topics, visualizes overall topic distribution, and
    provides an interactive tool to explore reviews and sentiment by selected topic.

    Args:
        df_processed: DataFrame with processed review data, including 'dominant_lda_topic'.
        lda_model: Trained LDA model object.
        feature_names: List of feature names (vocabulary) for the LDA model.
        num_topics_config: Configured number of topics for LDA.
        topic_labels_config: Dictionary mapping topic indices to labels.
    """
    st.header("🔑Topic Modeling Insights")
    st.markdown("""
    This section illuminates the primary themes and subjects that customers discuss in their reviews. 
    By employing unsupervised machine learning, specifically **Latent Dirichlet Allocation (LDA)**, 
    we can automatically discover and categorize these underlying topics from unstructured text data.
    """)

    with st.expander("Understanding the Methodology: TF-IDF and LDA", expanded=True):
        st.markdown(f"""
        **1. What is TF-IDF (Term Frequency-Inverse Document Frequency)?**
        TF-IDF is a crucial preprocessing step that transforms text into a numerical representation suitable for machine learning algorithms. It works by:
        * **Term Frequency (TF):** Calculating how often a word appears in a single document (review).
        * **Inverse Document Frequency (IDF):** Measuring how common or rare a word is across all documents (the entire review corpus). Words that are very common everywhere (like "the", "a" - though stopwords are usually removed first) get a low IDF score, while words rare across documents but potentially frequent in a specific one get a high IDF score.
        The TF-IDF score for a word in a document is the product of its TF and IDF.
        *In this project, TF-IDF converts the processed customer review texts into a numerical matrix where each row represents a review and each column represents a unique word, with the cell values being the TF-IDF scores. This matrix highlights words that are characteristic and distinguishing for each review, which then serves as input for the LDA topic model.*

        **2. What is Topic Modeling with Latent Dirichlet Allocation (LDA)?**
        Topic Modeling is an unsupervised machine learning technique designed to scan a collection of documents (our customer reviews), identify word and phrase patterns within them, and automatically cluster word groups that best characterize a set of underlying, latent (hidden) themes or topics.
        **Latent Dirichlet Allocation (LDA)** is a widely-used probabilistic generative model for this purpose. It assumes that:
        * Each document is a mixture of various topics.
        * Each topic is a mixture of various words.
        LDA attempts to work backward from the documents to infer these topic-word and document-topic distributions.
        
        *In this project, LDA is applied to the TF-IDF matrix derived from customer reviews. It has been configured to identify **{num_topics_config} distinct underlying themes**. Each discovered theme is statistically represented by a distribution of words (its most prominent keywords). These keyword sets are then manually interpreted and assigned meaningful labels (e.g., "{topic_labels_config.get(1, "Sizing & Fit")}", "{topic_labels_config.get(2, "Fabric Quality")}") to make them business-relevant.*
        
        **Business Impact & Value of Topic Modeling:**
        * **Uncover Latent Customer Concerns & Praises:** Discover what topics customers are *truly* discussing, even if not explicitly asked.
        * **Automated Feedback Categorization:** Systematically group vast amounts of unstructured feedback into meaningful themes for efficient analysis and routing.
        * **Identify Specific Pain Points & Delighters:** Understand recurring issues (e.g., "poor stitching," "difficult returns") or frequently praised aspects (e.g., "beautiful design," "great comfort") related to products or services.
        * **Inform Strategic Initiatives:** Provide data-driven guidance for product development, marketing campaign messaging, customer service protocols, and website/UX improvements.
        * **Track Evolving Trends:** (With ongoing analysis) Monitor how discussion topics shift over time or in response to business changes.
        """.replace("{num_topics_config}", str(num_topics_config))) # Ensure num_topics_config is string for .replace
    st.markdown("---")

    if not all([obj is not None for obj in [df_processed, lda_model]]) or \
       (feature_names is None or len(feature_names) == 0) or \
       (df_processed is not None and 'dominant_lda_topic' not in df_processed.columns):
        st.warning(
            "Essential components for topic modeling (Processed Data with 'dominant_lda_topic', LDA model, or Feature Names) "
            "are not available or loaded. The Topic Modeling Explorer section cannot be fully displayed. "
            "Please ensure all artifacts are correctly loaded and the data processing pipeline has run successfully."
        )
        return

    st.subheader(f"💬 Interpreted Customer Discussion Themes (Based on {num_topics_config} Topics)")
    display_lda_topics_for_view(lda_model, feature_names, 10, num_topics_config, topic_labels_config)
    st.caption("The top 10 keywords are displayed for each theme to aid in its interpretation.")
    
    st.markdown("---")
    st.subheader("📊 Overall Distribution of Dominant Topics in Reviews")
    st.markdown("""
    This chart visualises the prevalence of each discovered theme across the entire dataset of customer reviews.
    Each review is assigned a "dominant" topic, i.e., the theme it most strongly aligns with based on its content.
    **Value:** Understanding this distribution helps identify which topics are most frequently discussed by customers overall.
    This, in turn, can guide businesses in prioritising areas for attention, whether for product improvement,
    marketing focus, or customer service enhancements.
    """)
    
    if 'dominant_lda_topic' in df_processed.columns:
        topic_counts_df_display = df_processed['dominant_lda_topic'].value_counts().reset_index()
        topic_counts_df_display.columns = ['dominant_lda_topic', 'Number of Reviews']
        topic_counts_df_display['Interpreted Topic Label'] = topic_counts_df_display['dominant_lda_topic'].map(topic_labels_config).fillna(topic_counts_df_display['dominant_lda_topic'].apply(lambda x: f"Topic {x} (Unlabeled)"))
        topic_counts_df_display = topic_counts_df_display.sort_values(by='dominant_lda_topic')

        fig_topic_dist_plotly = px.bar(
            topic_counts_df_display, 
            x='Interpreted Topic Label', 
            y='Number of Reviews',
            color='Interpreted Topic Label', 
            title=f'Frequency of {num_topics_config} Dominant Customer Discussion Themes',
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            text_auto=True,
            labels={'Interpreted Topic Label': 'Customer Theme', 'Number of Reviews': 'Volume of Reviews'}
        )
        fig_topic_dist_plotly.update_layout(
            title_x=0.5, showlegend=False, height=600,
            xaxis={'categoryorder':'array', 
                   'categoryarray': [topic_labels_config.get(i, f"Topic {i} (Unlabeled)") for i in sorted(topic_counts_df_display['dominant_lda_topic'].unique())]},
            yaxis_title="Number of Reviews Associated with Theme",
            xaxis_title="Interpreted Customer Theme"
        )
        fig_topic_dist_plotly.update_xaxes(tickangle=-45, tickfont=dict(size=10)) 
        fig_topic_dist_plotly.update_traces(texttemplate='%{y:,}', textposition='outside', marker_line_width=1.5, marker_line_color="black")
        st.plotly_chart(fig_topic_dist_plotly, use_container_width=True)
    else:
        st.info("Column 'dominant_lda_topic' is not available in the data. Cannot display overall topic distribution.")

    st.markdown("---")
    st.subheader("🔎 Deep Dive: Explore Reviews & Sentiment by Selected Topic")
    st.markdown("""
    This interactive section allows for a more granular exploration of each specific theme. 
    By selecting a topic from the dropdown, you can:
    1.  View sample customer reviews that are predominantly associated with that chosen theme.
    2.  Analyse the sentiment distribution (Positive, Neutral, Negative) specifically *within* those selected reviews.
    **Value:** This deep dive helps in understanding the detailed context and emotional tone surrounding each key customer discussion point. 
    For example, is the "Sizing & Fit" topic usually discussed with negative sentiment, or are discussions around "Style & Appearance" typically positive?
    """)
    
    if 'dominant_lda_topic' in df_processed.columns:
        unique_topics_in_data_list = sorted(df_processed['dominant_lda_topic'].unique())
        selectbox_options_map = {
            topic_labels_config.get(num, f"Topic {num} (Unlabeled)"): num 
            for num in unique_topics_in_data_list
        }
        sorted_selectbox_labels = sorted(selectbox_options_map.keys(), key=lambda label: selectbox_options_map[label])

        if not selectbox_options_map:
            st.warning("No dominant topics found in the current dataset to populate the selector, or `topic_labels_config` might not cover these topics.")
        else:
            selected_topic_label_ui = st.selectbox(
                "Select a Customer Theme to Explore in Detail:", 
                options=sorted_selectbox_labels, 
                key="topic_selector_topic_view_final_v2", # Ensure unique key
                help="Choose a theme to see associated reviews and their sentiment breakdown."
            )

            selected_numeric_topic_val = selectbox_options_map.get(selected_topic_label_ui)
            
            if selected_numeric_topic_val is not None:
                topic_specific_df_view = df_processed[df_processed['dominant_lda_topic'] == selected_numeric_topic_val]
                
                st.markdown(f"#### Insights for Theme: **{selected_topic_label_ui}**")
                
                if 'vader_sentiment_label' not in topic_specific_df_view.columns:
                    st.warning(f"Sentiment data ('vader_sentiment_label') missing for reviews under topic '{selected_topic_label_ui}'. Cannot display full details.")
                else:
                    col_rev_topic_ui, col_sent_dist_topic_ui = st.columns([3,2], gap="large")

                    with col_rev_topic_ui:
                        st.markdown(f"##### Representative Customer Reviews (Max 5 Samples):")
                        if not topic_specific_df_view.empty:
                            for _, review_row in topic_specific_df_view[['Review Text', 'Rating', 'vader_sentiment_label']].head().iterrows():
                                rating_stars = '⭐' * int(review_row['Rating']) if pd.notna(review_row['Rating']) and review_row['Rating'] > 0 else 'N/A'
                                sentiment_label_review = review_row.get('vader_sentiment_label', 'N/A')
                                
                                border_color_review = NEUTRAL_GREY 
                                if sentiment_label_review == 'Positive': border_color_review = SUCCESS_GREEN
                                elif sentiment_label_review == 'Negative': border_color_review = NEGATIVE_SENTIMENT_COLOR

                                # Using BOX_BACKGROUND_COLOR defined at the top of the file
                                st.markdown(f"""
                                <div style="border-left: 5px solid {border_color_review}; background-color: {BOX_BACKGROUND_COLOR}; 
                                            padding: 12px 15px; margin-bottom: 12px; border-radius: 5px;
                                            box-shadow: 1px 1px 3px #ddd;">
                                    <small><b>Rating: {rating_stars}</b> | VADER Sentiment: <b>{sentiment_label_review}</b></small><br>
                                    <p style="font-style: italic; margin-top: 5px;">"{review_row['Review Text']}"</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(f"No reviews were predominantly categorized under the theme: '{selected_topic_label_ui}'.")
                    
                    with col_sent_dist_topic_ui:
                        st.markdown(f"##### Sentiment Distribution within '{selected_topic_label_ui}':")
                        if not topic_specific_df_view.empty and 'vader_sentiment_label' in topic_specific_df_view.columns:
                            topic_sent_counts_view = topic_specific_df_view['vader_sentiment_label'].value_counts().reset_index()
                            topic_sent_counts_view.columns = ['Sentiment Label', 'Number of Reviews'] # Renamed for clarity
                            
                            if not topic_sent_counts_view.empty:
                                fig_sentiment_per_topic = px.bar(
                                    topic_sent_counts_view, 
                                    x='Sentiment Label', y='Number of Reviews', 
                                    color='Sentiment Label',
                                    category_orders={"Sentiment Label": ['Positive', 'Neutral', 'Negative', 'Error']},
                                    color_discrete_map={
                                        'Positive': POSITIVE_SENTIMENT_COLOR, 
                                        'Neutral': NEUTRAL_SENTIMENT_COLOR, 
                                        'Negative': NEGATIVE_SENTIMENT_COLOR,
                                        'Error': '#F39C12' 
                                    },
                                    template="plotly_white", 
                                    text_auto=True,
                                    labels={'Number of Reviews': 'Review Count'}
                                )
                                fig_sentiment_per_topic.update_layout(
                                    showlegend=False, 
                                    yaxis_title="Number of Reviews", 
                                    xaxis_title="Sentiment within this Topic",
                                    height=400, title_text=None, 
                                    margin=dict(t=20, b=10, l=10, r=10)
                                )
                                fig_sentiment_per_topic.update_traces(texttemplate='%{y}', textposition='outside')
                                st.plotly_chart(fig_sentiment_per_topic, use_container_width=True)
                            else:
                                st.info(f"No sentiment data to display for reviews under '{selected_topic_label_ui}'.")
                        elif topic_specific_df_view.empty:
                             st.info(f"No reviews found for topic '{selected_topic_label_ui}' to analyze sentiment.")
            else:
                st.warning(f"Selected topic '{selected_topic_label_ui}' could not be mapped to a numeric topic ID. Please check topic configurations.")
    else:
        st.info("Column 'dominant_lda_topic' is not available in the data. Interactive topic exploration cannot be performed.")