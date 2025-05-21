# ui_sections/topic_modeling_view.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Helper function to display topics within this view
# It uses the configurations passed from app.py
def display_lda_topics_for_view(lda_model_obj, features_list, num_top_words, 
                                num_topics_config_view, topic_labels_config_view):
    if lda_model_obj is not None and features_list is not None:
        st.markdown("##### Key Themes Discovered (Top Keywords):")
        # Determine number of columns for topic display dynamically
        num_display_cols = min(num_topics_config_view, 3) if num_topics_config_view > 1 else 1
        topic_cols_display_func_view = st.columns(num_display_cols)
        
        col_idx_func_view = 0
        for topic_idx, topic_weights in enumerate(lda_model_obj.components_):
            current_col_func_view = topic_cols_display_func_view[col_idx_func_view % num_display_cols]
            topic_num = topic_idx + 1 # Topics are 1-indexed for users
            label = topic_labels_config_view.get(topic_num, f"Topic {topic_num} (Unlabeled)") # Get label from dict
            
            with current_col_func_view:
                # CORRECTED LINE: removed the extraneous 'open'
                with st.expander(f"**{label}**", expanded=True): 
                    top_word_indices = topic_weights.argsort()[:-num_top_words - 1:-1]
                    top_words = [features_list[i] for i in top_word_indices]
                    st.markdown(f"✨ `{', '.join(top_words)}`") 
            col_idx_func_view +=1
    else:
        # This warning will appear if models aren't loaded when this function is called
        st.warning("LDA model or feature names not available. Cannot display topic keywords.")


def render_topic_modeling(df_processed, lda_model, feature_names, num_topics_config, topic_labels_config):
    st.header("🔑 Discovered Customer Discussion Topics")

    # Explanation for TF-IDF and Topic Modeling (LDA) is directly visible
    st.subheader("Understanding the Methodology") 
    st.markdown("""
    **What is TF-IDF?**
    TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document within a collection (or "corpus"). It assigns higher weights to words that are frequent in a specific document but rare across all documents, helping to highlight characteristic terms.
    *In this project, TF-IDF converts the processed review text into a numerical matrix, making it ready for topic modeling algorithms.*

    **What is Topic Modeling (LDA)?**
    Topic Modeling is an unsupervised machine learning technique used to scan a set of documents (like customer reviews), detect word and phrase patterns, and automatically cluster groups of words that best characterize underlying themes or topics. Latent Dirichlet Allocation (LDA) is a popular probabilistic algorithm for this purpose.
    
    **How it's used in this project:**
    LDA is applied to the TF-IDF matrix of customer reviews to automatically identify **{num_topics_config} distinct underlying themes** that customers frequently discuss. Each discovered theme is represented by a collection of its most prominent keywords, which are then manually interpreted to assign a meaningful label (e.g., "Sizing & Fit," "Fabric Quality").
    
    **Business Impact:**
    * **Uncover Hidden Themes:** Discover what customers are *really* talking about.
    * **Categorize Feedback:** Automatically group vast amounts of feedback for easier analysis.
    * **Identify Pain Points & Praises:** Understand recurring issues or positive aspects related to products or services.
    * **Inform Strategy:** Guide product development, marketing, and customer service improvements.
    """.replace("{num_topics_config}", str(num_topics_config))) # Dynamically insert NUM_TOPICS
    st.markdown("---")

    # Check if all necessary components for this section are loaded
    if not all([obj is not None for obj in [lda_model, feature_names, df_processed]]) or \
       ('dominant_lda_topic' not in df_processed.columns):
        st.warning("Topic modeling artifacts (LDA model, features) or the 'dominant_lda_topic' column in the data are not available. Cannot display the Topic Modeling Explorer section.")
        return # Exit the function if prerequisites aren't met

    st.markdown(f"The analysis identified **{num_topics_config} primary themes**. Each topic below lists its most representative keywords:")
    display_lda_topics_for_view(lda_model, feature_names, 10, num_topics_config, topic_labels_config) # num_top_words can be adjusted
    
    st.markdown("<hr style='border:1px solid #eee; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.subheader("📊 Overall Distribution of Dominant Topics")
    
    topic_counts_df = df_processed['dominant_lda_topic'].value_counts().reset_index()
    topic_counts_df.columns = ['dominant_lda_topic', 'Number of Reviews']
    # Use the passed topic_labels_config for consistent labeling
    topic_counts_df['Interpreted Topic Label'] = topic_counts_df['dominant_lda_topic'].map(topic_labels_config).fillna(topic_counts_df['dominant_lda_topic'].astype(str))
    topic_counts_df = topic_counts_df.sort_values(by='dominant_lda_topic')

    fig_td_plotly = px.bar(topic_counts_df, 
                         x='Interpreted Topic Label', 
                         y='Number of Reviews',
                         color='Interpreted Topic Label', 
                         title=f'Overall Distribution of {num_topics_config} Dominant Topics',
                         color_discrete_sequence=px.colors.qualitative.Vivid, 
                         text_auto=True) 
    fig_td_plotly.update_layout(
        xaxis_title="Interpreted Topic Label", 
        yaxis_title="Number of Reviews", 
        title_x=0.5, # Center title
        showlegend=False, 
        height=550,
        xaxis={'categoryorder':'array', 'categoryarray': [topic_labels_config.get(i, str(i)) for i in sorted(topic_counts_df['dominant_lda_topic'].unique())]} 
    )
    fig_td_plotly.update_xaxes(tickangle=-45) 
    fig_td_plotly.update_traces(texttemplate='%{y:,}', textposition='outside')
    st.plotly_chart(fig_td_plotly, use_container_width=True)

    st.markdown("<hr style='border:1px solid #eee; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.subheader("🔎 Explore Reviews & Sentiment by Selected Topic")
    
    # Ensure unique topics from data are used for selectbox options if not all topics are present
    unique_topics_in_data = sorted(df_processed['dominant_lda_topic'].unique())
    selectbox_options = {num: label for num, label in topic_labels_config.items() if num in unique_topics_in_data}

    if not selectbox_options:
        st.warning("No dominant topics found in the current dataset to populate the selector, or `topic_labels_dict` might not cover these topics.")
    else:
        selected_topic_label = st.selectbox(
            "Select a Topic to Explore:", 
            options=list(selectbox_options.values()), 
            key="topic_selector_in_topic_view_corrected" 
        )

        selected_numeric_topic = None
        for num_key, str_label_val in selectbox_options.items():
            if str_label_val == selected_topic_label:
                selected_numeric_topic = num_key
                break
        
        if selected_numeric_topic is not None:
            topic_specific_df = df_processed[df_processed['dominant_lda_topic'] == selected_numeric_topic]
            
            st.markdown(f"#### Displaying Insights for: **{selected_topic_label}**")
            col_rev_topic, col_sent_dist_topic = st.columns([3,2]) # Reviews column wider

            with col_rev_topic:
                st.markdown(f"##### Sample Reviews (Up to 5):")
                if not topic_specific_df.empty:
                    for _, row_review in topic_specific_df[['Review Text', 'Rating', 'vader_sentiment_label']].head().iterrows():
                        # Using a more styled container for each review
                        st.markdown(f"""
                        <div style="border-left: 5px solid #1abc9c; background-color: #f8f9f9; 
                                    padding: 12px; margin-bottom: 12px; border-radius: 5px;
                                    box-shadow: 2px 2px 5px #eee;">
                            <small><b>Rating: {'⭐'*row_review['Rating']}</b> | Sentiment: {row_review['vader_sentiment_label']}</small><br>
                            <i>{row_review['Review Text']}</i>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No reviews are predominantly categorized under this topic.")
            
            with col_sent_dist_topic:
                st.markdown(f"##### Sentiment Distribution:")
                topic_sent_counts = topic_specific_df['vader_sentiment_label'].value_counts().reset_index()
                topic_sent_counts.columns = ['Sentiment Label', 'Count']
                if not topic_sent_counts.empty:
                    fig_ts_sel_topic = px.bar(topic_sent_counts, x='Sentiment Label', y='Count', 
                                         color='Sentiment Label',
                                         category_orders={"Sentiment Label": ['Positive', 'Neutral', 'Negative']},
                                         color_discrete_map={'Positive':'#2ECC71', 'Neutral':'#BDC3C7', 'Negative':'#E74C3C'},
                                         template="plotly_white", 
                                         text_auto=True)
                    fig_ts_sel_topic.update_layout(showlegend=False, yaxis_title="Number of Reviews", 
                                               height=400, title_text=None, margin=dict(t=10, b=0, l=0, r=0)) # Compact layout
                    fig_ts_sel_topic.update_traces(texttemplate='%{y}', textposition='outside')
                    st.plotly_chart(fig_ts_sel_topic, use_container_width=True)
                else:
                    st.info("No sentiment data to display for this topic's reviews.")
        else:
            st.warning(f"Could not find data for the selected topic: {selected_topic_label}")