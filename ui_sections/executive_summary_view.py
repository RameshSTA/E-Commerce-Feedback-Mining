
import streamlit as st
import pandas as pd

# Define consistent colors (can be imported from a central config if you have one)
# For now, defining them here to match potential global theme colors we discussed.
PRIMARY_BLUE = "#007bff" # Example: Bootstrap Primary Blue
SUCCESS_GREEN = "#28a745" # Example: Bootstrap Success Green
BOX_BACKGROUND_COLOR = "#F8F9FA" # A light, neutral background

def render_executive_summary(df_processed: pd.DataFrame | None, num_topics_config: int):
    """
    Renders the Executive Summary page for the E-Commerce Feedback Mining dashboard.

    This page provides a high-level overview of the project's purpose,
    key insights derived from customer reviews, the business challenges
    it addresses, and the strategic impact of leveraging these NLP-driven insights.
    It also displays key aggregate metrics from the processed dataset.

    Args:
        df_processed (pd.DataFrame | None): The main DataFrame containing the processed
                                            review data with sentiment labels,
                                            dominant topics, etc. Expected to be None if
                                            data loading failed.
        num_topics_config (int): The configured number of topics from the LDA model.
    """
    st.header("🎯Unlocking Customer Voice with AI")
    st.markdown(f"""
    Welcome to the **E-Commerce Feedback Mining Platform**. This executive summary highlights how
    Natural Language Processing (NLP), sentiment analysis, and topic modelling can transform raw
    customer reviews into **actionable business intelligence**. The insights presented empower
    data-driven decision-making, aiming to significantly enhance customer satisfaction, refine product
    offerings, and drive strategic growth within the competitive e-commerce clothing sector.

    Below, you'll find a snapshot of key aggregate metrics, the primary business challenges this analytical
    approach addresses, and the strategic value it delivers.
    """)
    st.markdown("---")

    st.subheader("📈 Key Performance Indicators at a Glance:")
    st.markdown("These metrics provide an immediate overview of the dataset and the overall sentiment landscape.")
    if df_processed is not None and not df_processed.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(label="Total Reviews Analysed", value=f"{len(df_processed):,}")

        if 'vader_sentiment_label' in df_processed.columns:
            positive_reviews = df_processed['vader_sentiment_label'] == 'Positive'
            negative_reviews = df_processed['vader_sentiment_label'] == 'Negative'
            
            positive_percentage = positive_reviews.mean() * 100 if not positive_reviews.empty else 0
            negative_percentage = negative_reviews.mean() * 100 if not negative_reviews.empty else 0
            
            col_m2.metric(label="Positive Sentiment 👍", value=f"{positive_percentage:.1f}%",
                          help="Percentage of reviews classified with positive sentiment by VADER.")
            col_m3.metric(label="Negative Sentiment 👎", value=f"{negative_percentage:.1f}%",
                          delta_color="inverse" if negative_percentage > 10 else "normal", # Example: highlight if >10%
                          help="Percentage of reviews classified with negative sentiment. Highlighted if significant.")
        else:
            col_m2.metric(label="Positive Sentiment 👍", value="N/A", help="VADER sentiment data unavailable.")
            col_m3.metric(label="Negative Sentiment 👎", value="N/A", help="VADER sentiment data unavailable.")
        
        col_m4.metric(label="Discovered Customer Themes 📝", value=num_topics_config,
                      help=f"Number of distinct topics identified from customer reviews using LDA topic modelling (configured as {num_topics_config}).")
    else:
        st.warning("Processed dataset is not available. Key metrics cannot be displayed.")
    st.markdown("---")

    col1_sum, col2_sum = st.columns(2)
    with col1_sum:
        st.markdown(f"#### <span style='color:{PRIMARY_BLUE};'>Key Challenges This Analysis Addresses:</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color:{BOX_BACKGROUND_COLOR}; padding:18px; border-radius:8px; border-left: 5px solid {PRIMARY_BLUE}; font-size: 0.95em; line-height: 1.7;'>
        Effectively understanding and acting upon customer feedback at scale presents several challenges for e-commerce businesses:
        <ul>
            <li style="margin-bottom: 8px;"><strong>Volume & Velocity of Feedback:</strong> Manually sifting through thousands of customer reviews is impractical and time-consuming, leading to missed insights.</li>
            <li style="margin-bottom: 8px;"><strong>Identifying Underlying Themes:</strong> Pinpointing the core topics and recurring issues or praises within unstructured text can be difficult without systematic analysis.</li>
            <li style="margin-bottom: 8px;"><strong>Gauging True Sentiment:</strong> Star ratings alone often don't capture the nuance of customer feelings. NLP provides a deeper understanding of sentiment.</li>
            <li style="margin-bottom: 8px;"><strong>Understanding Interconnections:</strong> Discovering how different aspects of the customer experience (e.g., product fit and return process) are related and discussed together.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2_sum:
        st.markdown(f"#### <span style='color:{SUCCESS_GREEN};'>Strategic Business Impact & Value:</span>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color:{BOX_BACKGROUND_COLOR}; padding:18px; border-radius:8px; border-left: 5px solid {SUCCESS_GREEN}; font-size: 0.95em; line-height: 1.7;'>
        Leveraging insights from this NLP-driven feedback analysis can generate significant business value:
        <ul>
            <li style="margin-bottom: 8px;"><strong>Data-Informed Product Development & Quality Assurance:</strong> Identify specific product features that delight or frustrate customers, guiding design improvements and flagging quality control issues proactively.</li>
            <li style="margin-bottom: 8px;"><strong>Optimised Marketing & Merchandising:</strong> Understand the language customers use to describe products they love, refining marketing messaging and product descriptions for better resonance and conversion.</li>
            <li style="margin-bottom: 8px;"><strong>Enhanced Customer Service Excellence:</strong> Pinpoint recurring pain points in the customer journey (e.g., shipping, returns, support interactions) to improve service protocols and agent training.</li>
            <li style="margin-bottom: 8px;"><strong>Increased Customer Loyalty & Retention:</strong> By proactively addressing identified issues and amplifying positive aspects, businesses can improve overall customer satisfaction, leading to stronger loyalty and reduced churn.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.checkbox("Show Sample of Processed Data for Context", 
                   value=True, # Default to unchecked to keep summary concise
                   help="Display the first 5 rows of the dataset used for these analyses, showing key input and output columns.", 
                   key="cb_sample_data_exec_summary_final"): # Unique key
        if df_processed is not None and not df_processed.empty:
            st.markdown("Below is a small sample of the processed data, illustrating the raw review text alongside key analytical outputs like VADER sentiment and the dominant LDA topic assigned to each review:")
            display_cols = ['Review Text', 'Rating', 'vader_sentiment_label', 'dominant_lda_topic']
            # Ensure columns exist before trying to display them
            cols_to_show_in_sample = [col for col in display_cols if col in df_processed.columns]
            if cols_to_show_in_sample:
                st.dataframe(
                    df_processed[cols_to_show_in_sample].head(5).style.set_properties(
                        **{'font-size': '10pt', 'text-align': 'left', 'max-width': '200px'} # Added max-width for text
                    )
                )
            else:
                st.info("Key columns for the sample display are not available in the processed data.")
        else:
            st.info("Processed data is not available to display a sample.")