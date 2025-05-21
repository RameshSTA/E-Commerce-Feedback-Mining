# ui_sections/executive_summary_view.py
import streamlit as st
import pandas as pd

def render_executive_summary(df_processed, num_topics_config): 
    st.header("🎯 Executive Summary & Business Value")
    st.markdown("""
    This interactive dashboard demonstrates how Natural Language Processing (NLP) and Network Analysis 
    can transform raw customer reviews into **actionable business intelligence**, enabling data-driven 
    decisions for enhanced customer satisfaction and strategic growth in the e-commerce clothing sector.
    """)
    
    st.subheader("📈 Key Metrics at a Glance:")
    if df_processed is not None:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(label="Total Reviews Analyzed", value=f"{len(df_processed):,}")
        if 'vader_sentiment_label' in df_processed.columns:
            positive_percentage = (df_processed['vader_sentiment_label'] == 'Positive').mean() * 100
            negative_percentage = (df_processed['vader_sentiment_label'] == 'Negative').mean() * 100
            col_m2.metric(label="Positive Sentiment 👍", value=f"{positive_percentage:.1f}%")
            col_m3.metric(label="Negative Sentiment 👎", value=f"{negative_percentage:.1f}%", delta_color="inverse")
        else:
            col_m2.metric(label="Positive Sentiment 👍", value="N/A")
            col_m3.metric(label="Negative Sentiment 👎", value="N/A")
        col_m4.metric(label="Discovered Customer Themes 📝", value=num_topics_config) # Use passed num_topics
    else:
        st.warning("Dataframe not available to display metrics.")
    st.markdown("<hr style='border:1px solid #eee'>", unsafe_allow_html=True)

    col1_sum, col2_sum = st.columns(2)
    with col1_sum:
        st.markdown("#### <span style='color:#2980B9;'>Challenges Addressed:</span>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#E8F8F5; padding:15px; border-radius:8px; border: 2px solid #A2D9CE;font-size: 0.75em;'>
        <ul>
            <li><strong>Scaling Customer Understanding:</strong> Efficiently analyzes thousands of reviews.</li>
            <li><strong>Identifying Core Themes:</strong> Automatically pinpoints consistent topics.</li>
            <li><strong>Nuanced Sentiment Insights:</strong> Deeper sentiment understanding beyond ratings.</li>
            <li><strong>Uncovering Connections:</strong> Reveals interrelations between feedback aspects.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2_sum:
        st.markdown("#### <span style='color:#27AE60;'>Strategic Business Impact:</span>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#EBF5FB; padding:15px; border-radius:8px; border: 2px solid #AED6F1;font-size: 0.75em;'>
        <ul>
            <li><strong>Product Development & QA:</strong> Guides improvements and flags quality issues.</li>
            <li><strong>Targeted Marketing:</strong> Refines messaging using customer language.</li>
            <li><strong>Service Excellence:</strong> Highlights recurring issues for support training.</li>
            <li><strong>Enhanced Retention:</strong> Proactively addresses pain points for loyalty.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.checkbox("Show Sample of Processed Data", True, help="Display the first 5 rows of the data used for analysis.", key="cb_sample_data_exec_view_final"):
        if df_processed is not None:
            st.dataframe(df_processed[['Review Text', 'Rating', 'vader_sentiment_label', 'dominant_lda_topic']].head(5).style.set_properties(**{'font-size': '10pt', 'text-align': 'left'}))
        else:
            st.info("Data not loaded.")