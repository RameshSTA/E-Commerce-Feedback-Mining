# ui_sections/sentiment_view.py
import streamlit as st
import pandas as pd
import plotly.express as px


def render_sentiment_analysis(df_processed, analyzer):
    st.header("🎭 Sentiment Analysis Insights")
    
    st.info("""
    **What is Sentiment Analysis?**
    Sentiment Analysis is an NLP technique used to determine the emotional tone (positive, negative, or neutral) 
    behind a body of text, such as customer reviews.

    **How it's used in this project:**
    We utilize VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based sentiment analysis tool 
    specifically attuned to sentiments expressed in social media and online content. Each review is assigned a compound 
    sentiment score, which is then categorized into Positive, Negative, or Neutral.

    **Business Impact:**
    Understanding customer sentiment allows businesses to:
    * Quickly gauge overall customer satisfaction or dissatisfaction.
    * Identify products or service aspects that are particularly well-received or problematic.
    * Correlate sentiment with explicit ratings to uncover deeper insights (e.g., why a 3-star review might still have positive sentiment).
    * Prioritize areas for improvement to directly enhance customer experience and loyalty.
    """)
    st.markdown("---")

    if df_processed is None:
        st.warning("Processed data is not available. Cannot display sentiment analysis.")
        return
    
    if analyzer is None:
        st.warning("Sentiment Analyzer (VADER) not initialized. Cannot perform interactive sentiment test.")
       

    col1_sent_main, col2_sent_main = st.columns([2,3]) # Ratio for pie chart and box plot
    
    with col1_sent_main:
        st.subheader("Overall Sentiment Breakdown")
        if 'vader_sentiment_label' in df_processed.columns:
            sentiment_counts_df = df_processed['vader_sentiment_label'].value_counts().reset_index()
            sentiment_counts_df.columns = ['Sentiment Label', 'Number of Reviews']
            
            # Display metrics above the pie chart
            for _, row in sentiment_counts_df.iterrows():
                label = row['Sentiment Label']
                count = row['Number of Reviews']
                percentage = (count / len(df_processed)) * 100
                delta_color = "normal"
                if label == "Negative": delta_color = "inverse"
                elif label == "Neutral": delta_color = "off"
                st.metric(label=f"{label} Reviews", value=f"{count:,}", delta=f"{percentage:.1f}%", delta_color=delta_color if label != "Neutral" else "off")

            fig_pie_sent = px.pie(sentiment_counts_df, 
                                names='Sentiment Label', 
                                values='Number of Reviews',
                                title='Overall Sentiment Distribution', 
                                hole=0.4, # Doughnut chart
                                color='Sentiment Label',
                                color_discrete_map={'Positive':'#2ECC71', 'Neutral':'#BDC3C7', 'Negative':'#E74C3C'},
                                template="plotly_white") # Use a clean template
            fig_pie_sent.update_traces(textposition='outside', textinfo='percent+label', pull=[0.03, 0, 0.03])
            fig_pie_sent.update_layout(legend_title_text='Sentiment', title_x=0.5, height=400, margin=dict(t=60, b=20, l=0, r=0))
            st.plotly_chart(fig_pie_sent, use_container_width=True)
        else:
            st.warning("Column 'vader_sentiment_label' not found in the data. Cannot display sentiment breakdown.")

    with col2_sent_main:
        if 'Rating' in df_processed.columns and 'compound' in df_processed.columns:
            st.subheader("Sentiment Score vs. Product Rating")
            fig_box_sent = px.box(df_processed, x='Rating', y='compound', color='Rating',
                                title='VADER Compound Sentiment Score by Product Rating',
                                labels={'compound': 'VADER Compound Score', 'Rating': 'Product Rating (Stars)'},
                                color_discrete_sequence=px.colors.sequential.Plasma_r, 
                                points="all",
                                notched=True) 
            fig_box_sent.update_layout(title_x=0.5, height=450, 
                                   yaxis_title="VADER Compound Score", 
                                   xaxis_title="Product Rating (Stars)")
            st.plotly_chart(fig_box_sent, use_container_width=True)
            st.caption("This box plot shows the distribution of VADER compound sentiment scores (ranging from -1 for most negative to +1 for most positive) for each star rating given by customers. Notches indicate 95% confidence intervals around the median.")
        else:
            st.warning("Columns 'Rating' or 'compound' not found in the data. Cannot display sentiment vs. rating plot.")

    st.markdown("<hr style='border:1px solid #eee; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.subheader("🔬 Test Sentiment for Your Own Text:")
    user_text_input = st.text_area(
        "Enter sample review text below (e.g., 'This product is amazing and works perfectly!', or 'I am very disappointed with the quality.'):", 
        "The dress is stunning, perfect for the occasion I bought it for! The fabric is also very nice.", 
        height=100, 
        key="user_text_sentiment_view_final_key" 
    )

    if st.button("✨ Analyze My Text Sentiment", key="analyze_button_sentiment_view_final_key"): # Unique key
        if not user_text_input.strip():
            st.warning("Please enter some text to analyze.")
        elif analyzer is None:
            st.error("Sentiment Analyzer (VADER) is not available. Cannot perform analysis.")
        else:
            scores = analyzer.polarity_scores(user_text_input)
            compound = scores['compound']
            
            if compound >= 0.05: 
                label = 'Positive 😊'
                color = "green"
                icon = "👍"
            elif compound <= -0.05: 
                label = 'Negative 😠'
                color = "red"
                icon = "👎"
            else: 
                label = 'Neutral 😐'
                color = "gray" 
                icon = "🤔"
            
            st.markdown(f"**Sentiment Result:** <span style='color:{color}; font-size: 1.3em; font-weight:bold;'>{icon} {label}</span> (Compound Score: **{compound:.3f}**)", unsafe_allow_html=True)
            
            with st.expander("Show Detailed Score Breakdown (VADER)"):
                st.json(scores)