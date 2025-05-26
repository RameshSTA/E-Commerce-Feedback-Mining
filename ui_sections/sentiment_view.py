import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Ensure VADER is imported

def render_sentiment_analysis(df_processed: pd.DataFrame | None, analyzer: SentimentIntensityAnalyzer | None):
    """
    Renders the Sentiment Analysis Insights page for the E-Commerce Feedback Mining dashboard.

    This page explains sentiment analysis, showcases the overall sentiment distribution
    from customer reviews using VADER, visualizes the relationship between VADER
    compound scores and explicit product ratings, and provides an interactive tool
    for users to test VADER sentiment on their own text.

    Args:
        df_processed (pd.DataFrame | None): The main DataFrame containing processed review
                                            data, including a 'vader_sentiment_label'
                                            column and a 'compound' VADER score column.
                                            Expected to be None if data loading failed.
        analyzer (SentimentIntensityAnalyzer | None): An initialized VADER sentiment
                                                      analyzer object. Expected to be
                                                      None if VADER failed to initialize.
    """
    st.header("🎭Understanding Customer Emotions")
    
    st.markdown("""
    This section delves into the emotional undercurrents of customer feedback. By systematically analysing
    the language used in reviews, we can move beyond simple star ratings to gain a more nuanced
    understanding of customer satisfaction and specific areas of concern or delight.
    """)

    with st.expander("🔍 What is Sentiment Analysis & How is it Used Here?", expanded=True):
        st.markdown("""
        **Sentiment Analysis**, also known as opinion mining, is a Natural Language Processing (NLP) technique used to 
        determine the emotional tone conveyed within a piece of text. It aims to classify the expressed opinion
        as **positive, negative, or neutral**.

        **Our Approach: VADER (Valence Aware Dictionary and sEntiment Reasoner)**
        * In this project, we employ **VADER**, a lexicon and rule-based sentiment analysis tool.
        * VADER is specifically attuned to sentiments expressed in social media and online content, making it well-suited for analysing customer reviews.
        * For each review, VADER calculates a **compound score** ranging from -1 (most negative) to +1 (most positive).
        * This compound score is then used to categorize each review into one of three distinct sentiment labels:
            * **Positive:** Compound score >= 0.05
            * **Negative:** Compound score <= -0.05
            * **Neutral:** Compound score between -0.05 and 0.05

        **Why is this important for Business?**
        Understanding aggregate and individual customer sentiment allows businesses to:
        * **Rapidly Gauge Overall Customer Satisfaction:** Get a quick pulse on how customers feel about products or services.
        * **Identify Problematic Areas or Praised Features:** Pinpoint specific aspects that are driving negative or positive experiences.
        * **Uncover Deeper Insights Beyond Star Ratings:** For example, a 3-star review might contain predominantly positive language about certain aspects, or a 5-star review might highlight a minor concern.
        * **Prioritize Improvement Efforts:** Focus resources on addressing issues that generate the most negative sentiment, thereby directly impacting customer experience and fostering loyalty.
        * **Track Sentiment Over Time:** (Future enhancement) Monitor changes in sentiment in response to product updates, marketing campaigns, or service improvements.
        """)
    st.markdown("---")

    if df_processed is None or df_processed.empty:
        st.warning("Processed data is not available. Cannot display sentiment analysis insights.")
        return
    
    # Check for essential columns for this page
    required_sentiment_cols = ['vader_sentiment_label', 'compound', 'Rating']
    missing_cols = [col for col in required_sentiment_cols if col not in df_processed.columns]
    if missing_cols:
        st.warning(f"The following essential columns for sentiment analysis are missing: {', '.join(missing_cols)}. Some visualisations may not be available.")
        # Allow partial rendering if some columns are present

    # Layout for key metrics and pie chart
    col1_sent_viz, col2_sent_viz = st.columns([0.8, 1.2], gap="large") # Adjusted ratio

    with col1_sent_viz:
        st.subheader("📊 Overall Sentiment Breakdown")
        st.markdown("Distribution of positive, negative, and neutral sentiments across all analysed customer reviews.")
        if 'vader_sentiment_label' in df_processed.columns:
            sentiment_counts = df_processed['vader_sentiment_label'].value_counts()
            sentiment_counts_df_display = sentiment_counts.reset_index()
            sentiment_counts_df_display.columns = ['Sentiment Label', 'Number of Reviews']
            
            # Display metrics for each sentiment category
            total_reviews = len(df_processed)
            st.markdown("**Sentiment Proportions:**")
            # Using columns for a cleaner metric layout
            metric_cols = st.columns(len(sentiment_counts_df_display))
            for i, (_, row) in enumerate(sentiment_counts_df_display.iterrows()):
                label = row['Sentiment Label']
                count = row['Number of Reviews']
                percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
                
                delta_color_metric = "normal"
                icon_metric = "😐" # Neutral default
                if label == "Positive": delta_color_metric = "normal"; icon_metric = "👍"
                elif label == "Negative": delta_color_metric = "inverse"; icon_metric = "👎"
                
                with metric_cols[i]:
                    st.metric(
                        label=f"{icon_metric} {label}", 
                        value=f"{count:,}", 
                        delta=f"{percentage:.1f}% of total", 
                        delta_color=delta_color_metric if label != "Neutral" else "off",
                        help=f"{count:,} out of {total_reviews:,} reviews ({percentage:.1f}%) were classified as '{label}'."
                    )
            
            st.markdown("<br>", unsafe_allow_html=True) # Spacer

            # Pie chart for visual distribution
            fig_pie_sent_display = px.pie(
                sentiment_counts_df_display, 
                names='Sentiment Label', 
                values='Number of Reviews',
                title='Visual Sentiment Distribution', 
                hole=0.45, # Doughnut chart effect
                color='Sentiment Label',
                color_discrete_map={'Positive':'#2ECC71', 'Neutral':'#BDC3C7', 'Negative':'#E74C3C', 'Error':'#F39C12'}, # Added Error color
                template="plotly_white"
            )
            fig_pie_sent_display.update_traces(
                textposition='outside', textinfo='percent+label', pull=[0.05, 0.02, 0.05], # Pull slices slightly
                marker=dict(line=dict(color='#000000', width=1)) # Add border to slices
            )
            fig_pie_sent_display.update_layout(
                legend_title_text='Sentiment Category', title_x=0.5, height=420, 
                margin=dict(t=70, b=30, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5) # Horizontal legend
            )
            st.plotly_chart(fig_pie_sent_display, use_container_width=True)
            st.caption("This doughnut chart visually represents the proportion of reviews falling into each sentiment category.")
        else:
            st.info("The 'vader_sentiment_label' column is not available in the dataset. Overall sentiment breakdown cannot be displayed.")

    with col2_sent_viz:
        if 'Rating' in df_processed.columns and 'compound' in df_processed.columns:
            st.subheader("🔬 Sentiment Score vs. Star Rating")
            st.markdown("""
            This visualisation explores the relationship between the **explicit star rating** provided by customers 
            and the **nuanced sentiment score** calculated by VADER (ranging from -1 for most negative, to +1 for most positive).
            **Value:** It helps identify discrepancies, such as highly-rated reviews with negative sentiment undertones
            or lower-rated reviews that still express some positive aspects. Such insights are valuable for understanding
            the true customer voice beyond simple ratings.
            """)
            # Prepare data, ensuring Rating is treated as categorical for distinct boxes
            df_for_boxplot = df_processed.copy()
            df_for_boxplot['Rating'] = df_for_boxplot['Rating'].astype('category')

            fig_box_sent_display = px.box(
                df_for_boxplot, 
                x='Rating', 
                y='compound', 
                color='Rating', # Color by rating for clarity
                title='VADER Compound Sentiment Score Distribution by Product Star Rating',
                labels={'compound': 'VADER Compound Score (-1 to +1)', 'Rating': 'Product Rating (Stars)'},
                color_discrete_sequence=px.colors.sequential.Tealgrn, # Changed color sequence
                points="outliers", # Show only outliers, or "all" if preferred
                notched=True, # Indicates confidence interval around median
                category_orders={"Rating": sorted(df_for_boxplot['Rating'].unique())} # Ensure ratings are sorted
            )
            fig_box_sent_display.update_layout(
                title_x=0.5, height=500, # Increased height slightly
                yaxis_title="VADER Compound Score", 
                xaxis_title="Product Rating (Stars)",
                xaxis=dict(type='category'), # Explicitly set x-axis type
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_box_sent_display, use_container_width=True)
            st.caption("""
            **How to Interpret:** Each box shows the distribution of VADER compound scores for a given star rating.
            The line in the box is the median sentiment score. The box spans the interquartile range (IQR).
            Notches represent the 95% confidence interval for the median. Observe if higher star ratings consistently
            correlate with higher (more positive) compound scores and tighter distributions.
            """)
        else:
            st.info("The 'Rating' or 'compound' (VADER score) columns are not available. Sentiment vs. Rating plot cannot be displayed.")

    st.markdown("---")
    st.subheader("✍️ Test VADER Sentiment on Your Own Text")
    st.markdown("""
    Curious about how VADER interprets sentiment? Enter any text below to get an instant sentiment analysis,
    including the overall classification (Positive, Negative, Neutral) and the underlying compound score.
    This is a direct application of the same VADER analyzer used on the review data.
    """)
    
    if analyzer is None:
        st.warning("Sentiment Analyzer (VADER) could not be initialized. Interactive sentiment testing is unavailable. Please check NLTK resource downloads (vader_lexicon).")
    else:
        user_text_input_val = st.text_area(
            "Enter sample text (e.g., a customer review snippet):", 
            "The design is beautiful and the color is exactly as pictured. However, it ran a bit small.", 
            height=120, 
            key="user_text_sentiment_view_final_v2", # Ensure unique key
            help="Type or paste text here to see its VADER sentiment score."
        )

        if st.button("✨ Analyse Text Sentiment", key="analyze_button_sentiment_view_final_v2", type="primary"):
            if not user_text_input_val.strip():
                st.warning("Please enter some text to analyse its sentiment.")
            else:
                with st.spinner("Analysing sentiment..."):
                    scores = analyzer.polarity_scores(user_text_input_val)
                    compound_score = scores['compound']
                    
                    sentiment_label_interactive = "Neutral 😐"
                    sentiment_color_interactive = "gray"
                    
                    if compound_score >= 0.05: 
                        sentiment_label_interactive = 'Positive 😊'
                        sentiment_color_interactive = SUCCESS_GREEN # Use defined color
                    elif compound_score <= -0.05: 
                        sentiment_label_interactive = 'Negative 😠'
                        sentiment_color_interactive = "#E74C3C" # Example red
                    
                    st.markdown(f"**Sentiment Result:** <span style='color:{sentiment_color_interactive}; font-size: 1.2em; font-weight:bold;'>{sentiment_label_interactive}</span>", unsafe_allow_html=True)
                    st.markdown(f"**VADER Compound Score:** `{compound_score:.4f}` (Ranges from -1 to +1)")
                    
                    with st.expander("Show Detailed VADER Score Breakdown"):
                        # Provide a more descriptive presentation of scores
                        st.markdown(f"""
                        - **Positive Score (`pos`):** `{scores['pos']:.3f}` (Intensity of positive sentiment)
                        - **Neutral Score (`neu`):** `{scores['neu']:.3f}` (Intensity of neutral sentiment)
                        - **Negative Score (`neg`):** `{scores['neg']:.3f}` (Intensity of negative sentiment)
                        """)
                        st.caption("These scores represent the proportion of text that falls into each sentiment category.")