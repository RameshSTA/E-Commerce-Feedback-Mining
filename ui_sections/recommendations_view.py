
import streamlit as st
import pandas as pd # Pandas import was missing, though not directly used in this refined version's rendering logic beyond type hints

# Define consistent colors (align with app.py or define locally if this module is standalone)
# These are examples; ideally, import from a central theme/config.
PRIMARY_BLUE = "#007bff"
SUCCESS_GREEN = "#28a745"
WARNING_ORANGE = "#FFA726" # Color for "Problem Focus"
RECOMMENDATION_GREEN = "#66BB6A" # Color for "Strategic Recommendation"
KEY_FINDINGS_BG = "#FFF9E6" # Light yellow for key findings box
KEY_FINDINGS_BORDER = "#FFD54F" # Amber for key findings border

def render_recommendations(num_topics_config: int, topic_labels_config: dict):
    """
    Renders the Strategic Business Recommendations page.

    This page synthesizes key findings from the sentiment and topic modeling
    analyses and translates them into actionable recommendations aimed at
    driving business improvements, enhancing customer satisfaction, and
    fostering strategic growth in the e-commerce clothing sector.

    The recommendations provided are illustrative and should be further validated
    and customized based on deeper dives into the project's specific results.

    Args:
        num_topics_config (int): The configured number of topics from the LDA model.
                                 Used to inform the context of the findings.
        topic_labels_config (dict): A dictionary mapping topic indices (1-based)
                                    to human-interpretable labels. Used to make
                                    findings and recommendations specific and relatable.
    """
    st.header("💡 Actionable Insights & Strategic Recommendations")
    st.markdown(f"""
    This crucial section bridges the gap between data analysis and tangible business strategy.
    Based on the insights gathered from sentiment analysis and the **{num_topics_config} customer discussion themes** identified
    through topic modelling, we propose the following strategic recommendations. These are designed to
    address specific customer pain points, leverage areas of positive feedback, and ultimately drive
    product improvement, marketing effectiveness, and customer loyalty.
    """)
    st.markdown("---")

    # --- Illustrative Key Findings (To be driven by actual analysis results) ---
    st.subheader("Summary of Illustrative Key Findings")
    st.markdown("The following findings are examples of what insights might drive the recommendations. *For a real application, these would be dynamically generated or directly reflect your specific analytical outcomes.*")

    # Using plausible, illustrative data for the placeholders
    # These should ideally be calculated from df_processed in a real scenario if passed here
    # or summarized from other pages.
    illustrative_positive_sentiment_percentage = 72.5
    illustrative_problem_topic_1_label = topic_labels_config.get(3, "Fabric & Material Concerns") # Example: Topic 3 is problematic
    illustrative_problem_topic_2_label = topic_labels_config.get(1, "Sizing & Fit Issues")    # Example: Topic 1 is problematic
    illustrative_problem_topic_compound_score = -0.28
    illustrative_total_reviews_analyzed = "23,450+" # Example string
    illustrative_most_prevalent_topic_label = topic_labels_config.get(2, "Style & Appearance") # Example: Topic 2
    illustrative_second_prevalent_topic_label = topic_labels_config.get(7, "Overall Satisfaction") # Example: Topic 7
    illustrative_interdependent_topic_A_label = topic_labels_config.get(1, "Sizing & Fit Issues") # Example: Topic 1
    illustrative_interdependent_topic_B_label = topic_labels_config.get(5, "Value & Returns")      # Example: Topic 5


    st.markdown(f"""
    <div style="background-color: {KEY_FINDINGS_BG}; border-left: 6px solid {KEY_FINDINGS_BORDER}; padding: 18px 22px; margin-bottom: 25px; border-radius: 5px; line-height: 1.7;">
    Based on a comprehensive analysis of **{illustrative_total_reviews_analyzed} customer reviews**, several key patterns emerged:
    <ul>
        <li><strong>Sentiment Landscape:</strong> While overall sentiment leans positive at approximately <strong>{illustrative_positive_sentiment_percentage:.1f}%</strong>, specific themes such as "<em>{illustrative_problem_topic_1_label}</em>" and "<em>{illustrative_problem_topic_2_label}</em>" consistently exhibit significantly lower sentiment scores (average VADER compound: <strong>{illustrative_problem_topic_compound_score:.2f}</strong>), and are strongly correlated with lower product ratings. This pinpoints areas needing immediate attention.</li>
        <li><strong>Dominant Customer Conversations:</strong> The {num_topics_config} identified themes reveal that "<em>{illustrative_most_prevalent_topic_label}</em>" and "<em>{illustrative_second_prevalent_topic_label}</em>" are the most frequently discussed topics, indicating high customer engagement and importance in these areas.</li>
        <li><strong>Critical Topic Interdependencies:</strong> Network analysis highlighted a significant co-occurrence between "<em>{illustrative_interdependent_topic_A_label}</em>" and "<em>{illustrative_interdependent_topic_B_label}</em>". This suggests that dissatisfaction with product fit often directly translates to concerns about value and an increased likelihood of returns.</li>
        <li><strong>Praise for Specific Attributes:</strong> Reviews associated with "<em>{topic_labels_config.get(2, "Style & Appearance")}</em>" frequently contain highly positive language related to design aesthetics and compliments received, indicating strong product appeal in this dimension.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🚀 Actionable Recommendations for Business Enhancement")
    st.markdown("The following strategic recommendations are derived from the key findings above, aiming to address identified issues and capitalize on strengths:")

    # Using your topic_labels_config to make recommendations more dynamic and relevant
    # These are illustrative; you should tailor them based on YOUR actual dominant topics and their sentiments.
    recommendations_data_ui = {
        f"1. Refine Product Fit & Sizing Guides (Targeting: `{topic_labels_config.get(1, 'Sizing & Fit Issues')}` 👚)": {
            "problem": f"The theme '`{topic_labels_config.get(1, 'Sizing & Fit Issues')}`' is a prevalent concern, often associated with negative sentiment (e.g., an average VADER compound score around {illustrative_problem_topic_compound_score:.2f}) and frequently co-occurs with discussions related to '`{topic_labels_config.get(5, 'Value & Returns')}`'. This indicates a direct impact on customer satisfaction and return rates.",
            "recommendation": """
            **Strategy:** Implement a multi-pronged approach to improve sizing accuracy and customer guidance.
            <ul>
                <li>Develop highly detailed, item-specific sizing charts with comprehensive body measurements.</li>
                <li>Integrate a "Customer Fit Feedback" feature on product pages, allowing users to share how items fit on different body types (e.g., "Runs small," "True to size for athletic builds").</li>
                <li>Explore AI-powered virtual try-on solutions or advanced fit predictor tools based on customer profiles to provide personalized sizing recommendations.</li>
                <li>Enhance product imagery to include models with diverse body shapes wearing various sizes.</li>
            </ul>
            **Expected Outcome:** Significant reduction in fit-related returns, improved customer satisfaction for '`{topic_labels_config.get(1, 'Sizing & Fit Issues')}`', and a positive ripple effect on perceptions of '`{topic_labels_config.get(5, 'Value & Returns')}`'.
            """.strip()
        },
        f"2. Elevate Material Quality & Transparency (Targeting: `{topic_labels_config.get(3, 'Fabric & Material Concerns')}` 🧵)": {
            "problem": f"Discussions around '`{topic_labels_config.get(3, 'Fabric & Material Concerns')}`' (e.g., comments on 'thin fabric,' 'poor stitching,' 'material not as expected') are strongly linked to negative sentiment and lower product ratings. This directly impacts perceived value and brand trust.",
            "recommendation": """
            **Strategy:** Conduct a targeted quality audit and enhance product information transparency.
            <ul>
                <li>Perform a quality control review of SKUs consistently flagged under this theme.</li>
                <li>Re-evaluate material sourcing partners and manufacturing standards for identified problem items.</li>
                <li>Update product descriptions with highly detailed, accurate, and transparent information about fabric composition, weight, texture, care instructions, and sourcing ethics if applicable. Use high-resolution imagery and videos showcasing material detail.</li>
            </ul>
            **Expected Outcome:** Improved customer perception of product quality and value, reduction in negative reviews related to materials, and increased brand credibility.
            """.strip()
        },
        f"3. Amplify Positive Attributes in Marketing (Leveraging: `{topic_labels_config.get(2, 'Style & Appearance')}` ✨)": {
            "problem": f"The theme '`{topic_labels_config.get(2, 'Style & Appearance')}`' consistently receives high praise, positive sentiment, and is often linked to overall customer satisfaction. This represents a key brand strength that may be underleveraged.",
            "recommendation": """
            **Strategy:** Capitalize on these positively perceived attributes in marketing and product presentation.
            <ul>
                <li>Incorporate authentic customer review snippets (verbatims) that highlight style and appearance into marketing copy, social media campaigns, and product pages.</li>
                <li>Ensure product photography and videography prominently feature the aesthetic qualities customers love (e.g., 'vibrant colors,' 'flattering designs,' 'unique details').</li>
                <li>Consider influencer collaborations focusing on the stylistic aspects praised by customers.</li>
                <li>Use insights from this theme to guide future design directions and new product development.</li>
            </ul>
            **Expected Outcome:** Increased brand appeal, higher engagement with marketing content, improved conversion rates for stylistically-driven purchases, and reinforcement of brand identity.
            """.strip()
        },
        # Add a 4th recommendation or modify existing ones based on your actual 7 topics
        f"4. Proactively Manage Expectations for `{topic_labels_config.get(6, 'Colour Accuracy & Print Detail')}` 🎨": {
             "problem": f"If '`{topic_labels_config.get(6, 'Colour Accuracy & Print Detail')}`' emerges with mixed or negative sentiment, it indicates a potential disconnect between online product representation and the actual item received by the customer.",
             "recommendation": """
             **Strategy:** Enhance visual accuracy and descriptive clarity for products where colour/print is a key feature.
             <ul>
                <li>Invest in high-fidelity, colour-calibrated product photography and ensure images are displayed accurately across devices.</li>
                <li>Provide detailed textual descriptions of colours and print patterns, perhaps including Pantone references or comparisons to well-known shades.</li>
                <li>Utilise customer-submitted photos (with permission) to showcase items in different lighting conditions.</li>
                <li>Offer clear disclaimers if colours may appear slightly different due to screen variations, but strive for maximum accuracy.</li>
             </ul>
             **Expected Outcome:** Reduced customer disappointment related to colour/print discrepancies, fewer returns for these reasons, and improved trust in product representation.
             """.strip()
        }
    }

    # Ensure we only try to display a number of recommendations that makes sense
    # For example, if you have 7 topics, you might have 3-5 key recommendations
    # This loop will display all defined in the dictionary above.

    for i, (title, details) in enumerate(recommendations_data_ui.items()):
        # Use globally consistent block styling if available, or maintain local for now
        st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True) # Using h5 for recommendation titles
        with st.container(): # Use st.container for better grouping if needed
            # Using slightly different background colors for problem and recommendation for clarity
            st.markdown(f"""
            <div style="background-color: #FFF3E0; border-left: 5px solid {WARNING_ORANGE}; padding: 12px 15px; border-radius: 4px; margin-bottom: 8px; font-size: 0.95em;">
                <strong>Problem Focus:</strong> {details['problem']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #E8F5E9; border-left: 5px solid {RECOMMENDATION_GREEN}; padding: 12px 15px; border-radius: 4px; margin-bottom: 20px; font-size: 0.95em;">
                <strong>Strategic Recommendation:</strong> {details['recommendation']}
            </div>
            """, unsafe_allow_html=True)
        if i < len(recommendations_data_ui) - 1: # Add separator if not the last item
            st.markdown("---")


    st.success(
        "**Conclusion:** By strategically addressing these data-driven insights derived from the customer's own voice, "
        "the business can significantly enhance product offerings, refine marketing messages, improve customer satisfaction, "
        "reduce negative feedback, foster loyalty, and make more informed decisions for sustainable growth and a stronger market position."
    )