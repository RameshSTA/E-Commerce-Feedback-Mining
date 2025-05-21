# ui_sections/recommendations_view.py
import streamlit as st

def render_recommendations(num_topics_config, topic_labels_config):
    st.header("💡 Strategic Business Recommendations")
    st.markdown("Leveraging customer voice analytics for data-driven improvements and growth. *(These recommendations are illustrative - **REPLACE with insights from YOUR specific project results and use YOUR topic labels from `topic_labels_config`**)*")
    st.markdown("---")
    

    st.subheader("Summary of Key Findings (Example - Customize with your data!):")

    st.markdown(f"""
    <div style="background-color: #FFF9C4; border-left: 6px solid #FFEB3B; padding: 15px; margin-bottom: 15px; border-radius: 5px;">
    <ul>
        <li><strong>Sentiment Hotspots:</strong> While overall sentiment is `[Your specific % - e.g., 72% positive]`, specific themes like `{topic_labels_config.get(3, "Problematic Topic A (e.g., Fabric Quality)")}` and `{topic_labels_config.get(1, "Problematic Topic B (e.g., Sizing)")}` consistently show significantly lower sentiment scores (average compound: `[Score e.g., -0.15]`), strongly correlating with lower product ratings.</li>
        <li><strong>Dominant Customer Conversations:</strong> The analysis of over `[Number e.g., 20,000]` reviews pinpointed {num_topics_config} primary discussion themes. `{topic_labels_config.get(2, "Most Prevalent Topic (e.g., Style)")}` and `{topic_labels_config.get(7, "Second Prevalent Topic (e.g., Overall Satisfaction)")}` emerged as the most frequently discussed.</li>
        <li><strong>Crucial Topic Interdependencies:</strong> The co-occurrence network highlighted that `{topic_labels_config.get(1, "Topic A (e.g., Sizing)")}` is frequently discussed alongside `{topic_labels_config.get(5, "Topic B (e.g., Returns)")}`. This suggests dissatisfaction with `[Theme A e.g., product fit]` often directly impacts `[Theme B e.g., value and likelihood to return]`.</li>
        <li>*(Add 1-2 more KEY, QUANTIFIABLE findings from YOUR specific analysis)*</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Actionable Recommendations for Business Growth:")
    
    recommendations_data_ui = {
        f"1. Enhance Sizing Accuracy & Fit Guidance (for `{topic_labels_config.get(1, 'Your Sizing Topic')}` 👚)": { 
            "problem": f"The `{topic_labels_config.get(1, 'Sizing Topic')}` is highly prevalent with `[e.g., mixed/often negative]` sentiment, often co-occurring with discussions on `[e.g., Returns or Disappointment]`.",
            "recommendation": "Invest in detailed, model-specific sizing charts; integrate user-submitted photos showcasing fit on diverse body types; add customer Q&A on fit. Explore AI-powered virtual try-on or fit recommendation tools to proactively reduce fit-related issues and subsequent returns."
        },
        f"2. Address Quality & Material Concerns (for `{topic_labels_config.get(3, 'Your Fabric/Quality Topic')}` 🧵)": { 
            "problem": f"Products frequently discussed under `{topic_labels_config.get(3, 'Fabric/Quality Topic')}` (e.g., items described as 'thin' or 'cheap material') often receive negative sentiment and lower star ratings.",
            "recommendation": "Conduct a targeted quality review for SKUs consistently linked to this topic. Re-evaluate material sourcing and manufacturing standards. Update product descriptions with transparent and accurate details about fabric composition, weight, and feel to set realistic customer expectations."
        },
        f"3. Amplify Positive Attributes & Marketing (for `{topic_labels_config.get(2, 'Your Positive Topic')}` ✨)":{ 
            "problem": f"Themes like `{topic_labels_config.get(2, 'Positive Topic')}` (e.g., 'Style, Appearance & Compliments') consistently receive high praise and positive sentiment.",
            "recommendation": "Leverage these strong points in marketing copy, social media campaigns, and product imagery. Highlight the specific attributes (e.g., 'vibrant colors,' 'flattering design,' 'unique style') that customers demonstrably love to attract new customers and reinforce brand perception."
        },
    
        f"4. Strategic Response to Co-occurring Issues (e.g., `{topic_labels_config.get(1, 'Sizing')}` & `{topic_labels_config.get(5, 'Returns')}` 🔗)":{
            "problem": f"The network analysis shows that discussions about `{topic_labels_config.get(1, 'Sizing')}` frequently lead to or are coupled with discussions about `{topic_labels_config.get(5, 'Returns')}`, often with negative sentiment.",
            "recommendation": "By improving the former (e.g., sizing accuracy), the business can likely reduce the negative impact and frequency of the latter (returns). This demonstrates how interconnected feedback can guide targeted interventions."
        }
    }

    for title, details in recommendations_data_ui.items():
        st.markdown(f"##### {title}") 
        with st.container():
            st.markdown(f"<div style='background-color: #FFF3E0; border-left: 5px solid #FFA726; padding: 10px; border-radius: 4px; margin-bottom: 5px;'><b>Problem Focus:</b> {details['problem']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #E8F5E9; border-left: 5px solid #66BB6A; padding: 10px; border-radius: 4px; margin-bottom: 15px;'><b>Strategic Recommendation:</b> {details['recommendation']}</div>", unsafe_allow_html=True)
        st.markdown("---")

    st.success("By strategically addressing these data-driven insights, the business can significantly enhance customer satisfaction, reduce negative feedback, foster loyalty, and make more informed decisions for sustainable growth.")