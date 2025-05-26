# E-Commerce Feedback Mining:Sentiment, Topics & Co-occurrence Networks

<p align="left">
  <a href="https://e-commerce-feedback-mining.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
  <a href="#15-license">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT">
  </a>
</p>

<p align="left">
  <a href="https://www.python.org" target="_blank"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="https://streamlit.io" target="_blank"><img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://pandas.pydata.org" target="_blank"><img src="https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org" target="_blank"><img src="https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org" target="_blank"><img src="https://img.shields.io/badge/Scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"></a>
  <a href="https://www.nltk.org" target="_blank"><img src="https://img.shields.io/badge/NLTK-%230C59A3.svg?style=for-the-badge&logo=nltk&logoColor=white" alt="NLTK"></a>
  <a href="https://spacy.io" target="_blank"><img src="https://img.shields.io/badge/spaCy-%2309A3D5.svg?style=for-the-badge&logo=spaCy&logoColor=white" alt="spaCy"></a>
  <a href="https://plotly.com/python/" target="_blank"><img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"></a>
  <a href="https://networkx.org" target="_blank"><img src="https://img.shields.io/badge/NetworkX-%230D5A9A.svg?style=for-the-badge&logo=networkx&logoColor=white" alt="NetworkX"></a>
  <a href="https://jupyter.org" target="_blank"><img src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter"></a>
</p>

This project leverages **Natural Language Processing (NLP)** and **Network Analysis** to automatically extract actionable insights from women's e-commerce clothing reviews. By performing sentiment analysis, identifying key discussion topics, and mapping their interconnections, the platform aims to provide data-driven recommendations for business improvement, product development, marketing strategy, and enhanced customer understanding.

---

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Live Demo](#2-live-demo)
3.  [Key Features & Capabilities](#3-key-features--capabilities)
4.  [Business Problems Addressed & Value Proposition](#4-business-problems-addressed--value-proposition)
5.  [Methodology & Analytical Workflow](#5-methodology--analytical-workflow)
6.  [Technology Stack](#6-technology-stack)
7.  [Dataset Utilized](#7-dataset-utilized)
8.  [Installation & Setup Guide](#8-installation--setup-guide)
9.  [Usage Instructions](#9-usage-instructions)
10. [Illustrative Key Findings & Visualizations](#10-illustrative-key-findings--visualizations)
11. [Actionable Business Recommendations](#11-actionable-business-recommendations)
12. [Project Limitations](#12-project-limitations)
13. [Future Work & Enhancements](#13-future-work--enhancements)
14. [Developer Information](#14-developer-information)
15. [License](#15-license)
16. [Acknowledgements](#16-acknowledgements)

---

## 1. Introduction
In the highly competitive e-commerce landscape, deeply understanding the "voice of the customer" is no longer a luxury but a strategic imperative for sustainable growth and success. This project delves into a dataset of over **23,000 customer reviews** from a women's clothing platform, moving beyond anecdotal feedback to uncover statistically significant patterns and insights.

Utilizing advanced Natural Language Processing (NLP) techniques—specifically **VADER for nuanced sentiment analysis**, **Latent Dirichlet Allocation (LDA) for unsupervised topic modeling** (applied to TF-IDF vectorized text features), and **NetworkX for visualizing topic co-occurrences**—this initiative systematically deciphers prevalent customer discussion themes, gauges the sentiment associated with them, and reveals intricate relationships between different aspects of the customer experience. The core objective is to transform vast quantities of unstructured text into **strategic, actionable intelligence**, providing a clear pathway for data-informed decision-making.

---

## 2. Live Demo
Experience the interactive insights and capabilities of this project through the deployed Streamlit application:

➡️ **Explore the Dashboard: [https://e-commerce-feedback-mining.streamlit.app/](https://e-commerce-feedback-mining.streamlit.app/)**

---

## 3. Key Features & Capabilities
* **Automated Sentiment Analysis:** Dynamically categorizes each customer review into **Positive, Negative, or Neutral** sentiment using NLTK's VADER, providing an immediate and scalable measure of overall and granular customer satisfaction.
* **Advanced Topic Modeling:** Employs Latent Dirichlet Allocation (LDA) to automatically identify **7 distinct underlying themes** from the review corpus (e.g., "👚 Sizing & Fit Issues," "💖 Style & Appearance," "🧵 Fabric Quality & Material Concerns"), revealing what customers are most frequently discussing.
* **Interactive Topic Exploration:** The Streamlit dashboard allows users to dynamically filter reviews by identified topics, view representative sample reviews, and analyze sentiment distributions *within* each specific theme for deeper contextual understanding.
* **Topic Co-occurrence Network Visualization:** Utilizes NetworkX and Plotly to generate an interactive network graph. This visually represents how different customer discussion topics are related and frequently co-occur within the same reviews, uncovering complex interdependencies.
* **Quantifiable Insights Dashboard:** Presents key metrics such as overall sentiment breakdown, prevalence percentages for each topic, sentiment distribution per topic, and centrality measures of themes within the discussion network.
* **Data-Driven Recommendations Engine:** Translates complex analytical findings from sentiment, topic, and network analyses into clear, actionable business recommendations aimed at product improvement, marketing optimization, and enhanced customer service strategies.

---

## 4. Business Problems Addressed & Value Proposition

This project directly addresses critical business challenges inherent in understanding and leveraging large volumes of customer feedback in the e-commerce domain:

* **Scaling Customer Feedback Analysis:** Manually processing thousands of reviews is inefficient and prone to bias. This platform automates the analysis.
    * **Value:** Enables businesses to **rapidly identify widespread issues or popular product attributes**, facilitating quicker responses and strategic adjustments.
* **Identifying Specific Customer Pain Points & Praises:** Moves beyond generic feedback to pinpoint exact aspects, such as "🧵 Fabric Quality & Material Concerns" or "💖 Style & Appearance," that significantly drive customer sentiment.
    * **Value:** Informs **targeted product development cycles, precise quality control interventions, and highly effective marketing messaging** that resonates with customer preferences.
* **Understanding Complex Feedback Interconnections:** Uncovers how different aspects of the customer journey and product experience are linked (e.g., "the co-occurrence of '👚 Sizing & Fit Issues' with '💰 Value & Returns' suggests that fit problems are a primary driver of returns and impact perceived value").
    * **Value:** Allows businesses to **address root causes rather than isolated symptoms**, leading to more impactful and holistic solutions that improve the overall customer experience.
* **Data-Driven Prioritization & Resource Allocation:** Provides quantitative evidence for which issues or positive attributes are most significant to the customer base.
    * **Value:** Helps businesses **allocate resources (time, budget, effort) more effectively** to areas that will yield the greatest improvement in customer satisfaction, loyalty, and ultimately, business outcomes like conversion rates and repeat purchases.

---

## 5. Methodology & Analytical Workflow
The project adheres to a systematic data science process to ensure robust and reliable insights:

1.  **Data Acquisition & Preparation:**
    * Loaded the "Women's E-Commerce Clothing Reviews" dataset using Pandas.
    * Performed initial data cleaning, handling of missing values, and data type conversions.
    * **Text Preprocessing (NLTK & spaCy):** A critical phase involving:
        * Lowercasing all review text for consistency.
        * Removal of punctuation, numerical digits, and special characters.
        * Tokenization (splitting text into individual words/tokens).
        * Stop-word removal (eliminating common words like "the," "is," "and" that add little semantic value for topic modeling).
        * Lemmatization using spaCy to reduce words to their dictionary base form (e.g., "running" to "run"), standardizing the vocabulary.
2.  **Sentiment Analysis:**
    * Applied NLTK's **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to each preprocessed review. VADER is chosen for its effectiveness with social media style text and online reviews.
    * Calculated a compound sentiment score (-1 to +1) and categorized reviews into **Positive, Negative, or Neutral** based on predefined thresholds (Positive: >= 0.05, Negative: <= -0.05, Neutral: between -0.05 and 0.05).
3.  **Feature Extraction for Topic Modeling:**
    * Utilized Scikit-learn's `TfidfVectorizer` to convert the lemmatized text corpus into a **TF-IDF (Term Frequency-Inverse Document Frequency) matrix**. This numerical representation emphasizes words that are important to a specific review relative to the entire collection of reviews.
4.  **Topic Modeling with LDA:**
    * Implemented **Latent Dirichlet Allocation (LDA)** using Scikit-learn on the TF-IDF matrix to discover **7 latent topics** within the reviews. LDA is a probabilistic model that assumes documents are a mixture of topics and topics are a mixture of words.
    * Manually interpreted and assigned meaningful business-relevant labels to each of the 7 topics (e.g., "👚 Sizing & Fit Issues," "🧵 Fabric Quality & Material Concerns") by examining their top constituent keywords.
5.  **Insight Generation & Visualization:**
    * Analyzed the distribution of sentiment (Positive, Negative, Neutral) within each discovered topic.
    * Correlated dominant topics with original product star ratings to understand alignment.
    * Developed interactive visualizations (bar charts, pie charts) using Plotly Express for presentation within the Streamlit application, making insights easily digestible.
6.  **Topic Co-occurrence Network Analysis:**
    * Identified topics that frequently appear together within the same customer reviews. This was achieved by assessing the probability distribution of topics for each review (output by LDA) and noting co-occurrences above a defined probability threshold.
    * Constructed a network graph using **NetworkX**, where nodes represent the identified topics and edges (with weights) represent the strength or frequency of their co-occurrence.
    * Visualized this interactive network using Plotly, allowing exploration of topic relationships and identification of central or highly connected themes.
7.  **Interactive Dashboard Development:**
    * Built a multi-page, user-friendly web application using **Streamlit** (`app.py` orchestrating modular UI sections) to present all analytical findings, facilitate interactive data exploration, and deliver actionable recommendations.

---

## 6. Technology Stack
* **Programming Language:** Python (3.9+)
* **Core Data Science Libraries:**
    * **Pandas:** Data manipulation, cleaning, and analysis.
    * **NumPy:** Numerical operations and array processing.
    * **Scikit-learn:** TF-IDF vectorization, Latent Dirichlet Allocation (LDA), machine learning utilities.
* **Natural Language Processing (NLP):**
    * **NLTK (Natural Language Toolkit):** VADER sentiment analysis, tokenization, stopword lists.
    * **spaCy:** Efficient and accurate lemmatization.
* **Network Analysis:**
    * **NetworkX:** Creation, manipulation, and study of complex networks (for topic co-occurrence).
* **Data Visualization:**
    * **Plotly & Plotly Express:** Interactive charts and graphs for the Streamlit application.
    * *(Matplotlib & Seaborn may have been used for initial exploratory plots in notebooks).*
* **Web Application Framework:**
    * **Streamlit:** Building and deploying the interactive data application.
    * **Streamlit Option Menu:** For custom navigation menus.
* **Development Environment & Utilities:**
    * Jupyter Notebooks: Initial analysis, experimentation, and model development.
    * VS Code (or preferred IDE).
    * Joblib: For saving and loading Python objects (e.g., trained models, vectorizers).
    * `ast`, `re`, `os`, `sys`: Standard Python libraries for utility functions.
* **Version Control:** Git & GitHub.

---

## 7. Dataset Utilized
* **Name:** Women's E-Commerce Clothing Reviews
* **Source:** Based on a publicly available dataset commonly found on platforms like Kaggle (e.g., "Women's E-Commerce Clothing Reviews" dataset). *For this project, a representative sample or a version of such a dataset was used.*
* **Details:** Comprises approximately **23,500 customer reviews**. Key features leveraged include 'Review Text' (primary for NLP), 'Rating' (1-5 stars), 'Recommended IND', 'Class Name', and 'Department Name'.
* **Focus:** The 'Review Text' column formed the core input for all sentiment analysis, topic modeling, and subsequent insights.

---

## 8. Installation & Setup Guide
To set up and run this project locally, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/RameshSTA/E-Commerce-Feedback-Mining.git](https://github.com/RameshSTA/E-Commerce-Feedback-Mining.git)
    cd E-Commerce-Feedback-Mining
    ```
2.  **Create and Activate a Python Virtual Environment** (Highly Recommended):
    ```bash
    python3 -m venv nlp_feedback_env
    source nlp_feedback_env/bin/activate  # On macOS/Linux
    # nlp_feedback_env\Scripts\activate    # On Windows PowerShell
    ```
3.  **Install Required Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` is up-to-date and present in your repository.)*
4.  **Download NLTK Resources:**
    Execute the following in a Python interpreter within your activated environment:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('stopwords') # For text preprocessing
    nltk.download('punkt')     # For tokenization by NLTK components
    nltk.download('wordnet')   # Often used by lemmatizers
    nltk.download('omw-1.4')   # Open Multilingual Wordnet, for WordNet
    ```

---

## 9. Usage Instructions
The project contains Jupyter Notebooks for the analytical pipeline and a Streamlit application for interactive presentation of results.

* **Jupyter Notebooks (located in the `notebooks/` directory):**
    These notebooks detail the data ingestion, preprocessing, EDA, sentiment analysis, topic modeling, and artifact generation steps. It's recommended to run them sequentially if you wish to regenerate the data artifacts used by the Streamlit app:
    1.  `01_Data_Acquisition_and_EDA.ipynb`: Initial data loading, cleaning, and basic EDA.
    2.  `02_NLP_Preprocessing.ipynb`: Detailed text preprocessing steps.
    3.  `03_Sentiment_Topic_Modeling.ipynb`: Sentiment analysis, TF-IDF, LDA model training, topic interpretation, and generation of `reviews_final_for_streamlit.csv` and model artifacts (`.joblib`, `.gexf`).
* **Streamlit Application (`app.py`):**
    1.  Ensure all required data and model artifacts (generated from Notebook 03, particularly `reviews_final_for_streamlit.csv`, `lda_model.joblib`, `tfidf_vectorizer.joblib`, `tfidf_feature_names.joblib`, and `topic_network.gexf`) are correctly placed in their respective `data/` and `artifacts/` folders within your project structure.
    2.  Ensure your project logo (e.g., `logo.png`) is in the `assets/` folder if you are using one.
    3.  Navigate to the project's root directory in your terminal.
    4.  Launch the Streamlit application by running:
        ```bash
        streamlit run app.py
        ```
    5.  The application will typically open automatically in your default web browser (usually at `http://localhost:8501`).

---

## 10. Illustrative Key Findings & Visualizations
*(This section provides examples of insights. **Please replace these with YOUR specific, quantifiable findings** from your analysis. For a live README, embed 2-3 key visualizations by saving them as images from your Streamlit app or notebooks, placing them in a folder like `assets/readme_images/`, and linking them using Markdown image syntax.)*

* **Overall Sentiment Profile:** Analysis of **~23,500 reviews** revealed a predominantly positive sentiment landscape, with approximately **72% Positive**, **18% Neutral**, and **10% Negative** reviews. This suggests a generally satisfied customer base, but with a significant minority expressing concerns.
    * ``

* **Dominant Customer Discussion Themes:** The LDA model successfully identified **7 distinct topics**. The most prevalent themes were:
    1.  **"💖 Style & Appearance"**: Accounting for an estimated **25%** of dominant topic assignments.
    2.  **"👚 Sizing & Fit Issues"**: Comprising around **20%** of assignments.
    3.  **"💯 Overall Satisfaction & General Positive Feedback"**: Representing approximately **18%** of assignments.
    * ``

* **Topic-Specific Sentiment & Star Rating Correlations:**
    * The theme **"💖 Style & Appearance"** was strongly associated with highly positive sentiment (average VADER compound score of **+0.78**) and predominantly **5-star ratings**.
    * Conversely, **"🧵 Fabric Quality & Material Concerns"** showed a high concentration of negative sentiment (average VADER compound score of **-0.45**) and was frequently linked to **1 and 2-star reviews**.

* **Key Topic Interconnections & Network Insights:**
    The topic co-occurrence network visually demonstrated strong relationships between themes. For instance, discussions concerning **"👚 Sizing & Fit Issues"** frequently co-occurred with **"💰 Value & Returns"** (e.g., appearing together in **over 600 reviews** where both topics were strongly present), strongly suggesting that fit problems are a major driver for returns and significantly impact customers' perception of value. The theme **"😌 Comfort & Wearability"** emerged as a highly central topic, connecting with various aspects of product satisfaction.
    * ``
    *(**Developer Note:** To embed images, create an `assets/readme_images/` folder in your GitHub repo, add your saved plot images there, and then use Markdown like `![Alt Text](assets/readme_images/your_image_name.png)`)*

---

## 11. Actionable Business Recommendations
*(Based on the illustrative findings above. **Tailor these to YOUR actual dominant topics and quantified insights.**)*

1.  **Prioritize Sizing Accuracy & Fit Guidance for "👚 Sizing & Fit Issues":**
    * Given its high prevalence (~20%) and strong linkage to "💰 Value & Returns," invest in enhanced, item-specific sizing charts, integrate user-submitted fit photos for diverse body types, and explore AI fit recommendation tools. This could potentially **reduce fit-related returns by an estimated 10-15%** and improve satisfaction for this key customer segment.
2.  **Address Material Quality Concerns in "🧵 Fabric Quality & Material Concerns":**
    * For product categories frequently associated with negative feedback on fabric (e.g., "thin material," "poor texture"), conduct immediate quality reviews and re-evaluate material sourcing. Improve product descriptions with transparent details about fabric composition and feel to manage expectations and **mitigate an estimated 5-8% of negative reviews** in this theme.
3.  **Amplify Positive Feedback on "💖 Style & Appearance" in Marketing & Merchandising:**
    * Capitalize on the high positive sentiment (>80% positive for this theme) by featuring associated keywords (e.g., "beautiful design," "lovely color," "unique style") and authentic customer testimonials in marketing campaigns, social media, and product highlights to attract new customers and reinforce desirable brand attributes.
4.  **Investigate Critical Topic Co-occurrences for Holistic Solutions:**
    * The strong link between "👚 Sizing & Fit Issues" and "💰 Value & Returns" suggests that proactively enhancing sizing information and fit prediction can also alleviate pressures on the returns process and improve overall value perception. Address interconnected issues for greater impact.

---

## 12. Project Limitations
* **Dataset Scope:** Analysis is based on a static, historical dataset; real-time customer opinions and emerging product issues can evolve.
* **Sentiment Analysis Nuances:** VADER, while effective for online text, may not perfectly capture highly nuanced linguistic expressions such as sarcasm, irony, or complex conditional statements.
* **Topic Model Configuration:** The determination of **7 topics** for LDA is based on a balance of coherence and interpretability; different numbers of topics or alternative modeling techniques might yield varied thematic structures.
* **Co-occurrence vs. Causation:** The topic co-occurrence network reveals associations between themes but does not inherently imply direct causation. Further qualitative and quantitative investigation would be needed to confirm causal relationships.
* **Generalizability:** Insights are specific to the analyzed dataset (women's e-commerce clothing); direct application to other product categories or demographics would require model retraining and validation.

---

## 13. Future Work & Enhancements
* Implement **Aspect-Based Sentiment Analysis (ABSA)** for more granular insights into sentiment towards specific product features (e.g., "the *collar* is too tight," "the *color* is vibrant") within broader topics.
* Develop **dynamic topic modeling (e.g., DTM, TTM)** to track how customer discussion themes and sentiments evolve over different time periods, seasons, or in response to product launches/marketing campaigns.
* Build **predictive models** leveraging derived NLP features and topic assignments (e.g., to predict products likely to receive low ratings, forecast return rates, or identify customers at risk of churn).
* Enhance the Streamlit dashboard with more advanced interactive filtering options (e.g., by product category, rating combined with sentiment), keyword search functionality within specific topics, and potentially user-specific dashboards if user data were integrated.
* Explore **causal inference techniques** to more rigorously investigate the drivers behind identified topic co-occurrences and their impact on business KPIs.
* Integrate **explainability methods (XAI)** for topic models (e.g., LIME, SHAP adapted for text) to better understand why certain documents are assigned to specific topics.

---

## 14. Developer Information
* **Name:** Ramesh Shrestha
* **Email:** `shrestha.ramesh000@gmail.com`
* **LinkedIn:** [https://linkedin.com/in/rameshsta](https://linkedin.com/in/rameshsta)
* **GitHub Project Repository:** [https://github.com/RameshSTA/E-Commerce-Feedback-Mining](https://github.com/RameshSTA/E-Commerce-Feedback-Mining)

---

## 15. License

Copyright (c) 2025 Ramesh Shrestha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
