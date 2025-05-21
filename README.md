# E-Commerce Feedback Mining: Sentiment, Topics & Co-occurrence Networks

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen.svg)]([YOUR_STREAMLIT_APP_URL_HERE - REPLACE]) [![License: MIT](https://img.shields.io/badge/License-Inquire-lightgrey.svg)](LICENSE.md) An advanced NLP and Network Analysis project to automatically extract actionable insights from women's e-commerce clothing reviews, enabling data-driven business decisions.

---

## Table of Contents
1.  [Introduction](#introduction)
2.  [Live Demo](#live-demo)
3.  [Key Features](#key-features)
4.  [Business Problems Addressed & Value Proposition](#business-problems-addressed--value-proposition)
5.  [Methodology & Workflow](#methodology--workflow)
6.  [Technologies Used](#technologies-used)
7.  [Dataset](#dataset)
8.  [Installation & Setup](#installation--setup)
9.  [Usage](#usage)
10. [Key Findings & Visualizations (Quantifiable)](#key-findings--visualizations-quantifiable)
11. [Actionable Business Recommendations](#actionable-business-recommendations)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Developer](#developer)
15. [License & Permissions](#license--permissions)
16. [Acknowledgements](#acknowledgements)

---

## 1. Introduction
In today's competitive e-commerce landscape, understanding the voice of the customer is paramount. This project dives deep into over `[e.g., 23,000]` customer reviews from a women's clothing e-commerce platform. By applying sophisticated Natural Language Processing (NLP) techniques, including sentiment analysis, Latent Dirichlet Allocation (LDA) for topic modeling, and network analysis for topic co-occurrences, this project uncovers hidden patterns, key discussion themes, and customer sentiments. The ultimate goal is to transform unstructured feedback into structured, actionable intelligence that can drive product improvement, enhance customer experience, and inform strategic business decisions.

---

## 2. Live Demo
Explore the interactive findings of this project through the deployed Streamlit application:

➡️ **[Link to Your Deployed Streamlit App - REPLACE WITH YOUR URL e.g., on Streamlit Community Cloud]**

*(Once your Streamlit app is deployed, put the link here. If it's not deployed yet, you can say "Streamlit application for interactive exploration is available in `app.py`.")*

---

## 3. Key Features
* **Automated Sentiment Analysis:** Classifies reviews into positive, negative, or neutral categories using VADER.
* **Advanced Topic Modeling:** Identifies `[Your NUM_TOPICS]` distinct underlying themes discussed by customers using LDA.
* **Interactive Topic Exploration:** Allows users to explore sample reviews and sentiment distribution for each discovered topic.
* **Topic Co-occurrence Network:** Visualizes the relationships and common co-occurrences between different topics.
* **Quantifiable Insights:** Presents key metrics, sentiment breakdowns per topic, and prevalent themes.
* **Actionable Recommendations:** Translates analytical findings into strategic business recommendations.
* **Interactive UI:** A user-friendly Streamlit dashboard for dynamic exploration of results.

---

## 4. Business Problems Addressed & Value Proposition

This project directly addresses several key business challenges:

* **Understanding Customer Satisfaction at Scale:** Manually processing thousands of reviews is infeasible. This project provides an automated way to gauge overall sentiment and identify drivers of satisfaction and dissatisfaction.
    * **Value:** Enables quick identification of problem areas and aspects customers love, allowing for targeted improvements and marketing.
* **Identifying Key Product/Service Attributes:** Pinpoints what specific features, qualities, or issues (e.g., sizing, fabric, style, returns) are most salient to customers.
    * **Value:** Informs product development, quality assurance, and inventory management.
* **Uncovering Complex Customer Feedback Patterns:** Moves beyond simple keyword searches to understand how different issues or praises are interconnected (e.g., how discussions about "fit" might relate to "returns" or "fabric type").
    * **Value:** Provides a holistic view of the customer experience, allowing for more effective, multi-faceted solutions.
* **Data-Driven Strategic Decision Making:** Provides quantifiable evidence to support business strategies related to product, marketing, and customer service.
    * **Value:** Reduces reliance on anecdotal evidence, leading to more impactful and efficient business operations.

---

## 5. Methodology & Workflow
The project follows a structured data science workflow:

1.  **Data Acquisition & Preparation:**
    * Loaded the "Women's E-Commerce Clothing Reviews" dataset.
    * Performed extensive text preprocessing: lowercasing, punctuation/number removal, stop-word removal (NLTK), and lemmatization (spaCy).
2.  **Sentiment Analysis:**
    * Applied VADER (Valence Aware Dictionary and sEntiment Reasoner) to calculate polarity scores (positive, negative, neutral, compound) for each review.
3.  **Feature Extraction:**
    * Converted processed text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
4.  **Topic Modeling:**
    * Implemented Latent Dirichlet Allocation (LDA) on the TF-IDF matrix to discover `[Your NUM_TOPICS]` latent topics.
    * Manually interpreted and labeled each topic based on its top constituent keywords.
5.  **Insight Generation & Visualization:**
    * Analyzed sentiment distribution per topic and correlation with product ratings.
    * Visualized findings using Plotly for interactive charts (bar charts, pie charts, box plots).
6.  **Topic Co-occurrence Network Analysis:**
    * Identified topics frequently discussed together within the same reviews (based on a probability threshold from LDA output).
    * Constructed and visualized an interactive network graph using NetworkX and Plotly to map these inter-topic relationships.
7.  **Streamlit Application Development:**
    * Built an interactive dashboard to present all findings and allow user exploration.

---

## 6. Technologies Used
* **Programming Language:** Python 3.x
* **Data Manipulation & Analysis:** Pandas, NumPy
* **NLP:** NLTK (VADER, tokenization, stopwords), spaCy (lemmatization)
* **Machine Learning:** Scikit-learn (TfidfVectorizer, LatentDirichletAllocation)
* **Network Analysis:** NetworkX
* **Data Visualization:** Plotly, Matplotlib, Seaborn (primarily Plotly in the Streamlit app)
* **Web Application Framework:** Streamlit
* **Development Environment:** Jupyter Notebooks, VS Code (or your IDE)
* **Version Control:** Git & GitHub

---

## 7. Dataset
The project utilizes the **"Women's E-Commerce Clothing Reviews"** dataset, which is publicly available (often found on platforms like Kaggle).
* **Source:** [Link to the dataset source if available, e.g., Kaggle link - REPLACE]
* **Size:** Approximately 23,000 reviews.
* **Key Columns Used:** 'Review Text', 'Rating', 'Recommended IND', 'Class Name', 'Department Name'.
* **Preprocessing:** The 'Review Text' was the primary focus for NLP tasks.

---

## 8. Installation & Setup
To set up this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git # REPLACE
    cd [YOUR_REPO_NAME]
    ```
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have a comprehensive `requirements.txt` file in your repository. You can generate it using `pip freeze > requirements.txt` from your activated virtual environment.)*

4.  **Download necessary NLTK resources:**
    Run the following Python code once (e.g., in an interactive Python session or a setup script):
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('stopwords') # If used explicitly beyond spaCy/sklearn
    nltk.download('punkt')     # For tokenization if used explicitly
    # nltk.download('wordnet') # If you used NLTK lemmatizer
    # nltk.download('omw-1.4') # For WordNet
    ```
    *(The Streamlit app also attempts to download 'vader_lexicon' if not found, but it's good practice to pre-download.)*

---

## 9. Usage
The project consists of Jupyter Notebooks for analysis and a Streamlit application for interactive visualization.

* **Jupyter Notebooks:**
    Located in the `notebooks/` directory. Run them sequentially for data processing, model training, and artifact generation:
    1.  `01_Data_Acquisition_and_EDA.ipynb`
    2.  `02_NLP_Preprocessing.ipynb`
    3.  `03_Sentiment_Topic_Modeling.ipynb` (This notebook also saves the final data and models used by the Streamlit app).
* **Streamlit Application:**
    1.  Ensure all artifacts (`.csv`, `.joblib`, `.gexf` files) are saved in the `data/` and `artifacts/` folders as generated by Notebook 03.
    2.  Ensure your `logo.png` is in the `assets/` folder.
    3.  Navigate to the project's root directory in your terminal.
    4.  Run the command:
        ```bash
        streamlit run app.py
        ```
    5.  The application will open in your default web browser.

---

## 10. Key Findings & Visualizations (Quantifiable)
*(This section should be a highlight reel from your "Project Summary & Conclusions" in the Streamlit app. **REPLACE the placeholders with YOUR specific, quantifiable findings and embed 2-3 key visualizations.**)*

* **Overall Sentiment:** `[e.g., 72.5%]` of reviews were classified as Positive, `[e.g., 15.3%]` Negative, and `[e.g., 12.2%]` Neutral, indicating a generally favorable customer base but with specific areas of concern.
    * *(Consider embedding your Plotly pie chart of overall sentiment distribution here as an image)*
    `![Overall Sentiment Distribution](path/to/your/sentiment_pie_chart.png)`

* **Dominant Topics Identified:** The LDA model successfully identified `[Your NUM_TOPICS]` distinct topics. The most prevalent topics were:
    1.  **`[Your Topic Label 1]`** (e.g., "👚 Sizing & Fit"): Accounting for `[e.g., 28%]` of dominant topic assignments.
    2.  **`[Your Topic Label 2]`** (e.g., "💖 Style & Appearance"): Making up `[e.g., 22%]` of assignments.
    3.  **`[Your Topic Label 3]`** (e.g., "🧵 Fabric & Material"): Representing `[e.g., 18%]` of assignments.
    * *(Consider embedding your Plotly bar chart of topic distribution here as an image)*
    `![Topic Distribution](path/to/your/topic_distribution_chart.png)`

* **Topic Sentiment Correlation:**
    * Topic `[Your Most Positive Topic Label]` (e.g., "💯 Overall Satisfaction") exhibited the highest average positive sentiment (VADER compound score of `[e.g., +0.85]`).
    * Topic `[Your Most Negative Topic Label]` (e.g., "🧵 Fabric & Material Concerns") was strongly associated with negative sentiment (VADER compound score of `[e.g., -0.45]`) and lower product ratings.

* **Key Topic Co-occurrences:** The network analysis revealed significant relationships. For example, `[Your Topic Label A]` (e.g., "Sizing & Fit") frequently co-occurred with `[Your Topic Label B]` (e.g., "Value & Returns") `[N]` times, suggesting a direct link between fit issues and return considerations.
    * *(Consider embedding a simplified view or screenshot of your Plotly network graph here)*
    `![Topic Network Snippet](path/to/your/network_graph_snippet.png)`

---

## 11. Actionable Business Recommendations
*(Summarize 2-3 key recommendations from your Streamlit app's conclusion. **REPLACE with YOUR specific recommendations.**)*

1.  **Enhance Sizing & Fit Guidance for `[Your Sizing Topic Label]`:** Given its prevalence and link to `[e.g., returns/negative sentiment]`, invest in detailed sizing charts, user-generated fit photos, and AI fit predictors.
2.  **Address Quality & Material Concerns for `[Your Fabric/Quality Topic Label]`:** For items linked to this negative theme, review material sourcing and update product descriptions to set accurate expectations.
3.  **Leverage Positive Attributes from `[Your Positive Topic Label]` in Marketing:** Highlight aspects customers love in campaigns to attract and convert.

---

## 12. Limitations
* Analysis based on a static dataset; trends may evolve.
* Sentiment analysis (VADER) might not capture all linguistic nuances like sarcasm perfectly.
* The chosen number of topics (`[Your NUM_TOPICS]`) is based on interpretability; further quantitative evaluation could refine this.
* Topic co-occurrence shows association, not necessarily direct causation.

---

## 13. Future Work
* Implement Aspect-Based Sentiment Analysis (ABSA) for more granular insights.
* Develop dynamic topic modeling to track theme evolution over time.
* Build predictive models (e.g., for product returns) using NLP features.
* Enhance the Streamlit dashboard with more advanced filtering and comparative analysis features.

---

## 14. Developer
* **Name:** Ramesh Shrestha
* **Email:** `shrestha.ramesh000@gmail.com`
* **LinkedIn:** `https://linkedin.com/in/rameshsta`
* **GitHub:** `https://github.com/RameshSTA` 

---

## 15. License & Permissions

Copyright (c) 2024 Ramesh Shrestha. All Rights Reserved.

The code and work presented in this repository are for demonstration and portfolio purposes. If you wish to use, modify, or distribute any part of this project, please contact the developer, Ramesh Shrestha, at `shrestha.ramesh000@gmail.com` to request permission.

*(Alternatively, if you want to make it open source, you could choose a license like MIT and add a `LICENSE.md` file:)*
---

## 16. Acknowledgements (Optional)
* Mention the source of the dataset if appropriate (e.g., "Dataset sourced from Kaggle: [Dataset Name and Link]").
* Any specific libraries or tools you found particularly helpful beyond the main ones.
* Anyone who provided significant guidance (if applicable).
