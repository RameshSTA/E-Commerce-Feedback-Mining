# E-Commerce Feedback Mining: Sentiment, Topics & Co-occurrence Networks

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

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-brightgreen.svg?style=for-the-badge)]([YOUR_STREAMLIT_APP_URL_HERE - REPLACE])
[![License: Inquire](https://img.shields.io/badge/License-Inquire-lightgrey.svg?style=for-the-badge)](#15-license--permissions) 

This project leverages Natural Language Processing (NLP) and Network Analysis to automatically extract actionable insights from women's e-commerce clothing reviews. By analyzing sentiment, identifying key discussion topics, and mapping their interconnections, the goal is to provide data-driven recommendations for business improvement and enhanced customer understanding.

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
In the dynamic e-commerce sector, deeply understanding customer feedback is crucial for success. This project analyzes over `[e.g., 23,000 - REPLACE WITH YOUR ACTUAL COUNT]` customer reviews from a women's clothing platform to move beyond anecdotal evidence. Using advanced NLP techniques—VADER for sentiment analysis, Latent Dirichlet Allocation (LDA) for topic modeling from TF-IDF features, and NetworkX for visualizing topic co-occurrences—this initiative uncovers prevalent themes, gauges customer sentiment towards them, and reveals how different aspects of the customer experience are interconnected. The core aim is to convert unstructured text into strategic, actionable intelligence.

---

## 2. Live Demo
Explore the interactive findings of this project through the deployed Streamlit application:

➡️ **[Link to Your Deployed Streamlit App - REPLACE WITH YOUR URL, e.g., on Streamlit Community Cloud]**

*(If not yet deployed, state: "The interactive Streamlit application can be run locally by following the 'Usage' instructions below.")*

---

## 3. Key Features
* **Automated Sentiment Analysis:** Categorizes reviews into positive, negative, or neutral using VADER, providing an immediate sense of customer satisfaction.
* **Advanced Topic Modeling:** Identifies `[Your NUM_TOPICS - REPLACE]` distinct underlying themes (e.g., "[Your Topic Label 1]", "[Your Topic Label 2]") using LDA, revealing what customers discuss most.
* **Interactive Topic Exploration:** Allows dynamic filtering and viewing of sample reviews and sentiment distributions for each identified topic within the Streamlit app.
* **Topic Co-occurrence Network Visualization:** Employs NetworkX and Plotly to create an interactive graph showing how different topics are related and discussed together.
* **Quantifiable Insights:** Presents key metrics such as sentiment breakdown per topic, prevalence of topics, and centrality of themes in the discussion network.
* **Data-Driven Recommendations:** Translates complex analytical findings into clear, actionable business recommendations.

---

## 4. Business Problems Addressed & Value Proposition

This project directly addresses critical business challenges in understanding customer feedback:

* **Scaling Feedback Analysis:** Overcomes the inefficiency of manually processing thousands of reviews, enabling rapid insight generation.
    * **Value:** Allows businesses to quickly identify widespread issues or popular features, leading to faster response times.
* **Identifying Specific Customer Pain Points & Praises:** Pinpoints exact aspects like `[Your specific problematic topic, e.g., "Fabric Quality"]` or `[Your specific positive topic, e.g., "Style & Appearance"]` that drive customer sentiment.
    * **Value:** Informs targeted product development, quality control adjustments, and effective marketing messaging.
* **Understanding Complex Feedback Interconnections:** Reveals how different issues are linked (e.g., "the co-occurrence of `[Topic A Label]` with `[Topic B Label]` suggests that `[Your Interpretation of this link]`").
    * **Value:** Enables businesses to address root causes rather than isolated symptoms, leading to more impactful solutions.
* **Data-Driven Prioritization:** Provides quantitative backing for which issues or positive aspects are most significant to customers.
    * **Value:** Helps allocate resources effectively to areas that will most improve customer satisfaction and business outcomes.

---

## 5. Methodology & Workflow
The project followed a systematic data science process:

1.  **Data Acquisition & Preparation:**
    * Loaded the dataset using Pandas.
    * Performed initial cleaning and handling of missing values.
    * **Text Preprocessing (NLTK, spaCy):**
        * Lowercasing, removal of punctuation, numbers, and special characters.
        * Tokenization.
        * Stop-word removal.
        * Lemmatization to reduce words to their base forms.
2.  **Sentiment Analysis:**
    * Applied NLTK's VADER to calculate compound sentiment scores and categorize reviews as Positive, Negative, or Neutral.
3.  **Feature Extraction:**
    * Utilized Scikit-learn's `TfidfVectorizer` to convert the processed text corpus into a TF-IDF matrix, highlighting important words.
4.  **Topic Modeling:**
    * Implemented Latent Dirichlet Allocation (LDA) with Scikit-learn on the TF-IDF matrix to discover `[Your NUM_TOPICS - REPLACE]` latent topics.
    * Manually interpreted and assigned meaningful labels to each topic (e.g., "`[Your Topic Label 1 - REPLACE]`", "`[Your Topic Label 2 - REPLACE]`") based on their top constituent keywords.
5.  **Insight Generation & Visualization:**
    * Analyzed sentiment distribution per discovered topic.
    * Correlated topics with original product ratings.
    * Developed interactive visualizations using Plotly for presentation in the Streamlit app.
6.  **Topic Co-occurrence Network Analysis:**
    * Identified topics that frequently appear together within the same reviews (based on LDA topic probabilities exceeding a defined threshold).
    * Constructed a network graph using NetworkX, where nodes are topics and edges represent co-occurrence strength.
    * Visualized the interactive network using Plotly.
7.  **Interactive Dashboard Development:**
    * Built a user-friendly web application using Streamlit (`app.py` and modular UI sections) to present findings and allow interactive exploration.

---

## 6. Technologies Used
* **Programming Language:** Python (3.9+)
* **Core Data Science Libraries:**
    * **Pandas:** For data manipulation and analysis.
    * **NumPy:** For numerical operations.
    * **Scikit-learn:** For TF-IDF vectorization and LDA topic modeling.
* **Natural Language Processing (NLP):**
    * **NLTK:** For VADER sentiment analysis, tokenization, stopwords.
    * **spaCy:** For efficient lemmatization.
* **Network Analysis:**
    * **NetworkX:** For creating, manipulating, and analyzing network graphs.
* **Data Visualization:**
    * **Plotly & Plotly Express:** For interactive charts and graphs in the Streamlit app.
    * **Matplotlib & Seaborn:** For initial exploration/static plots (if any shown in notebooks).
    * **WordCloud:** For generating word cloud visualizations (if used in notebooks).
* **Web Application Framework:**
    * **Streamlit:** For building and deploying the interactive data application.
* **Development Environment:**
    * Jupyter Notebooks: For initial analysis, experimentation, and model development.
    * VS Code (or your preferred IDE).
* **Version Control:**
    * Git & GitHub.

---

## 7. Dataset
* **Name:** Women's E-Commerce Clothing Reviews
* **Source:** Publicly available [e.g., "from Kaggle" - Add specific Kaggle dataset name and link if you have it - REPLACE]
* **Details:** Contains approximately `[Actual number from your df_processed, e.g., 23,486 - REPLACE]` customer reviews. Key features utilized include 'Review Text', 'Rating', 'Recommended IND', 'Class Name', and 'Department Name'.
* **Focus:** The 'Review Text' column was the primary source for NLP analysis.

---

## 8. Installation & Setup
To set up this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/RameshSTA/E-Commerce-Feedback-Mining.git](https://github.com/RameshSTA/E-Commerce-Feedback-Mining.git) # REPLACE with your repo URL if different
    cd E-Commerce-Feedback-Mining # REPLACE with your repo name if different
    ```
2.  **Create a Python Virtual Environment** (recommended):
    ```bash
    python3 -m venv venv_nlp_project
    source venv_nlp_project/bin/activate  # macOS/Linux
    # venv_nlp_project\Scripts\activate    # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` is complete and in your repository. Generate with `pip freeze > requirements.txt`)*
4.  **Download NLTK Resources:**
    Execute the following in a Python interpreter within your activated environment:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('stopwords') # If used explicitly
    nltk.download('punkt')     # For tokenization
    # nltk.download('wordnet') # Only if you explicitly used NLTK's lemmatizer
    # nltk.download('omw-1.4') # For WordNet
    ```

---

## 9. Usage
The project includes Jupyter Notebooks for the analytical pipeline and a Streamlit application for interactive presentation.

* **Jupyter Notebooks (`notebooks/` directory):**
    These detail the data processing, model training, and artifact generation steps. They should be run sequentially if regenerating artifacts:
    1.  `01_Data_Acquisition_and_EDA.ipynb`
    2.  `02_NLP_Preprocessing.ipynb`
    3.  `03_Sentiment_Topic_Modeling.ipynb` (This notebook saves the final data and models for the Streamlit app)
* **Streamlit Application (`app.py`):**
    1.  Ensure all required artifacts (generated from Notebook 03) are in the `data/` and `artifacts/` folders.
    2.  Ensure your `logo.png` is in the `assets/` folder.
    3.  Navigate to the project's root directory in your terminal.
    4.  Run:
        ```bash
        streamlit run app.py
        ```
    5.  The application will open in your default web browser.

---

## 10. Key Findings & Visualizations (Quantifiable)
*(This section is CRITICAL. **REPLACE all bracketed placeholders with YOUR specific, quantifiable findings.** Embed 2-3 key visualizations from your Streamlit app or notebooks by saving them as images (e.g., in `assets/readme_images/`) and linking them.)*

* **Overall Sentiment:** The sentiment analysis of `[Total Number of Reviews Analyzed - REPLACE]` reviews revealed that **`[X.X]%` were Positive**, `[Y.Y]%` Negative, and `[Z.Z]%` Neutral. This indicates `[Your brief interpretation, e.g., a generally positive customer base but with a notable segment expressing dissatisfaction for specific reasons - REPLACE]`.
    * `![Overall Sentiment Distribution](assets/readme_images/sentiment_distribution_plot.png)` *(Save your Plotly pie chart as an image and link it here. Create the folder if needed.)*

* **Dominant Customer Themes:** `[Your NUM_TOPICS - REPLACE]` distinct topics were identified through LDA modeling. The most prevalent themes discussed by customers were:
    1.  **`[Your Topic Label 1 - REPLACE]`**: Accounting for `[e.g., XX% - REPLACE]` of dominant topic assignments.
    2.  **`[Your Topic Label 2 - REPLACE]`**: Making up `[e.g., YY% - REPLACE]` of assignments.
    3.  **`[Your Topic Label 3 - REPLACE]`**: Representing `[e.g., ZZ% - REPLACE]` of assignments.
    * `![Topic Distribution Chart](assets/readme_images/topic_distribution_plot.png)` *(Save your Plotly bar chart of topic distribution)*

* **Topic-Specific Sentiment & Ratings:**
    * The topic **`[Your Most Positive Topic Label - REPLACE]`** was strongly associated with positive sentiment (average VADER compound score of `[e.g., +0.82 - REPLACE]`) and predominantly `[e.g., 5-star - REPLACE]` ratings.
    * Conversely, **`[Your Most Negative Topic Label - REPLACE]`** (e.g., related to "Fabric Quality Concerns") showed a high concentration of negative sentiment (average compound score `[e.g., -0.55 - REPLACE]`) and was frequently linked to `[e.g., 1 and 2-star - REPLACE]` reviews.

* **Key Topic Interconnections:** The topic co-occurrence network highlighted significant relationships. For example, discussions about **`[Your Topic Label A - REPLACE]`** frequently appeared alongside **`[Your Topic Label B - REPLACE]`** `[N - REPLACE]` times, suggesting a strong correlation where `[brief interpretation, e.g., fit issues are a major driver for returns and affect value perception - REPLACE]`. The topic **`[Your Most Central Topic Label - REPLACE]`** was identified as a key hub, connecting diverse aspects of customer feedback.
    * `![Topic Network Visualization](assets/readme_images/topic_network_plot.png)` *(Save your Plotly network graph as an image)*

---

## 11. Actionable Business Recommendations
*(Summarize 2-4 key, impactful recommendations derived from YOUR findings. Make them specific and link them back to your topic labels and quantitative findings.)*

1.  **Prioritize Sizing & Fit Accuracy for `[Your Sizing Topic Label - REPLACE]`:** Given its high prevalence (`[XX% - REPLACE]`) and link to `[e.g., returns/negative sentiment - be specific from your data]`, invest heavily in detailed sizing charts, user-generated fit photos, and AI fit predictors to potentially reduce returns by an estimated `[Target % - REPLACE]` and improve satisfaction for this key customer concern.
2.  **Address Material Quality Concerns in `[Your Fabric/Quality Topic Label - REPLACE]`:** For product categories frequently associated with negative feedback on `[Specific keywords like 'thin fabric', 'cheap material' from your topic - REPLACE]`, conduct urgent quality reviews. Set clear expectations in product descriptions regarding fabric type and durability to mitigate `[e.g., YY% - REPLACE]` of negative reviews in this theme.
3.  **Amplify Positive Feedback on `[Your Positive Topic Label - REPLACE]` in Marketing:** Capitalize on the high positive sentiment (`[e.g., >80% positive - REPLACE]`) for `[Your Positive Topic Label - REPLACE]` by featuring associated keywords (`[e.g., 'beautiful design', 'lovely color' - from your topic keywords - REPLACE]`) in marketing campaigns and product highlights.
4.  **Investigate Critical Co-occurrences for Proactive Solutions:** The strong link between `[Topic X Label - REPLACE]` and `[Topic Y Label - REPLACE]` suggests that addressing `[Root Cause related to Topic X]` could also alleviate issues related to `[Topic Y]`. This holistic approach can lead to more efficient problem-solving.

---

## 12. Limitations
* Analysis based on a static dataset; customer opinions and product issues can evolve over time.
* Sentiment analysis (VADER) may not capture highly nuanced linguistic expressions like sarcasm or complex conditional statements perfectly.
* The determination of `[Your NUM_TOPICS - REPLACE]` topics for LDA is partly subjective and based on interpretability; further quantitative evaluation (e.g., coherence scores) could refine this optimal number.
* The topic co-occurrence network shows association between topics, not necessarily direct causation. Further investigation would be needed to confirm causal links.

---

## 13. Future Work
* Implement **Aspect-Based Sentiment Analysis (ABSA)** for more granular insights into specific product features within broader topics.
* Develop **dynamic topic modeling** to track how customer discussion themes and sentiments change over different time periods or seasons.
* Build **predictive models** (e.g., to predict product returns, low ratings, or customer churn) using the derived NLP features and topic assignments.
* Enhance the Streamlit dashboard with more advanced filtering options, keyword search within topics, and potentially user-specific insights if user data were available.
* Explore **causal inference techniques** to better understand the drivers behind identified topic co-occurrences.

---

## 14. Developer
* **Name:** Ramesh Shrestha
* **Email:** `shrestha.ramesh000@gmail.com`
* **LinkedIn:** `https://linkedin.com/in/rameshsta` * **GitHub:** `https://github.com/RameshSTA/E-Commerce-Feedback-Mining` ---

## 15. License & Permissions

Copyright (c) 2024 Ramesh Shrestha. All Rights Reserved.

The code and work presented in this repository are for demonstration and portfolio purposes. If you wish to use, modify, or distribute any part of this project, please contact the developer, Ramesh Shrestha, at `shrestha.ramesh000@gmail.com` to request permission.

*(If you choose an open-source license like MIT, replace the above with:)*
<!--
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
-->
*(And then add a `LICENSE.md` file with the chosen license text.)*

---

