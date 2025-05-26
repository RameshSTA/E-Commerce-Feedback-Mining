
import streamlit as st
import os
import base64

def _get_project_root_from_passed_dir(base_dir_passed: str) -> str:
    """
    Confirms or determines the project root directory based on the passed base_dir.
    This helps in reliably locating assets.

    Args:
        base_dir_passed (str): The directory path passed from the main app,
                               expected to be the project root.

    Returns:
        str: The absolute path to the project root directory.
    """
    # Assuming base_dir_passed from app.py IS the intended project root.
    # Further checks could be added if the structure is more complex.
    return os.path.abspath(base_dir_passed)

def render_about_me(project_root_dir: str, assets_dir_passed: str):
    """
    Renders the "About Me" page with professional articulation and detailed content.

    This page showcases a personal and professional introduction, including:
    - A profile image with consistent styling.
    - Clearly presented contact information and links to professional profiles.
    - A comprehensive "Professional Summary" detailing skills, experiences, and specialisations.
    - "Key Credentials & Experience Highlights" outlining formative academic and practical achievements.
    - "Career Aspirations & Professional Goals" demonstrating ambition and focus.

    The content is structured using globally defined CSS classes (expected in main app.py)
    for consistent visual presentation (e.g., bordered content blocks).

    Args:
        project_root_dir (str): The absolute path to the project's root directory.
                                (Used if `assets_dir_passed` needs resolving).
        assets_dir_passed (str): The absolute path to the project's assets directory.
    """
    st.header("👋Hi there,This is Ramesh Shrestha")
    st.markdown("---")

    # Resolve asset path robustly
    # If assets_dir_passed is not absolute, make it relative to project_root_dir
    if not os.path.isabs(assets_dir_passed):
        true_assets_dir = os.path.join(project_root_dir, assets_dir_passed)
    else:
        true_assets_dir = assets_dir_passed

    profile_image_filename = "ramesh.jpeg" # Ensure this image is in your assets directory
    profile_image_path = os.path.join(true_assets_dir, profile_image_filename)

    col_img, col_text = st.columns([1.2, 2.8], gap="large") # Adjusted column ratio

    with col_img:
        # Profile Image Display
        if os.path.exists(profile_image_path):
            try:
                with open(profile_image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()
                # Specific CSS for the profile image element
                st.markdown(f"""
                    <style>
                        .profile-image-about-page {{ /* More specific class name */
                            display: flex;
                            justify-content: center;
                            align-items: flex-start; /* Aligns image to the top of its column cell */
                            padding-top: 15px;      /* Reduced top padding */
                            padding-bottom: 20px;
                        }}
                        .profile-image-about-page img {{
                            width: 200px; /* Consistent size */
                            height: 200px;
                            border-radius: 50%;
                            object-fit: cover;
                            border: 4px solid #ECEFF1; /* Lighter, more subtle border */
                            box-shadow: 0 5px 15px rgba(0,0,0,0.12), 0 7px 25px rgba(0,0,0,0.08); /* Refined shadow */
                        }}
                    </style>
                    <div class="profile-image-about-page">
                        <img src="data:image/jpeg;base64,{b64_string}" alt="Ramesh Shrestha - Profile Photo">
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e: # pragma: no cover
                st.error(f"Error loading profile image: {e}")
                st.markdown(f"<p style='text-align:center; color:red;'><i>Photo not loaded. Ensure '{profile_image_filename}' is in `{true_assets_dir}`.</i></p>", unsafe_allow_html=True)
        else: # pragma: no cover
            st.markdown(
                f"<p style='text-align:center; color:orange;'><i>Profile photo ('{profile_image_filename}') not found in `{true_assets_dir}`. Placeholder shown.</i></p>",
                unsafe_allow_html=True)
            st.markdown( # Placeholder if image is not found
                """
                <div style='width:200px; height:200px; border-radius:50%; background-color:#E0E0E0;
                            display:flex; justify-content:center; align-items:center; margin: 15px auto 20px auto;
                            border: 4px solid #ECEFF1;'>
                    <span style='font-size:50px; color: #757575; font-weight: bold;'>RS</span>
                </div>
                """, unsafe_allow_html=True)

    with col_text:
        st.subheader("Ramesh Shrestha")
        st.markdown("##### Aspiring Data Scientist & Machine Learning Engineer")
        st.markdown("Sydney, NSW, Australia")
        st.markdown("---")
        st.markdown("""
        <div style="line-height: 2.3; font-size: 0.98em;">
            <div>
                <span style="font-size: 1.1em; vertical-align: middle;">📧</span>&nbsp;
                <a href="mailto:shrestha.ramesh000@gmail.com" style="text-decoration:none; color: #007bff; vertical-align: middle;">shrestha.ramesh000@gmail.com</a>
            </div>
            <div>
                <span style="font-size: 1.1em; vertical-align: middle;">📞</span>&nbsp;
                <span style="vertical-align: middle; font-family: monospace; color: #333;">+61 452 083 046</span>
            </div>
            <div style="margin-top: 12px;">
                <a href="https://linkedin.com/in/rameshsta" target="_blank" style="text-decoration: none; margin-right: 20px; color: #0077B5; font-weight:500;">
                    <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" width="21" height="21" style="vertical-align:middle; margin-right:7px;">LinkedIn Profile
                </a>
                <a href="https://github.com/RameshSTA" target="_blank" style="text-decoration: none; color: #181717; font-weight:500;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="21" height="21" style="vertical-align:middle; margin-right:7px;">GitHub Portfolio
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Sections using globally defined CSS classes
    st.markdown("#### Professional Summary")
    st.markdown("""
    <div class="content-block content-block-info">
        <p>A highly motivated and analytical professional, holding a Bachelor of Software Engineering with a Specialisation in AI, and currently advancing practical expertise through a Professional Year in IT. My core strength lies in translating complex data into actionable insights and developing robust, end-to-end AI/ML solutions.</p>
        <p>I possess a comprehensive skill set in Python, SQL, and R, complemented by a deep understanding of machine learning algorithms (including predictive analytics, classification, and deep learning), Natural Language Processing (NLP), statistical methods, and data visualisation techniques. This theoretical knowledge is solidified by hands-on internship experience involving data analysis and the practical application of machine learning models. My project portfolio, particularly "SupplyChainAI IntelligenX" and "E-Commerce Feedback Mining," demonstrates a capacity for independent development and deployment of interactive, data-driven applications using tools like Streamlit.</p>
        <p>Committed to continuous professional development—as evidenced by multiple industry-recognised certifications—I am passionate about contributing to innovative projects within collaborative, agile environments that drive measurable business impact.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Key Credentials & Experience Highlights")
    st.markdown("""
    <div class="content-block content-block-neutral">
        My journey into data science and AI is built upon a strong academic foundation, consistently enhanced by specialised training and practical application:
        <ul>
            <li><strong>Academic Foundation:</strong> Bachelor of Software Engineering (Specialisation in Artificial Intelligence) from Torrens University Australia, providing comprehensive knowledge in software development principles and core AI concepts. Currently augmenting this with a Professional Year in IT at QIBA, focusing on industry best practices.</li>
            <li><strong>Specialised Certifications:</strong> Actively pursued advanced credentials to deepen expertise, including the <em>Google Advanced Data Analytics Professional Certificate</em>, the <em>Deep Learning Specialization (DeepLearning.AI)</em>, and the <em>Machine Learning Specialization (Stanford University / DeepLearning.AI)</em>. Further practical experience gained via a <em>Software Engineering Job Simulation with JPMorgan Chase & Co. (Forage)</em>.</li>
            <li><strong>Data Analysis & ML Internship (Hightech Mastermind Pty Ltd):</strong> Gained valuable hands-on experience in a professional setting, contributing to data solution design, assisting in the development and validation of predictive machine learning models, performing in-depth exploratory data analysis, and automating analytical reporting processes.</li>
            <li><strong>End-to-End AI Project Development:</strong> Independently conceptualised, developed, and deployed AI-powered applications such as "SupplyChainAI IntelligenX" (for demand forecasting, inventory optimisation, and NLP-based risk alerting) and "E-Commerce Feedback Mining" (for sentiment analysis and LDA topic modelling). These projects showcase proficiency in Python, NLP, forecasting, MLOps fundamentals, and creating insightful, interactive dashboards with Streamlit.</li>
            <li><strong>Mentorship & Applied Communication:</strong> Honed communication, technical explanation, and leadership skills by mentoring participants at industry-aligned Business Analyst and Data Analyst Hackathons. Developed strong interpersonal, problem-solving, and adaptability skills through dynamic customer-facing roles.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Career Aspirations & Professional Goals")
    st.markdown("""
    <div class="content-block content-block-success">
        My primary career objective is to secure a challenging and impactful role as a Data Scientist or Machine Learning Engineer, where I can leverage my technical acumen and analytical mindset to contribute to innovative, data-driven solutions that yield significant business outcomes. I aim to:
        <ul>
            <li>Apply and continuously expand my expertise in machine learning, deep learning, and NLP to extract meaningful insights, build robust predictive models, and create tangible business value from complex, large-scale datasets.</li>
            <li>Thrive within dynamic, cross-functional teams, collaborating effectively to tackle ambitious projects, solve intricate problems, and learn from seasoned professionals and thought leaders in the AI and data science domain.</li>
            <li>Actively contribute to the full lifecycle of machine learning projects, from meticulous ideation and data exploration through to rigorous model development, deployment, monitoring, and operationalisation (MLOps best practices).</li>
            <li>Dedicate myself to ongoing learning and professional growth, particularly in cutting-edge AI technologies (such as LLMs and Generative AI), advanced cloud-based data solutions, and the ethical, responsible application of AI.</li>
            <li>Contribute positively to an organisational culture that champions innovation, data-informed strategic decision-making, and the unwavering pursuit of excellence in creating impactful technological advancements.</li>
        </ul>
        I am enthusiastic about bringing my dedication, robust analytical capabilities, and proactive learning approach to an organisation where I can make substantial contributions and evolve alongside industry pioneers.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__": # pragma: no cover
    # This block allows for standalone testing of this page.
    # For assets to load correctly and global styles to apply,
    # it's best to run the main app.py.
    test_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Construct a plausible assets path assuming this file is in app/page_content
    test_assets_dir = os.path.join(test_project_root, "app", "assets")
    
    st.set_page_config(layout="wide", page_title="About Ramesh Shrestha")
    st.warning("Standalone Run: Global CSS for bordered content blocks (defined in app.py) will not be applied here.")
    render_about_me(project_root_dir=test_project_root, assets_dir_passed=test_assets_dir)