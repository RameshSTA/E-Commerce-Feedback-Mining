# ui_sections/about_me_view.py
import streamlit as st
import os
import base64 

def render_about_me(base_dir, assets_dir):
    st.header("👋 About Me & My Aspirations")
    st.markdown("---")
    
    col_img, col_text = st.columns([1, 2.5]) 
    
    with col_img:
        logo_filename = "ramesh.jpeg" 
        logo_path = os.path.join(assets_dir, logo_filename)

        if os.path.exists(logo_path):
            try:
                with open(logo_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()
                
                # CSS for round image with a subtle border and shadow
                st.markdown(f"""
                    <style>
                        .profile-image-container {{
                            display: flex;
                            justify-content: center; 
                            align-items: center; 
                            padding-top: 10px; 
                            padding-bottom: 10px;
                        }}
                        .profile-image-container img {{
                            width: 180px;  
                            height: 180px; 
                            border-radius: 50%;
                            object-fit: cover; 
                            border: 3px solid #E0E0E0;
                            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.1); 
                        }}
                    </style>
                    <div class="profile-image-container">
                        <img src="data:image/png;base64,{b64_string}" alt="Ramesh Shrestha - Profile">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                # Caption below the image, also centered
                st.markdown(f"<p style='text-align:center; font-size: smaller; color: #555;'>Ramesh Shrestha</p>", unsafe_allow_html=True)

            except FileNotFoundError:
                st.markdown(f"<p style='text-align:center; color:red;'><i>Profile photo placeholder - Image not found at `{logo_path}`. Ensure `{logo_filename}` is in the `assets` folder.</i></p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading profile image: {e}")
        else:
            st.markdown(f"<p style='text-align:center; color:orange;'><i>Profile photo (`{logo_filename}`) not found in `{assets_dir}`.</i></p>", unsafe_allow_html=True)


    with col_text:
        st.subheader("Ramesh Shrestha")
        st.markdown("**Aspiring Data Scientist | NLP & Machine Learning Enthusiast**") 
        st.markdown("📧 `shrestha.ramesh000@gmail.com`")
        st.markdown("📞 `0452 083 046`")
        st.markdown("---")
        st.markdown("""
            <a href="https://linkedin.com/in/rameshsta" target="_blank" style="text-decoration: none; margin-right: 15px;">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" width="28" height="28" style="vertical-align:middle; margin-right:5px;">LinkedIn Profile
            </a>
            <a href="https://github.com/RameshSTA" target="_blank" style="text-decoration: none;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="28" height="28" style="vertical-align:middle; margin-right:5px;">GitHub Profile
            </a>
        """, unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("#### A Little About Me:")
   
    st.markdown("""
    <div style="background-color: #f0f8ff; border-left: 6px solid #1e90ff; padding: 15px; border-radius: 5px; margin-bottom:15px; font-size: 0.75em;">
    I’m Ramesh Shrestha, an aspiring Data Scientist and ML&AL enthusiast based in Sydney with a strong foundation in machine learning, NLP, and MLOps. With hands-on experience in developing scalable data pipelines, deploying models via REST APIs, and applying advanced statistical techniques, I bring both technical depth and business insight. My work spans customer analytics, recommendation systems, and workforce intelligence—supported by real-world experience in retail at Woolworths and internship roles in the tech sector. I’m passionate about solving complex problems that bridge data and decision-making, and I thrive in Agile, cross-functional teams driving measurable impact.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### My Aspirations:")
   
    st.markdown("""
    <div style="background-color: #f0fff0; border-left: 6px solid #3cb371; padding: 15px; border-radius: 5px; font-size: 0.75em; font-family :roboto">
   I’m Ramesh Shrestha, a Data Scientist based in Sydney with a strong foundation in machine learning, NLP, and MLOps. With hands-on experience in developing scalable data pipelines, deploying models via REST APIs, and applying advanced statistical techniques, I bring both technical depth and business insight. My work spans customer analytics, recommendation systems, and workforce intelligence—supported by real-world experience in retail at Woolworths and internship roles in the tech sector. I’m passionate about solving complex problems that bridge data and decision-making, and I thrive in Agile, cross-functional teams driving measurable impact.

    I am eager to:
    <ul>
        <li>Apply my NLP and machine learning skills to create tangible value and drive data-informed decisions.</li>
        <li>Collaborate within a dynamic team to tackle complex challenges and learn from experienced professionals.</li>
        <li>Continuously expand my knowledge base in cutting-edge AI technologies, including deep learning and MLOps.</li>
        <li>Contribute to a company culture that values innovation, data, and impactful outcomes.</li>
    </ul>
    I am ready to bring my dedication, analytical mindset, and proactive learning approach to make a significant contribution.
    </div>
    """, unsafe_allow_html=True)