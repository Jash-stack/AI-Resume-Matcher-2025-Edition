import os
import warnings
import torch._dynamo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
torch._dynamo.config.suppress_errors = True

import streamlit as st
from components.chatbot import generate_chat_response, format_context
from components.resume_parser import parse_resume_and_skills
from components.job_scraper import get_all_jobs
from components.job_matcher import rank_jobs_by_embedding
from components.job_clustering import cluster_jobs
from components.report_generator import generate_report

# --- Custom CSS ---
st.set_page_config(
    page_title="Resume Job Matcher", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS
st.markdown("""
<style>
    /* Main colors */
    :root {
        --burgundy: #800020;
        --navy: #000080;
        --light-burgundy: #aa5f73;
        --light-navy: #3a3a8c;
        --off-white: #f8f8f8;
        --light-gray: #f0f0f0;
    }
    
    /* Header styling */
    .main-header {
        color: var(--burgundy);
        font-weight: 600;
        padding-bottom: 20px;
        border-bottom: 2px solid var(--navy);
        margin-bottom: 30px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--light-gray);
        padding: 10px 10px 0 10px;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: white;
        border: 1px solid #ddd;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--navy) !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background-color: var(--burgundy);
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: var(--light-burgundy);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Dataframe styling */
    .dataframe th {
        background-color: var(--navy);
        color: white;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: var(--light-gray);
    }
    
    /* Chat message styling */
    .user-message {
        background-color: var(--light-navy);
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        display: inline-block;
    }
    
    .assistant-message {
        background-color: var(--light-gray);
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        display: inline-block;
    }
    
    /* Success message */
    .success-banner {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    /* Info/warning message */
    .info-banner {
        background-color: #cce5ff;
        color: #004085;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 5px solid #0066ff;
        margin: 10px 0;
    }
    
    .warning-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    .badge-primary {
        background-color: var(--navy);
        color: white;
    }
    
    .badge-secondary {
        background-color: var(--burgundy);
        color: white;
    }
    
    /* Section styling */
    .section-header {
        color: var(--navy);
        border-bottom: 2px solid var(--burgundy);
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit Setup ---
st.markdown("<h1 class='main-header'>Resume Matcher Pro</h1>", unsafe_allow_html=True)
st.markdown("<p>AI-powered resume analysis and job matching for career success</p>", unsafe_allow_html=True)

# --- Tabs ---
upload_tab, match_tab, clusters_tab, report_tab, chat_tab = st.tabs([
    "📤 Upload Resume", "🔍 Job Matches", "🧠 Clusters", "📥 Career Report", "💬 Career Chat"
])

# --- Global State ---
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'skills' not in st.session_state:
    st.session_state.skills = []
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'ranked_df' not in st.session_state:
    st.session_state.ranked_df = None
if 'clustered_df' not in st.session_state:
    st.session_state.clustered_df = None

# --- Upload Tab ---
with upload_tab:
    st.markdown("<h3 class='section-header'>Upload Your Resume</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select your resume file (PDF or DOCX)", type=["pdf", "docx"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p>Supported formats:</p>", unsafe_allow_html=True)
        st.markdown("• PDF<br>• DOCX", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            resume_text, extracted_skills = parse_resume_and_skills(uploaded_file)
            st.session_state.resume_text = resume_text
            st.session_state.skills = extracted_skills
        
        st.markdown("<div class='success-banner'>Resume successfully analyzed!</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Extracted Skills</h4>", unsafe_allow_html=True)
            skills_html = ""
            for skill in extracted_skills:
                skills_html += f"<span class='badge badge-primary'>{skill}</span>"
            st.markdown(skills_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            with st.expander("📄 Resume Text Preview"):
                st.text_area("", resume_text[:3000], height=200)

# --- Match Tab ---
with match_tab:
    st.markdown("<h3 class='section-header'>Find Your Perfect Job Match</h3>", unsafe_allow_html=True)
    
    if st.session_state.resume_text and st.session_state.skills:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.button("🔎 Search Jobs & Rank Matches"):
            with st.spinner("Searching for matching jobs..."):
                jobs_df = get_all_jobs(st.session_state.skills)
                ranked_df, missing = rank_jobs_by_embedding(
                    st.session_state.resume_text, jobs_df, st.session_state.skills
                )
                st.session_state.jobs_df = jobs_df
                st.session_state.ranked_df = ranked_df
                st.session_state.top_jobs = ranked_df.head(5).to_dict('records')
            
            st.markdown(f"<div class='success-banner'>Found {len(ranked_df)} potential job matches</div>", unsafe_allow_html=True)
            
            if missing:
                missing_skills_html = ", ".join([f"<span class='badge badge-secondary'>{skill}</span>" for skill in missing])
                st.markdown(f"<div class='warning-banner'>Top jobs require these skills that weren't found in your resume: {missing_skills_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.ranked_df is not None:
            st.markdown("<h4>Top 10 Job Matches</h4>", unsafe_allow_html=True)
            
            # Safe column selection
            try:
                display_columns = []
                available_columns = st.session_state.ranked_df.columns.tolist()
                
                # Map expected columns to actual columns
                column_mapping = {
                    'Job Title': ['Job Title', 'job_title', 'title', 'position', 'role'],
                    'Company': ['Company', 'company', 'company_name', 'employer'],
                    'Location': ['Location', 'location', 'job_location', 'city'],
                    'Match Score': ['Match Score', 'match_score', 'score', 'similarity', 'match'],
                    'Apply Link': ['Apply Link', 'apply_link', 'url', 'link', 'job_url']
                }
                
                # Build columns to display
                display_df = st.session_state.ranked_df.copy()
                
                # Add columns if needed or rename existing ones
                for display_name, possible_names in column_mapping.items():
                    found = False
                    for col_name in possible_names:
                        if col_name in available_columns:
                            if col_name != display_name and col_name in display_df.columns:
                                display_df[display_name] = display_df[col_name]
                            display_columns.append(display_name)
                            found = True
                            break
                    if not found:
                        display_df[display_name] = "N/A"
                        display_columns.append(display_name)
                
                st.dataframe(
                    display_df[display_columns].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error displaying job matches: {e}")
                st.write("Available columns:", ", ".join(st.session_state.ranked_df.columns.tolist()))
    else:
        st.markdown("<div class='info-banner'>Please upload and analyze your resume first</div>", unsafe_allow_html=True)

# --- Clusters Tab ---
# Inside the cluster analysis button logic in main_app.py (already structured):

with clusters_tab:
    st.markdown("<h3 class='section-header'>Job Clusters Analysis</h3>", unsafe_allow_html=True)

    if st.session_state.ranked_df is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p>Group similar jobs together to identify career paths and opportunities</p>", unsafe_allow_html=True)
        num_clusters = st.slider("Number of Job Clusters", 2, 6, 3)

        if st.button("🔍 Run Cluster Analysis"):
            try:
                with st.spinner("Running cluster analysis..."):
                    if 'Description' not in st.session_state.ranked_df.columns:
                        desc_alternatives = ['description', 'job_description', 'JobDescription']
                        found = False
                        for col in desc_alternatives:
                            if col in st.session_state.ranked_df.columns:
                                st.session_state.ranked_df['Description'] = st.session_state.ranked_df[col].fillna("")
                                found = True
                                break
                        if not found:
                            st.session_state.ranked_df['Description'] = st.session_state.ranked_df.apply(
                                lambda row: f"{row.get('Job Title', 'Unknown')} at {row.get('Company', 'Unknown')}", axis=1
                            )
                    else:
                        st.session_state.ranked_df['Description'] = st.session_state.ranked_df['Description'].fillna("")

                    clustered_df, fig = cluster_jobs(st.session_state.ranked_df, num_clusters=num_clusters)
                    st.session_state.clustered_df = clustered_df

                st.plotly_chart(fig, use_container_width=True)

                for cluster_id in sorted(clustered_df['Cluster'].unique()):
                    cluster_df = clustered_df[clustered_df['Cluster'] == cluster_id]

                    with st.expander(f"Cluster {cluster_id + 1} - {len(cluster_df)} jobs"):
                        try:
                            display_columns = []
                            available_columns = cluster_df.columns.tolist()

                            column_mapping = {
                                'Job Title': ['Job Title', 'job_title', 'title', 'position', 'role'],
                                'Company': ['Company', 'company', 'company_name', 'employer'],
                                'Location': ['Location', 'location', 'job_location', 'city'],
                                'Apply Link': ['Apply Link', 'apply_link', 'url', 'link', 'job_url']
                            }

                            display_df = cluster_df.copy()

                            for display_name, possible_names in column_mapping.items():
                                found = False
                                for col_name in possible_names:
                                    if col_name in available_columns:
                                        if col_name != display_name and col_name in display_df.columns:
                                            display_df[display_name] = display_df[col_name]
                                        display_columns.append(display_name)
                                        found = True
                                        break
                                if not found:
                                    display_df[display_name] = "N/A"
                                    display_columns.append(display_name)

                            st.dataframe(
                                display_df[display_columns],
                                use_container_width=True,
                                hide_index=True
                            )
                        except Exception as e:
                            st.error(f"Error displaying cluster data: {e}")
                            st.write("Available columns:", ", ".join(cluster_df.columns.tolist()))
            except Exception as e:
                st.error(f"Error during clustering: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-banner'>Please run the job match step first</div>", unsafe_allow_html=True)

# --- Report Tab ---
with report_tab:
    st.markdown("<h3 class='section-header'>Career Opportunities Report</h3>", unsafe_allow_html=True)
    
    if st.session_state.clustered_df is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <p>Generate a comprehensive career report with:</p>
        <ul>
            <li>Resume analysis summary</li>
            <li>Top skill matches and gaps</li>
            <li>Personalized job recommendations</li>
            <li>Career path visualization</li>
        </ul>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Generate Career Report"):
            try:
                with st.spinner("Creating your personalized career report..."):
                    report_path = generate_report(
                        st.session_state.resume_text,
                        st.session_state.skills,
                        st.session_state.ranked_df,
                        st.session_state.clustered_df
                    )
                
                st.markdown("<div class='success-banner'>Career report generated successfully!</div>", unsafe_allow_html=True)
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Career Report", 
                        f, 
                        file_name="career_report.html",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error generating report: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-banner'>Complete clustering first to unlock the report</div>", unsafe_allow_html=True)

# --- Chat Tab ---
with chat_tab:
    st.markdown("<h3 class='section-header'>Career Guidance Assistant</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Ask questions about your resume, skills, job matches, or get career advice</p>", unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_input("Your question:", placeholder="e.g., What skills should I improve to get better job matches?")
    
    if user_input:
        try:
            with st.spinner("Thinking..."):
                context = format_context(st.session_state.resume_text, matched_jobs=st.session_state.get("top_jobs", []))
                response = generate_chat_response(user_input, context)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", response))
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Assistant", "I'm sorry, I encountered an error while processing your request."))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div style='text-align: right;'><div class='user-message'>{message}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div><div class='assistant-message'>{message}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)