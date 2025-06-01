🤖 AI Resume Matcher – 2025 Edition
Capstone Project | Stevens Institute of Technology
Team: Anugya Sharma, Maruthi Kunchala, Jash Shah, Yash Kalpesh Shah
Duration: Jan 2025 – May 2025

📘 Overview
The AI Resume Matcher is a full-stack intelligent platform that analyzes resumes and matches them to live job postings using natural language processing (NLP), machine learning, and real-time job feeds. It delivers personalized job recommendations, highlights skill gaps, and offers career advice through an integrated AI chatbot.

🔍 Features
📄 Resume Parsing & Skill Extraction
Extracts text from .pdf and .docx resumes using PyPDF2 and docx2txt

Uses regex and dictionary-based techniques to identify relevant skills

Achieved 88% precision in skill extraction

🌐 Real-Time Job Fetching
Integrated with JSearch, RemoteOK, and Remotive APIs

Matches jobs dynamically based on parsed skill sets

🔗 Resume-to-Job Matching Logic
Uses TF-IDF vectorization and cosine similarity to compare resume skills with job descriptions

Ranks jobs and identifies missing skills for targeted improvement

📊 Job Clustering & Visualization
Uses K-Means clustering and PCA for visualizing job market segments and alternate career paths

🤖 Career Advisor Chatbot
Dual-model setup:

FLAN-T5 for fast, private queries

GPT-3.5 Turbo for deep contextual insights

Provides contextual career advice based on resume and cluster data

📄 Downloadable Career Report
Generates a clean, downloadable HTML report using Jinja2 templating

Summarizes top job matches, skill gaps, and career recommendations

🛠 Tech Stack
Languages & Frameworks: Python, Streamlit, Flask

NLP & ML Libraries: spaCy, transformers, sentence-transformers, scikit-learn, torch

Job APIs: JSearch, RemoteOK, Remotive

Parsing Tools: PyPDF2, python-docx, regex, beautifulsoup4

Visualization: Plotly, PCA

AI Models: FLAN-T5, GPT-3.5 Turbo (via OpenAI API)

Templating & Output: Jinja2, HTML

📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/ai-resume-matcher.git
cd ai-resume-matcher
Create a virtual environment and install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run main_app.py
📈 Results
Skill extraction achieved 88% precision

Job relevance alignment was 76% compared to human reviewers

Chatbot provided fast, context-aware responses across multiple use cases

🚧 Challenges & Future Work
Improve handling of stylized/resume-formatted skills (e.g., ReactJS, Next.js)

Incorporate feedback learning to improve ranking accuracy

Enhance UI for better interactivity and accessibility

📬 Contact
For questions or collaboration:
📧 jshah67@stevens.edu
