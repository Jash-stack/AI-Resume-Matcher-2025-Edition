import docx2txt
import PyPDF2
import re
import string

# --- Unified Skill Set ---
SKILL_KEYWORDS = list(set([
    "java", "python", "sql", "aws", "azure", "html", "css", "javascript", "react", "nodejs", "tensorflow",
    "keras", "pandas", "numpy", "scikit-learn", "git", "docker", "kubernetes", "flask", "django", "spark",
    "hadoop", "nlp", "c++", "c#", "xgboost", "airflow", "jira", "postman", "ci/cd", "unit testing", "linux"
]))

# --- Text Extraction ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

# --- Skill Extraction ---
def extract_skills(text, keywords=SKILL_KEYWORDS):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return list({skill for skill in keywords if skill.lower() in text})

# --- Entry Point ---
def parse_resume_and_skills(file):
    text = extract_text(file)
    skills = extract_skills(text)
    return text, skills
