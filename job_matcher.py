import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- Match and Score Jobs ---
def rank_jobs_by_embedding(resume_text, job_df, skill_list):
    """
    Rank jobs by similarity to resume text with enhanced debugging
    """
    print(f"\n--- Starting job ranking process ---")
    print(f"Resume text length: {len(resume_text)} characters")
    print(f"Skills list: {skill_list}")
    
    # Check if job dataframe is empty
    if job_df is None or job_df.empty:
        print("Error: Job dataframe is empty or None")
        return pd.DataFrame(), []
        
    print(f"Job dataframe shape: {job_df.shape}")
    print(f"Job dataframe columns: {job_df.columns.tolist()}")
    
    # Check if Description column exists
    if 'Description' not in job_df.columns:
        print("Warning: 'Description' column not found in job dataframe")
        print("Available columns:", job_df.columns.tolist())
        
        # Try to find an alternative description column
        desc_alternatives = ['description', 'job_description', 'JobDescription']
        found = False
        for col in desc_alternatives:
            if col in job_df.columns:
                print(f"Using '{col}' instead of 'Description'")
                job_df['Description'] = job_df[col]
                found = True
                break
                
        if not found:
            # Create a placeholder description from job title
            print("Creating placeholder descriptions from job titles")
            job_df['Description'] = job_df.apply(
                lambda row: f"{row.get('Job Title', '')} at {row.get('Company', '')}", 
                axis=1
            )
    
    # Check for any empty descriptions
    empty_desc_count = job_df['Description'].isnull().sum()
    if empty_desc_count > 0:
        print(f"Warning: {empty_desc_count} jobs have empty descriptions")
        job_df = job_df.dropna(subset=['Description'])
        print(f"After dropping NaN descriptions: {len(job_df)} jobs remain")
    
    # Create document collection for TF-IDF
    print("Creating TF-IDF vectorizer...")
    docs = [resume_text] + job_df['Description'].tolist()
    
    # Create and apply TF-IDF vectorizer
    try:
        vec = TfidfVectorizer(stop_words='english')
        vectors = vec.fit_transform(docs)
        print(f"TF-IDF vectorization successful. Matrix shape: {vectors.shape}")
        
        # Calculate similarity scores
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        print(f"Calculated similarity scores. Range: {scores.min():.4f} to {scores.max():.4f}")
        
        # Add scores to dataframe and sort
        job_df['Match Score'] = scores
        job_df = job_df.sort_values(by='Match Score', ascending=False)
        
    except Exception as e:
        print(f"Error during vectorization or similarity calculation: {str(e)}")
        return job_df, []
    
    # Find missing skills
    print("Analyzing skill gaps...")
    clean_text = re.sub(r"[\W_]+", " ", resume_text.lower())
    resume_skills = [skill for skill in skill_list if skill.lower() in clean_text]
    
    print(f"Skills found in resume: {len(resume_skills)}/{len(skill_list)}")
    print(f"Found skills: {resume_skills}")
    
    # Identify missing skills that appear in top jobs
    top_descriptions = " ".join(job_df['Description'].head(5)).lower()
    missing = [s for s in skill_list if s.lower() not in resume_skills and s.lower() in top_descriptions]
    
    print(f"Skills missing from resume but present in top jobs: {missing}")
    print("--- Job ranking complete ---")
    
    return job_df, missing