import pandas as pd
import sys
import traceback

# Test script to diagnose job scraping and matching issues
print("===== JOB MATCHER DIAGNOSTICS =====")

# Import the debugging versions - replace with your actual imports
try:
    # First, try to import your actual modules
    try:
        print("Trying to import original modules...")
        from components.job_scraper import get_all_jobs
        from components.job_matcher import rank_jobs_by_embedding
    except ImportError:
        print("Couldn't import from components, trying direct imports...")
        # If that fails, import directly from current directory
        from job_scraper import get_all_jobs
        from job_matcher import rank_jobs_by_embedding
        
    print("Modules imported successfully")
except Exception as e:
    print(f"Import error: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

# Test data
print("\n1. SETTING UP TEST DATA")
test_skills = ["python", "data analysis", "machine learning", "sql", "javascript"]
print(f"Test skills: {test_skills}")

test_resume = """
John Doe
Data Scientist

Skills:
- Python programming
- Data analysis and visualization
- Machine learning models
- SQL database management
- Project management

Experience:
- Data Scientist at XYZ Corp (2020-Present)
- Data Analyst at ABC Inc (2018-2020)
"""
print(f"Test resume length: {len(test_resume)} characters")

# Test job scraping
print("\n2. TESTING JOB SCRAPER")
try:
    print("Calling get_all_jobs()...")
    jobs_df = get_all_jobs(test_skills)
    
    if jobs_df is None:
        print("ERROR: get_all_jobs returned None")
    elif jobs_df.empty:
        print("WARNING: get_all_jobs returned empty DataFrame")
    else:
        print(f"SUCCESS: Found {len(jobs_df)} jobs")
        print(f"DataFrame columns: {jobs_df.columns.tolist()}")
        print("\nFirst job sample:")
        if len(jobs_df) > 0:
            sample_job = jobs_df.iloc[0]
            for col in jobs_df.columns:
                val = sample_job[col]
                if col == 'Description':
                    val = val[:100] + "..." if isinstance(val, str) and len(val) > 100 else val
                print(f"  {col}: {val}")
except Exception as e:
    print(f"Job scraper error: {str(e)}")
    print(traceback.format_exc())
    # Continue to next test rather than exiting

# Test job matching
print("\n3. TESTING JOB MATCHER")
try:
    # Create a fake jobs dataframe if the real one failed
    if 'jobs_df' not in locals() or jobs_df is None or jobs_df.empty:
        print("Using test data for job matching since scraper failed")
        jobs_df = pd.DataFrame({
            'Job Title': ['Data Scientist', 'Data Analyst', 'ML Engineer'],
            'Company': ['Test Co', 'Sample Inc', 'Demo LLC'],
            'Location': ['Remote', 'New York', 'San Francisco'],
            'Description': [
                'Looking for a Python expert with machine learning skills',
                'Need SQL and data analysis expert',
                'AI/ML engineer with deep learning experience'
            ]
        })
    
    print(f"Running job matching with {len(jobs_df)} jobs...")
    ranked_df, missing = rank_jobs_by_embedding(test_resume, jobs_df, test_skills)
    
    if ranked_df is None:
        print("ERROR: rank_jobs_by_embedding returned None for ranked_df")
    elif ranked_df.empty:
        print("WARNING: rank_jobs_by_embedding returned empty DataFrame")
    else:
        print(f"SUCCESS: Ranked {len(ranked_df)} jobs")
        print(f"Missing skills: {missing}")
        print("\nTop 3 matches:")
        
        # Show top 3 matches
        top_matches = ranked_df.head(3)
        for i, (_, job) in enumerate(top_matches.iterrows()):
            print(f"\nMatch #{i+1}: {job.get('Job Title', 'Untitled')} at {job.get('Company', 'Unknown')}")
            print(f"  Match Score: {job.get('Match Score', 'N/A'):.4f}")
            
except Exception as e:
    print(f"Job matcher error: {str(e)}")
    print(traceback.format_exc())

print("\n===== DIAGNOSTICS COMPLETE =====")