import requests
import random
import pandas as pd
import traceback
import time

# --- Live Job Scraper ---
def get_all_jobs(skills):
    """
    Fetch jobs based on skills from RapidAPI JSSearch
    With added debugging information
    """
    print(f"Starting job search with skills: {skills}")
    
    if not skills or len(skills) == 0:
        print("Error: No skills provided for job search")
        return pd.DataFrame()
    
    headers = {
        "X-RapidAPI-Key": "36e3982643msh28252de82063dc1p1779cbjsn060c60c0604f",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    url = "https://jsearch.p.rapidapi.com/search"
    
    # Create search queries from skills (pairs of skills)
    queries = [" ".join(skills[i:i+2]) for i in range(0, len(skills), 2)]
    if not queries:
        # Fallback to individual skills if pairing doesn't work
        queries = skills
    
    print(f"Generated {len(queries)} search queries: {queries}")
    
    all_jobs = []
    seen = set()
    
    # Take a random sample of queries to avoid excessive API calls
    sample_size = min(len(queries), 5)
    query_sample = random.sample(queries, sample_size)
    print(f"Sampling {sample_size} queries: {query_sample}")
    
    for q in query_sample:
        print(f"\nSearching for jobs with query: '{q}'")
        for page in range(1, 3):  # Check first 2 pages
            print(f"  Checking page {page}...")
            params = {"query": q, "page": str(page), "num_pages": "1"}
            
            try:
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
                # Make the API request
                resp = requests.get(url, headers=headers, params=params)
                
                # Check response status
                if resp.status_code != 200:
                    print(f"  Error: API returned status code {resp.status_code}")
                    print(f"  Response: {resp.text[:200]}...")
                    continue
                
                # Parse response
                response_json = resp.json()
                jobs = response_json.get("data", [])
                
                if not jobs:
                    print(f"  No jobs found on page {page}")
                    continue
                
                print(f"  Found {len(jobs)} jobs on page {page}")
                
                # Process jobs
                for job in jobs:
                    link = job.get("job_apply_link")
                    if not link or link in seen:
                        continue
                    
                    seen.add(link)
                    all_jobs.append({
                        "Skill": q,
                        "Job Title": job.get("job_title"),
                        "Company": job.get("employer_name"),
                        "Location": job.get("job_city"),
                        "Apply Link": f"[Click here to apply]({link})",
                        "Description": job.get("job_description", "")
                    })
                
            except Exception as e:
                print(f"  Error processing query '{q}' page {page}:")
                print(f"  {str(e)}")
                print(traceback.format_exc())
                continue
    
    print(f"\nTotal jobs collected: {len(all_jobs)}")
    
    # Return DataFrame or empty DataFrame if no jobs found
    if all_jobs:
        return pd.DataFrame(all_jobs)
    else:
        print("No jobs found across all queries")
        return pd.DataFrame()