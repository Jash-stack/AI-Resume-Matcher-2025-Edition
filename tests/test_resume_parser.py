"""Tests for the resume parser and job matching pipeline."""
import pytest


@pytest.fixture
def sample_resume_text():
    return """
    John Doe
    john@example.com | (555) 123-4567
    Skills: Python, Machine Learning, NLP, spaCy, scikit-learn, SQL, Docker
    Experience: 3 years as Data Scientist at TechCorp
    Education: MS Computer Science, Stanford University
    """


@pytest.fixture
def sample_job_description():
    return """
    Senior Data Scientist
    Requirements: Python, Machine Learning, NLP, PyTorch, SQL, 3+ years experience
    Nice to have: Docker, Kubernetes, MLflow
    """


class TestSkillExtraction:
    def test_extracts_known_skills(self, sample_resume_text):
        from resume_parser import extract_skills
        skills = extract_skills(sample_resume_text)
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_returns_lowercase_skills(self, sample_resume_text):
        from resume_parser import extract_skills
        skills = extract_skills(sample_resume_text)
        for skill in skills:
            assert skill == skill.lower()

    def test_extracts_python(self, sample_resume_text):
        from resume_parser import extract_skills
        skills = extract_skills(sample_resume_text)
        assert "python" in skills

    def test_empty_text_returns_empty_list(self):
        from resume_parser import extract_skills
        skills = extract_skills("")
        assert skills == []


class TestJobMatching:
    def test_match_score_between_0_and_1(self, sample_resume_text, sample_job_description):
        from job_matcher import compute_match_score
        score = compute_match_score(sample_resume_text, sample_job_description)
        assert 0.0 <= score <= 1.0

    def test_identical_texts_score_near_1(self, sample_resume_text):
        from job_matcher import compute_match_score
        score = compute_match_score(sample_resume_text, sample_resume_text)
        assert score >= 0.9

    def test_unrelated_texts_score_near_0(self):
        from job_matcher import compute_match_score
        resume = "I am a chef with 10 years of culinary experience."
        jd = "Quantum physicist needed for particle accelerator research."
        score = compute_match_score(resume, jd)
        assert score < 0.5


class TestSkillGapDetection:
    def test_returns_missing_skills(self, sample_resume_text, sample_job_description):
        from resume_parser import extract_skills
        from job_matcher import find_skill_gaps
        resume_skills = extract_skills(sample_resume_text)
        jd_skills = extract_skills(sample_job_description)
        gaps = find_skill_gaps(resume_skills, jd_skills)
        assert isinstance(gaps, list)

    def test_no_gaps_when_resume_covers_jd(self):
        from job_matcher import find_skill_gaps
        resume_skills = ["python", "sql", "pytorch", "docker"]
        jd_skills = ["python", "sql"]
        gaps = find_skill_gaps(resume_skills, jd_skills)
        assert gaps == []

    def test_identifies_pytorch_gap(self, sample_resume_text, sample_job_description):
        from resume_parser import extract_skills
        from job_matcher import find_skill_gaps
        resume_skills = extract_skills(sample_resume_text)
        jd_skills = extract_skills(sample_job_description)
        gaps = find_skill_gaps(resume_skills, jd_skills)
        # PyTorch is in JD but not resume
        assert "pytorch" in gaps
