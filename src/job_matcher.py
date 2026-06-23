"""Job description matching engine for AI Resume Matcher.

Computes semantic and keyword-based similarity between a parsed resume and
one or more job descriptions, returning ranked match scores with explainable
skill-gap analysis.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Match score and analysis for one resume-JD pair."""

    jd_title: str
    score: float
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    keyword_overlap: float = 0.0

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"[{self.score:.1%}] {self.jd_title} | "
            f"matched={len(self.matched_skills)} missing={len(self.missing_skills)}"
        )


def _tokenize(text: str) -> set[str]:
    """Lowercase and split text into word tokens, removing punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(set_a: set, set_b: set) -> float:
    """Return the Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_jd_skills(jd_text: str, skill_vocab: list[str] | None = None) -> list[str]:
    """Extract required skills from a job description.

    Parameters
    ----------
    jd_text : str
        Raw job description text.
    skill_vocab : list of str, optional
        Skill vocabulary (same list used in resume_parser.extract_skills).

    Returns
    -------
    list of str
        Skills mentioned in the JD.
    """
    if skill_vocab is None:
        skill_vocab = [
            "python", "sql", "r", "java", "scala",
            "machine learning", "deep learning", "nlp",
            "tensorflow", "pytorch", "scikit-learn", "keras",
            "pandas", "numpy", "spark", "hadoop",
            "docker", "kubernetes", "aws", "gcp", "azure",
            "data visualisation", "tableau", "power bi",
            "statistics", "a/b testing", "experiment design",
        ]
    lower = jd_text.lower()
    return [s for s in skill_vocab if s in lower]


def compute_tfidf_vectors(
    resume_text: str, jd_texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TF-IDF vectors for resume and job descriptions.

    Parameters
    ----------
    resume_text : str
        Raw resume text.
    jd_texts : list of str
        List of raw job description texts.

    Returns
    -------
    tuple of (resume_vector, jd_matrix)
        resume_vector shape (1, vocab_size), jd_matrix shape (n_jd, vocab_size).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    corpus = [resume_text] + jd_texts
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus).toarray()
    return matrix[0:1], matrix[1:]


def cosine_similarity_scores(resume_vec: np.ndarray, jd_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between resume vector and each JD vector.

    Parameters
    ----------
    resume_vec : np.ndarray
        Shape (1, n_features).
    jd_matrix : np.ndarray
        Shape (n_jds, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_jds,) with cosine similarity scores in [0, 1].
    """
    resume_norm = np.linalg.norm(resume_vec)
    jd_norms = np.linalg.norm(jd_matrix, axis=1)
    if resume_norm == 0:
        return np.zeros(len(jd_matrix))
    dots = jd_matrix @ resume_vec.T
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(jd_norms > 0, dots.flatten() / (jd_norms * resume_norm), 0.0)
    return scores


def match_resume_to_jobs(
    resume_text: str,
    jd_texts: list[str],
    jd_titles: list[str] | None = None,
    skill_vocab: list[str] | None = None,
    top_k: int = 5,
) -> list[MatchResult]:
    """Rank job descriptions by match score against a resume.

    Combines TF-IDF cosine similarity (70%) with keyword Jaccard overlap (30%)
    for a robust hybrid score.

    Parameters
    ----------
    resume_text : str
        Raw resume text.
    jd_texts : list of str
        Raw job description texts to rank.
    jd_titles : list of str, optional
        Human-readable titles for each JD. Defaults to "Job 1", "Job 2", ...
    skill_vocab : list of str, optional
        Custom skill vocabulary.
    top_k : int
        Number of top matches to return.

    Returns
    -------
    list of MatchResult
        Top-k results sorted by descending score.
    """
    if not jd_texts:
        return []
    if jd_titles is None:
        jd_titles = [f"Job {i + 1}" for i in range(len(jd_texts))]
    if len(jd_titles) != len(jd_texts):
        raise ValueError("jd_titles and jd_texts must have the same length")

    # TF-IDF cosine component
    resume_vec, jd_matrix = compute_tfidf_vectors(resume_text, jd_texts)
    cosine_scores = cosine_similarity_scores(resume_vec, jd_matrix)

    # Jaccard keyword component
    resume_tokens = _tokenize(resume_text)
    jaccard_scores = np.array([
        _jaccard(resume_tokens, _tokenize(jd)) for jd in jd_texts
    ])

    # Hybrid score
    hybrid_scores = 0.7 * cosine_scores + 0.3 * jaccard_scores

    # Extract skill gap analysis
    resume_skills_set = set(extract_jd_skills(resume_text, skill_vocab))
    results = []
    for i, (title, score) in enumerate(zip(jd_titles, hybrid_scores)):
        jd_skills = set(extract_jd_skills(jd_texts[i], skill_vocab))
        matched = sorted(resume_skills_set & jd_skills)
        missing = sorted(jd_skills - resume_skills_set)
        results.append(MatchResult(
            jd_title=title,
            score=float(score),
            matched_skills=matched,
            missing_skills=missing,
            keyword_overlap=float(jaccard_scores[i]),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    logger.info("Ranked %d JDs; top match: %s (%.1%%)", len(results), results[0].jd_title, results[0].score * 100)
    return results[:top_k]
