"""Resume parsing utilities for AI Resume Matcher.

Extracts structured information from PDF, DOCX, and plain-text resumes.
All public functions accept a file path and return typed data-classes so
downstream matchers can operate on a stable schema.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?[0-9][\s\-.]?){7,15}")
_SECTION_HEADERS = {
    "experience", "work experience", "employment", "professional experience",
    "education", "skills", "technical skills", "projects", "certifications",
    "summary", "objective", "publications", "awards",
}


@dataclass
class ParsedResume:
    """Structured representation of a resume."""

    raw_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    sections: dict = field(default_factory=dict)


def extract_text_from_pdf(path: Path) -> str:
    """Extract raw text from a PDF file using pdfminer.

    Parameters
    ----------
    path : Path
        Absolute path to the PDF file.

    Returns
    -------
    str
        Concatenated text content of all pages.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path))
        logger.info("Extracted %d chars from PDF: %s", len(text), path.name)
        return text
    except ImportError as exc:
        raise ImportError("pdfminer.six is required: pip install pdfminer.six") from exc


def extract_text_from_docx(path: Path) -> str:
    """Extract raw text from a .docx file using python-docx.

    Parameters
    ----------
    path : Path
        Absolute path to the DOCX file.

    Returns
    -------
    str
        Concatenated paragraph text.
    """
    try:
        from docx import Document
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        logger.info("Extracted %d chars from DOCX: %s", len(text), path.name)
        return text
    except ImportError as exc:
        raise ImportError("python-docx is required: pip install python-docx") from exc


def load_resume(path: str | Path) -> str:
    """Load raw resume text from PDF, DOCX, or TXT files.

    Parameters
    ----------
    path : str or Path
        Path to the resume file. Supported extensions: .pdf, .docx, .txt.

    Returns
    -------
    str
        Raw text content of the resume.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported resume format: {suffix!r}")


def extract_contact_info(text: str) -> dict:
    """Extract name, email, and phone from raw resume text.

    Uses simple regex heuristics. For production use, consider a Named
    Entity Recognition (NER) model such as spaCy en_core_web_trf.

    Parameters
    ----------
    text : str
        Raw resume text.

    Returns
    -------
    dict
        Keys: "email", "phone". Values are None if not found.
    """
    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0).strip() if phone_match else None,
    }


def extract_skills(text: str, skill_vocab: list[str] | None = None) -> list[str]:
    """Extract skills from resume text via keyword matching.

    Parameters
    ----------
    text : str
        Raw resume text (lowercased internally).
    skill_vocab : list of str, optional
        Custom skill vocabulary. Defaults to a curated DS/ML skill list.

    Returns
    -------
    list of str
        Unique skills found in the text, in vocabulary order.
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
    lower = text.lower()
    found = [skill for skill in skill_vocab if skill in lower]
    logger.debug("Extracted %d skills from resume", len(found))
    return found


def parse_resume(path: str | Path, skill_vocab: list[str] | None = None) -> ParsedResume:
    """Parse a resume file into a structured ParsedResume object.

    End-to-end pipeline: load text, extract contact info and skills.

    Parameters
    ----------
    path : str or Path
        Path to the resume file.
    skill_vocab : list of str, optional
        Custom skill vocabulary passed to extract_skills.

    Returns
    -------
    ParsedResume
        Structured resume with raw text, contact info, and skills.
    """
    raw = load_resume(path)
    contact = extract_contact_info(raw)
    skills = extract_skills(raw, skill_vocab=skill_vocab)
    return ParsedResume(
        raw_text=raw,
        email=contact["email"],
        phone=contact["phone"],
        skills=skills,
    )
