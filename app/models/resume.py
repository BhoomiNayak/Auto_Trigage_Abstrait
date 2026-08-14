"""Pydantic models for resume/CV document extraction."""

from typing import Optional

from pydantic import BaseModel, Field

from app.models.base import DocumentType, ExtractionResult


class Experience(BaseModel):
    """A single work experience entry."""

    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    duration: Optional[str] = Field(
        None, description="Duration (e.g., '2 years', 'Jan 2020 - Mar 2022')"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Key achievements or responsibilities (max 3)",
    )


class Education(BaseModel):
    """A single education entry."""

    institution: str = Field(..., description="School/university name")
    degree: str = Field(
        ..., description="Degree and field (e.g., 'B.S. Computer Science')"
    )
    year: Optional[str] = Field(None, description="Graduation year or date range")


class ResumeData(BaseModel):
    """Structured data extracted from a resume/CV."""

    # Personal info
    full_name: str = Field(..., description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City/state/country")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL if present")
    portfolio_url: Optional[str] = Field(
        None, description="Portfolio/website URL if present"
    )

    # Professional summary
    summary: Optional[str] = Field(
        None,
        max_length=300,
        description="Professional summary or objective statement",
    )

    # Skills
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Technical skills (programming languages, tools, frameworks)",
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description="Soft skills (leadership, communication, etc.)",
    )

    # Experience
    total_years_experience: Optional[float] = Field(
        None, description="Estimated total years of professional experience"
    )
    experiences: list[Experience] = Field(
        default_factory=list,
        description="Work experience entries (most recent first, max 5)",
    )

    # Education
    education: list[Education] = Field(
        default_factory=list,
        description="Education entries",
    )

    # Extras
    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages spoken",
    )

    # AI assessment
    fit_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall candidate strength score (0.0 = poor fit, 1.0 = excellent)",
    )
    fit_reasoning: str = Field(
        ...,
        description="Brief explanation of the fit score",
    )


class ResumeExtraction(ExtractionResult):
    """Complete resume extraction response."""

    document_type: DocumentType = DocumentType.resume
    data: ResumeData
