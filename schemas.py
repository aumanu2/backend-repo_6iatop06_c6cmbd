"""
Database Schemas for Autism Prediction App

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name (e.g., Patient -> "patient").
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, EmailStr


class Patient(BaseModel):
    """Patients collection schema"""
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="Hashed password (server-side only)")
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, description="Male/Female/Other")


class Doctor(BaseModel):
    """Doctors collection schema"""
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="Hashed password (server-side only)")
    specialty: Optional[str] = Field(None, description="Specialty")
    hospital: Optional[str] = Field(None, description="Hospital/Clinic name")


class Assessment(BaseModel):
    """Assessments submitted by patients for prediction"""
    patient_id: str = Field(..., description="ID of patient")
    features: Dict[str, float] = Field(default_factory=dict, description="Numeric features for model")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score used for probability")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of autism risk")
    result_label: str = Field(..., description="Low/Moderate/High Risk")
    notes: Optional[str] = Field(None, description="Optional patient notes")
    reviewed_by: Optional[str] = Field(None, description="Doctor ID if reviewed")
    feedback_id: Optional[str] = Field(None, description="Linked feedback document ID")


class Feedback(BaseModel):
    """Doctor feedback linked to an assessment"""
    doctor_id: str = Field(..., description="Doctor ID")
    assessment_id: str = Field(..., description="Assessment ID")
    message: str = Field(..., description="Feedback message")
    severity: Optional[str] = Field(None, description="Low/Moderate/High")
    recommendations: Optional[List[str]] = Field(default_factory=list)


class Session(BaseModel):
    """Simple session tokens for auth"""
    user_id: str = Field(...)
    role: str = Field(..., description="patient or doctor")
    token: str = Field(...)
