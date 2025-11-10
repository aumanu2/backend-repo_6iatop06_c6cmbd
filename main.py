import os
import hashlib
import secrets
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from database import db, create_document, get_documents
from schemas import Patient, Doctor, Assessment, Feedback, Session

app = FastAPI(title="Autism Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utilities

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == password_hash


# Request models
class RegisterRequest(BaseModel):
    role: str  # 'patient' or 'doctor'
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    role: str
    email: EmailStr
    password: str


class PredictionRequest(BaseModel):
    # Example features for a simple rule-based mock model
    # values are 0 or 1, representing questionnaire answers
    eye_contact: float
    speech_delay: float
    repetitive_behavior: float
    sensory_sensitivity: float
    social_interaction_difficulty: float
    notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    assessment_id: str
    message: str
    severity: Optional[str] = None
    recommendations: Optional[List[str]] = None


# Simple in-db session management

def create_session(user_id: str, role: str) -> str:
    token = secrets.token_urlsafe(24)
    create_document("session", Session(user_id=user_id, role=role, token=token))
    return token


def get_user_by_email(role: str, email: str):
    coll = "patient" if role == "patient" else "doctor"
    users = get_documents(coll, {"email": email}, limit=1)
    return users[0] if users else None


@app.get("/")
async def root():
    return {"message": "Autism Prediction API running"}


@app.post("/auth/register")
async def register(req: RegisterRequest):
    role = req.role.lower()
    if role not in ("patient", "doctor"):
        raise HTTPException(400, "Role must be patient or doctor")

    if get_user_by_email(role, req.email):
        raise HTTPException(409, "Email already registered")

    password_hash = hash_password(req.password)
    if role == "patient":
        doc = Patient(name=req.name, email=req.email, password_hash=password_hash)
        user_id = create_document("patient", doc)
    else:
        doc = Doctor(name=req.name, email=req.email, password_hash=password_hash)
        user_id = create_document("doctor", doc)

    token = create_session(user_id, role)
    return {"token": token, "role": role, "user_id": user_id, "name": req.name, "email": req.email}


@app.post("/auth/login")
async def login(req: LoginRequest):
    role = req.role.lower()
    if role not in ("patient", "doctor"):
        raise HTTPException(400, "Role must be patient or doctor")

    user = get_user_by_email(role, req.email)
    if not user or not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(401, "Invalid credentials")
    token = create_session(str(user.get("_id")), role)
    return {"token": token, "role": role, "user_id": str(user.get("_id")), "name": user.get("name"), "email": user.get("email")}


@app.post("/predict")
async def predict(req: PredictionRequest, token: Optional[str] = None, user_id: Optional[str] = None):
    # Basic check: must supply either token or user_id for association
    if not token and not user_id:
        raise HTTPException(401, "Authentication required")

    # Simple mock scoring algorithm combining features
    weights = {
        "eye_contact": 0.18,
        "speech_delay": 0.22,
        "repetitive_behavior": 0.22,
        "sensory_sensitivity": 0.2,
        "social_interaction_difficulty": 0.18,
    }
    total = 0.0
    for k, w in weights.items():
        total += getattr(req, k) * w
    probability = max(0.0, min(1.0, total))
    if probability < 0.33:
        label = "Low Risk"
    elif probability < 0.66:
        label = "Moderate Risk"
    else:
        label = "High Risk"

    # If token provided, try to resolve to patient_id from session
    pid = user_id
    if token and not pid:
        sessions = get_documents("session", {"token": token}, limit=1)
        if sessions:
            sess = sessions[0]
            if sess.get("role") == "patient":
                pid = sess.get("user_id")

    features = {
        "eye_contact": req.eye_contact,
        "speech_delay": req.speech_delay,
        "repetitive_behavior": req.repetitive_behavior,
        "sensory_sensitivity": req.sensory_sensitivity,
        "social_interaction_difficulty": req.social_interaction_difficulty,
    }

    assessment = Assessment(
        patient_id=str(pid) if pid else "unknown",
        features=features,
        score=probability,
        probability=probability,
        result_label=label,
        notes=req.notes or "",
    )
    assessment_id = create_document("assessment", assessment)

    return {
        "assessment_id": assessment_id,
        "probability": probability,
        "label": label,
    }


@app.get("/patient/assessments")
async def patient_assessments(token: str):
    sessions = get_documents("session", {"token": token}, limit=1)
    if not sessions or sessions[0].get("role") != "patient":
        raise HTTPException(401, "Unauthorized")
    pid = sessions[0].get("user_id")
    items = get_documents("assessment", {"patient_id": pid})
    return items


@app.get("/doctor/assessments")
async def doctor_assessments(token: str):
    sessions = get_documents("session", {"token": token}, limit=1)
    if not sessions or sessions[0].get("role") != "doctor":
        raise HTTPException(401, "Unauthorized")
    items = get_documents("assessment", {})
    return items


@app.post("/doctor/feedback")
async def doctor_feedback(req: FeedbackRequest, token: str):
    sessions = get_documents("session", {"token": token}, limit=1)
    if not sessions or sessions[0].get("role") != "doctor":
        raise HTTPException(401, "Unauthorized")
    doctor_id = sessions[0].get("user_id")

    fb = Feedback(
        doctor_id=doctor_id,
        assessment_id=req.assessment_id,
        message=req.message,
        severity=req.severity,
        recommendations=req.recommendations or [],
    )
    feedback_id = create_document("feedback", fb)
    return {"feedback_id": feedback_id}


@app.get("/schema")
async def get_schema_info():
    # Simple way to reveal available collections to the built-in DB viewer
    return {
        "collections": [
            "patient",
            "doctor",
            "assessment",
            "feedback",
            "session",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
