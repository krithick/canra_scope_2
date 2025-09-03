# Question System Models - Add these to your existing models

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any 
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
import uuid
# FastAPI Models
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()

class ChatSession(BaseModel):
    _id:str
    extra:str
    session_id: str
    scenario_name: str
    conversation_history: List[Message]
    created_at: datetime = datetime.now()
    last_updated: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str = Form(...)
    session_id: Optional[str] = Form(default=None)
    scenario_name: Optional[str] = Form(default=None)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    emotion:str
    complete:bool
    conversation_history: List[Message]

class ChatReport(BaseModel):
    session_id: str
    conversation_id: str
    timestamp: datetime = datetime.now()
    overall_score: float
    category_scores: Dict[str, float]
    detailed_feedback: Dict[str, List[str]]
    recommendations: List[str]
    
class BotConfig(BaseModel):
    """
    Represents the configuration for a bot stored in MongoDB
    """
    bot_id: str
    bot_name: str
    bot_description: str
    bot_role:str
    bot_role_alt:str
    system_prompt: str
    is_active: bool = True
    bot_class: Optional[str] = None
    llm_model: str

class BotConfigAnalyser(BaseModel):
    """
    Represents the configuration for a analyser bot stored in MongoDB
    """
    bot_id: str
    bot_name: str
    bot_description: str
    bot_schema:dict
    system_prompt:str
    is_active: bool = True
    llm_model:str
# 

class QuestionScenarioDoc(BaseModel):
    """MongoDB document for storing question sets"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str  # Must match bot_description in BotConfig
    scenario_description: str
    scenario_context: str  # Background context for LLM prompts
    questions: List[Dict]  # Your existing question JSON structure
    competency_framework: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

class ParaphrasingRequest(BaseModel):
    """Request model for creating question paraphrases"""
    scenario_name: str
    difficulty_level: str  # "easy" or "hard"
    question_ids: Optional[List[str]] = None  # If None, paraphrase all questions
    force_regenerate: bool = False  # Regenerate even if paraphrases exist

class QuestionAttemptRecord(BaseModel):
    question_id: str
    original_question: Dict  # Original question from database
    paraphrased_question: Dict  # What user actually saw (shuffled options)
    user_answer: str  # A, B, C, D (what they selected)
    user_answer_text: str  # Actual text of the option they selected
    correct_answer_original: str  # From original question (A, B, C, D)
    correct_answer_paraphrased: str  # What was correct in shuffled version
    is_correct: bool
    user_explanation: Optional[str] = None
    explanation_validation: Optional[Dict] = None
    ai_feedback: str = ""
    time_taken_seconds: int = 30
    timestamp: datetime = Field(default_factory=datetime.now)
class ParaphrasedQuestionCache(BaseModel):
    """Cache for paraphrased questions to avoid regeneration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_question_id: str
    scenario_name: str
    difficulty: str  # "easy" or "hard"
    paraphrased_data: Dict  # Contains paraphrased question and options
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
# class QuestionSession(BaseModel):
#     """Main session model for conversational question training"""
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     scenario_name: str
#     difficulty: str  # "easy" or "hard"
#     user_id: Optional[str] = None
    
#     # Core data
#     questions_data: List[Dict] = []  # Original questions from DB
#     current_question_index: int = 0
#     conversation_history: List[Message] = []  # Reuse existing Message model
    
#     # Progress tracking
#     question_attempts: List[Dict] = []  # Each attempt with answer, explanation, validation
#     current_state: str = "start"  # start, awaiting_answer, awaiting_explanation, completed
#     score: int = 0
#     total_questions: int = 0
    
#     # Session metadata
#     is_completed: bool = False
#     created_at: datetime = Field(default_factory=datetime.now)
#     last_updated: datetime = Field(default_factory=datetime.now)
#     conversation_history: List[Message] = []  # SAVE ALL CONVERSATION
#     question_attempts: List[QuestionAttemptRecord] = []  # DETAILED TRACKING
#     paraphrased_questions_used: List[Dict] = []  # What user actually saw
class QuestionSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str
    difficulty: str  # "easy" or "hard"
    user_id: Optional[str] = None
    
    # Core data
    questions_data: List[Dict] = []  # Original questions from DB
    paraphrased_questions_used: List[Dict] = []  # What user actually saw
    current_question_index: int = 0
    conversation_history: List[Message] = []  # FULL CONVERSATION TRACKING
    
    # Progress tracking - ENHANCED
    question_attempts: List[Dict] = []  # DETAILED TRACKING
    current_state: str = "start"  # start, awaiting_answer, awaiting_explanation, completed
    score: int = 0
    total_questions: int = 0
    
    # Session metadata
    is_completed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)