from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, Depends
import importlib
import inspect
from fastapi import FastAPI, HTTPException, Depends, Form
from pydantic import BaseModel,Field
from typing import Dict, List, Optional
from datetime import datetime
import motor.motor_asyncio
import uuid
from fastapi.middleware.cors import CORSMiddleware
import json
import random
import uvicorn
import re
import os
from dotenv import load_dotenv
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import time
from google import genai
import base64
from models import (
    Message, ChatSession, ChatResponse, ChatReport, BotConfig, BotConfigAnalyser,
    QuestionScenarioDoc, ParaphrasedQuestionCache, QuestionSession)
class MongoDB:
    def __init__(self,MONGO_URL,DATABASE_NAME):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        self.db = self.client[DATABASE_NAME]
        self.sessions = self.db.sessions
        self.analysis=self.db.analysis
        self.bot_configs=self.db.bot_configs
        self.bot_configs_analyser=self.db.bot_configs_analyser
        self.question_scenarios = self.db.question_scenarios
        self.question_chat_sessions = self.db.question_chat_sessions  
        self.paraphrased_questions = self.db.paraphrased_questions
        
    async def create_session(self, session: ChatSession) -> str:
        await self.sessions.insert_one(session.dict())
        return session.session_id
    async def create_conversation_analysis(self,report:ChatReport) -> str:
        await self.analysis.insert_one(report.dict())
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        session_data = await self.sessions.find_one({"session_id": session_id})
        if session_data:
            return ChatSession(**session_data)
        return None
    async def get_session_raw(self, session_id: str) -> Optional[ChatSession]:
        session_data = await self.sessions.find_one({"session_id": session_id})
        if session_data:
            return session_data
        return None
    async def get_session_analysis(self, session_id: str) -> Optional[ChatSession]:
        session_data = await self.analysis.find_one({"session_id": session_id})
        if session_data:
            return ChatReport(**session_data)
        return None
    async def update_session(self, session: ChatSession):
        session.last_updated = datetime.now()
        await self.sessions.update_one(
            {"session_id": session.session_id},
            {"$set": session.dict()}
        )
    async def create_bot(self,bot_config:BotConfig):
        bot= await self.bot_configs.insert_one(bot_config.dict())
        if bot :
            return bot
        else:
            HTTPException(status_code=400,detail="Error creating Bot")
    async def create_bot_analyser(self,bot_config:BotConfigAnalyser):
        bot= await self.bot_configs_analyser.insert_one(bot_config.dict())
        if bot :
            return bot
        else:
            HTTPException(status_code=400,detail="Error creating Bot")
    async def get_scenario_questions(self, scenario_name: str) -> List[Dict]:
        """Get all questions for a scenario from database"""
        try:
            scenario = await self.question_scenarios.find_one({
                "scenario_name": scenario_name,
                "is_active": True
            })
            return scenario["questions"] if scenario else []
        except Exception as e:
            print(f"Error getting scenario questions: {e}")
            return []

    async def get_scenario_context(self, scenario_name: str) -> str:
        """Get scenario context for LLM prompts"""
        try:
            scenario = await self.question_scenarios.find_one({
                "scenario_name": scenario_name,
                "is_active": True
            })
            return scenario["scenario_context"] if scenario else ""
        except Exception as e:
            print(f"Error getting scenario context: {e}")
            return ""

    async def save_question_session(self, session: QuestionSession):
        """Save or update question session"""
        try:
            session.last_updated = datetime.now()
            await self.question_chat_sessions.update_one(
                {"session_id": session.session_id},
                {"$set": session.dict()},
                upsert=True
            )
        except Exception as e:
            print(f"Error saving question session: {e}")
            raise

    async def get_question_session(self, session_id: str) -> Optional[QuestionSession]:
        """Get question session by ID"""
        try:
            session_data = await self.question_chat_sessions.find_one({"session_id": session_id})
            return QuestionSession(**session_data) if session_data else None
        except Exception as e:
            print(f"Error getting question session: {e}")
            return None

    async def get_question_session_by_conversation(self, conversation_history: List[Message]) -> Optional[QuestionSession]:
        """Get session by analyzing conversation history"""
        try:
            # Look for session markers in conversation or create logic to identify session
            # For now, get the most recent session for this scenario
            if len(conversation_history) >= 2:
                # Try to find session based on recent messages
                recent_sessions = await self.question_chat_sessions.find().sort("last_updated", -1).limit(5).to_list(length=None)
                
                for session_data in recent_sessions:
                    session = QuestionSession(**session_data)
                    # Match based on conversation similarity or other logic
                    if not session.is_completed:
                        return session
            
            return None
        except Exception as e:
            print(f"Error getting session by conversation: {e}")
            return None

    async def create_question_scenario(self, scenario: QuestionScenarioDoc) -> str:
        """Create new question scenario"""
        try:
            result = await self.question_scenarios.insert_one(scenario.dict())
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating question scenario: {e}")
            raise

    async def get_paraphrased_question(self, original_question_id: str, scenario_name: str, difficulty: str) -> Optional[Dict]:
        """Get cached paraphrased question"""
        try:
            cached = await self.paraphrased_questions.find_one({
                "original_question_id": original_question_id,
                "scenario_name": scenario_name,
                "difficulty": difficulty,
                "is_active": True
            })
            return cached["paraphrased_data"] if cached else None
        except Exception as e:
            print(f"Error getting paraphrased question: {e} hererer")
            return None

    async def save_paraphrased_question(self, cache_doc: ParaphrasedQuestionCache):
        """Save paraphrased question to cache"""
        try:
            await self.paraphrased_questions.insert_one(cache_doc.dict())
        except Exception as e:
            print(f"Error saving paraphrased question: {e}")
            raise

    async def delete_scenario_paraphrases(self, scenario_name: str, difficulty: Optional[str] = None):
        """Delete paraphrased questions for regeneration"""
        try:
            query = {"scenario_name": scenario_name}
            if difficulty:
                query["difficulty"] = difficulty
                
            result = await self.paraphrased_questions.delete_many(query)
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting paraphrases: {e}")
            return 0

    async def get_session_analytics(self, scenario_name: str, days: int = 7) -> Dict:
        """Get basic analytics for question sessions"""
        try:
            from datetime import timedelta
            start_date = datetime.now() - timedelta(days=days)
            
            sessions = await self.question_chat_sessions.find({
                "scenario_name": scenario_name,
                "created_at": {"$gte": start_date}
            }).to_list(length=None)
            
            total_sessions = len(sessions)
            completed_sessions = [s for s in sessions if s.get("is_completed", False)]
            
            return {
                "total_sessions": total_sessions,
                "completed_sessions": len(completed_sessions),
                "completion_rate": len(completed_sessions) / total_sessions * 100 if total_sessions > 0 else 0,
                "average_score": sum(s["score"] for s in completed_sessions) / len(completed_sessions) if completed_sessions else 0
            }
        except Exception as e:
            print(f"Error getting session analytics: {e}")
            return {"error": str(e)}

    # Index creation for performance
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Question scenarios indexes
            await self.question_scenarios.create_index("scenario_name")
            await self.question_scenarios.create_index([("scenario_name", 1), ("is_active", 1)])
            
            # Question sessions indexes  
            await self.question_chat_sessions.create_index("session_id")
            await self.question_chat_sessions.create_index("scenario_name")
            await self.question_chat_sessions.create_index([("created_at", -1)])
            
            # Paraphrased questions indexes
            await self.paraphrased_questions.create_index([
                ("original_question_id", 1), 
                ("scenario_name", 1), 
                ("difficulty", 1)
            ])
            await self.paraphrased_questions.create_index([("scenario_name", 1), ("difficulty", 1)])
            
            print("Database indexes created successfully")
            
        except Exception as e:
            print(f"Error creating indexes: {e}")

    # Data validation helpers
    def validate_question_structure(question: Dict) -> bool:
        """Validate that question has required structure"""
        required_fields = ["id", "question_text", "options", "correct_answer", "explanation"]
        
        for field in required_fields:
            if field not in question:
                return False
        
        # Validate options
        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            return False
        
        # Validate correct answer
        if question["correct_answer"] not in ["A", "B", "C", "D"]:
            return False
        
        # Validate explanation structure
        explanation = question["explanation"]
        if not isinstance(explanation, dict):
            return False
        
        if "correct_explanation" not in explanation:
            return False
        
        return True

    def validate_scenario_data(scenario_data: Dict) -> List[str]:
        """Validate complete scenario data and return list of issues"""
        issues = []
        
        required_fields = ["scenario_name", "questions"]
        for field in required_fields:
            if field not in scenario_data:
                issues.append(f"Missing required field: {field}")
        
        # if "questions" in scenario_data:
        #     questions = scenario_data["questions"]
        #     if not isinstance(questions, list):
        #         issues.append("Questions must be a list")
        #     else:
        #         for i, question in enumerate(questions):
        #             if not validate_question_structure(question):
        #                 issues.append(f"Invalid question structure at index {i}")
        
        # return issues

        