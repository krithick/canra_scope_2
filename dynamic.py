from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, Depends
import importlib
import inspect
from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
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

import base64
from mongo import MongoDB
from factory import DynamicBotFactory
from models import (
    Message, ChatSession, ChatResponse, ChatReport, BotConfig, BotConfigAnalyser,
    QuestionScenarioDoc, ParaphrasedQuestionCache, QuestionSession)
from question_bot import QuestionBot
load_dotenv('.env')
# MongoDB configuration
MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
print(MONGO_URL,DATABASE_NAME)



        

 

# FastAPI Application Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = MongoDB(MONGO_URL,DATABASE_NAME)
@app.get("/api/check")
async def say_hi():
    return {"message": "hi"}	
# Create bot factory
bot_factory = DynamicBotFactory(
    mongodb_uri=os.getenv("MONGO_URL"), 
    database_name=os.getenv("DATABASE_NAME")
)
bot_factory_analyser = DynamicBotFactory(
    mongodb_uri=os.getenv("MONGO_URL"), 
    database_name=os.getenv("DATABASE_NAME")
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize bots when application starts
    """
    await bot_factory.initialize_bots()
    await bot_factory_analyser.initialize_bots_analyser()
# Dependency to get database
async def get_db():
    return db
def replace_name(original_text, your_name,replace):
    if replace in original_text:
        return original_text.replace(replace, your_name)
    return original_text

@app.post("/api/chat")
async def chat(message: str = Form(...),
    session_id: Optional[str] = Form(default=None),
    scenario_name: Optional[str] = Form(default=None),
    name: Optional[str] = Form(default=None),
    spouse_name: Optional[str] = Form(default=None),
    db: MongoDB = Depends(get_db)
):
    """
    Chat endpoint that dynamically routes to appropriate bot
    
    :param bot_id: ID of the bot to use
    :param conversation_history: Conversation to get response for
    :return: Bot's response
    """

    if not session_id:
        if not scenario_name:
            raise HTTPException(status_code=400,detail="scenario_name is required for new sessions")
        session = ChatSession(
            extra=str(uuid.uuid4()),
            _id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            scenario_name=scenario_name,
            conversation_history=[]
        )
        await db.create_session(session)
        
    else:
        session= await db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=400,detail="Session not found")
    bot = await bot_factory.get_bot(session.scenario_name)
    scenario_name = session.scenario_name
    new_message= Message(role=f"{bot.bot_role_alt}",content=message)
    session.conversation_history.append(new_message)
    response=await bot.get_farmer_response(
            message,session.scenario_name,session.conversation_history
        )
    updated_message = replace_name(response, name,"[Your Name]") if name else response
    updated_message = replace_name(updated_message, spouse_name,"[Spouse Name]") if spouse_name else updated_message
    bot_message = Message(role=f"{bot.bot_role}", content=updated_message)
    session.conversation_history.append(bot_message)
    await db.update_session(session)

    # Parse response for answer and emotion
    result = re.split(r'\$(.*?)\$', updated_message)
    if len(result) >= 3:
        emotion = result[1]
        answer = result[2]
    else:
        emotion = "neutral"  # Default emotion if parsing fails
        answer = updated_message  # Use full response as answer if parsing fails
    if "[FINISH]"  in updated_message: 
        complete=True
        answer= answer.replace("[FINISH]", " ")    
    else:
        complete=False
    return ChatResponse(
        session_id=session.session_id,
        response=answer,
        emotion=emotion,
        complete=complete,
        conversation_history=session.conversation_history
    )

@app.put("/api/bots/{bot_id}")
async def update_bot(bot_id: str, update_data: Dict):
    """
    Update bot configuration
    
    :param bot_id: Bot ID to update
    :param update_data: Configuration update details
    :return: Update confirmation
    """
    await bot_factory.update_bot_config(bot_id, update_data)
    return {"message": "Bot configuration updated successfully"}

@app.get("/api/available_bots")
async def get_available_bots():
    """
    Get list of available active bots
    
    :return: List of active bot IDs
    """
    print(list(bot_factory.bots.keys()),bot_factory.bots)
    return list(bot_factory.bots.keys())

@app.post("/api/createBot")
async def createBot(
    bot_name: str=Form(default=None),
    bot_description: str=Form(default=None),
    bot_role:str=Form(default=None),
    bot_role_alt:str=Form(default=None),
    system_prompt: str=Form(default=None),
    is_active: bool = Form(default=True),
    bot_class: Optional[str] = Form(default=None),
    llm_model: str=Form(default='gemini-1.5-flash-002')):
 
                  
    bot_ = BotConfig(bot_id=str(uuid.uuid4()),
                    bot_name=bot_name,
                    bot_description=bot_description,
                    bot_role=bot_role,
                    bot_role_alt=bot_role_alt,
                    system_prompt=system_prompt,
                    is_active=is_active,
                    bot_class=bot_class,
                  
                    llm_model=llm_model)
    await db.create_bot(bot_)
    # await bot_factory.create_bot(bot_)
    await bot_factory.initialize_bots()
    return bot_
    
@app.post("/api/createBotAnalyser")
async def createBotAnalyser(
    bot_name: str=Form(default=None),
    bot_description: str=Form(default=None),
    bot_schema:str=Form(default=None),
    system_prompt: str=Form(default=None),
    is_active: bool = Form(default=True),
    llm_model: str=Form(default='gemini-1.5-flash-002')):
 
    test=json.loads(bot_schema)
    bot_ = BotConfigAnalyser(bot_id=str(uuid.uuid4()),
                    bot_name=bot_name,
                    bot_description=bot_description,
                    bot_schema=test,
                    system_prompt=system_prompt,
                    is_active=is_active,
                    llm_model=llm_model)
    await db.create_bot_analyser(bot_)
    # await bot_factory.create_bot(bot_)
    await bot_factory_analyser.initialize_bots_analyser()
    return bot_
    
    
@app.get("/api/sessionAnalyser/{session_id}")
    
async def get_session_analysis(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    session2 = await db.get_session_raw(session_id)
    analysis= await db.get_session_analysis(session_id)
    if not session2:
        raise HTTPException(status_code=404, detail="Session not found")
    if not analysis:
    # Access the conversation_history
        conversation_history = session2['conversation_history']


        conversation = {"conversation_history":conversation_history}
        analyzer= await bot_factory_analyser.get_bot_analyser(session2['scenario_name'])
        print(analyzer)
        results = await analyzer.analyze_conversation(conversation)
        results['session_id']=session2['session_id']
        results['conversation_id']=str(uuid.uuid4())
        results['timestamp']=datetime.now()
        category_scores=results['category_scores']
        # results['overall_score']=category_scores['language_and_communication']+category_scores['product_knowledge']+category_scores['empathy_and_trust']+category_scores['process_clarity']+category_scores['product_suitability']
        report = ChatReport(**results)
        model= await db.create_conversation_analysis(report)
        return report
    return analysis

    
@app.get("/api/refreshBots")
async def refresh_bots():
    await bot_factory.initialize_bots()
    await bot_factory_analyser.initialize_bots_analyser()

#  ===== SCENARIO MANAGEMENT APIS =====

@app.post("/api/question-scenarios/create")
async def create_question_scenario(
    scenario_name: str = Form(...),
    scenario_description: str = Form(...),
    scenario_context: str = Form(..., description="Background context for LLM"),
    questions_json: str = Form(..., description="JSON array of questions"),
    competency_framework: str = Form(default="[]", description="JSON array of competencies"),
    db: MongoDB = Depends(get_db)
):
    """Create a new question scenario"""
    try:
        questions = json.loads(questions_json)
        competencies = json.loads(competency_framework)
        
        scenario = QuestionScenarioDoc(
            scenario_name=scenario_name,
            scenario_description=scenario_description,
            scenario_context=scenario_context,
            questions=questions,
            competency_framework=competencies
        )
        
        await db.question_scenarios.insert_one(scenario.dict())
        
        # Create corresponding bot config
        bot_config = BotConfig(
            bot_id=str(uuid.uuid4()),
            bot_name=f"{scenario_name} Question Bot",
            bot_description=scenario_name,  # This links to scenario
            bot_role="assistant",
            bot_role_alt="user",
            system_prompt=f"You are a training facilitator for {scenario_name}. Help learners understand leadership concepts through interactive questioning and explanation validation.",
            is_active=True,
            bot_class="QuestionBot",  # Specify our custom class
            llm_model="gemini-2.0-flash"
        )
        
        await db.create_bot(bot_config)
        await bot_factory.initialize_bots()  # Refresh bot factory
        
        return {
            "message": "Question scenario created successfully",
            "scenario_name": scenario_name,
            "total_questions": len(questions),
            "bot_config_id": bot_config.bot_id
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating scenario: {str(e)}")

@app.post("/api/question-scenarios/upload-from-json")
async def upload_scenario_from_json(
    file: UploadFile = File(...),
    db: MongoDB = Depends(get_db)
):
    """Upload scenario from existing JSON structure"""
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Extract from JSON structure
        training_data = data["training_scenarios"]
        metadata = training_data["scenario_metadata"]
        
        # Convert JSON to our format
        all_questions = []
        for level_name, level_data in training_data["question_sets"].items():
            for q in level_data["questions"]:
                # Add difficulty marker to question
                q["source_difficulty"] = "easy" if "basic" in level_name.lower() else "hard"
                all_questions.append(q)
        
        scenario = QuestionScenarioDoc(
            scenario_name=metadata["scenario_name"],
            scenario_description=metadata["scenario_description"],
            scenario_context=metadata["scenario_description"],
            questions=all_questions,
            competency_framework=training_data["metadata"]["competency_framework"]
        )
        
        await db.question_scenarios.insert_one(scenario.dict())
        
        # Create bot config
        bot_config = BotConfig(
            bot_id=str(uuid.uuid4()),
            bot_name=f"{metadata['scenario_name']} Question Bot",
            bot_description=metadata["scenario_name"],
            bot_role="assistant",
            bot_role_alt="user",
            system_prompt=f"""You are an expert trainer for {metadata['scenario_name']}. 
            
Your role is to:
1. Guide learners through questions with paraphrased versions
2. Validate their explanations when they answer correctly
3. Provide constructive feedback to build leadership skills
4. Maintain engaging, educational conversation

Context: {metadata['scenario_description']}""",
            is_active=True,
            bot_class="QuestionBot",
            llm_model="gemini-2.0-flash"
        )
        
        await db.create_bot(bot_config)
        await bot_factory.initialize_bots()
        
        return {
            "message": "Scenario uploaded successfully from JSON",
            "scenario_name": metadata["scenario_name"],
            "total_questions": len(all_questions),
            "file_name": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading scenario: {str(e)}")

@app.get("/api/question-scenarios/list")
async def list_question_scenarios(db: MongoDB = Depends(get_db)):
    """List all available question scenarios"""
    try:
        scenarios = await db.question_scenarios.find(
            {"is_active": True}, 
            {"scenario_name": 1, "scenario_description": 1, "questions": 1}
        ).to_list(length=None)
        
        return [{
            "scenario_name": s["scenario_name"],
            "description": s["scenario_description"],
            "question_count": len(s.get("questions", [])),
            "available_difficulties": ["easy", "hard"]
        } for s in scenarios]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing scenarios: {str(e)}")

@app.get("/api/question-scenarios/{scenario_name}/questions")
async def get_scenario_questions(
    scenario_name: str,
    include_paraphrases: bool = False,
    db: MongoDB = Depends(get_db)
):
    """Get all questions for a scenario"""
    try:
        scenario = await db.question_scenarios.find_one({"scenario_name": scenario_name})
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        response_data = {
            "scenario_name": scenario_name,
            "description": scenario["scenario_description"],
            "total_questions": len(scenario["questions"]),
            "questions": scenario["questions"]
        }
        
        if include_paraphrases:
            # Get paraphrases for each question
            paraphrases = await db.paraphrased_questions.find({
                "scenario_name": scenario_name
            }).to_list(length=None)
            
            # Group paraphrases by question_id and difficulty
            paraphrase_map = {}
            for p in paraphrases:
                q_id = p["original_question_id"]
                difficulty = p["difficulty"]
                if q_id not in paraphrase_map:
                    paraphrase_map[q_id] = {}
                paraphrase_map[q_id][difficulty] = p["paraphrased_data"]
            
            response_data["paraphrases"] = paraphrase_map
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting questions: {str(e)}")

# ===== PARAPHRASING MANAGEMENT =====

@app.post("/api/question-scenarios/{scenario_name}/generate-paraphrases")
async def generate_all_paraphrases(
    scenario_name: str,
    difficulty: str = Form(..., description="easy or hard"),
    force_regenerate: bool = Form(default=False),
    db: MongoDB = Depends(get_db)
):
    """Generate paraphrased versions for all questions in a scenario"""
    try:
        bot = await bot_factory.get_bot(scenario_name)
        if not hasattr(bot, '_paraphrase_with_llm'):
            raise HTTPException(status_code=400, detail="Bot doesn't support question paraphrasing")
        
        generated_count = 0
        errors = []
        
        for question in bot.scenario_questions:
            try:
                # Check if exists
                if not force_regenerate:
                    existing = await db.paraphrased_questions.find_one({
                        "original_question_id": question["id"],
                        "scenario_name": scenario_name,
                        "difficulty": difficulty
                    })
                    if existing:
                        continue
                
                # Generate paraphrase
                paraphrased = await bot._paraphrase_with_llm(question, difficulty)
                
                # Save to cache
                cache_doc = ParaphrasedQuestionCache(
                    original_question_id=question["id"],
                    scenario_name=scenario_name,
                    difficulty=difficulty,
                    paraphrased_data=paraphrased
                )
                await db.paraphrased_questions.insert_one(cache_doc.dict())
                generated_count += 1
                
            except Exception as e:
                errors.append(f"Question {question['id']}: {str(e)}")
        
        return {
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "generated_count": generated_count,
            "total_questions": len(bot.scenario_questions),
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating paraphrases: {str(e)}")

@app.get("/api/questions/{scenario_name}/paraphrases")
async def get_scenario_paraphrases(
    scenario_name: str,
    difficulty: str = "easy",
    db: MongoDB = Depends(get_db)
):
    """Get all paraphrased questions for a scenario"""
    try:
        paraphrases = await db.paraphrased_questions.find({
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "is_active": True
        }).to_list(length=None)
        
        return {
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "paraphrases": paraphrases
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting paraphrases: {str(e)}")

# ===== CORE QUESTION SESSION APIS =====

@app.post("/api/question-sessions/start")
async def start_question_session(
    scenario_name: str = Form(..., description="Scenario name to start"),
    difficulty: str = Form(default="easy", description="easy or hard"),
    user_id: Optional[str] = Form(default=None),
    db: MongoDB = Depends(get_db)
):
    """Start a new question training session"""
    try:
        # Get the QuestionBot for this scenario
        bot = await bot_factory.get_bot(scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Not a question bot scenario")
        
        # Start new session using the bot
        result = await bot.start_new_session(difficulty)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/api/question-sessions/{session_id}/answer")
async def submit_question_answer(
    session_id: str,
    user_input: str = Form(..., description="Speech text or option letter"),
    time_taken: int = Form(default=30, description="Time taken in seconds"),
    is_timeout: bool = Form(default=False, description="Whether this is a timeout submission"),
    db: MongoDB = Depends(get_db)
):
    """Submit answer to current question (speech or timeout)"""
    try:
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the appropriate bot
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Submit answer through bot
        result = await bot.submit_answer(session_id, user_input, time_taken, is_timeout)
        
        return {
            "success": True,
            "data": result
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting answer: {str(e)}")

@app.post("/api/question-sessions/{session_id}/timeout")
async def handle_question_timeout(
    session_id: str,
    time_taken: int = Form(default=30, description="Time taken before timeout"),
    db: MongoDB = Depends(get_db)
):
    """Handle question timeout - show correct answer and move to next"""
    try:
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the appropriate bot
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Handle timeout through bot
        result = await bot.submit_answer(session_id, "", time_taken, is_timeout=True)
        
        return {
            "success": True,
            "data": result
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling timeout: {str(e)}")

@app.post("/api/question-sessions/{session_id}/explain")  
async def submit_explanation(
    session_id: str,
    explanation: str = Form(..., description="User's explanation of their answer choice"),
    db: MongoDB = Depends(get_db)
):
    """Submit and validate user's explanation"""
    try:
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Get the appropriate bot
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Submit explanation through bot
        result = await bot.submit_explanation(session_id, explanation)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing explanation: {str(e)}")

@app.get("/api/question-sessions/{session_id}/results")
async def get_session_final_results(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get final results and analysis for completed session"""
    try:
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the appropriate bot
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Get results through bot
        result = await bot.get_session_results(session_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.get("/api/question-sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get current session status and progress"""
    try:
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        current_question = None
        if (session.current_question_index < len(session.paraphrased_questions_used) and 
            not session.is_completed):
            current_question = session.paraphrased_questions_used[session.current_question_index]
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "current_question": current_question,
                "question_number": session.current_question_index + 1,
                "total_questions": session.total_questions,
                "current_score": session.score,
                "state": session.current_state,
                "is_completed": session.is_completed,
                "difficulty": session.difficulty,
                "scenario_name": session.scenario_name
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

# ===== ANALYTICS =====

@app.get("/api/analytics/question-sessions/summary")
async def get_question_sessions_summary(
    scenario_name: Optional[str] = None,
    days: int = 7,
    db: MongoDB = Depends(get_db)
):
    """Get summary analytics for question sessions"""
    try:
        from datetime import timedelta
        start_date = datetime.now() - timedelta(days=days)
        
        # Build query
        query = {"created_at": {"$gte": start_date}}
        if scenario_name:
            query["scenario_name"] = scenario_name
        
        sessions = await db.question_chat_sessions.find(query).to_list(length=None)
        
        if not sessions:
            return {"message": "No sessions found", "total_sessions": 0}
        
        # Calculate analytics
        total_sessions = len(sessions)
        completed_sessions = [s for s in sessions if s.get("is_completed", False)]
        completion_rate = len(completed_sessions) / total_sessions * 100 if total_sessions > 0 else 0
        
        # Score analytics
        if completed_sessions:
            scores = [(s["score"] / s["total_questions"]) * 100 for s in completed_sessions if s["total_questions"] > 0]
            avg_score = sum(scores) / len(scores) if scores else 0
        else:
            avg_score = 0
        
        # Difficulty breakdown
        easy_sessions = len([s for s in sessions if s.get("difficulty") == "easy"])
        hard_sessions = len([s for s in sessions if s.get("difficulty") == "hard"])
        
        return {
            "period_days": days,
            "scenario_name": scenario_name or "All scenarios",
            "summary": {
                "total_sessions": total_sessions,
                "completed_sessions": len(completed_sessions),
                "completion_rate_percent": round(completion_rate, 1),
                "average_score_percent": round(avg_score, 1)
            },
            "difficulty_breakdown": {
                "easy": easy_sessions,
                "hard": hard_sessions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

# ===== BOT MANAGEMENT (Minimal) =====

@app.get("/api/available-scenarios")
async def get_available_scenarios():
    """Get list of available question scenarios"""
    try:
        bots = list(bot_factory.bots.keys())
        return {
            "available_scenarios": bots,
            "total_count": len(bots)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scenarios: {str(e)}")

@app.post("/api/refresh-bots")
async def refresh_question_bots():
    """Refresh bot factory - reload all question bots"""
    try:
        await bot_factory.initialize_bots()
        return {
            "message": "Question bots refreshed successfully",
            "loaded_scenarios": list(bot_factory.bots.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing bots: {str(e)}")