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

@app.post("/llm/chat")
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

@app.put("/llm/bots/{bot_id}")
async def update_bot(bot_id: str, update_data: Dict):
    """
    Update bot configuration
    
    :param bot_id: Bot ID to update
    :param update_data: Configuration update details
    :return: Update confirmation
    """
    await bot_factory.update_bot_config(bot_id, update_data)
    return {"message": "Bot configuration updated successfully"}

@app.get("/llm/available_bots")
async def get_available_bots():
    """
    Get list of available active bots
    
    :return: List of active bot IDs
    """
    print(list(bot_factory.bots.keys()),bot_factory.bots)
    return list(bot_factory.bots.keys())

@app.post("/llm/createBot")
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
    
@app.post("/llm/createBotAnalyser")
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
    
    
@app.get("/llm/sessionAnalyser/{session_id}")
    
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

    
@app.get("/llm/refreshBots")
async def refresh_bots():
    await bot_factory.initialize_bots()
    await bot_factory_analyser.initialize_bots_analyser()

# ===== SCENARIO MANAGEMENT APIS =====

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
    """Upload scenario from your existing JSON structure"""
    print('hiii')
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Extract from your JSON structure
        training_data = data["training_scenarios"]
        metadata = training_data["scenario_metadata"]
        
        # Convert your JSON to our format
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

# ===== PARAPHRASING MANAGEMENT =====

@app.post("/api/questions/generate-paraphrases")
async def generate_question_paraphrases(
    scenario_name: str = Form(...),
    difficulty: str = Form(..., description="easy or hard"),
    force_regenerate: bool = Form(default=False),
    db: MongoDB = Depends(get_db)
):
    """Generate paraphrased versions of all questions in a scenario"""
    try:
        # Get the bot for this scenario
        bot = await bot_factory.get_bot(scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Not a question bot scenario")
        
        results = []
        for question in bot.scenario_questions:
            # Check if paraphrase already exists
            if not force_regenerate:
                existing = await db.paraphrased_questions.find_one({
                    "original_question_id": question["id"],
                    "scenario_name": scenario_name,
                    "difficulty": difficulty
                })
                if existing:
                    results.append({"question_id": question["id"], "status": "already_exists"})
                    continue
            
            # Generate new paraphrase
            try:
                paraphrased = await bot._paraphrase_with_llm(question, difficulty)
                
                # Cache the result
                cache_doc = ParaphrasedQuestionCache(
                    original_question_id=question["id"],
                    scenario_name=scenario_name,
                    difficulty=difficulty,
                    paraphrased_data=paraphrased
                )
                await db.paraphrased_questions.insert_one(cache_doc.dict())
                
                results.append({"question_id": question["id"], "status": "generated"})
                
            except Exception as e:
                results.append({"question_id": question["id"], "status": "error", "error": str(e)})
        
        return {
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "total_questions": len(bot.scenario_questions),
            "results": results,
            "summary": {
                "generated": len([r for r in results if r["status"] == "generated"]),
                "already_existed": len([r for r in results if r["status"] == "already_exists"]),
                "errors": len([r for r in results if r["status"] == "error"])
            }
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

# ===== SESSION ANALYTICS =====

@app.get("/api/question-sessions/{session_id}/progress")
async def get_question_session_progress(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get current progress of a question session"""
    try:
        session_data = await db.question_chat_sessions.find_one({"session_id": session_id})
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = QuestionSession(**session_data)
        
        return {
            "session_id": session_id,
            "scenario_name": session.scenario_name,
            "difficulty": session.difficulty,
            "progress": {
                "current_question": session.current_question_index + 1,
                "total_questions": session.total_questions,
                "current_score": session.score,
                "percentage": round((session.score / max(1, session.current_question_index)) * 100, 1),
                "state": session.current_state,
                "is_completed": session.is_completed
            },
            "attempts": session.question_attempts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")

@app.get("/api/question-sessions/{session_id}/conversation")
async def get_question_session_conversation(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get full conversation history for a question session"""
    try:
        session_data = await db.question_chat_sessions.find_one({"session_id": session_id})
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "conversation_history": session_data["conversation_history"],
            "question_attempts": session_data["question_attempts"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

# ===== UTILITY & TESTING =====

@app.post("/api/dev/setup-banking-scenario")
async def setup_banking_scenario_quick(db: MongoDB = Depends(get_db)):
    """Quick setup for testing - creates the banking scenario"""
    try:
        # Sample questions (you can replace with your full JSON)
        sample_questions = [
            {
                "id": "basic_q1",
                "question_number": 1,
                "category": "Crisis Recognition",
                "question_text": "What should be your first action when observing a customer argument?",
                "options": [
                    {"option_id": "A", "text": "Immediately intervene to understand the dispute"},
                    {"option_id": "B", "text": "Call security to control the situation"},
                    {"option_id": "C", "text": "Continue observing to assess the conflict"},
                    {"option_id": "D", "text": "Signal staff to move customer to private area"}
                ],
                "correct_answer": "A",
                "explanation": {
                    "correct_explanation": "Immediate intervention demonstrates proactive leadership and prevents escalation",
                    "incorrect_explanations": {
                        "B": "Calling security may escalate unnecessarily",
                        "C": "Continued observation allows situation to worsen",
                        "D": "Gesturing shows lack of direct leadership"
                    }
                },
                "competencies_tested": ["Crisis Management", "Leadership Presence"],
                "scenario_context": "Initial response to conflict",
                "source_difficulty": "easy"
            }
        ]
        
        scenario = QuestionScenarioDoc(
            scenario_name="Banking Leadership Questions",
            scenario_description="Leadership training for banking professionals",
            scenario_context="You are a branch manager dealing with customer service conflicts. Your decisions impact customer satisfaction, staff morale, and bank reputation.",
            questions=sample_questions,
            competency_framework=["Crisis Management", "Leadership Presence", "Communication"]
        )
        
        await db.question_scenarios.insert_one(scenario.dict())
        
        # Create bot config
        bot_config = BotConfig(
            bot_id=str(uuid.uuid4()),
            bot_name="Banking Leadership Question Bot",
            bot_description="Banking Leadership Questions",  # Must match scenario_name
            bot_role="assistant",
            bot_role_alt="user",
            system_prompt="""You are an expert banking leadership trainer. You guide learners through leadership scenarios using questions and explanations. You validate their understanding and provide constructive feedback to build their leadership skills.""",
            is_active=True,
            bot_class="QuestionBot",
            llm_model="gemini-2.0-flash"
        )
        
        await db.create_bot(bot_config)
        await bot_factory.initialize_bots()
        
        return {
            "message": "Banking scenario setup complete",
            "scenario_name": "Banking Leadership Questions",
            "sample_usage": {
                "start_session": "POST /llm/chat with message='start easy training' and scenario_name='Banking Leadership Questions'",
                "continue_chat": "Use normal /llm/chat endpoint to continue conversation"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up scenario: {str(e)}")

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
        print(bot)
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

# ===== SESSION ANALYTICS =====

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

# ===== TESTING & DEVELOPMENT =====

@app.post("/api/dev/test-question-flow")
async def test_question_flow(
    scenario_name: str = Form(default="Banking Leadership Questions"),
    difficulty: str = Form(default="easy"),
    db: MongoDB = Depends(get_db)
):
    """Test the complete question flow"""
    try:
        # Simulate conversation flow
        test_messages = [
            Message(role="user", content=f"start {difficulty} training"),
            Message(role="assistant", content="Starting session...")
        ]
        
        bot = await bot_factory.get_bot(scenario_name)
        response = await bot.get_farmer_response(f"start {difficulty} training", scenario_name, test_messages)
        
        return {
            "test_status": "success",
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "first_question_response": response,
            "next_steps": "Send 'A' or 'B' or 'C' or 'D' as next message to test answer flow"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing flow: {str(e)}")

# ===== DATABASE EXTENSIONS =====

# Missing Question Session APIs - Add these to your FastAPI application

# @app.post("/api/question-sessions/start")
# async def start_question_session(
#     scenario_id: str = Form(..., description="Scenario name to start"),
#     difficulty: str = Form(default="easy", description="easy or hard"),
#     user_id: Optional[str] = Form(default=None),
#     db: MongoDB = Depends(get_db)
# ):
#     """Start a new question training session"""
#     try:
#         # Create new session
#         session = QuestionSession(
#             scenario_name=scenario_id,
#             difficulty=difficulty,
#             user_id=user_id,
#             current_state="awaiting_answer"
#         )
        
#         # Load questions for this scenario
#         questions = await db.get_scenario_questions(scenario_id)
#         if not questions:
#             raise HTTPException(status_code=404, detail="No questions found for scenario")
            
#         session.questions_data = questions
#         session.total_questions = len(questions)
        
#         # Save session
#         await db.save_question_session(session)
        
#         # Get first question (paraphrased)
#         if len(questions) > 0:
#             bot = await bot_factory.get_bot(scenario_id)
#             if hasattr(bot, '_get_paraphrased_question'):
#                 first_question = await bot._get_paraphrased_question(questions[0], difficulty)
#             else:
#                 first_question = questions[0]
#         else:
#             first_question = None
        
#         return {
#             "session_id": session.session_id,
#             "scenario_name": session.scenario_name,
#             "difficulty": session.difficulty,
#             "total_questions": session.total_questions,
#             "current_question": first_question
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")
@app.post("/api/question-sessions/start")
async def start_question_session(
    scenario_id: str = Form(..., description="Scenario name to start"),
    difficulty: str = Form(default="easy", description="easy or hard"),
    user_id: Optional[str] = Form(default=None),
    db: MongoDB = Depends(get_db)
):
    """Start a new question training session"""
    try:
        # Get the QuestionBot for this scenario
        bot = await bot_factory.get_bot(scenario_id)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Not a question bot scenario")
        
        # Create new session using the bot
        session_id = str(uuid.uuid4())
        session = await bot._get_or_create_session(session_id, difficulty)
        
        # Get first question
        if len(session.paraphrased_questions_used) > 0:
            first_question = session.paraphrased_questions_used[0]
        else:
            raise HTTPException(status_code=500, detail="No questions available")
        
        return {
            "session_id": session.session_id,
            "scenario_name": session.scenario_name,
            "difficulty": session.difficulty,
            "total_questions": session.total_questions,
            "current_question": first_question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")
# @app.post("/api/question-sessions/{session_id}/answer")
# async def submit_question_answer(
#     session_id: str,
#     selected_option: str = Form(..., description="A, B, C, or D"),
#     time_taken: int = Form(default=30, description="Time taken in seconds"),
#     db: MongoDB = Depends(get_db)
# ):
#     """Submit answer to current question"""
#     try:
#         # Get session
#         session = await db.get_question_session(session_id)
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
            
#         if session.current_question_index >= len(session.questions_data):
#             raise HTTPException(status_code=400, detail="No more questions available")
            
#         current_q = session.questions_data[session.current_question_index]
#         is_correct = selected_option.upper() == current_q['correct_answer'].upper()
        
#         # Create attempt record
#         attempt = {
#             "question_id": current_q['id'],
#             "user_answer": selected_option.upper(),
#             "is_correct": is_correct,
#             "time_taken": time_taken,
#             "timestamp": datetime.now()
#         }
        
#         session.question_attempts.append(attempt)
        
#         if is_correct:
#             # Ask for explanation
#             session.current_state = "awaiting_explanation"
#             await db.save_question_session(session)
            
#             return {
#                 "is_correct": True,
#                 "feedback": f"Correct! You selected {selected_option.upper()}. Now please explain why this is the best choice.",
#                 "current_score": session.score,
#                 "total_questions": session.total_questions,
#                 "is_completed": False,
#                 "next_question_available": False,
#                 "awaiting_explanation": True
#             }
#         else:
#             # Move to next question
#             session.score += 0  # No points for incorrect
#             session.current_question_index += 1
            
#             # Check if completed
#             next_question = None
#             is_completed = session.current_question_index >= len(session.questions_data)
            
#             if not is_completed:
#                 # Get next paraphrased question
#                 bot = await bot_factory.get_bot(session.scenario_name)
#                 if hasattr(bot, '_get_paraphrased_question'):
#                     next_question = await bot._get_paraphrased_question(
#                         session.questions_data[session.current_question_index], 
#                         session.difficulty
#                     )
#                 else:
#                     next_question = session.questions_data[session.current_question_index]
#             else:
#                 session.is_completed = True
#                 session.current_state = "completed"
            
#             await db.save_question_session(session)
            
#             return {
#                 "is_correct": False,
#                 "correct_answer": current_q['correct_answer'],
#                 "explanation": current_q.get('explanation', {}),
#                 "feedback": f"The correct answer is {current_q['correct_answer']}. {current_q.get('explanation', {}).get('correct_explanation', '')}",
#                 "current_score": session.score,
#                 "total_questions": session.total_questions,
#                 "is_completed": is_completed,
#                 "next_question_available": not is_completed,
#                 "next_question": next_question
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error submitting answer: {str(e)}")
@app.post("/api/question-sessions/{session_id}/answer")
async def submit_question_answer(
    session_id: str,
    selected_option: str = Form(..., description="A, B, C, or D"),
    time_taken: int = Form(default=30, description="Time taken in seconds"),
    db: MongoDB = Depends(get_db)
):
    """Submit answer to current question"""
    try:
        # Get session
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        if session.current_question_index >= len(session.questions_data):
            raise HTTPException(status_code=400, detail="No more questions available")
        
        # Get the bot to process the answer
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Simulate conversation for the bot's answer processing
        fake_conversation = [
            Message(role="user", content=selected_option, timestamp=datetime.now())
        ]
        
        # Process through the bot
        response = await bot._handle_answer_submission(selected_option, fake_conversation)
        
        # Update session in database
        updated_session = await db.get_question_session(session_id)
        
        # Parse the response to determine next action
        is_correct = "Correct!" in response
        is_awaiting_explanation = updated_session.current_state == "awaiting_explanation"
        is_completed = updated_session.is_completed
        
        # Get next question if available and not awaiting explanation
        next_question = None
        if (not is_awaiting_explanation and not is_completed and 
            updated_session.current_question_index < len(updated_session.paraphrased_questions_used)):
            next_question = updated_session.paraphrased_questions_used[updated_session.current_question_index]
        
        return {
            "is_correct": is_correct,
            "feedback": response.replace("$encouraging$", "").replace("$educational$", "").replace("$neutral$", ""),
            "current_score": updated_session.score,
            "total_questions": updated_session.total_questions,
            "is_completed": is_completed,
            "next_question_available": not is_awaiting_explanation and not is_completed,
            "awaiting_explanation": is_awaiting_explanation,
            "next_question": next_question
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting answer: {str(e)}")
# @app.post("/api/question-sessions/{session_id}/explain")  
# async def submit_explanation(
#     session_id: str,
#     explanation: str = Form(..., description="User's explanation of their answer choice"),
#     db: MongoDB = Depends(get_db)
# ):
#     """Submit and validate user's explanation"""
#     try:
#         # Get session
#         session = await db.get_question_session(session_id)
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
            
#         if session.current_state != "awaiting_explanation":
#             raise HTTPException(status_code=400, detail="Not expecting explanation at this time")
            
#         current_q = session.questions_data[session.current_question_index]
        
#         # Validate explanation with LLM
#         bot = await bot_factory.get_bot(session.scenario_name)
#         if hasattr(bot, '_validate_explanation_with_llm'):
#             validation_result = await bot._validate_explanation_with_llm(current_q, explanation)
#         else:
#             # Fallback validation
#             validation_result = {
#                 "is_valid": True,
#                 "feedback": "Thank you for your explanation. Let's continue.",
#                 "key_points_covered": [],
#                 "missing_critical_points": []
#             }
        
#         # Update the last attempt with explanation
#         if session.question_attempts:
#             last_attempt = session.question_attempts[-1]
#             last_attempt["user_explanation"] = explanation
#             last_attempt["explanation_validation"] = validation_result
        
#         # Award point for correct answer + explanation
#         session.score += 1
#         session.current_question_index += 1
        
#         # Check if completed
#         next_question = None
#         is_completed = session.current_question_index >= len(session.questions_data)
        
#         if not is_completed:
#             # Get next paraphrased question
#             if hasattr(bot, '_get_paraphrased_question'):
#                 next_question = await bot._get_paraphrased_question(
#                     session.questions_data[session.current_question_index], 
#                     session.difficulty
#                 )
#             else:
#                 next_question = session.questions_data[session.current_question_index]
            
#             session.current_state = "awaiting_answer"
#         else:
#             session.is_completed = True
#             session.current_state = "completed"
        
#         await db.save_question_session(session)
        
#         return {
#             "explanation_valid": validation_result.get("is_valid", True),
#             "feedback": validation_result.get("feedback", "Good explanation!"),
#             "key_points_covered": validation_result.get("key_points_covered", []),
#             "missing_points": validation_result.get("missing_critical_points", []),
#             "current_score": session.score,
#             "total_questions": session.total_questions,
#             "is_completed": is_completed,
#             "next_question_available": not is_completed,
#             "next_question": next_question
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing explanation: {str(e)}")
@app.post("/api/question-sessions/{session_id}/explain")  
async def submit_explanation(
    session_id: str,
    explanation: str = Form(..., description="User's explanation of their answer choice"),
    db: MongoDB = Depends(get_db)
):
    """Submit and validate user's explanation"""
    try:
        # Get session
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        if session.current_state != "awaiting_explanation":
            raise HTTPException(status_code=400, detail="Not expecting explanation at this time")
        
        # Get the bot to process the explanation
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Simulate conversation for the bot's explanation processing
        fake_conversation = [
            Message(role="user", content=explanation, timestamp=datetime.now())
        ]
        
        # Process through the bot
        response = await bot._handle_explanation_submission(explanation, fake_conversation)
        
        # Get updated session
        updated_session = await db.get_question_session(session_id)
        
        # Get validation result from the last attempt
        validation_result = {}
        if updated_session.question_attempts:
            last_attempt = updated_session.question_attempts[-1]
            if hasattr(last_attempt, 'explanation_validation') and last_attempt.explanation_validation:
                validation_result = last_attempt.explanation_validation
        
        # Check if there's a next question
        next_question = None
        is_completed = updated_session.is_completed
        if (not is_completed and 
            updated_session.current_question_index < len(updated_session.paraphrased_questions_used)):
            next_question = updated_session.paraphrased_questions_used[updated_session.current_question_index]
        
        return {
            "explanation_valid": validation_result.get("is_valid", True),
            "feedback": response.replace("$proud$", "").replace("$educational$", "").replace("$neutral$", ""),
            "key_points_covered": validation_result.get("key_points_covered", []),
            "missing_points": validation_result.get("missing_critical_points", []),
            "current_score": updated_session.score,
            "total_questions": updated_session.total_questions,
            "is_completed": is_completed,
            "next_question_available": not is_completed,
            "next_question": next_question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing explanation: {str(e)}")

# @app.get("/api/question-sessions/{session_id}/current")
# async def get_current_question_state(
#     session_id: str,
#     db: MongoDB = Depends(get_db)
# ):
#     """Get current question and session state"""
#     try:
#         session = await db.get_question_session(session_id)
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
        
#         current_question = None
#         if (session.current_question_index < len(session.questions_data) and 
#             not session.is_completed):
            
#             current_q = session.questions_data[session.current_question_index]
            
#             # Get paraphrased version
#             bot = await bot_factory.get_bot(session.scenario_name)
#             if hasattr(bot, '_get_paraphrased_question'):
#                 current_question = await bot._get_paraphrased_question(current_q, session.difficulty)
#             else:
#                 current_question = current_q
        
#         return {
#             "session_id": session_id,
#             "current_question": current_question,
#             "question_number": session.current_question_index + 1,
#             "total_questions": session.total_questions,
#             "current_score": session.score,
#             "state": session.current_state,
#             "is_completed": session.is_completed,
#             "difficulty": session.difficulty
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting current state: {str(e)}")
@app.get("/api/question-sessions/{session_id}/current")
async def get_current_question_state(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get current question and session state"""
    try:
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        current_question = None
        if (session.current_question_index < len(session.paraphrased_questions_used) and 
            not session.is_completed):
            current_question = session.paraphrased_questions_used[session.current_question_index]
        
        return {
            "session_id": session_id,
            "current_question": current_question,
            "question_number": session.current_question_index + 1,
            "total_questions": session.total_questions,
            "current_score": session.score,
            "state": session.current_state,
            "is_completed": session.is_completed,
            "difficulty": session.difficulty
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current state: {str(e)}")
# @app.get("/api/question-sessions/{session_id}/results")
# async def get_session_final_results(
#     session_id: str,
#     db: MongoDB = Depends(get_db)
# ):
#     """Get final results and analysis for completed session"""
#     try:
#         session = await db.get_question_session(session_id)
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
            
#         if not session.is_completed:
#             raise HTTPException(status_code=400, detail="Session not yet completed")
        
#         # Calculate results
#         total_questions = session.total_questions
#         final_score = session.score
#         percentage = (final_score / total_questions * 100) if total_questions > 0 else 0
        
#         # Calculate competency performance
#         competency_scores = {}
#         for attempt in session.question_attempts:
#             question_id = attempt.get("question_id")
#             question = next((q for q in session.questions_data if q["id"] == question_id), None)
#             if question and attempt.get("is_correct"):
#                 competencies = question.get("competencies_tested", [])
#                 for comp in competencies:
#                     if comp not in competency_scores:
#                         competency_scores[comp] = {"correct": 0, "total": 0}
#                     competency_scores[comp]["correct"] += 1
#                     competency_scores[comp]["total"] += 1
#             elif question:
#                 competencies = question.get("competencies_tested", [])
#                 for comp in competencies:
#                     if comp not in competency_scores:
#                         competency_scores[comp] = {"correct": 0, "total": 0}
#                     competency_scores[comp]["total"] += 1
        
#         # Convert to percentages
#         competency_percentages = {
#             comp: (scores["correct"] / scores["total"] * 100) if scores["total"] > 0 else 0
#             for comp, scores in competency_scores.items()
#         }
        
#         return {
#             "session_id": session_id,
#             "scenario_name": session.scenario_name,
#             "difficulty": session.difficulty,
#             "final_score": final_score,
#             "total_questions": total_questions,
#             "percentage_score": round(percentage, 1),
#             "passed": percentage >= 70,
#             "excellence": percentage >= 85,
#             "competency_scores": competency_percentages,
#             "detailed_attempts": session.question_attempts,
#             "session_duration": (session.last_updated - session.created_at).total_seconds() / 60,  # minutes
#             "created_at": session.created_at,
#             "completed_at": session.last_updated
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")
@app.get("/api/question-sessions/{session_id}/results")
async def get_session_final_results(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get final results and analysis for completed session"""
    try:
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        if not session.is_completed:
            raise HTTPException(status_code=400, detail="Session not yet completed")
        
        # Calculate results
        total_questions = session.total_questions
        final_score = session.score
        percentage = (final_score / total_questions * 100) if total_questions > 0 else 0
        
        # Calculate competency performance from attempts
        competency_scores = {}
        for attempt in session.question_attempts:
            if hasattr(attempt, 'original_question'):
                question = attempt.original_question
            else:
                question = attempt.get('original_question', {})
                
            competencies = question.get("competencies_tested", [])
            is_correct = attempt.get('is_correct', False) if isinstance(attempt, dict) else attempt.is_correct
            
            for comp in competencies:
                if comp not in competency_scores:
                    competency_scores[comp] = {"correct": 0, "total": 0}
                competency_scores[comp]["total"] += 1
                if is_correct:
                    competency_scores[comp]["correct"] += 1
        
        # Convert to percentages
        competency_percentages = {
            comp: (scores["correct"] / scores["total"] * 100) if scores["total"] > 0 else 0
            for comp, scores in competency_scores.items()
        }
        
        return {
            "session_id": session_id,
            "scenario_name": session.scenario_name,
            "difficulty": session.difficulty,
            "final_score": final_score,
            "total_questions": total_questions,
            "percentage_score": round(percentage, 1),
            "passed": percentage >= 70,
            "excellence": percentage >= 85,
            "competency_scores": competency_percentages,
            "detailed_attempts": session.question_attempts,
            "conversation_history": session.conversation_history,
            "session_duration": (session.last_updated - session.created_at).total_seconds() / 60,  # minutes
            "created_at": session.created_at,
            "completed_at": session.last_updated
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")