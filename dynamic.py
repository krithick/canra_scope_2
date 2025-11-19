from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, Depends
import importlib
import inspect
from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
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
    QuestionScenarioDoc, ParaphrasedQuestionCache, QuestionSession, STTRequest, STTResponse)
from question_bot import QuestionBot
from stt_service import GoogleSTTService
from stt_tracker import STTTracker
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

# Initialize STT service and tracker
stt_service = GoogleSTTService(credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH"))
stt_tracker = STTTracker(db)

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

@app.delete("/api/questions/{scenario_name}/clear")
async def clear_generated_questions(
    scenario_name: str,
    difficulty: str = "all",
    db: MongoDB = Depends(get_db)
):
    """Clear generated questions for testing"""
    try:
        if difficulty == "all":
            result = await db.paraphrased_questions.delete_many({"scenario_name": scenario_name})
        else:
            result = await db.paraphrased_questions.delete_many({
                "scenario_name": scenario_name,
                "difficulty": difficulty
            })
        
        return {
            "message": f"Cleared {result.deleted_count} questions",
            "deleted_count": result.deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing questions: {str(e)}")

@app.post("/api/question-scenarios/{scenario_name}/generate-bulk-questions")
async def generate_bulk_questions(
    scenario_name: str,
    difficulty: str = Form(..., description="easy or hard"),
    target_count: int = Form(default=100, description="Number of questions to generate"),
    force_regenerate: bool = Form(default=False),
    db: MongoDB = Depends(get_db)
):
    """Generate 100 new questions from scenario context"""
    try:
        # Get scenario context
        scenario = await db.question_scenarios.find_one({"scenario_name": scenario_name})
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        scenario_context = scenario.get("scenario_context", "")
        if not scenario_context:
            raise HTTPException(status_code=400, detail="Scenario context required for question generation")
        
        bot = await bot_factory.get_bot(scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Clear existing if force regenerate
        if force_regenerate:
            await db.paraphrased_questions.delete_many({
                "scenario_name": scenario_name,
                "difficulty": difficulty
            })
        
        # Check existing count
        existing_count = await db.paraphrased_questions.count_documents({
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "is_active": True
        })
        
        if existing_count >= target_count and not force_regenerate:
            return {
                "message": f"Already have {existing_count} questions for {difficulty} mode",
                "existing_count": existing_count,
                "target_count": target_count
            }
        
        generated_count = 0
        errors = []
        questions_to_generate = target_count - existing_count
        
        # Generate in batches of 10 (balance between speed and reliability)
        batch_size = 10
        batches_needed = (questions_to_generate + batch_size - 1) // batch_size
        
        for batch_num in range(batches_needed):
            try:
                questions_in_batch = min(batch_size, questions_to_generate - generated_count)
                if questions_in_batch <= 0:
                    break
                
                # Generate batch of questions with variation instruction
                variation_prompt = f"Focus on batch {batch_num + 1}/10 - ensure unique questions different from previous batches. Vary the situations, stakeholders, and decision points."
                batch_questions = await bot.generate_bulk_questions_from_scenario(
                    scenario_context, difficulty, questions_in_batch, variation_prompt
                )
                
                # Save each question
                for i, question in enumerate(batch_questions):
                    try:
                        # Update question ID to be unique
                        question_id = f"{difficulty}_q{generated_count + i + 1}"
                        question["id"] = question_id
                        question["question_number"] = generated_count + i + 1
                        
                        # Save to database
                        cache_doc = ParaphrasedQuestionCache(
                            id=question_id,
                            original_question_id=question_id,
                            scenario_name=scenario_name,
                            difficulty=difficulty,
                            paraphrased_data=question
                        )
                        
                        result = await db.paraphrased_questions.insert_one(cache_doc.dict())
                        if result.inserted_id:
                            generated_count += 1
                            print(f"Saved question {question_id} to database")
                        else:
                            print(f"Failed to save question {question_id}")
                        
                    except Exception as e:
                        errors.append(f"Question {generated_count + i + 1}: {str(e)}")
                
                print(f"Generated batch {batch_num + 1}/{batches_needed}: {len(batch_questions)} questions")
                
            except Exception as e:
                errors.append(f"Batch {batch_num + 1}: {str(e)}")
        
        return {
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "generated_count": generated_count,
            "existing_count": existing_count,
            "total_count": existing_count + generated_count,
            "target_count": target_count,
            "batches_processed": batches_needed,
            "errors": errors[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating bulk questions: {str(e)}")

@app.post("/api/question-scenarios/{scenario_name}/generate-paraphrases")
async def generate_all_paraphrases(
    scenario_name: str,
    difficulty: str = Form(..., description="easy or hard"),
    force_regenerate: bool = Form(default=False),
    db: MongoDB = Depends(get_db)
):
    """Generate paraphrased versions for all questions in a scenario (legacy method)"""
    return await generate_bulk_questions(scenario_name, difficulty, 20, force_regenerate, db)

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
        
        # Convert ObjectId to string
        for p in paraphrases:
            if '_id' in p:
                p['_id'] = str(p['_id'])
        
        return {
            "scenario_name": scenario_name,
            "difficulty": difficulty,
            "count": len(paraphrases),
            "paraphrases": paraphrases
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting paraphrases: {str(e)}")

@app.get("/api/debug/questions/{scenario_name}")
async def debug_questions(
    scenario_name: str,
    db: MongoDB = Depends(get_db)
):
    """Debug endpoint to check what's in the database"""
    try:
        easy_count = await db.paraphrased_questions.count_documents({
            "scenario_name": scenario_name,
            "difficulty": "easy",
            "is_active": True
        })
        
        hard_count = await db.paraphrased_questions.count_documents({
            "scenario_name": scenario_name,
            "difficulty": "hard",
            "is_active": True
        })
        
        # Get sample questions
        easy_samples = await db.paraphrased_questions.find({
            "scenario_name": scenario_name,
            "difficulty": "easy",
            "is_active": True
        }).limit(3).to_list(length=None)
        
        hard_samples = await db.paraphrased_questions.find({
            "scenario_name": scenario_name,
            "difficulty": "hard",
            "is_active": True
        }).limit(3).to_list(length=None)
        
        # Convert ObjectIds
        for samples in [easy_samples, hard_samples]:
            for s in samples:
                if '_id' in s:
                    s['_id'] = str(s['_id'])
        
        return {
            "scenario_name": scenario_name,
            "easy_count": easy_count,
            "hard_count": hard_count,
            "easy_samples": [q.get("paraphrased_data", {}).get("question_text", "No text")[:100] for q in easy_samples],
            "hard_samples": [q.get("paraphrased_data", {}).get("question_text", "No text")[:100] for q in hard_samples]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

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
    retry_count: int = Form(default=0, description="Number of retries attempted"),
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
        print(retry_count,"retry_count - api")
        # Submit answer through bot
        result = await bot.submit_answer(session_id, user_input, time_taken, is_timeout, retry_count)
        
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
        result = await bot.submit_answer(session_id, "", time_taken, is_timeout=True, retry_count=0)
        
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

@app.get("/api/question-sessions/{session_id}/all-questions")
async def get_all_session_questions(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Get all 15 questions from a session for testing"""
    try:
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Return all paraphrased questions used in this session
        questions = session.paraphrased_questions_used
        
        # Clean ObjectIds
        for q in questions:
            if '_id' in q:
                q['_id'] = str(q['_id'])
        
        return {
            "session_id": session_id,
            "total_questions": len(questions),
            "questions": [{
                "id": q["id"],
                "question_text": q["question_text"][:100] + "...",
                "correct_answer": q["correct_answer"]
            } for q in questions]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session questions: {str(e)}")

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

@app.get("/api/question-sessions/{session_id}/analysis")
async def analyze_question_session(
    session_id: str,
    db: MongoDB = Depends(get_db)
):
    """Analyze completed question session performance"""
    try:
        # Check if analysis already exists
        existing_analysis = await db.question_analysis.find_one({"session_id": session_id})
        if existing_analysis:
            return {
                "success": True,
                "data": existing_analysis
            }
        
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not session.is_completed:
            raise HTTPException(status_code=400, detail="Session not completed yet")
        
        # Get scenario for competency framework
        scenario = await db.question_scenarios.find_one({"scenario_name": session.scenario_name})
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        # Create analysis using question bot
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        analysis = await bot.analyze_session_performance(session, scenario.get('competency_framework', []))
        
        # Save analysis to database
        await db.question_analysis.insert_one(analysis)
        
        return {
            "success": True,
            "data": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing session: {str(e)}")

# ===== SPEECH-TO-TEXT API ENDPOINTS =====

@app.post("/api/stt/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language_code: str = Form(default="en-US", description="Language code (e.g., en-US, es-ES)"),
    user_id: Optional[str] = Form(default=None, description="Optional user ID for tracking"),
    session_id: Optional[str] = Form(default=None, description="Optional session ID for tracking")
):
    """Transcribe uploaded audio file to text using Google STT"""
    audio_content = None
    try:
        # Read audio content
        audio_content = await audio_file.read()
        print(f"Audio file size: {len(audio_content)} bytes")
        
        # Create STT request config with defaults
        stt_config = STTRequest(
            language_code=language_code,
            sample_rate_hertz=16000,
            encoding="LINEAR16",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
            model="latest_long"
        )
        
        # Determine if we should use long-running operation based on file size
        # Files larger than 10MB or longer than 1 minute should use long-running
        if len(audio_content) > 10 * 1024 * 1024:  # 10MB
            result = await stt_service.transcribe_long_audio(audio_content, stt_config)
        else:
            result = await stt_service.transcribe_audio(audio_content, stt_config)
        
        # Track successful transcription
        # Generate temp session_id if none provided (for first message)
        tracking_session_id = session_id or f"temp_{uuid.uuid4().hex[:8]}"
        
        await stt_tracker.track_transcription(
            endpoint="/api/stt/transcribe",
            audio_content=audio_content,
            config=stt_config.dict(),
            result=result.dict(),
            session_id=tracking_session_id,
            user_id=user_id,
            success=True
        )
        
        # Add tracking session ID to response
        result_dict = result.dict()
        result_dict['tracking_session_id'] = tracking_session_id
        return result_dict
        
    except Exception as e:
        # Track failed transcription
        if audio_content:
            tracking_session_id = session_id or f"temp_{uuid.uuid4().hex[:8]}"
            
            await stt_tracker.track_transcription(
                endpoint="/api/stt/transcribe",
                audio_content=audio_content,
                config={"language_code": language_code},
                result={},
                session_id=tracking_session_id,
                user_id=user_id,
                success=False,
                error_message=str(e)
            )
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/api/stt/transcribe-base64", response_model=STTResponse)
async def transcribe_base64_audio(
    audio_base64: str = Form(..., description="Base64 encoded audio data"),
    language_code: str = Form(default="en-US"),
    sample_rate_hertz: int = Form(default=16000),
    encoding: str = Form(default="WEBM_OPUS"),
    enable_automatic_punctuation: bool = Form(default=True),
    model: str = Form(default="latest_long")
):
    """Transcribe base64 encoded audio to text"""
    try:
        # Decode base64 audio
        audio_content = base64.b64decode(audio_base64)
        
        # Create STT request config
        stt_config = STTRequest(
            language_code=language_code,
            sample_rate_hertz=sample_rate_hertz,
            encoding=encoding,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=model
        )
        
        # Transcribe audio
        result = await stt_service.transcribe_audio(audio_content, stt_config)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 transcription error: {str(e)}")

@app.get("/api/stt/supported-languages")
async def get_supported_languages():
    """Get list of supported languages for STT"""
    try:
        languages = stt_service.get_supported_languages()
        return {
            "supported_languages": languages,
            "total_count": len(languages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting languages: {str(e)}")

@app.get("/api/stt/supported-encodings")
async def get_supported_encodings():
    """Get list of supported audio encodings for STT"""
    try:
        encodings = stt_service.get_supported_encodings()
        return {
            "supported_encodings": encodings,
            "total_count": len(encodings)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting encodings: {str(e)}")

@app.post("/api/stt/question-answer")
async def transcribe_question_answer(
    session_id: str = Form(..., description="Question session ID"),
    audio_file: UploadFile = File(..., description="Audio file with user's answer"),
    language_code: str = Form(default="en-US"),
    time_taken: int = Form(default=30, description="Time taken to answer in seconds")
):
    """Transcribe audio answer and submit to question session"""
    try:
        # Read and transcribe audio
        audio_content = await audio_file.read()
        
        stt_config = STTRequest(
            language_code=language_code,
            sample_rate_hertz=16000,
            encoding="WEBM_OPUS",
            enable_automatic_punctuation=True,
            model="latest_short"  # Use short model for quick responses
        )
        
        # Transcribe the audio
        transcription = await stt_service.transcribe_audio(audio_content, stt_config)
        
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the appropriate bot and submit the transcribed answer
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Submit the transcribed text as the answer
        result = await bot.submit_answer(
            session_id, 
            transcription.transcript, 
            time_taken, 
            is_timeout=False, 
            retry_count=0
        )
        
        return {
            "success": True,
            "transcription": {
                "text": transcription.transcript,
                "confidence": transcription.confidence,
                "processing_time_ms": transcription.processing_time_ms
            },
            "question_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio answer: {str(e)}")

@app.post("/api/stt/explanation")
async def transcribe_explanation(
    session_id: str = Form(..., description="Question session ID"),
    audio_file: UploadFile = File(..., description="Audio file with user's explanation"),
    language_code: str = Form(default="en-US")
):
    """Transcribe audio explanation and submit to question session"""
    try:
        # Read and transcribe audio
        audio_content = await audio_file.read()
        
        stt_config = STTRequest(
            language_code=language_code,
            sample_rate_hertz=16000,
            encoding="WEBM_OPUS",
            enable_automatic_punctuation=True,
            model="latest_long"  # Use long model for detailed explanations
        )
        
        # Transcribe the audio
        transcription = await stt_service.transcribe_audio(audio_content, stt_config)
        
        # Get session to find the scenario
        session = await db.get_question_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the appropriate bot and submit the transcribed explanation
        bot = await bot_factory.get_bot(session.scenario_name)
        if not isinstance(bot, QuestionBot):
            raise HTTPException(status_code=400, detail="Invalid bot type")
        
        # Submit the transcribed explanation
        result = await bot.submit_explanation(session_id, transcription.transcript)
        
        return {
            "success": True,
            "transcription": {
                "text": transcription.transcript,
                "confidence": transcription.confidence,
                "processing_time_ms": transcription.processing_time_ms
            },
            "explanation_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio explanation: {str(e)}")

# ===== STT USAGE TRACKING & REPORTING ENDPOINTS =====

@app.get("/api/stt/usage/summary")
async def get_stt_usage_summary(
    days: int = 30,
    user_id: Optional[str] = None
):
    """Get STT usage summary for reporting"""
    try:
        start_date = datetime.now() - timedelta(days=days)
        summary = await stt_tracker.get_usage_summary(
            start_date=start_date,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting usage summary: {str(e)}")

@app.get("/api/stt/usage/detailed")
async def get_detailed_stt_usage(
    days: int = 7,
    limit: int = 100
):
    """Get detailed STT usage records"""
    try:
        start_date = datetime.now() - timedelta(days=days)
        records = await stt_tracker.get_detailed_usage(
            start_date=start_date,
            limit=limit
        )
        
        return {
            "success": True,
            "total_records": len(records),
            "data": records
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting detailed usage: {str(e)}")

@app.get("/api/stt/usage/export")
async def export_stt_usage(
    days: int = 30,
    format: str = "json"  # json or csv
):
    """Export STT usage data for external reporting"""
    try:
        start_date = datetime.now() - timedelta(days=days)
        
        if format.lower() == "csv":
            # Get detailed records for CSV export
            records = await stt_tracker.get_detailed_usage(
                start_date=start_date,
                limit=10000  # Large limit for export
            )
            
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if records:
                writer = csv.DictWriter(output, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            
            from fastapi.responses import Response
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=stt_usage_{days}days.csv"}
            )
        else:
            # JSON export with summary
            summary = await stt_tracker.get_usage_summary(start_date=start_date)
            records = await stt_tracker.get_detailed_usage(
                start_date=start_date,
                limit=1000
            )
            
            return {
                "export_date": datetime.now().isoformat(),
                "period_days": days,
                "summary": summary,
                "detailed_records": records
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting usage data: {str(e)}")

@app.get("/api/stt/dashboard", response_class=HTMLResponse)
async def stt_dashboard():
    """Serve STT usage dashboard"""
    try:
        with open("stt_dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Please ensure stt_dashboard.html exists in the project directory.</p>",
            status_code=404
        )

@app.get("/api/stt/dashboard.js")
async def stt_dashboard_js():
    """Serve dashboard JavaScript file"""
    try:
        with open("stt_dashboard.js", "r", encoding="utf-8") as f:
            from fastapi.responses import Response
            return Response(content=f.read(), media_type="application/javascript")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/api/stt/session/{session_id}")
async def get_session_stt_usage(session_id: str):
    """Get STT usage for a specific session"""
    try:
        usage_data = await stt_tracker.get_session_usage(session_id)
        return {"success": True, "data": usage_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stt/sessions")
async def get_sessions_stt_summary(days: int = 7):
    """Get summary of all sessions with STT usage"""
    try:
        sessions = await stt_tracker.get_sessions_summary(days)
        return {"success": True, "data": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stt/update-session")
async def update_stt_session_id(
    temp_session_id: str = Form(...),
    real_session_id: str = Form(...)
):
    """Update temp session ID with real session ID after chat session is created"""
    try:
        result = await db.stt_usage.update_many(
            {"session_id": temp_session_id},
            {"$set": {"session_id": real_session_id}}
        )
        return {
            "success": True, 
            "updated_records": result.modified_count,
            "message": f"Updated {result.modified_count} STT records from {temp_session_id} to {real_session_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)