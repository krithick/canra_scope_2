# 
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
from google import genai
import base64
from models import (
    Message, ChatSession, ChatResponse, ChatReport, BotConfig, BotConfigAnalyser,
    QuestionScenarioDoc, ParaphrasedQuestionCache, QuestionSession)
from google.genai import types
class BaseLLMBot(ABC):
    """
    Abstract base class for dynamically created LLM bots
    """
    def __init__(self, config: BotConfig, llm_client):
        """
        Initialize bot with configuration and LLM client
        
        :param config: Bot configuration from database
        :param llm_client: Initialized LLM client
        """
        self.bot_id = config.bot_id
        self.bot_name = config.bot_name
        self.bot_description = config.bot_description
        self.bot_role=config.bot_role
        self.bot_role_alt=config.bot_role_alt
        self.system_prompt = config.system_prompt
        self.is_active = config.is_active
        self.llm_model = "gemini-2.0-flash"#config.llm_model
        self.model=self._initialize_llm_model(config)
    def _initialize_llm_model(self, config: BotConfig):
        """
        Initialize the LLM model for the bot
        
        :param config: Bot configuration
        :return: Initialized generative model
        """
        try:
            # Configure the generative model with system instruction
            return genai.Client(
      vertexai=True,
      project="arvr-440711",
      location="global",
  )
        except Exception as e:
            print(f"Error initializing LLM for {config.bot_name}: {e}")
            raise
        
    @abstractmethod
    async def load_scenarios(self):
        """
        Load and preprocess scenarios specific to the bot type
        """
        pass


 
    def format_conversation(self, conversation_history: List[Message]) -> List[types.Content]:
        """
        Convert conversation history to Vertex AI Content format
        """
        contents = []
    
        # Add system prompt as the first user message if needed
        system_prompt = self.system_prompt
        if system_prompt:
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)]
            ))
    
        for message in conversation_history:
            # Map your roles to Vertex AI roles
            if message.role == self.bot_role:  # This is the AI/model response
                role = "model"
            else:  # This is user/human input
                role = "user"
            
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=message.content)]
            ))
    
        return contents 
   
    async def get_farmer_response(self,
                              officer_question: str,
                              scenario_name: str,
                              conversation_history: List[Message]) -> str:
        """
        Generate response using new Vertex AI API structure
        """
        start_time = time.time()
    
        try:
            # Format conversation history
            contents = self.format_conversation(conversation_history)
        
            # Add the new question as a user message
            contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=officer_question)]
            ))
        
            # Configure generation settings
            generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
            # Add other config parameters as needed
        )
        
            # Generate response using the new API
            response = await self.model.aio.models.generate_content(
            model=self.llm_model,  # Your model name/path
            contents=contents,
            # config=generate_content_config
        )
        
            # Extract text from response
            response_text = response.text
        
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"The function took {execution_time} seconds to run.")
        
            return response_text.strip()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")            
# 
class BaseAnalyserBot(ABC):
    def __init__(self,config:BotConfigAnalyser,llm_client):
        self.bot_id = config.bot_id
        self.bot_name = config.bot_name
        self.bot_description = config.bot_description    
        self.bot_schema = config.bot_schema    
        self.system_prompt = config.system_prompt
        self.is_active = config.is_active
        self.llm_model = "gemini-2.0-flash"#config.llm_model
        self.model=self._initialize_llm_model(config)       
    def _initialize_llm_model(self, config: BotConfig):
        """
        Initialize the LLM model for the bot
        
        :param config: Bot configuration
        :return: Initialized generative model
        """
        try:
            # Configure the generative model with system instruction
            
            return genai.Client(
      vertexai=True,
      project="arvr-440711",
      location="global",
  )
        except Exception as e:
            print(f"Error initializing LLM for {config.bot_name}: {e}")
            raise


    def _create_analysis_prompt(self, conversation: str) -> str:
        """Create the analysis prompt for Gemini."""
        prompt = f"""
        You are a conversation quality analyst specialized in banking services. 
        Analyze the following conversation using the provided evaluation schema.
        
        EVALUATION SCHEMA:
        {json.dumps(self.bot_schema, indent=2)}
        
        CONVERSATION TO ANALYZE:
        {conversation}
        
        Please provide a detailed analysis following these steps:
        1. Evaluate each criteria as not_present (0), partially_present (0.5), or fully_present (1)
        2. Calculate category scores using the weights provided in the schema
2. For each main category (language_and_communication, product_knowledge, empathy_and_trust, process_clarity, product_suitability), calculate a score out of 100 using these steps:
            a. For each sub-category within a main category, calculate a sub-score by:
                i. Summing the values (0, 0.5, or 1) for each present criteria.
                ii. Dividing that sum by the total number of criteria in that sub-category.
                iii. Multiplying the result by the sub-category's weight.
            b. Sum all the sub-category scores within a main category.
            c. Multiply the summed sub-category score by 100 and then divide it by the sum of all the weights of the sub-categories within that main category. This result is the category score (out of 100).
2.Make sure the score of category_scores are out of 100 
 3. Generate an overall score based on the weighted categories
        4. Identify specific strengths and areas for improvement and make sure they are not more than one point
        5. Provide actionable recommendations and make sure they are not more than one point
        
        FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE FOLLOWING STRUCTURE:
        {{
            "conversation_id": "unique_id",
            "timestamp": "current_datetime",
            "overall_score": number,
            "category_scores": {{
                "language_and_communication": number,
                "product_knowledge": number,
                "empathy_and_trust": number,
                "process_clarity": number,
                "product_suitability": number
            }},
            "detailed_feedback": {{
                "strengths": [strings],
                "areas_for_improvement": [strings],
                "critical_gaps": [strings]
            }},
            "recommendations": [strings]
        }}
        
        IMPORTANT: Ensure your response is ONLY the JSON object, with no additional text or explanation.
        """
        return prompt

    def _format_conversation_for_analysis(self, conversation_data: dict) -> str:
        """
        Convert the JSON conversation format into a readable string format for analysis.
        
        Args:
            conversation_data (dict): Conversation in JSON format
            
        Returns:
            str: Formatted conversation text
        """
        formatted_conversation = []
        
        for message in conversation_data["conversation_history"]:
            role = message["role"].replace("_", " ").title()
            content = message["content"]
            formatted_conversation.append(f"{role}: {content}")
            
        return "\n".join(formatted_conversation)
    
    async def analyze_conversation(self, conversation_data: dict) :
        """
        Analyze a conversation using Gemini and return structured feedback.
        
        Args:
            conversation_data (dict): Conversation in JSON format
            
        Returns:
            Dict[str, Any]: Analysis results in JSON format
        """
        try:
            # Validate input
            if not isinstance(conversation_data, dict) or "conversation_history" not in conversation_data:
                raise ValueError("Invalid conversation format. Expected dict with 'conversation_history' key")
            
            # Format conversation for analysis
            formatted_conversation = self._format_conversation_for_analysis(conversation_data)
            
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(formatted_conversation)
            contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
            # Get response from Gemini
            response = await self.model.aio.models.generate_content(model=self.llm_model,
            contents=contents,
)
            
            # Clean and parse the response
            cleaned_json_text = self._clean_gemini_response(response.text)
            
            try:
                analysis_result = json.loads(cleaned_json_text)
            except json.JSONDecodeError:
                print("Failed to parse JSON. Raw response:", response.text)
                raise Exception("Invalid JSON response from Gemini")
            
            # Add timestamp if not present
            if 'timestamp' not in analysis_result:
                analysis_result['timestamp'] = datetime.utcnow().isoformat()
                
            return analysis_result
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print("Raw response:", response.text if 'response' in locals() else "No response generated")
            raise
   
   

    def _clean_gemini_response(self, response_text: str) -> str:
        """
    Clean the Gemini response text to extract pure JSON.
    
    Args:
        response_text (str): Raw response text from Gemini
        
    Returns:
        str: Cleaned JSON text
    """
    # Remove code block markers if present
        cleaned_text = response_text.replace('```json', '').replace('```', '')
    
    # Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
    
        return cleaned_text

            
