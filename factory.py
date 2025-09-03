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
from question_bot import QuestionBot
from bots import BaseLLMBot , BaseAnalyserBot
class DynamicBotFactory:
    """
    Factory for creating dynamic bot instances based on database configurations
    """
    def __init__(self, mongodb_uri: str, database_name: str):
        """
        Initialize bot factory with MongoDB connection
        
        :param mongodb_uri: MongoDB connection string
        :param database_name: Name of the database
        """
        # vertexai.init(project="arvr-440711", location="us-central1")
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client[database_name]
        self.bots: Dict[str, BaseLLMBot] = {}
                # LLM client setup (replace with your preferred LLM client)
        self.llm_client = None  # Initialize your LLM client here

    async def create_dynamic_bot_class(self, config: BotConfig) -> Type[BaseLLMBot]:
        """
        Dynamically create a bot class based on configuration
        
        :param config: Bot configuration
        :return: Dynamically created bot class
        """
        # If a specific bot class is specified in the configuration
        if config.bot_class == "QuestionBot":
            return QuestionBot  
        if config.bot_class:
            try:
                # Attempt to import the specified bot class
                module_name = config.bot_class#.rsplit('.', 1)
                class_name = config.bot_class#.rsplit('.', 1)
                module = importlib.import_module(module_name)
                bot_class = getattr(module, class_name)
                # Validate that the class is a subclass of BaseLLMBot
                if not issubclass(bot_class, BaseLLMBot):
                    raise ValueError(f"Specified class {config.bot_class} must inherit from BaseLLMBot")
                
                return bot_class
            except (ImportError, AttributeError) as e:
                # Fallback to default bot class if import fails
                print(f"Could not import specified bot class: {e}")

        # Create a dynamic bot class if no specific class is specified
        class DynamicBot(BaseLLMBot):
           
            async def load_scenarios(self):
                # Default scenario loading 
                # Can be overridden by specific implementation if needed
                print(f"Loading scenarios for {self.bot_name}")
                # Implement your scenario loading logic here
  
        # Set the class name dynamically for better debugging
        DynamicBot.__name__ = f"{config.bot_name}Bot"
        return DynamicBot

    async def initialize_bots(self):
        """
        Load all active bots from database and instantiate
        """
        # Clear existing bots
        self.bots.clear()

        # Fetch active bot configurations
        bot_configs = await self.db.bot_configs.find({"is_active": True}).to_list(length=None)
        # print('Bot config',bot_configs)
        for config_dict in bot_configs:
            # print("%c Line:201 🌶 config_dict", "color:#6ec1c2", config_dict) 
            config = BotConfig(**config_dict)
            print(config.bot_name,"nameeees")
            # Dynamically create bot class
            bot_class = await self.create_dynamic_bot_class(config)
            
            # Instantiate the bot
            bot = bot_class(config, self.llm_client)
            await bot.load_scenarios()
            
            # Store the bot
            self.bots[config.bot_description] = bot

    async def get_bot(self, bot_description: str) -> BaseLLMBot:
        """
        Retrieve a specific bot by ID
        
        :param bot_id: Unique identifier for the bot
        :return: Bot instance
        """
        bot = self.bots.get(bot_description)
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        return bot
   
    async def create_bot(self):
        # await self.db.bot_config.insert_one(bot_config.dict())
        await self.initialize_bots()
  
    async def update_bot_config(self, bot_id: str, update_data: Dict):
        """
        Update bot configuration in database and reload
        
        :param bot_id: Bot ID to update
        :param update_data: Configuration update details
        """
        await self.db.bot_configs.update_one(
            {"bot_id": bot_id}, 
            {"$set": update_data}
        )
        await self.initialize_bots()
    

    async def create_dynamic_bot_analyser_class(self, config: BotConfigAnalyser) -> Type[BaseAnalyserBot]:
        """
        Dynamically create a bot class based on configuration
        
        :param config: Bot configuration
        :return: Dynamically created bot class
        """
        # Create a dynamic bot class if no specific class is specified
        class DynamicBot(BaseAnalyserBot):
           
            async def load_scenarios(self):
                # Default scenario loading 
                # Can be overridden by specific implementation if needed
                print(f"Loading scenarios for {self.bot_name}")
                # Implement your scenario loading logic here
  
        # Set the class name dynamically for better debugging
        DynamicBot.__name__ = f"{config.bot_name}Bot"
        return DynamicBot

    async def initialize_bots_analyser(self):
        """
        Load all active bots from database and instantiate
        """
        # Clear existing bots
        self.bots.clear()

        # Fetch active bot configurations
        bot_configs = await self.db.bot_configs_analyser.find({"is_active": True}).to_list(length=None)
        # print('Bot config',bot_configs)
        for config_dict in bot_configs:
            # print("%c Line:201 🌶 config_dict", "color:#6ec1c2", config_dict) 
            config = BotConfigAnalyser(**config_dict)
            print(config.bot_name,"nameeees")
            # Dynamically create bot class
            bot_class = await self.create_dynamic_bot_analyser_class(config)
            
            # Instantiate the bot
            bot = bot_class(config, self.llm_client)
            print(bot)
            await bot.load_scenarios()
            
            # Store the bot
            self.bots[config.bot_description] = bot


    async def get_bot_analyser(self, bot_description: str) -> BaseAnalyserBot:
        """
        Retrieve a specific bot by ID
        
        :param bot_id: Unique identifier for the bot
        :return: Bot instance
        """
        bot = self.bots.get(bot_description)
        print(self.bots)
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        return bot
  