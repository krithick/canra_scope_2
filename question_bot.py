# QuestionBot - Single class that extends your existing BaseLLMBot

import json
import re
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from datetime import datetime
from models import (
    Message, BotConfig, QuestionScenarioDoc, ParaphrasedQuestionCache,  QuestionSession,
    QuestionAttemptRecord           )
from bots import BaseLLMBot
import uuid
from database import get_db
class QuestionBot(BaseLLMBot):
    """
    Conversational question-based training bot
    Extends existing BaseLLMBot to work with your current factory
    """
    
    def __init__(self, config: BotConfig, llm_client):
        super().__init__(config, llm_client)
        self.scenario_questions = []  # Cache questions from DB
        self.scenario_context = ""   # Background context for LLM

    # ===== MAIN ORCHESTRATION METHOD =====
    async def _get_db(self):
        """Helper to get database instance when needed"""
        from database import get_db
        return await get_db()
        
    async def get_farmer_response(self, message: str, scenario_name: str, 
                                  conversation_history: List[Message],) -> str:
        """
        Main method that integrates with your existing chat endpoint
        Orchestrates the entire question flow
        """
        try:
            # Determine what the user is trying to do
            state = self._determine_state_from_conversation(conversation_history)
            
            if state == "start":
                return await self._handle_session_start(message, conversation_history)
            elif state == "awaiting_answer":
                return await self._handle_answer_submission(message, conversation_history)
            elif state == "awaiting_explanation":
                return await self._handle_explanation_submission(message, conversation_history)
            else:
                return self._format_response("I'm not sure how to help with that. Please start a new session.", "confused")
                
        except Exception as e:
            return self._format_response(f"Sorry, there was an error: {str(e)}", "apologetic")
    
    # ===== CORE FLOW METHODS =====
    
    async def _handle_session_start(self, message: str, conversation_history: List[Message]) -> str:
        """Handle session initialization and first question"""
        try:
            # Parse difficulty from message (easy/hard)
            difficulty = "easy" if "easy" in message.lower() else "hard"
            
            # Get or create session
            session_id = await self._extract_session_id_from_history(conversation_history)
            session = await self._get_or_create_session(session_id, difficulty)
            
            # Get first paraphrased question
            if session.current_question_index < len(session.questions_data):
                current_q = session.questions_data[session.current_question_index]
                paraphrased_q = await self._get_paraphrased_question(current_q, difficulty)
                
                response = f"""Welcome to the {self.bot_description} training!
                
Question {session.current_question_index + 1}/{len(session.questions_data)}:

{paraphrased_q['question_text']}

A) {paraphrased_q['options'][0]['text']}
B) {paraphrased_q['options'][1]['text']}
C) {paraphrased_q['options'][2]['text']}
D) {paraphrased_q['options'][3]['text']}

Please select your answer (A, B, C, or D)."""
                
                return self._format_response(response, "engaged")
            else:
                return self._format_response("No questions available for this scenario.", "neutral", True)
                
        except Exception as e:
            return self._format_response(f"Error starting session: {str(e)}", "apologetic")
    
#     async def _handle_answer_submission(self, answer: str, conversation_history: List[Message]) -> str:
#         """Process user's answer selection"""
#         try:
#             # Extract answer (A, B, C, D)
#             user_answer = self._extract_answer_from_message(answer)
#             if not user_answer:
#                 return self._format_response("Please select A, B, C, or D for your answer.", "neutral")
            
#             # Get session and validate answer
#             session = await self._get_session_from_history(conversation_history)
#             current_q = session.questions_data[session.current_question_index]
#             is_correct = user_answer.upper() == current_q['correct_answer'].upper()
            
#             if is_correct:
#                 # Ask for explanation
#                 response = f"""Correct! You selected {user_answer}.

# Now, please explain WHY this is the best choice. What leadership principles or reasoning led you to this decision?

# Your explanation will help me understand your thought process."""
                
#                 # Update session state
#                 session.current_state = "awaiting_explanation"
#                 await self._update_session(session)
                
#                 return self._format_response(response, "encouraging")
#             else:
#                 # Provide feedback and move to next question
#                 correct_explanation = current_q['explanation']['correct_explanation']
#                 incorrect_explanation = current_q['explanation']['incorrect_explanations'].get(user_answer, "This option is not optimal for this situation.")
                
#                 feedback = await self._generate_feedback_with_context(current_q, user_answer, False)
                
#                 response = f"""That's not quite right. You selected {user_answer}.

# The correct answer is {current_q['correct_answer']}.

# Here's why:
# ✓ Correct choice: {correct_explanation}
# ✗ Your choice: {incorrect_explanation}

# {feedback}"""
                
#                 # Move to next question or complete session
#                 next_response = await self._move_to_next_question(session)
#                 response += f"\n\n{next_response}"
                
#                 return self._format_response(response, "educational")
                
#         except Exception as e:
#             return self._format_response(f"Error processing answer: {str(e)}", "apologetic")
    
    def _map_paraphrased_to_original_option(self, paraphrased_answer: str, paraphrased_q: Dict, original_q: Dict) -> str:
        """Map shuffled option back to original for explanation lookup"""
        try:
            # Get the text of what user selected in paraphrased version
            paraphrased_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == paraphrased_answer
            )
        
            # Find which original option has similar meaning (this is approximate)
            # For now, just return the paraphrased answer - you might need better mapping
            return paraphrased_answer
        except:
            return paraphrased_answer
    async def _handle_answer_submission(self, answer: str, conversation_history: List[Message]) -> str:
        """Process user's answer selection with proper tracking"""
        try:
            # Extract answer (A, B, C, D)
            user_answer = self._extract_answer_from_message(answer)
            if not user_answer:
                return self._format_response("Please select A, B, C, or D for your answer.", "neutral")
        
            # Get session and current question
            session = await self._get_session_from_history(conversation_history)
            if session is None:
                return self._format_response("Session not found. Please start a new training session.", "apologetic")
            print(session)
            current_q_original = session.questions_data[session.current_question_index]
        
            # Get the paraphrased question they actually saw
            paraphrased_q = session.paraphrased_questions_used[session.current_question_index]
        
            # Check if answer is correct (use paraphrased question's correct answer)
            is_correct = user_answer.upper() == paraphrased_q['correct_answer'].upper()
        
            # Get the text of what they selected
            user_answer_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == user_answer.upper()
            )
        
            # Create detailed attempt record
            attempt = QuestionAttemptRecord(
            question_id=current_q_original['id'],
            original_question=current_q_original,
            paraphrased_question=paraphrased_q,
            user_answer=user_answer.upper(),
            user_answer_text=user_answer_text,
            correct_answer_original=current_q_original['correct_answer'],
            correct_answer_paraphrased=paraphrased_q['correct_answer'],
            is_correct=is_correct,
            time_taken_seconds=30  # You can track actual time later
            )
        
            session.question_attempts.append(attempt)
        
            if is_correct:
                # Ask for explanation
                response = f"""Correct! You selected {user_answer}: "{user_answer_text}"

Now, please explain WHY this is the best choice. What leadership principles or reasoning led you to this decision?

Your explanation will help me understand your thought process."""
            
                # Update session state
                session.current_state = "awaiting_explanation"
            
                # Add conversation to history
                user_msg = Message(role="user", content=answer, timestamp=datetime.now())
                bot_msg = Message(role="assistant", content=response, timestamp=datetime.now())
                session.conversation_history.extend([user_msg, bot_msg])
            
                await self._update_session(session)
                return self._format_response(response, "encouraging")
            
            else:
                # Wrong answer - provide feedback and move to next
                correct_option_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == paraphrased_q['correct_answer']
                )
            
                # Get explanation from original question
                if 'explanation' in current_q_original:
                    correct_explanation = current_q_original['explanation']['correct_explanation']
                    incorrect_explanations = current_q_original['explanation'].get('incorrect_explanations', {})
                else:
                    correct_explanation = current_q_original.get('explanation_correct', '')
                    incorrect_explanations = current_q_original.get('explanations_incorrect', {})
            
                # Map back to original option ID for explanation
                original_user_choice = self._map_paraphrased_to_original_option(
                    user_answer, paraphrased_q, current_q_original
                )
                incorrect_explanation = incorrect_explanations.get(original_user_choice, "This option is not optimal.")
            
                # Generate AI feedback
                feedback = await self._generate_feedback_with_context(current_q_original, user_answer, False)
            
                response = f"""That's not quite right. You selected {user_answer}: "{user_answer_text}"

The correct answer is {paraphrased_q['correct_answer']}: "{correct_option_text}"

Here's why:
✅ Correct choice: {correct_explanation}
❌ Your choice: {incorrect_explanation}

{feedback}"""
            
                # Update attempt with feedback
                attempt.ai_feedback = feedback
                session.question_attempts[-1] = attempt  # Update the last attempt
            
                # Add conversation to history
                user_msg = Message(role="user", content=answer, timestamp=datetime.now())
                bot_msg = Message(role="assistant", content=response, timestamp=datetime.now())
                session.conversation_history.extend([user_msg, bot_msg])
            
                # Move to next question
                next_response = await self._move_to_next_question(session)
                full_response = f"{response}\n\n{next_response}"
            
                return self._format_response(full_response, "educational")
            
        except Exception as e:
            return self._format_response(f"Error processing answer: {str(e)}", "apologetic")    
#     async def _handle_explanation_submission(self, explanation: str, conversation_history: List[Message]) -> str:
#         """Process and validate user's explanation"""
#         try:
#             session = await self._get_session_from_history(conversation_history)
#             current_q = session.questions_data[session.current_question_index]
            
#             # Validate explanation with LLM
#             validation_result = await self._validate_explanation_with_llm(current_q, explanation)
            
#             # Generate comprehensive feedback
#             if validation_result['is_valid']:
#                 feedback = f"""Excellent explanation! {validation_result['feedback']}

# Key points you covered:
# {chr(10).join('• ' + point for point in validation_result['key_points_covered'])}"""
#                 emotion = "proud"
#             else:
#                 feedback = f"""Your explanation shows some understanding, but let me help clarify:

# {validation_result['feedback']}

# Key points to consider:
# {chr(10).join('• ' + point for point in validation_result.get('missing_critical_points', []))}"""
#                 emotion = "educational"
            
#             # Record attempt
#             attempt = QuestionAttemptRecord(
#                 question_id=current_q['id'],
#                 original_question=current_q,
#                 paraphrased_question={},  # Would store the paraphrased version used
#                 user_answer=current_q['correct_answer'],  # They got it right to reach this point
#                 is_correct=True,
#                 user_explanation=explanation,
#                 explanation_validation=validation_result,
#                 ai_feedback=feedback
#             )
            
#             session.question_attempts.append(attempt.dict())
#             session.score += 1
            
#             # Move to next question
#             next_response = await self._move_to_next_question(session)
#             full_response = f"{feedback}\n\n{next_response}"
            
#             return self._format_response(full_response, emotion)
            
#         except Exception as e:
#             return self._format_response(f"Error processing explanation: {str(e)}", "apologetic")
    
    async def _handle_explanation_submission(self, explanation: str, conversation_history: List[Message]) -> str:
        """Process and validate user's explanation with detailed tracking"""
        try:
            session = await self._get_session_from_history(conversation_history)
            current_q_original = session.questions_data[session.current_question_index]
            
            # Validate explanation with LLM
            validation_result = await self._validate_explanation_with_llm(current_q_original, explanation)
            print(validation_result)
            # Generate comprehensive feedback
            if validation_result.get('is_valid', False):
                feedback = f"""Excellent explanation! {validation_result['feedback']}

Key points you covered:
{chr(10).join('• ' + point for point in validation_result['key_points_covered'])}"""
                emotion = "proud"
            else:
                feedback = f"""Your explanation shows some understanding, but let me help clarify:

{validation_result['feedback']}

Key points to consider:
{chr(10).join('• ' + point for point in validation_result.get('missing_critical_points', []))}"""
                emotion = "educational"
            
            # Update the last attempt with explanation and validation
            if session.question_attempts:
                last_attempt = session.question_attempts[-1]
                last_attempt.user_explanation = explanation
                last_attempt.explanation_validation = validation_result
                last_attempt.ai_feedback = feedback
                session.question_attempts[-1] = last_attempt
            
            # Award point for correct answer + explanation
            session.score += 1
            
            # Add conversation to history
            user_msg = Message(role="user", content=explanation, timestamp=datetime.now())
            bot_msg = Message(role="assistant", content=feedback, timestamp=datetime.now())
            session.conversation_history.extend([user_msg, bot_msg])
            
            # Move to next question
            next_response = await self._move_to_next_question(session)
            full_response = f"{feedback}\n\n{next_response}"
            
            return self._format_response(full_response, emotion)
            
        except Exception as e:
            return self._format_response(f"Error processing explanation: {str(e)}", "apologetic")    
    # ===== CORE HELPER METHODS =====
    
    async def load_scenarios(self):
        """Load questions from MongoDB for this scenario"""
        try:
            db = await self._get_db()
            # Use self.bot_description as scenario_name to get questions
            scenario_doc = await db.question_scenarios.find_one({
                "scenario_name": self.bot_description,
                "is_active": True
            })
            
            if scenario_doc:
                scenario = QuestionScenarioDoc(**scenario_doc)
                self.scenario_questions = scenario.questions
                self.scenario_context = scenario.scenario_context
                print(f"Loaded {len(self.scenario_questions)} questions for {self.bot_description}")
            else:
                print(f"No questions found for scenario: {self.bot_description}")
                self.scenario_questions = []
                
        except Exception as e:
            print(f"Error loading scenarios for {self.bot_description}: {e}")
            self.scenario_questions = []
    
    # async def _get_paraphrased_question(self, original_question: Dict, difficulty: str) -> Dict:
    #     """Get or generate paraphrased version of question"""
    #     try:
    #         db = await self._get_db()
    #         # Check cache first
    #         cached = await db.paraphrased_questions.find_one({
    #             "original_question_id": original_question['id'],
    #             "scenario_name": self.bot_description,
    #             "difficulty": difficulty,
    #             "is_active": True
    #         })
            
    #         if cached:
    #             return cached['paraphrased_data']
            
    #         # Generate new paraphrase using LLM
    #         paraphrased = await self._paraphrase_with_llm(original_question, difficulty)
            
    #         # Cache the result
    #         cache_doc = ParaphrasedQuestionCache(
    #             original_question_id=original_question['id'],
    #             scenario_name=self.bot_description,
    #             difficulty=difficulty,
    #             paraphrased_data=paraphrased
    #         )
    #         await db.paraphrased_questions.insert_one(cache_doc.dict())
            
    #         return paraphrased
            
    #     except Exception as e:
    #         print(f"Error getting paraphrased question: {e}")
    #         # Fallback to original question
    #         return original_question
    # 
    async def _get_paraphrased_question(self, original_question: Dict, difficulty: str) -> Dict:
        """Always generate fresh paraphrased version - no caching"""
        try:
            # Always generate new paraphrase using LLM
            paraphrased = await self._paraphrase_with_llm(original_question, difficulty)
            return paraphrased
        
        except Exception as e:
            print(f"Error getting paraphrased question: {e}")
            # Fallback to original question
            return original_question
    # 

    # async def _paraphrase_with_llm(self, original_question: Dict, difficulty: str) -> Dict:
    #     try:
    #         prompt_template = self._get_paraphrasing_prompt(difficulty)

    #         prompt = prompt_template.format(
    #         question_text=original_question['question_text'],
    #         options=json.dumps([opt for opt in original_question['options']], indent=2),
    #         scenario_context=self.scenario_context,
    #         correct_answer=original_question['correct_answer'],
    #         category=original_question.get('category', ''),
    #         competencies=', '.join(original_question.get('competencies_tested', []))
    #         )

    #         contents = [
    #         types.Content(
    #             role="user",
    #             parts=[types.Part.from_text(text=prompt)]
    #         )
    #         ]

    #         response = await self.model.aio.models.generate_content(
    #         model=self.llm_model,
    #         contents=contents,
    #         )

    #         response_text = response.text.strip()
    #         response_text = response_text.replace('```json', '').replace('```', '').strip()
    #         paraphrased_data = json.loads(response_text)

    #         # SHUFFLE OPTIONS AND TRACK CORRECT ANSWER
    #         import random
    #         original_options = original_question['options']
    #         correct_option_text = next(
    #             opt['text'] for opt in original_options if opt['option_id'] == original_question['correct_answer']
    #         )

    #         # Create list of (paraphrased_option, is_correct) pairs
    #         options_with_correct = []
    #         for i, paraphrased_opt in enumerate(paraphrased_data['options']):
    #             original_opt = original_options[i]  # Same index mapping
    #             is_correct = (original_opt['option_id'] == original_question['correct_answer'])
    #             options_with_correct.append((paraphrased_opt['text'], is_correct))

    #         # Shuffle the pairs
    #         random.shuffle(options_with_correct)

    #         # Rebuild with new option IDs and find new correct answer
    #         new_options = []
    #         new_correct_answer = None
    #         for i, (option_text, is_correct) in enumerate(options_with_correct):
    #             new_id = chr(65 + i)  # A, B, C, D
    #             new_options.append({"option_id": new_id, "text": option_text})
    #             if is_correct:
    #                 new_correct_answer = new_id

    #         # Build final paraphrased question with shuffled options
    #         final_paraphrased = {
    #         'id': original_question['id'],
    #         'question_text': paraphrased_data['question_text'],
    #         'options': new_options,
    #         'correct_answer': new_correct_answer,
    #         'category': original_question.get('category', ''),
    #         'explanation': {
    #             'correct_explanation': original_question.get('explanation_correct', ''),
    #             'incorrect_explanations': original_question.get('explanations_incorrect', {})
    #             }
    #         }

    #         return final_paraphrased

    #     except Exception as e:
    #         print(f"Error in LLM paraphrasing: {e}")
    #         return {
    #         'id': original_question['id'],
    #         'question_text': original_question['question_text'],
    #         'options': original_question['options'],
    #         'correct_answer': original_question['correct_answer'],
    #         'category': original_question.get('category', ''),
    #         'explanation': {
    #             'correct_explanation': original_question.get('explanation_correct', ''),
    #             'incorrect_explanations': original_question.get('explanations_incorrect', {})
    #             }
    #         }
    async def _paraphrase_with_llm(self, original_question: Dict, difficulty: str) -> Dict:
        """Use LLM to paraphrase question based on difficulty"""
        try:
            prompt_template = self._get_paraphrasing_prompt(difficulty)
            
            prompt = prompt_template.format(
                question_text=original_question['question_text'],
                options=json.dumps([opt for opt in original_question['options']], indent=2),
                scenario_context=self.scenario_context,
                correct_answer=original_question['correct_answer'],
                category=original_question.get('category', ''),
                competencies=', '.join(original_question.get('competencies_tested', []))
            )
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
            
            response = await self.model.aio.models.generate_content(
                model=self.llm_model,
                contents=contents,
            )
            
            # IMPROVED JSON PARSING
            response_text = response.text.strip()
            
            # Remove code blocks and extra formatting
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Print raw response for debugging
            print(f"Raw LLM Response: {response_text}")
            
            try:
                paraphrased_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"Problematic text: {response_text[max(0, e.pos-50):e.pos+50]}")
                
                # TRY TO FIX COMMON JSON ISSUES
                # Fix trailing commas
                response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
                # Fix missing quotes around keys
                response_text = re.sub(r'(\w+):', r'"\1":', response_text)
                # Try parsing again
                try:
                    paraphrased_data = json.loads(response_text)
                    print("Fixed JSON successfully!")
                except:
                    print("Could not fix JSON, using original question")
                    raise
            
            # SHUFFLE OPTIONS with better error handling
            import random
            
            if 'options' not in paraphrased_data or len(paraphrased_data['options']) != 4:
                print("Invalid options in paraphrased data, using original")
                return self._format_fallback_question(original_question)
            
            # Create shuffleable pairs
            original_options = original_question['options']
            paraphrased_options = paraphrased_data['options']
            
            if len(original_options) != len(paraphrased_options):
                print("Option count mismatch, using original")
                return self._format_fallback_question(original_question)
            
            # Create (paraphrased_text, is_correct) pairs
            options_with_correct = []
            for i in range(len(original_options)):
                original_opt = original_options[i]
                paraphrased_opt = paraphrased_options[i]
                is_correct = (original_opt['option_id'] == original_question['correct_answer'])
                options_with_correct.append((paraphrased_opt['text'], is_correct))
            
            # Shuffle
            random.shuffle(options_with_correct)
            
            # Rebuild with new IDs
            new_options = []
            new_correct_answer = None
            for i, (option_text, is_correct) in enumerate(options_with_correct):
                new_id = chr(65 + i)  # A, B, C, D
                new_options.append({"option_id": new_id, "text": option_text})
                if is_correct:
                    new_correct_answer = new_id
            
            # Build final result
            final_paraphrased = {
                'id': original_question['id'],
                'question_text': paraphrased_data.get('question_text', original_question['question_text']),
                'options': new_options,
                'correct_answer': new_correct_answer,
                'category': original_question.get('category', ''),
                'explanation': {
                    'correct_explanation': original_question.get('explanation_correct', ''),
                    'incorrect_explanations': original_question.get('explanations_incorrect', {})
                }
            }
            
            return final_paraphrased
            
        except Exception as e:
            print(f"Error in LLM paraphrasing: {e}")
            return self._format_fallback_question(original_question)

    def _format_fallback_question(self, original_question: Dict) -> Dict:
        """Format original question with proper structure when paraphrasing fails"""
        return {
            'id': original_question['id'],
            'question_text': original_question['question_text'],
            'options': original_question['options'],
            'correct_answer': original_question['correct_answer'],
            'category': original_question.get('category', ''),
            'explanation': {
                'correct_explanation': original_question.get('explanation_correct', ''),
                'incorrect_explanations': original_question.get('explanations_incorrect', {})
            }
        }
    # async def _update_session_with_conversation(self, session: QuestionSession, 
    #                                       user_message: str, bot_response: str):
    #     # Add to conversation history
    #     user_msg = Message(role="user", content=user_message, timestamp=datetime.now())
    #     bot_msg = Message(role="assistant", content=bot_response, timestamp=datetime.now())
    
    #     session.conversation_history.extend([user_msg, bot_msg])
    #     await self._update_session(session)
    async def _update_session_with_conversation(self, session: QuestionSession, user_message: str, bot_response: str):
        """Add conversation messages to session history"""
        user_msg = Message(role="user", content=user_message, timestamp=datetime.now())
        bot_msg = Message(role="assistant", content=bot_response, timestamp=datetime.now())
        session.conversation_history.extend([user_msg, bot_msg])
        await self._update_session(session)

    
#     async def _validate_explanation_with_llm(self, question: Dict, user_explanation: str) -> Dict:
#         """Use LLM to validate user's explanation"""
#         try:
#             official_justification = question['explanation']['correct_explanation']
#             prompt = f"""
# EVALUATE USER'S EXPLANATION:

# QUESTION: {question['question_text']}
# CORRECT ANSWER: {question['correct_answer']}
# OFFICIAL REASON: {question['explanation']['correct_explanation']}
# USER'S EXPLANATION: {user_explanation}

# Return JSON with:
# 1. is_valid: true/false
# 2. feedback: WHY their explanation is right/wrong 
# 3. what_they_got_right: specific correct points
# 4. what_they_missed: specific missing points

# {{
#     "is_valid": true/false,
#     "feedback": "Detailed explanation of why their reasoning is correct/incorrect",
#     "what_they_got_right": ["point1", "point2"], 
#     "what_they_missed": ["missing1", "missing2"],
#     "overall_understanding": "summary of their grasp of the concept"
# }}"""
# #             prompt = f"""
# # You are evaluating a banking leadership trainee's explanation.

# # SCENARIO CONTEXT: {self.scenario_context}

# # QUESTION: {question['question_text']}
# # CORRECT ANSWER: {question['correct_answer']}
# # OFFICIAL JUSTIFICATION: {official_justification}

# # USER'S EXPLANATION: {user_explanation}

# # Evaluate if the user's explanation demonstrates understanding of the key leadership concepts.
# # Focus on whether they understand the WHY behind the correct answer.

# # Return ONLY JSON:
# # {{
# #     "is_valid": true/false,
# #     "feedback": "detailed feedback on their reasoning (2-3 sentences)",
# #     "key_points_covered": ["point1", "point2"],
# #     "missing_critical_points": ["missing1", "missing2"]
# # }}
# # """
            
#             contents = [
#                 types.Content(
#                     role="user",
#                     parts=[types.Part.from_text(text=prompt)]
#                 )
#             ]
            
#             response = await self.model.aio.models.generate_content(
#                 model=self.llm_model,
#                 contents=contents,
#             )
            
#             response_text = response.text.strip()
#             response_text = response_text.replace('```json', '').replace('```', '').strip()
            
#             return json.loads(response_text)
            
#         except Exception as e:
#             print(f"Error validating explanation: {e}")
#             return {
#                 "is_valid": True,
#                 "feedback": "Thank you for your explanation. Let's continue with the next question.",
#                 "key_points_covered": [],
#                 "missing_critical_points": []
#             }
    
    async def _validate_explanation_with_llm(self, question: Dict, user_explanation: str) -> Dict:
        try:
            # FIX: Handle your explanation structure
            if 'explanation' in question:
                official_justification = question['explanation']['correct_explanation']
            else:
                official_justification = question.get('explanation_correct', '')
        
            prompt = f"""
You are evaluating a banking leadership trainee's explanation.

SCENARIO CONTEXT: {self.scenario_context}

QUESTION: {question['question_text']}
CORRECT ANSWER: {question['correct_answer']}
OFFICIAL JUSTIFICATION: {official_justification}

USER'S EXPLANATION: {user_explanation}

Evaluate if the user's explanation demonstrates understanding of the key leadership concepts.
Focus on whether they understand the WHY behind the correct answer.

Return ONLY JSON:
{{
    "is_valid": true/false,
    "feedback": "Detailed explanation of WHY their reasoning is correct or incorrect. Be specific about what they understood or missed.",
    "key_points_covered": ["specific point 1", "specific point 2"],
    "missing_critical_points": ["what they missed 1", "what they missed 2"]
}}
"""
        
            contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
            ]
        
            response = await self.model.aio.models.generate_content(
            model=self.llm_model,
            contents=contents,
            )
        
            response_text = response.text.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
            return json.loads(response_text)
        
        except Exception as e:
            print(f"Error validating explanation: {e}")
            return {
            "is_valid": True,
            "feedback": "Thank you for your explanation. Let's continue with the next question.",
            "key_points_covered": [],
            "missing_critical_points": []
            }
    
    # ===== SESSION MANAGEMENT =====
    

    # async def _get_or_create_session(self, session_id: str, difficulty: str) -> QuestionSession:
    #     """Create session with filtered questions based on difficulty"""
    #     try:
    #         db = await self._get_db()
            
    #         # Try to get existing session first
    #         session_data = await db.question_chat_sessions.find_one({"session_id": session_id})
    #         if session_data:
    #             return QuestionSession(**session_data)
            
    #         # FILTER QUESTIONS BY DIFFICULTY
    #         if difficulty == "easy":
    #             # Get questions marked as basic/easy
    #             filtered_questions = [q for q in self.scenario_questions 
    #                                 if q.get('source_difficulty') == 'easy' or 
    #                                    'basic' in q.get('category', '').lower()]
    #         else:  # hard
    #             # Get questions marked as advanced/hard  
    #             filtered_questions = [q for q in self.scenario_questions 
    #                                 if q.get('source_difficulty') == 'hard' or 
    #                                    'advanced' in q.get('category', '').lower()]
            
    #         # If no filtered questions, take first 15
    #         if not filtered_questions:
    #             filtered_questions = self.scenario_questions[:15]
    #         elif len(filtered_questions) > 15:
    #             filtered_questions = filtered_questions[:15]
            
    #         print(f"Creating session with {len(filtered_questions)} {difficulty} questions")
            
    #         # Generate paraphrased versions of filtered questions
    #         paraphrased_questions = []
    #         for question in filtered_questions:
    #             paraphrased_q = await self._get_paraphrased_question(question, difficulty)
    #             paraphrased_questions.append(paraphrased_q)
            
    #         session = QuestionSession(
    #             session_id=session_id,
    #             scenario_name=self.bot_description,
    #             difficulty=difficulty,
    #             questions_data=filtered_questions,  # Only easy or hard questions
    #             paraphrased_questions_used=paraphrased_questions,
    #             total_questions=len(filtered_questions),
    #             current_state="awaiting_answer"
    #         )
            
    #         await db.question_chat_sessions.insert_one(session.dict())
    #         print(f"Session created with {len(filtered_questions)} {difficulty} questions")
            
    #         return session
            
    #     except Exception as e:
    #         print(f"Error creating session: {e}")
    #         # Fallback: use first 15 questions
    #         fallback_questions = self.scenario_questions[:15]
    #         session = QuestionSession(
    #             session_id=session_id,
    #             scenario_name=self.bot_description,
    #             difficulty=difficulty,
    #             questions_data=fallback_questions,
    #             paraphrased_questions_used=fallback_questions,
    #             total_questions=len(fallback_questions),
    #             current_state="awaiting_answer"
    #         )
    #         return session    
    async def _get_or_create_session(self, session_id: str, difficulty: str) -> QuestionSession:
        """Create session with filtered questions based on difficulty - ASYNC VERSION"""
        try:
            db = await self._get_db()
            
            # Try to get existing session first
            session_data = await db.question_chat_sessions.find_one({"session_id": session_id})
            if session_data:
                return QuestionSession(**session_data)
            
            # Filter questions by difficulty (same logic)
            if difficulty == "easy":
                filtered_questions = [q for q in self.scenario_questions 
                                    if q.get('source_difficulty') == 'easy' or 
                                       'basic' in q.get('category', '').lower()]
            else:
                filtered_questions = [q for q in self.scenario_questions 
                                    if q.get('source_difficulty') == 'hard' or 
                                       'advanced' in q.get('category', '').lower()]
            
            if not filtered_questions:
                filtered_questions = self.scenario_questions[:15]
            elif len(filtered_questions) > 15:
                filtered_questions = filtered_questions[:15]
            
            print(f"Creating session with {len(filtered_questions)} {difficulty} questions")
            
            # CREATE ALL PARAPHRASES CONCURRENTLY - MUCH FASTER
            import asyncio
            
            async def paraphrase_single(question):
                return await self._get_paraphrased_question(question, difficulty)
            
            # Run all paraphrasing tasks concurrently
            paraphrasing_tasks = [paraphrase_single(q) for q in filtered_questions]
            paraphrased_questions = await asyncio.gather(*paraphrasing_tasks, return_exceptions=True)
            
            # Handle any failed paraphrases
            final_paraphrased = []
            for i, result in enumerate(paraphrased_questions):
                if isinstance(result, Exception):
                    print(f"Paraphrasing failed for question {i}: {result}")
                    # Use fallback formatting for failed questions
                    final_paraphrased.append(self._format_fallback_question(filtered_questions[i]))
                else:
                    final_paraphrased.append(result)
            
            session = QuestionSession(
                session_id=session_id,
                scenario_name=self.bot_description,
                difficulty=difficulty,
                questions_data=filtered_questions,
                paraphrased_questions_used=final_paraphrased,
                total_questions=len(filtered_questions),
                current_state="awaiting_answer"
            )
            
            await db.question_chat_sessions.insert_one(session.dict())
            print(f"Session created with {len(final_paraphrased)} questions in parallel")
            
            return session
            
        except Exception as e:
            print(f"Error creating session: {e}")
            # Fallback to original questions
            fallback_questions = self.scenario_questions[:15]
            session = QuestionSession(
                session_id=session_id,
                scenario_name=self.bot_description,
                difficulty=difficulty,
                questions_data=fallback_questions,
                paraphrased_questions_used=fallback_questions,
                total_questions=len(fallback_questions),
                current_state="awaiting_answer"
            )
            return session    
    
    async def _update_session(self, session: QuestionSession):
        """Update session in database"""
        db = await self._get_db()
        session.last_updated = datetime.now()
        await db.question_chat_sessions.update_one(
            {"session_id": session.session_id},
            {"$set": session.dict()}
        )
    
#     async def _move_to_next_question(self, session: QuestionSession) -> str:
#         """Move to next question or complete session"""
#         try:
#             session.current_question_index += 1
            
#             if session.current_question_index >= len(session.questions_data):
#                 # Session completed
#                 session.is_completed = True
#                 session.current_state = "completed"
#                 await self._update_session(session)
                
#                 final_score = (session.score / session.total_questions) * 100
#                 return f"""
# 🎉 Training Complete! 

# Final Score: {session.score}/{session.total_questions} ({final_score:.1f}%)
# {'Excellent work!' if final_score >= 80 else 'Good effort! Consider reviewing the areas for improvement.'}

# [FINISH]"""
            
#             else:
#                 # Get next question
#                 next_q = session.questions_data[session.current_question_index]
#                 paraphrased_q = await self._get_paraphrased_question(next_q, session.difficulty)
                
#                 session.current_state = "awaiting_answer"
#                 await self._update_session(session)
                
#                 return f"""
# Next Question ({session.current_question_index + 1}/{session.total_questions}):

# {paraphrased_q['question_text']}

# A) {paraphrased_q['options'][0]['text']}
# B) {paraphrased_q['options'][1]['text']}
# C) {paraphrased_q['options'][2]['text']}
# D) {paraphrased_q['options'][3]['text']}

# Please select your answer (A, B, C, or D)."""
        
#         except Exception as e:
#             return f"Error moving to next question: {str(e)}"
    
    async def _move_to_next_question(self, session: QuestionSession) -> str:
        """Move to next question using pre-generated paraphrased questions"""
        try:
            session.current_question_index += 1
            
            if session.current_question_index >= len(session.questions_data):
                # Session completed
                session.is_completed = True
                session.current_state = "completed"
                
                # Add completion message to conversation
                completion_msg = Message(
                    role="assistant", 
                    content="Training completed!", 
                    timestamp=datetime.now()
                )
                session.conversation_history.append(completion_msg)
                
                await self._update_session(session)
                
                final_score = (session.score / session.total_questions) * 100
                return f"""
🎉 Training Complete! 

Final Score: {session.score}/{session.total_questions} ({final_score:.1f}%)
{'Excellent work!' if final_score >= 80 else 'Good effort! Review the feedback to improve.'}

[FINISH]"""
            
            else:
                # Get next paraphrased question (already generated)
                next_q = session.paraphrased_questions_used[session.current_question_index]
                
                session.current_state = "awaiting_answer"
                await self._update_session(session)
                
                next_question_text = f"""
Next Question ({session.current_question_index + 1}/{session.total_questions}):

{next_q['question_text']}

A) {next_q['options'][0]['text']}
B) {next_q['options'][1]['text']}
C) {next_q['options'][2]['text']}
D) {next_q['options'][3]['text']}

Please select your answer (A, B, C, or D)."""
                
                # Add question to conversation history
                question_msg = Message(
                    role="assistant", 
                    content=next_question_text, 
                    timestamp=datetime.now()
                )
                session.conversation_history.append(question_msg)
                await self._update_session(session)
                
                return next_question_text
        
        except Exception as e:
            return f"Error moving to next question: {str(e)}"    
    # ===== LLM INTEGRATION METHODS =====
    
    def _get_paraphrasing_prompt(self, difficulty: str) -> str:
        """Get appropriate prompt template for question paraphrasing"""
        
        if difficulty == "easy":
            return """
Rephrase this banking leadership question to be EASIER for new managers:
- Add context clues and helpful hints
- Use simpler, clearer language  
- Include more descriptive scenario details
- Make the correct option more distinguishable
- Add explanatory phrases that guide thinking

SCENARIO CONTEXT: {scenario_context}
ORIGINAL QUESTION: {question_text}
ORIGINAL OPTIONS: {options}
CORRECT ANSWER: {correct_answer}
COMPETENCIES BEING TESTED: {competencies}

Create a version that helps guide new managers toward the right thinking.

Return ONLY JSON:
{{
    "question_text": "rephrased easier question with context clues",
    "options": [
        {{"option_id": "A", "text": "option with more context"}},
        {{"option_id": "B", "text": "option with more context"}},
        {{"option_id": "C", "text": "option with more context"}}, 
        {{"option_id": "D", "text": "option with more context"}}
    ]
}}
"""
        else:  # hard
            return """
Rephrase this banking leadership question to be HARDER for experienced managers:
- Remove obvious hints and context clues
- Use advanced professional terminology
- Add complexity and subtle nuances
- Make incorrect options more plausible and tempting
- Require deeper leadership knowledge to differentiate

SCENARIO CONTEXT: {scenario_context}  
ORIGINAL QUESTION: {question_text}
ORIGINAL OPTIONS: {options}
CORRECT ANSWER: {correct_answer}
COMPETENCIES BEING TESTED: {competencies}

Create a version that challenges experienced managers' expertise.

Return ONLY JSON:
{{
    "question_text": "rephrased harder question without hints",
    "options": [
        {{"option_id": "A", "text": "subtle professional option"}},
        {{"option_id": "B", "text": "subtle professional option"}},
        {{"option_id": "C", "text": "subtle professional option"}},
        {{"option_id": "D", "text": "subtle professional option"}}
    ]
}}
"""
    
    async def _generate_feedback_with_context(self, question: Dict, user_answer: str, is_correct: bool) -> str:
        """Generate contextual feedback using LLM"""
        try:
            prompt = f"""
You are a banking leadership expert providing feedback to a trainee.

SCENARIO CONTEXT: {self.scenario_context}
QUESTION: {question['question_text']}
TRAINEE'S CHOICE: {user_answer}
CORRECT ANSWER: {question['correct_answer']}
WAS CORRECT: {is_correct}

COMPETENCIES BEING TESTED: {', '.join(question.get('competencies_tested', []))}

Provide encouraging, educational feedback (2-3 sentences) that:
- Acknowledges their thinking process
- Explains the key leadership principle involved
- Encourages continued learning
- Relates to real banking leadership scenarios

Keep it positive and constructive, not judgmental.
"""
            
            contents = [
                types.Content(
                    role="user", 
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
            
            response = await self.model.aio.models.generate_content(
                model=self.llm_model,
                contents=contents,
            )
            
            return response.text.strip()
            
        except Exception as e:
            return "Keep learning from each question - every challenge helps build stronger leadership skills!"
    
    # ===== UTILITY METHODS =====
    
    def _determine_state_from_conversation(self, conversation_history: List[Message]) -> str:
        """Determine current conversation state"""
        if not conversation_history:
            return "start"
        
        last_message = conversation_history[-1].content.lower()
        
        if "start" in last_message or len(conversation_history) <= 1:
            return "start"
        elif any(char in last_message for char in ['a)', 'b)', 'c)', 'd)']):
            return "awaiting_answer"
        elif "explain" in conversation_history[-2].content.lower() if len(conversation_history) >= 2 else False:
            return "awaiting_explanation"
        else:
            return "awaiting_answer"
    
    def _extract_answer_from_message(self, message: str) -> Optional[str]:
        """Extract A/B/C/D from user message"""
        message = message.upper()
        # Look for patterns like "A", "A)", "OPTION A", etc.
        patterns = [r'\b([ABCD])\)', r'\b([ABCD])\b', r'OPTION\s*([ABCD])']
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        return None
    
    async def _extract_session_id_from_history(self, conversation_history: List[Message]) -> str:
        """Extract session ID from conversation or generate new one"""
        db = await self._get_db()
        print(conversation_history)
        # You might have session ID in your existing chat system
        # For now, generate based on conversation length or use existing method
        return str(uuid.uuid4())
    
    def _format_response(self, content: str, emotion: str = "neutral", complete: bool = False) -> str:
        """Format response to match your existing chat format"""
        if complete:
            return f"${emotion}${content}[FINISH]"
        else:
            return f"${emotion}${content}"
    
    async def _get_session_from_history(self, conversation_history: List[Message]) -> QuestionSession:
        """Get current session from conversation history"""
        db = await self._get_db()
        # Implementation depends on how you track sessions in your current system
        # This is a placeholder - you'll need to adapt to your session tracking
        session_id =await self._extract_session_id_from_history(conversation_history)
        session_data = await db.question_chat_sessions.find_one({"session_id": session_id})
        return QuestionSession(**session_data) if session_data else None
    
    # ===== API METHODS FOR QUESTION MANAGEMENT =====
    
    async def create_paraphrase_set(self, question_id: str, difficulties: List[str] = ["easy", "hard"]) -> Dict:
        """API method: Generate paraphrases for a specific question"""
        try:
            db = await self._get_db()
            original_question = next((q for q in self.scenario_questions if q['id'] == question_id), None)
            if not original_question:
                raise Exception("Question not found")
            
            results = {}
            for difficulty in difficulties:
                paraphrased = await self._paraphrase_with_llm(original_question, difficulty)
                results[difficulty] = paraphrased
                
                # Cache the result
                cache_doc = ParaphrasedQuestionCache(
                    original_question_id=question_id,
                    scenario_name=self.bot_description,
                    difficulty=difficulty,
                    paraphrased_data=paraphrased
                )
                await db.paraphrased_questions.insert_one(cache_doc.dict())
            
            return {
                "question_id": question_id,
                "paraphrases_generated": results,
                "message": "Paraphrases created successfully"
            }
            
        except Exception as e:
            raise Exception(f"Error creating paraphrase set: {str(e)}")