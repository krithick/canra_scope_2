# question_bot.py - Simplified for API-only usage
import json
import re
import uuid
import asyncio
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from datetime import datetime
from models import (
    Message, BotConfig, QuestionScenarioDoc, ParaphrasedQuestionCache,
    QuestionSession, QuestionAttemptRecord
)
from bots import BaseLLMBot


class QuestionBot(BaseLLMBot):
    """
    API-focused question training bot - NO chat integration
    Only used through dedicated /api/question-sessions/* endpoints
    """
    
    def __init__(self, config: BotConfig, llm_client):
        super().__init__(config, llm_client)
        self.scenario_questions = []
        self.scenario_context = ""

    async def _get_db(self):
        """Helper to get database instance"""
        from database import get_db
        return await get_db()

    # ===== CORE API METHODS =====

    async def start_new_session(self, difficulty: str = "easy") -> Dict:
        """
        API: POST /api/question-sessions/start
        Creates new training session with first question
        """
        try:
            # Filter questions by difficulty
            filtered_questions = self._filter_questions_by_difficulty(difficulty)
            if not filtered_questions:
                raise Exception("No questions available for this difficulty")

            # Generate paraphrased questions concurrently
            paraphrased_questions = await self._generate_all_paraphrases(filtered_questions, difficulty)
            
            # Create session
            session = QuestionSession(
                session_id=str(uuid.uuid4()),
                scenario_name=self.bot_description,
                difficulty=difficulty,
                questions_data=filtered_questions,
                paraphrased_questions_used=paraphrased_questions,
                total_questions=len(filtered_questions),
                current_state="awaiting_answer"
            )
            
            # Save to database
            db = await self._get_db()
            await db.create_question_session(session)
            
            # Return first question
            first_question = paraphrased_questions[0]
            
            return {
                "session_id": session.session_id,
                "scenario_name": session.scenario_name,
                "difficulty": session.difficulty,
                "total_questions": session.total_questions,
                "current_question": first_question,
                "question_number": 1
            }
            
        except Exception as e:
            raise Exception(f"Error starting session: {str(e)}")

    async def submit_answer(self, session_id: str, user_input: str, time_taken: int = 30, is_timeout: bool = False, retry_count: int = 0) -> Dict:
        """
        API: POST /api/question-sessions/{session_id}/answer
        Processes answer submission (speech or timeout)
        """
        try:
            db = await self._get_db()
            session = await db.get_question_session(session_id)
            
            if not session:
                raise Exception("Session not found")
                
            if session.current_question_index >= len(session.questions_data):
                raise Exception("No more questions available")

            # Get current questions
            current_q_original = session.questions_data[session.current_question_index]
            paraphrased_q = session.paraphrased_questions_used[session.current_question_index]
            
            # Handle timeout case
            if is_timeout:
                return await self._handle_timeout_or_invalid(session, current_q_original, paraphrased_q, time_taken, db, "timeout")
            
            # Process speech input to extract option
            user_answer = await self._extract_option_from_speech(user_input, paraphrased_q)
            
            if not user_answer:
                print(retry_count,"retry_count - could not extract answer")
                # Check retry limit (max 1 retry)
                if retry_count >= 1:
                    return await self._handle_timeout_or_invalid(session, current_q_original, paraphrased_q, time_taken, db, "invalid")
                
                feedback_msg = "I couldn't understand which option you selected. Please clearly state A, B, C, or D, or mention the option text."
                return {
                    "is_valid_answer": False,
                    "feedback": feedback_msg,
                    "current_score": session.score,
                    "question_number": session.current_question_index + 1,
                    "total_questions": session.total_questions,
                    "retry_allowed": True,
                    "retry_count": retry_count + 1
                }
            
            # Check correctness
            is_correct = user_answer.upper() == paraphrased_q['correct_answer'].upper()
            
            # Get answer text
            user_answer_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == user_answer
            )
            
            # Create attempt record
            attempt = QuestionAttemptRecord(
                question_id=current_q_original['id'],
                original_question=current_q_original,
                paraphrased_question=paraphrased_q,
                user_answer=user_answer,
                user_answer_text=user_answer_text,
                correct_answer_original=current_q_original['correct_answer'],
                correct_answer_paraphrased=paraphrased_q['correct_answer'],
                is_correct=is_correct,
                time_taken_seconds=time_taken
            )
            
            session.question_attempts.append(attempt.dict())
            
            if is_correct:
                # Correct - ask for explanation
                session.current_state = "awaiting_explanation"
                await db.update_question_session(session)
                
                return {
                    "is_correct": True,
                    "feedback": f"Correct! You selected {user_answer}: \"{user_answer_text}\"",
                    "awaiting_explanation": True,
                    "explanation_prompt": "Please explain WHY this is the best choice. What leadership principles led you to this decision?",
                    "current_score": session.score,
                    "question_number": session.current_question_index + 1,
                    "total_questions": session.total_questions
                }
                
            else:
                # Incorrect - provide feedback and move to next
                correct_option_text = next(
                    opt['text'] for opt in paraphrased_q['options'] 
                    if opt['option_id'] == paraphrased_q['correct_answer']
                )
                
                feedback = await self._generate_incorrect_feedback(current_q_original, user_answer, paraphrased_q)
                
                # Move to next question
                session.current_question_index += 1
                session.current_state = "awaiting_answer"
                
                # Check if session completed
                is_completed = session.current_question_index >= len(session.questions_data)
                if is_completed:
                    session.is_completed = True
                    session.current_state = "completed"
                
                await db.update_question_session(session)
                
                # Get next question if available
                next_question = None
                if not is_completed:
                    next_question = session.paraphrased_questions_used[session.current_question_index]
                
                return {
                    "is_correct": False,
                    "feedback": feedback,
                    "correct_answer": paraphrased_q['correct_answer'],
                    "correct_answer_text": correct_option_text,
                    "awaiting_explanation": False,
                    "current_score": session.score,
                    "question_number": session.current_question_index,
                    "total_questions": session.total_questions,
                    "is_completed": is_completed,
                    "next_question": next_question
                }
                
        except Exception as e:
            raise Exception(f"Error submitting answer: {str(e)}")

    async def submit_explanation(self, session_id: str, explanation: str) -> Dict:
        """
        API: POST /api/question-sessions/{session_id}/explain
        Validates explanation and moves to next question
        """
        try:
            db = await self._get_db()
            session = await db.get_question_session(session_id)
            
            if not session:
                raise Exception("Session not found")
                
            if session.current_state != "awaiting_explanation":
                raise Exception("Not expecting explanation at this time")
            
            # Get current question for validation
            current_q = session.questions_data[session.current_question_index]
            
            # Validate explanation with AI
            validation_result = await self._validate_explanation_with_llm(current_q, explanation)
            
            # Update last attempt with explanation
            if session.question_attempts:
                last_attempt = session.question_attempts[-1]
                last_attempt['user_explanation'] = explanation
                last_attempt['explanation_validation'] = validation_result
                session.question_attempts[-1] = last_attempt
            
            # Award point for correct answer + explanation
            session.score += 1
            
            # Move to next question
            session.current_question_index += 1
            session.current_state = "awaiting_answer"
            
            # Check if completed
            is_completed = session.current_question_index >= len(session.questions_data)
            if is_completed:
                session.is_completed = True
                session.current_state = "completed"
            
            await db.update_question_session(session)
            
            # Get next question if available
            next_question = None
            if not is_completed:
                next_question = session.paraphrased_questions_used[session.current_question_index]
            
            return {
                "explanation_valid": validation_result.get("is_valid", True),
                "feedback": validation_result.get("feedback", "Thank you for your explanation."),
                "key_points_covered": validation_result.get("key_points_covered", []),
                "missing_points": validation_result.get("missing_critical_points", []),
                "current_score": session.score,
                "question_number": session.current_question_index,
                "total_questions": session.total_questions,
                "is_completed": is_completed,
                "next_question": next_question
            }
            
        except Exception as e:
            raise Exception(f"Error processing explanation: {str(e)}")

    async def get_session_results(self, session_id: str) -> Dict:
        """
        API: GET /api/question-sessions/{session_id}/results
        Returns complete session analytics
        """
        try:
            db = await self._get_db()
            session = await db.get_question_session(session_id)
            
            if not session:
                raise Exception("Session not found")
                
            if not session.is_completed:
                raise Exception("Session not yet completed")
            
            # Calculate results
            final_score = session.score
            total_questions = session.total_questions
            percentage = (final_score / total_questions * 100) if total_questions > 0 else 0
            
            # Competency analysis
            competency_scores = self._calculate_competency_scores(session.question_attempts)
            
            # Performance insights
            insights = self._generate_performance_insights(session.question_attempts, percentage)
            
            return {
                "session_id": session_id,
                "scenario_name": session.scenario_name,
                "difficulty": session.difficulty,
                "final_score": final_score,
                "total_questions": total_questions,
                "percentage_score": round(percentage, 1),
                "passed": percentage >= 70,
                "excellence": percentage >= 85,
                "competency_scores": competency_scores,
                "performance_insights": insights,
                "detailed_attempts": session.question_attempts,
                "session_duration_minutes": (session.last_updated - session.created_at).total_seconds() / 60,
                "created_at": session.created_at.isoformat(),
                "completed_at": session.last_updated.isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Error getting results: {str(e)}")

    # ===== HELPER METHODS =====

    async def load_scenarios(self):
        """Load questions from MongoDB for this scenario"""
        try:
            db = await self._get_db()
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
            print(f"Error loading scenarios: {e}")
            self.scenario_questions = []

    def _filter_questions_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filter questions by difficulty level"""
        if difficulty == "easy":
            filtered = [q for q in self.scenario_questions 
                       if q.get('source_difficulty') == 'easy' or 
                          'basic' in q.get('category', '').lower()]
        else:
            filtered = [q for q in self.scenario_questions 
                       if q.get('source_difficulty') == 'hard' or 
                          'advanced' in q.get('category', '').lower()]
        
        if not filtered:
            filtered = self.scenario_questions[:15]
        elif len(filtered) > 15:
            filtered = filtered[:15]
            
        return filtered

    async def _generate_all_paraphrases(self, questions: List[Dict], difficulty: str) -> List[Dict]:
        """Generate paraphrases for all questions concurrently"""
        async def paraphrase_single(question):
            return await self._paraphrase_with_llm(question, difficulty)
        
        tasks = [paraphrase_single(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        paraphrased = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Paraphrasing failed for question {i}: {result}")
                paraphrased.append(self._format_fallback_question(questions[i]))
            else:
                paraphrased.append(result)
        
        return paraphrased

    async def _paraphrase_with_llm(self, original_question: Dict, difficulty: str) -> Dict:
        """Generate paraphrased question using LLM"""
        try:
            prompt = self._get_paraphrasing_prompt(difficulty).format(
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
            
            response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            paraphrased_data = json.loads(response_text)
            
            # Shuffle options and track correct answer
            import random
            original_options = original_question['options']
            paraphrased_options = paraphrased_data['options']
            
            options_with_correct = []
            for i in range(len(original_options)):
                original_opt = original_options[i]
                paraphrased_opt = paraphrased_options[i]
                is_correct = (original_opt['option_id'] == original_question['correct_answer'])
                options_with_correct.append((paraphrased_opt['text'], is_correct))
            
            random.shuffle(options_with_correct)
            
            new_options = []
            new_correct_answer = None
            for i, (option_text, is_correct) in enumerate(options_with_correct):
                new_id = chr(65 + i)  # A, B, C, D
                new_options.append({"option_id": new_id, "text": option_text})
                if is_correct:
                    new_correct_answer = new_id
            
            return {
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
            
        except Exception as e:
            print(f"Error paraphrasing: {e}")
            return self._format_fallback_question(original_question)

    def _format_fallback_question(self, original_question: Dict) -> Dict:
        """Format original question when paraphrasing fails"""
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

    async def _generate_incorrect_feedback(self, question: Dict, user_answer: str, paraphrased_q: Dict) -> str:
        """Generate AI feedback for incorrect answers"""
        try:
            correct_explanation = question.get('explanation_correct', '')
            incorrect_explanations = question.get('explanations_incorrect', {})
            
            user_explanation = incorrect_explanations.get(user_answer, "This option is not optimal.")
            correct_answer_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == paraphrased_q['correct_answer']
            )
            
            return f"""That's not quite right. You selected {user_answer}.

The correct answer is {paraphrased_q['correct_answer']}: "{correct_answer_text}"

Here's why:
Correct choice: {correct_explanation}
Your choice: {user_explanation}

Remember: Focus on the core leadership principles when making decisions."""
            
        except Exception as e:
            return "That's not correct. Please review the leadership principles and try to understand the reasoning behind the correct choice."

    async def _validate_explanation_with_llm(self, question: Dict, explanation: str) -> Dict:
        """Validate user explanation using AI"""
        try:
            official_justification = question.get('explanation_correct', '')
            
            prompt = f"""
You are evaluating a leadership trainee's explanation.

CONTEXT: {self.scenario_context}
QUESTION: {question['question_text']}
CORRECT ANSWER: {question['correct_answer']}
OFFICIAL REASON: {official_justification}
USER'S EXPLANATION: {explanation}

Evaluate if their explanation shows understanding of key leadership concepts.

Return ONLY JSON:
{{
    "is_valid": true/false,
    "feedback": "detailed feedback on their reasoning",
    "key_points_covered": ["point1", "point2"],
    "missing_critical_points": ["missing1", "missing2"]
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
            
            response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(response_text)
            
        except Exception as e:
            return {
                "is_valid": True,
                "feedback": "Thank you for your explanation.",
                "key_points_covered": [],
                "missing_critical_points": []
            }

    def _calculate_competency_scores(self, attempts: List[Dict]) -> Dict:
        """Calculate performance by competency"""
        competency_scores = {}
        for attempt in attempts:
            if 'original_question' in attempt:
                competencies = attempt['original_question'].get('competencies_tested', [])
                is_correct = attempt.get('is_correct', False)
                
                for comp in competencies:
                    if comp not in competency_scores:
                        competency_scores[comp] = {"correct": 0, "total": 0}
                    competency_scores[comp]["total"] += 1
                    if is_correct:
                        competency_scores[comp]["correct"] += 1
        
        return {
            comp: (scores["correct"] / scores["total"] * 100) if scores["total"] > 0 else 0
            for comp, scores in competency_scores.items()
        }

    def _generate_performance_insights(self, attempts: List[Dict], percentage: float) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if percentage >= 85:
            insights.append("Excellent performance! You demonstrate strong leadership understanding.")
        elif percentage >= 70:
            insights.append("Good work! You have a solid grasp of leadership principles.")
        else:
            insights.append("Consider reviewing leadership fundamentals to improve performance.")
            
        # Add specific insights based on attempts
        incorrect_count = sum(1 for attempt in attempts if not attempt.get('is_correct', False))
        if incorrect_count > 0:
            insights.append(f"Focus on areas where you answered incorrectly ({incorrect_count} questions).")
            
        return insights

    async def _extract_option_from_speech(self, speech_text: str, question_data: Dict) -> Optional[str]:
        """Extract A/B/C/D option from speech using LLM"""
        try:
            options_text = "\n".join([f"{opt['option_id']}: {opt['text']}" for opt in question_data['options']])
            
            prompt = f"""User said: "{speech_text}"

Question options:
{options_text}

Extract which option (A, B, C, or D) the user selected. They might say:
- The letter directly: "A", "B", "C", "D"
- Option text: part of the actual option content
- Descriptive: "first one", "second option", etc.

If the user's input is completely unrelated to the question (like asking about weather, food, etc.), return "IRRELEVANT".
If unclear but seems question-related, return "UNCLEAR".
Otherwise return ONLY the letter (A, B, C, or D)."""
            
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
            
            extracted = response.text.strip().upper()
            if extracted in ['A', 'B', 'C', 'D']:
                return extracted
            elif extracted in ['IRRELEVANT', 'UNCLEAR']:
                return None  # Will trigger retry or timeout
            else:
                return None
            
        except Exception as e:
            print(f"Error extracting option from speech: {e}")
            return None

    async def _handle_timeout_or_invalid(self, session, current_q_original: Dict, paraphrased_q: Dict, time_taken: int, db, reason: str = "timeout") -> Dict:
        """Handle timeout or invalid answer scenario - show correct answer and move to next"""
        try:
            correct_option_text = next(
                opt['text'] for opt in paraphrased_q['options'] 
                if opt['option_id'] == paraphrased_q['correct_answer']
            )
            
            # Create attempt record based on reason
            if reason == "timeout":
                user_answer = "TIMEOUT"
                user_answer_text = "No answer provided (timeout)"
                feedback = f"Time's up! The correct answer was {paraphrased_q['correct_answer']}: \"{correct_option_text}\""
            else:  # irrelevant or invalid
                user_answer = "INVALID"
                user_answer_text = "Invalid or irrelevant answer"
                feedback = f"Your answer was not relevant to the question. The correct answer was {paraphrased_q['correct_answer']}: \"{correct_option_text}\""
            
            attempt = QuestionAttemptRecord(
                question_id=current_q_original['id'],
                original_question=current_q_original,
                paraphrased_question=paraphrased_q,
                user_answer=user_answer,
                user_answer_text=user_answer_text,
                correct_answer_original=current_q_original['correct_answer'],
                correct_answer_paraphrased=paraphrased_q['correct_answer'],
                is_correct=False,
                time_taken_seconds=time_taken
            )
            
            session.question_attempts.append(attempt.dict())
            
            # Move to next question
            session.current_question_index += 1
            session.current_state = "awaiting_answer"
            
            # Check if session completed
            is_completed = session.current_question_index >= len(session.questions_data)
            if is_completed:
                session.is_completed = True
                session.current_state = "completed"
            
            await db.update_question_session(session)
            
            # Get next question if available
            next_question = None
            if not is_completed:
                next_question = session.paraphrased_questions_used[session.current_question_index]
            
            return {
                "is_timeout": reason == "timeout",
                "is_invalid": reason != "timeout",
                "feedback": feedback,
                "correct_answer": paraphrased_q['correct_answer'],
                "correct_answer_text": correct_option_text,
                "awaiting_explanation": False,
                "current_score": session.score,
                "question_number": session.current_question_index,
                "total_questions": session.total_questions,
                "is_completed": is_completed,
                "next_question": next_question
            }
            
        except Exception as e:
            raise Exception(f"Error handling {reason}: {str(e)}")

    def _get_paraphrasing_prompt(self, difficulty: str) -> str:
        """Get prompt for question paraphrasing"""
        if difficulty == "easy":
            return """
Rephrase this leadership question to be EASIER:
- Add context clues and helpful hints
- Use clearer, more descriptive language
- Make the correct choice more distinguishable

CONTEXT: {scenario_context}
QUESTION: {question_text}
OPTIONS: {options}
CORRECT: {correct_answer}

Return ONLY JSON:
{{
    "question_text": "easier rephrased question",
    "options": [
        {{"option_id": "A", "text": "clearer option"}},
        {{"option_id": "B", "text": "clearer option"}},
        {{"option_id": "C", "text": "clearer option"}},
        {{"option_id": "D", "text": "clearer option"}}
    ]
}}
"""
        else:
            return """
Rephrase this leadership question to be HARDER:
- Remove obvious hints
- Use advanced terminology
- Make incorrect options more plausible
- Require deeper expertise to differentiate

CONTEXT: {scenario_context}
QUESTION: {question_text}
OPTIONS: {options}
CORRECT: {correct_answer}

Return ONLY JSON:
{{
    "question_text": "harder rephrased question",
    "options": [
        {{"option_id": "A", "text": "subtle option"}},
        {{"option_id": "B", "text": "subtle option"}},
        {{"option_id": "C", "text": "subtle option"}},
        {{"option_id": "D", "text": "subtle option"}}
    ]
}}
"""