# Question Training System API Documentation

**Base URL:** `http://localhost:8000`

**Version:** 1.0.0

---

## Authentication
Currently no authentication required. All endpoints are public.

---

## Response Format

All endpoints return responses in this format:

```json
{
  "success": true,
  "data": {
    // Actual response data here
  }
}
```

Error responses:
```json
{
  "detail": "Error message here"
}
```

---

## Core Training Flow APIs

### 1. Start Training Session

**Endpoint:** `POST /api/question-sessions/start`

**Description:** Initiates a new question training session for a specific scenario and difficulty level.

**Request Body (Form Data):**
- `scenario_name` (string, required): Name of the training scenario
- `difficulty` (string, optional): "easy" or "hard" (default: "easy")
- `user_id` (string, optional): User identifier for tracking

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "uuid-string",
    "scenario_name": "Banking Leadership Questions",
    "difficulty": "easy",
    "total_questions": 15,
    "current_question": {
      "id": "q1",
      "question_text": "What should be your first action when observing a customer argument?",
      "options": [
        {"option_id": "A", "text": "Immediately intervene to understand the dispute"},
        {"option_id": "B", "text": "Call security to control the situation"},
        {"option_id": "C", "text": "Continue observing to assess the conflict"},
        {"option_id": "D", "text": "Signal staff to move customer to private area"}
      ],
      "correct_answer": "A",
      "category": "Crisis Recognition"
    },
    "question_number": 1
  }
}
```

**Example:**
```javascript
const formData = new FormData();
formData.append('scenario_name', 'Banking Leadership Questions');
formData.append('difficulty', 'easy');
formData.append('user_id', 'user_123');

const response = await fetch('/api/question-sessions/start', {
  method: 'POST',
  body: formData
});
```

---

### 2. Submit Answer

**Endpoint:** `POST /api/question-sessions/{session_id}/answer`

**Description:** Submits user's answer (speech text, typed input, or timeout).

**Path Parameters:**
- `session_id` (string, required): Session identifier from start endpoint

**Request Body (Form Data):**
- `user_input` (string, required): Speech text, option letter, or empty for timeout
- `time_taken` (integer, optional): Time taken in seconds (default: 30)
- `is_timeout` (boolean, optional): Whether this is a timeout submission (default: false)

**Response for Correct Answer:**
```json
{
  "success": true,
  "data": {
    "is_correct": true,
    "feedback": "Correct! You selected A: \"Immediately intervene to understand the dispute\"",
    "awaiting_explanation": true,
    "explanation_prompt": "Please explain WHY this is the best choice. What leadership principles led you to this decision?",
    "current_score": 0,
    "question_number": 1,
    "total_questions": 15
  }
}
```

**Response for Incorrect Answer:**
```json
{
  "success": true,
  "data": {
    "is_correct": false,
    "feedback": "That's not quite right. You selected B.\n\nThe correct answer is A: \"Immediately intervene to understand the dispute\"\n\nHere's why:\n✅ Correct choice: Immediate intervention demonstrates proactive leadership\n❌ Your choice: Calling security may escalate unnecessarily",
    "correct_answer": "A",
    "correct_answer_text": "Immediately intervene to understand the dispute",
    "awaiting_explanation": false,
    "current_score": 0,
    "question_number": 2,
    "total_questions": 15,
    "is_completed": false,
    "next_question": {
      "id": "q2",
      "question_text": "Next question text...",
      "options": [...],
      "correct_answer": "B"
    }
  }
}
```

**Response for Invalid Speech Input:**
```json
{
  "success": true,
  "data": {
    "is_valid_answer": false,
    "feedback": "I couldn't understand which option you selected. Please clearly state A, B, C, or D.",
    "current_score": 0,
    "question_number": 1,
    "total_questions": 15,
    "retry_allowed": true
  }
}
```

**Response for Timeout:**
```json
{
  "success": true,
  "data": {
    "is_timeout": true,
    "feedback": "Time's up! The correct answer was A: \"Immediately intervene to understand the dispute\"",
    "correct_answer": "A",
    "correct_answer_text": "Immediately intervene to understand the dispute",
    "awaiting_explanation": false,
    "current_score": 0,
    "question_number": 2,
    "total_questions": 15,
    "is_completed": false,
    "next_question": {...}
  }
}
```

**Examples:**
```javascript
// Speech input
const formData = new FormData();
formData.append('user_input', 'I think the answer is A');
formData.append('time_taken', '15');
formData.append('is_timeout', false);

// Timeout
const formData = new FormData();
formData.append('user_input', '');
formData.append('time_taken', '30');
formData.append('is_timeout', true);
```

---

### 3. Submit Explanation

**Endpoint:** `POST /api/question-sessions/{session_id}/explain`

**Description:** Submits user's explanation for their correct answer choice.

**Path Parameters:**
- `session_id` (string, required): Session identifier

**Request Body (Form Data):**
- `explanation` (string, required): User's reasoning for their answer choice

**Response:**
```json
{
  "success": true,
  "data": {
    "explanation_valid": true,
    "feedback": "Excellent explanation! You demonstrated understanding of proactive leadership principles.\n\nKey points you covered:\n• Immediate intervention prevents escalation\n• Shows leadership presence\n• Demonstrates customer-first approach",
    "key_points_covered": [
      "Immediate intervention prevents escalation",
      "Shows leadership presence", 
      "Demonstrates customer-first approach"
    ],
    "missing_points": [],
    "current_score": 1,
    "question_number": 2,
    "total_questions": 15,
    "is_completed": false,
    "next_question": {
      "id": "q2",
      "question_text": "Next question...",
      "options": [...],
      "correct_answer": "C"
    }
  }
}
```

**Example:**
```javascript
const formData = new FormData();
formData.append('explanation', 'I chose this because immediate intervention shows proactive leadership and helps de-escalate the situation before it gets worse.');

const response = await fetch(`/api/question-sessions/${sessionId}/explain`, {
  method: 'POST',
  body: formData
});
```

---

### 4. Get Session Results

**Endpoint:** `GET /api/question-sessions/{session_id}/results`

**Description:** Retrieves complete results and analytics for a completed training session.

**Path Parameters:**
- `session_id` (string, required): Session identifier

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "uuid-string",
    "scenario_name": "Banking Leadership Questions",
    "difficulty": "easy",
    "final_score": 12,
    "total_questions": 15,
    "percentage_score": 80.0,
    "passed": true,
    "excellence": false,
    "competency_scores": {
      "Crisis Management": 85.5,
      "Leadership Presence": 75.0,
      "Communication": 90.0
    },
    "performance_insights": [
      "Good work! You have a solid grasp of leadership principles.",
      "Focus on areas where you answered incorrectly (3 questions)."
    ],
    "detailed_attempts": [
      {
        "question_id": "q1",
        "user_answer": "A",
        "is_correct": true,
        "user_explanation": "Shows proactive leadership...",
        "explanation_validation": {
          "is_valid": true,
          "feedback": "Great reasoning!"
        }
      }
    ],
    "session_duration_minutes": 12.5,
    "created_at": "2024-01-15T10:30:00Z",
    "completed_at": "2024-01-15T10:42:30Z"
  }
}
```

**Example:**
```javascript
const response = await fetch(`/api/question-sessions/${sessionId}/results`);
const result = await response.json();
console.log(`Score: ${result.data.percentage_score}%`);
```

---

## Session Management APIs

### Get Session Status

**Endpoint:** `GET /api/question-sessions/{session_id}/status`

**Description:** Get current session progress and state.

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "uuid-string",
    "current_question": {
      "id": "q3",
      "question_text": "Current question...",
      "options": [...]
    },
    "question_number": 3,
    "total_questions": 15,
    "current_score": 2,
    "state": "awaiting_answer",
    "is_completed": false,
    "difficulty": "easy",
    "scenario_name": "Banking Leadership Questions"
  }
}
```

**Session States:**
- `awaiting_answer`: User needs to select an answer
- `awaiting_explanation`: User needs to provide explanation (after correct answer)
- `completed`: Training session finished

---

## Scenario Management APIs

### List Scenarios

**Endpoint:** `GET /api/question-scenarios/list`

**Description:** Get all available training scenarios.

**Response:**
```json
[
  {
    "scenario_name": "Banking Leadership Questions",
    "description": "Leadership training for banking professionals",
    "question_count": 25,
    "available_difficulties": ["easy", "hard"]
  },
  {
    "scenario_name": "Customer Service Excellence",
    "description": "Advanced customer service training",
    "question_count": 30,
    "available_difficulties": ["easy", "hard"]
  }
]
```

### Create Scenario

**Endpoint:** `POST /api/question-scenarios/create`

**Description:** Create a new training scenario with questions.

**Request Body (Form Data):**
- `scenario_name` (string, required): Unique scenario name
- `scenario_description` (string, required): Description
- `scenario_context` (string, required): Context for AI paraphrasing
- `questions_json` (string, required): JSON array of questions
- `competency_framework` (string, optional): JSON array of competencies

**Questions JSON Format:**
```json
[
  {
    "id": "q1",
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
    "explanation_correct": "Immediate intervention demonstrates proactive leadership and prevents escalation",
    "explanations_incorrect": {
      "B": "Calling security may escalate unnecessarily",
      "C": "Continued observation allows situation to worsen",
      "D": "Gesturing shows lack of direct leadership"
    },
    "competencies_tested": ["Crisis Management", "Leadership Presence"],
    "source_difficulty": "easy"
  }
]
```

### Upload Scenario from JSON

**Endpoint:** `POST /api/question-scenarios/upload-from-json`

**Description:** Upload scenario from JSON file.

**Request Body (Multipart):**
- `file` (file, required): JSON file containing scenario data

---

## Analytics APIs

### Session Summary

**Endpoint:** `GET /api/analytics/question-sessions/summary`

**Description:** Get training analytics summary.

**Query Parameters:**
- `scenario_name` (string, optional): Filter by scenario
- `days` (integer, optional): Number of days to analyze (default: 7)

**Response:**
```json
{
  "period_days": 7,
  "scenario_name": "Banking Leadership Questions",
  "summary": {
    "total_sessions": 45,
    "completed_sessions": 38,
    "completion_rate_percent": 84.4,
    "average_score_percent": 76.5
  },
  "difficulty_breakdown": {
    "easy": 28,
    "hard": 17
  }
}
```

---

## Error Handling

**Common Error Responses:**

**404 - Not Found:**
```json
{
  "detail": "Session not found"
}
```

**400 - Bad Request:**
```json
{
  "detail": "Please select A, B, C, or D"
}
```

**500 - Server Error:**
```json
{
  "detail": "Error starting session: Database connection failed"
}
```

---

## Frontend Integration Notes

### Enhanced State Management

1. **Session Flow States:**
   - `not_started`: No active session
   - `answering`: User selecting answer (30s timer active)
   - `explaining`: User providing explanation
   - `completed`: Session finished

2. **Speech & Timer State:**
   ```javascript
   const [isRecording, setIsRecording] = useState(false);
   const [speechText, setSpeechText] = useState('');
   const [timeLeft, setTimeLeft] = useState(30);
   const [timer, setTimer] = useState(null);
   const recognitionRef = useRef(null);
   ```

3. **Progress Tracking:**
   ```javascript
   const [sessionProgress, setSessionProgress] = useState({
     questionNumber: 1,
     totalQuestions: 15,
     currentScore: 0
   });
   ```

4. **Speech Recognition Setup:**
   ```javascript
   const initializeSpeechRecognition = () => {
     if ('webkitSpeechRecognition' in window) {
       recognitionRef.current = new window.webkitSpeechRecognition();
       recognitionRef.current.continuous = false;
       recognitionRef.current.interimResults = false;
       recognitionRef.current.lang = 'en-US';
       
       recognitionRef.current.onresult = (event) => {
         const transcript = event.results[0][0].transcript;
         setSpeechText(transcript);
         submitAnswer(transcript);
       };
     }
   };
   ```

5. **Timer Management:**
   ```javascript
   const startTimer = () => {
     setTimeLeft(30);
     const newTimer = setInterval(() => {
       setTimeLeft(prev => {
         if (prev <= 1) {
           clearInterval(newTimer);
           handleTimeout();
           return 0;
         }
         return prev - 1;
       });
     }, 1000);
     setTimer(newTimer);
   };
   ```

6. **Response Data Extraction:**
   ```javascript
   const response = await fetch('/api/endpoint');
   const result = await response.json();
   const actualData = result.data; // Extract from success wrapper
   
   // Handle retry for invalid speech
   if (actualData.retry_allowed) {
     // Show retry message, keep timer running
     return;
   }
   ```

### Error Handling Best Practices

```javascript
const handleApiCall = async (url, options) => {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    const result = await response.json();
    return result.data; // Extract data from success wrapper
  } catch (error) {
    console.error('API Error:', error.message);
    // Show user-friendly error message
    alert(`Error: ${error.message}`);
    throw error;
  }
};
```

### Updated Training Flow

**Enhanced Flow with Speech & Timeout:**
1. **Start Session** → Get first question + start 30s timer
2. **User speaks/types** → Frontend converts speech to text
3. **Submit answer** → Backend extracts option from text using LLM
4. **If invalid speech** → Retry (timer continues)
5. **If timeout** → Show correct answer, auto-advance
6. **If correct** → Request explanation
7. **If incorrect** → Show feedback, advance to next

**API Flow:**
1. **List scenarios** → `GET /api/question-scenarios/list`
2. **Start session** → `POST /api/question-sessions/start` (starts 30s timer)
3. **Answer questions (with speech/timeout handling):**
   - User speaks → Frontend: speech-to-text → Submit: `POST /api/question-sessions/{id}/answer`
   - Backend: LLM extracts option from speech text
   - **If invalid speech:** Return retry_allowed=true, timer continues
   - **If timeout:** Auto-submit with is_timeout=true, show correct answer, advance
   - **If correct:** Submit explanation → `POST /api/question-sessions/{id}/explain`
   - **If incorrect:** Show feedback, auto-advance to next question
4. **Get results** → `GET /api/question-sessions/{id}/results`

**Speech Processing Flow:**
```
User Speech → Browser Speech-to-Text → Send Text to Backend → 
LLM Extracts Option → Validate Answer → Continue Flow
```

**Timeout Handling:**
```
30s Timer Expires → Auto-submit with is_timeout=true → 
Show Correct Answer → Move to Next Question (No explanation required)
```

---

## Rate Limits

Currently no rate limits implemented. Consider implementing in production:
- 100 requests per minute per IP
- 10 session starts per hour per user

---

## CORS Configuration

The API is configured to accept requests from any origin (`*`) for development. Update for production:

```javascript
// Current: allow_origins=["*"]
// Production: allow_origins=["https://yourfrontend.com"]
```