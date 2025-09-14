# Updated Question Training Flow

## Overview
The system now supports **speech input** and **timeout handling** with automatic question progression.

## Key Changes

### 1. Speech Processing
- **Frontend**: Browser speech-to-text conversion
- **Backend**: LLM extracts A/B/C/D from speech text
- **Fallback**: Manual text input option

### 2. Timeout Handling
- **30-second timer** per question
- **Auto-advance** on timeout (no answer required)
- **Show correct answer** when time expires

### 3. Question Progression
- **No blocking**: Questions advance automatically on timeout/incorrect answers
- **Explanation required**: Only for correct answers
- **Retry logic**: For invalid speech input only

## Updated Flow Diagram

```
START SESSION
    ↓
SHOW QUESTION + START 30s TIMER
    ↓
USER SPEAKS/TYPES
    ↓
FRONTEND: Speech → Text
    ↓
BACKEND: LLM Extracts Option
    ↓
┌─────────────────────────────────────┐
│ INVALID SPEECH?                     │
│ → Return retry_allowed=true         │
│ → Timer continues, show retry msg   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ TIMEOUT (30s)?                      │
│ → Show correct answer               │
│ → Auto-advance to next question     │
│ → No explanation required           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ CORRECT ANSWER?                     │
│ → Request explanation               │
│ → User speaks/types explanation     │
│ → Advance to next question          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ INCORRECT ANSWER?                   │
│ → Show feedback + correct answer    │
│ → Auto-advance to next question     │
│ → No explanation required           │
└─────────────────────────────────────┘
    ↓
REPEAT UNTIL ALL QUESTIONS COMPLETE
    ↓
SHOW FINAL RESULTS
```

## API Changes

### Submit Answer Endpoint
**Before:**
```javascript
formData.append('selected_option', 'A');
```

**After:**
```javascript
formData.append('user_input', 'I think the answer is A');
formData.append('is_timeout', false);
```

### New Response Types

1. **Invalid Speech Response:**
```json
{
  "is_valid_answer": false,
  "retry_allowed": true,
  "feedback": "Please clearly state A, B, C, or D"
}
```

2. **Timeout Response:**
```json
{
  "is_timeout": true,
  "feedback": "Time's up! Correct answer was A",
  "next_question": {...}
}
```

## Frontend Changes

### Speech Recognition
```javascript
// Initialize speech recognition
const recognition = new webkitSpeechRecognition();
recognition.onresult = (event) => {
  const transcript = event.results[0][0].transcript;
  submitAnswer(transcript);
};
```

### Timer Management
```javascript
// 30-second countdown timer
const startTimer = () => {
  setTimeLeft(30);
  const timer = setInterval(() => {
    setTimeLeft(prev => {
      if (prev <= 1) {
        handleTimeout();
        return 0;
      }
      return prev - 1;
    });
  }, 1000);
};
```

### Enhanced UI
- **Visual timer**: Shows countdown with color coding
- **Recording buttons**: Start/stop speech input
- **Speech feedback**: "You said: ..." display
- **Dual input**: Speech + manual text input

## Backend Processing

### Speech Option Extraction
```python
async def _extract_option_from_speech(self, speech_text: str, question_data: Dict):
    prompt = f"""User said: "{speech_text}"
    
Question options:
A: {option_a_text}
B: {option_b_text}
C: {option_c_text}  
D: {option_d_text}

Return ONLY the letter (A, B, C, or D) or "INVALID" if unclear."""
    
    # LLM processes and returns extracted option
```

### Timeout Processing
```python
async def _handle_timeout(self, session, question_data: Dict):
    # Create timeout attempt record
    # Show correct answer in feedback
    # Auto-advance to next question
    # No explanation required
```

## Benefits

1. **Natural Interaction**: Users can speak naturally ("I think it's A" vs just "A")
2. **No Blocking**: Sessions always progress, even on timeout
3. **Flexible Input**: Speech + manual text input options
4. **Better UX**: Visual timer, clear feedback, retry for unclear speech
5. **Robust Handling**: Graceful timeout and error management

## Migration Notes

- **Existing sessions**: Will continue to work with typed input
- **API compatibility**: Old `selected_option` parameter still supported
- **Progressive enhancement**: Speech features degrade gracefully on unsupported browsers