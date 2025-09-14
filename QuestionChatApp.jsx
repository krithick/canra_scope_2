import React, { useState, useEffect, useRef } from 'react';
import { Plus, Trash2, Upload, MessageCircle, Send, BookOpen, Brain, Play, CheckCircle, XCircle, Clock } from 'lucide-react';

const QuestionChatApp = () => {
  const [activeTab, setActiveTab] = useState('scenarios');
  const [scenarios, setScenarios] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [sessionProgress, setSessionProgress] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [userExplanation, setUserExplanation] = useState('');
  const [feedback, setFeedback] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionState, setSessionState] = useState('not_started'); // not_started, answering, explaining, completed
  const [isRecording, setIsRecording] = useState(false);
  const [speechText, setSpeechText] = useState('');
  const [timeLeft, setTimeLeft] = useState(30);
  const [timer, setTimer] = useState(null);
  const feedbackRef = useRef(null);
  const recognitionRef = useRef(null);

  // Create Scenario State
  const [scenarioForm, setScenarioForm] = useState({
    scenario_name: '',
    scenario_description: '',
    scenario_context: '',
    questions: []
  });

  const [currentQuestionForm, setCurrentQuestionForm] = useState({
    question_text: '',
    category: '',
    options: [
      { option_id: 'A', text: '' },
      { option_id: 'B', text: '' },
      { option_id: 'C', text: '' },
      { option_id: 'D', text: '' }
    ],
    correct_answer: 'A',
    correct_explanation: '',
    incorrect_explanations: { A: '', B: '', C: '', D: '' },
    competencies_tested: '',
    source_difficulty: 'easy'
  });

  useEffect(() => {
    loadScenarios();
    initializeSpeechRecognition();
  }, []);

  const initializeSpeechRecognition = () => {
    if ('webkitSpeechRecognition' in window) {
      recognitionRef.current = new window.webkitSpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setSpeechText(transcript);
        if (sessionState === 'answering') {
          submitAnswer(transcript);
        } else if (sessionState === 'explaining') {
          submitExplanation(transcript);
        }
      };
      
      recognitionRef.current.onerror = () => {
        setSpeechText('Speech recognition failed. Try again.');
        setIsRecording(false);
      };
    }
  };

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

  const stopTimer = () => {
    if (timer) {
      clearInterval(timer);
      setTimer(null);
    }
  };

  const handleTimeout = async () => {
    stopRecording();
    const timeTaken = 30 - timeLeft;
    
    try {
      const formData = new FormData();
      formData.append('user_input', '');
      formData.append('time_taken', timeTaken);
      formData.append('is_timeout', true);

      const response = await fetch(`http://localhost:4000/api/question-sessions/${currentSession.sessionId}/answer`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        handleAnswerResponse(result.data);
      }
    } catch (error) {
      console.error('Timeout error:', error);
    }
  };

  const startRecording = () => {
    if (!recognitionRef.current) {
      alert('Speech recognition not supported');
      return;
    }
    
    setIsRecording(true);
    setSpeechText('Listening...');
    recognitionRef.current.start();
  };

  const stopRecording = () => {
    if (recognitionRef.current && isRecording) {
      recognitionRef.current.stop();
    }
    setIsRecording(false);
  };

  useEffect(() => {
    if (feedback && feedbackRef.current) {
      feedbackRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [feedback]);

  const loadScenarios = async () => {
    try {
      const response = await fetch('http://localhost:4000/api/question-scenarios/list');
      if (response.ok) {
        const data = await response.json();
        setScenarios(data);
      }
    } catch (error) {
      console.error('Error loading scenarios:', error);
    }
  };

  const addQuestion = () => {
    if (!currentQuestionForm.question_text || !currentQuestionForm.correct_explanation) {
      alert('Please fill in question text and correct explanation');
      return;
    }

    const question = {
      id: `q${scenarioForm.questions.length + 1}`,
      question_number: scenarioForm.questions.length + 1,
      ...currentQuestionForm,
      competencies_tested: currentQuestionForm.competencies_tested.split(',').map(c => c.trim()),
      explanation_correct: currentQuestionForm.correct_explanation,
      explanations_incorrect: currentQuestionForm.incorrect_explanations
    };

    setScenarioForm(prev => ({
      ...prev,
      questions: [...prev.questions, question]
    }));

    // Reset form
    setCurrentQuestionForm({
      question_text: '',
      category: '',
      options: [
        { option_id: 'A', text: '' },
        { option_id: 'B', text: '' },
        { option_id: 'C', text: '' },
        { option_id: 'D', text: '' }
      ],
      correct_answer: 'A',
      correct_explanation: '',
      incorrect_explanations: { A: '', B: '', C: '', D: '' },
      competencies_tested: '',
      source_difficulty: 'easy'
    });
  };

  const updateQuestionOption = (index, text) => {
    const newOptions = [...currentQuestionForm.options];
    newOptions[index].text = text;
    setCurrentQuestionForm(prev => ({ ...prev, options: newOptions }));
  };

  const createScenario = async () => {
    if (!scenarioForm.scenario_name || scenarioForm.questions.length === 0) {
      alert('Please provide scenario name and at least one question');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('scenario_name', scenarioForm.scenario_name);
      formData.append('scenario_description', scenarioForm.scenario_description);
      formData.append('scenario_context', scenarioForm.scenario_context);
      formData.append('questions_json', JSON.stringify(scenarioForm.questions));

      const response = await fetch('http://localhost:4000/api/question-scenarios/create', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        alert('Scenario created successfully!');
        setScenarioForm({ scenario_name: '', scenario_description: '', scenario_context: '', questions: [] });
        await loadScenarios();
        setActiveTab('scenarios');
      } else {
        const error = await response.text();
        alert(`Error: ${error}`);
      }
    } catch (error) {
      console.error('Error creating scenario:', error);
      alert('Error creating scenario');
    } finally {
      setIsLoading(false);
    }
  };

const startTraining = async (scenarioName, difficulty) => {
  setIsLoading(true);
  setFeedback('');
  try {
    const formData = new FormData();
    formData.append('scenario_name', scenarioName); // Changed from 'scenario_id'
    formData.append('difficulty', difficulty);
    formData.append('user_id', 'user_123');

    const response = await fetch('http://localhost:4000/api/question-sessions/start', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const result = await response.json();
      const data = result.data; // Extract data from new response format
      
      setCurrentSession({
        sessionId: data.session_id,
        scenarioName: data.scenario_name,
        difficulty: data.difficulty,
        totalQuestions: data.total_questions,
        currentScore: 0
      });
      
      setCurrentQuestion(data.current_question);
      setSessionProgress({
        questionNumber: data.question_number || 1,
        totalQuestions: data.total_questions,
        currentScore: 0
      });
      
      setSessionState('answering');
      setActiveTab('training');
      startTimer();
    }
  } catch (error) {
    console.error('Error starting training:', error);
    alert('Error starting training');
  } finally {
    setIsLoading(false);
  }
};

const submitAnswer = async (speechInput = null) => {
  const inputText = speechInput || userAnswer;
  if (!inputText.trim()) {
    alert('Please provide an answer');
    return;
  }

  stopTimer();
  stopRecording();
  setIsLoading(true);
  
  try {
    const timeTaken = 30 - timeLeft;
    const formData = new FormData();
    formData.append('user_input', inputText);
    formData.append('time_taken', timeTaken);
    formData.append('is_timeout', false);

    const response = await fetch(`http://localhost:4000/api/question-sessions/${currentSession.sessionId}/answer`, {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const result = await response.json();
      handleAnswerResponse(result.data);
    } else {
      const errorData = await response.json();
      alert(`Error: ${errorData.detail || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Error submitting answer:', error);
    alert('Error submitting answer');
  } finally {
    setIsLoading(false);
  }
};

const handleAnswerResponse = (data) => {
  const cleanFeedback = data.feedback?.replace(/\$\w+\$/g, '').trim() || '';
  setFeedback(cleanFeedback);
  
  setSessionProgress(prev => ({
    ...prev,
    questionNumber: data.question_number || prev?.questionNumber,
    currentScore: data.current_score
  }));
  
  if (data.retry_allowed) {
    // Invalid speech input - allow retry
    setTimeout(() => {
      setFeedback('');
      setSpeechText('');
      startTimer();
    }, 3000);
    return;
  }
  
  if (data.awaiting_explanation) {
    setSessionState('explaining');
    setUserAnswer('');
  } else if (data.next_question) {
    setTimeout(() => {
      setCurrentQuestion(data.next_question);
      setSessionState('answering');
      setUserAnswer('');
      setSpeechText('');
      startTimer();
    }, 3000);
  } else if (data.is_completed) {
    setSessionState('completed');
  }
};
const submitExplanation = async (speechInput = null) => {
  const explanationText = speechInput || userExplanation;
  if (!explanationText.trim()) {
    alert('Please provide an explanation');
    return;
  }

  stopRecording();
  setIsLoading(true);
  
  try {
    const formData = new FormData();
    formData.append('explanation', explanationText);

    const response = await fetch(`http://localhost:4000/api/question-sessions/${currentSession.sessionId}/explain`, {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const result = await response.json();
      const data = result.data;
      
      const cleanFeedback = data.feedback?.replace(/\$\w+\$/g, '').trim() || '';
      setFeedback(cleanFeedback);
      
      setSessionProgress(prev => ({
        ...prev,
        questionNumber: data.question_number || (prev?.questionNumber + 1),
        currentScore: data.current_score
      }));

      setUserExplanation('');
      setSpeechText('');
      
      if (data.next_question) {
        setTimeout(() => {
          setCurrentQuestion(data.next_question);
          setSessionState('answering');
          startTimer();
        }, 3000);
      } else if (data.is_completed) {
        setSessionState('completed');
      }
    } else {
      const errorData = await response.json();
      alert(`Error: ${errorData.detail || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Error submitting explanation:', error);
    alert('Error submitting explanation');
  } finally {
    setIsLoading(false);
  }
};
 const getSessionResults = async () => {
  if (!currentSession) return;
  
  try {
    const response = await fetch(`http://localhost:4000/api/question-sessions/${currentSession.sessionId}/results`);
    if (response.ok) {
      const result = await response.json();
      const data = result.data; // Extract data from new response format
      
      setFeedback(
        `Training Complete!\n\nFinal Score: ${data.final_score}/${data.total_questions} (${data.percentage_score}%)\n\n${
          data.passed ? 'Congratulations! You passed!' : 'Keep practicing to improve!'
        }\n\nSession Duration: ${Math.round(data.session_duration_minutes)} minutes`
      );
    }
  } catch (error) {
    console.error('Error getting results:', error);
    alert('Error getting session results');
  }
};

  const resetSession = () => {
    setCurrentSession(null);
    setCurrentQuestion(null);
    setSessionProgress(null);
    setUserAnswer('');
    setUserExplanation('');
    setFeedback('');
    setSessionState('not_started');
  };

  const uploadJsonFile = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:4000/api/question-scenarios/upload-from-json', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        alert('JSON scenario uploaded successfully!');
        await loadScenarios();
      } else {
        const error = await response.text();
        alert(`Error uploading: ${error}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file');
    } finally {
      setIsLoading(false);
      event.target.value = '';
    }
  };

  const renderCreateTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Create Question Scenario</h2>
        
        {/* Upload JSON Option */}
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h3 className="text-lg font-semibold mb-2 text-blue-800">Upload JSON File</h3>
          <label className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg cursor-pointer hover:bg-blue-700 w-fit">
            <Upload className="w-4 h-4" />
            <span>Upload JSON</span>
            <input type="file" accept=".json" onChange={uploadJsonFile} className="hidden" />
          </label>
        </div>

        {/* Manual Creation */}
        <div className="space-y-4">
          <input
            type="text"
            value={scenarioForm.scenario_name}
            onChange={(e) => setScenarioForm(prev => ({ ...prev, scenario_name: e.target.value }))}
            className="w-full px-3 py-2 border rounded-md"
            placeholder="Scenario Name"
          />
          
          <input
            type="text"
            value={scenarioForm.scenario_description}
            onChange={(e) => setScenarioForm(prev => ({ ...prev, scenario_description: e.target.value }))}
            className="w-full px-3 py-2 border rounded-md"
            placeholder="Description"
          />
          
          <textarea
            value={scenarioForm.scenario_context}
            onChange={(e) => setScenarioForm(prev => ({ ...prev, scenario_context: e.target.value }))}
            className="w-full px-3 py-2 border rounded-md h-20"
            placeholder="Context for AI paraphrasing..."
          />
        </div>

        {/* Question Builder */}
        <div className="mt-6 border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Add Question</h3>
          
          <div className="space-y-4">
            <textarea
              value={currentQuestionForm.question_text}
              onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, question_text: e.target.value }))}
              className="w-full px-3 py-2 border rounded-md h-16"
              placeholder="Question text..."
            />

            <div className="grid grid-cols-3 gap-4">
              <input
                type="text"
                value={currentQuestionForm.category}
                onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, category: e.target.value }))}
                className="px-3 py-2 border rounded-md"
                placeholder="Category"
              />
              <select
                value={currentQuestionForm.correct_answer}
                onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, correct_answer: e.target.value }))}
                className="px-3 py-2 border rounded-md"
              >
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
                <option value="D">D</option>
              </select>
              <select
                value={currentQuestionForm.source_difficulty}
                onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, source_difficulty: e.target.value }))}
                className="px-3 py-2 border rounded-md"
              >
                <option value="easy">Easy</option>
                <option value="hard">Hard</option>
              </select>
            </div>

            {/* Options */}
            <div className="space-y-2">
              {currentQuestionForm.options.map((option, index) => (
                <div key={option.option_id} className="flex items-center space-x-2">
                  <span className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center font-medium">
                    {option.option_id}
                  </span>
                  <input
                    type="text"
                    value={option.text}
                    onChange={(e) => updateQuestionOption(index, e.target.value)}
                    className="flex-1 px-3 py-2 border rounded-md"
                    placeholder={`Option ${option.option_id}...`}
                  />
                </div>
              ))}
            </div>

            <textarea
              value={currentQuestionForm.correct_explanation}
              onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, correct_explanation: e.target.value }))}
              className="w-full px-3 py-2 border rounded-md h-16"
              placeholder="Why is this the correct answer?"
            />

            <input
              type="text"
              value={currentQuestionForm.competencies_tested}
              onChange={(e) => setCurrentQuestionForm(prev => ({ ...prev, competencies_tested: e.target.value }))}
              className="w-full px-3 py-2 border rounded-md"
              placeholder="Competencies (comma-separated)"
            />

            <button
              onClick={addQuestion}
              className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700"
            >
              <Plus className="w-4 h-4" />
              <span>Add Question</span>
            </button>
          </div>
        </div>

        {/* Questions List */}
        {scenarioForm.questions.length > 0 && (
          <div className="mt-6 border-t pt-6">
            <h3 className="text-lg font-semibold mb-4">Added Questions ({scenarioForm.questions.length})</h3>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {scenarioForm.questions.map((question, index) => (
                <div key={index} className="bg-gray-50 p-3 rounded-md flex justify-between items-start">
                  <div>
                    <p className="font-medium text-sm">{question.question_text}</p>
                    <p className="text-xs text-gray-600">
                      Correct: {question.correct_answer} | {question.source_difficulty}
                    </p>
                  </div>
                  <button
                    onClick={() => setScenarioForm(prev => ({
                      ...prev,
                      questions: prev.questions.filter((_, i) => i !== index)
                    }))}
                    className="text-red-600 hover:text-red-800"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Create Button */}
        <button
          onClick={createScenario}
          disabled={isLoading}
          className="w-full bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 disabled:opacity-50 mt-6"
        >
          {isLoading ? 'Creating...' : 'Create Scenario'}
        </button>
      </div>
    </div>
  );

  const renderScenariosTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Available Scenarios</h2>
        
        {scenarios.length === 0 ? (
          <p className="text-gray-600">No scenarios available. Create one first!</p>
        ) : (
          <div className="grid gap-4">
            {scenarios.map((scenario, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="mb-3">
                  <h3 className="text-lg font-semibold text-gray-800">{scenario.scenario_name}</h3>
                  <p className="text-sm text-gray-600">{scenario.description}</p>
                  <p className="text-xs text-gray-500">{scenario.question_count} questions</p>
                </div>
                
                <div className="flex gap-2">
                  <button
                    onClick={() => startTraining(scenario.scenario_name, 'easy')}
                    className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 text-sm flex items-center space-x-1"
                    disabled={isLoading}
                  >
                    <Play className="w-4 h-4" />
                    <span>Start Easy</span>
                  </button>
                  
                  <button
                    onClick={() => startTraining(scenario.scenario_name, 'hard')}
                    className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 text-sm flex items-center space-x-1"
                    disabled={isLoading}
                  >
                    <Brain className="w-4 h-4" />
                    <span>Start Hard</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderTrainingTab = () => (
    <div className="space-y-6">
      {!currentSession ? (
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <MessageCircle className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <h2 className="text-2xl font-bold mb-2 text-gray-800">No Active Session</h2>
          <p className="text-gray-600 mb-4">Start a training session from the Scenarios tab</p>
          <button
            onClick={() => setActiveTab('scenarios')}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700"
          >
            View Scenarios
          </button>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-lg">
          {/* Session Header */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-t-lg">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-semibold">{currentSession.scenarioName}</h2>
                <p className="text-sm opacity-90">
                  {currentSession.difficulty.toUpperCase()} Mode
                  {sessionState === 'completed' && (
                    <span className="ml-2 bg-green-500 px-2 py-1 rounded text-xs">COMPLETED</span>
                  )}
                </p>
              </div>
              {sessionProgress && (
                <div className="text-right">
                  <div className="text-sm opacity-90">
                    Question {sessionProgress.questionNumber}/{sessionProgress.totalQuestions}
                  </div>
                  <div className="text-sm opacity-90">
                    Score: {sessionProgress.currentScore}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-6">
            {sessionState === 'completed' && (
              <div className="text-center space-y-4">
                <CheckCircle className="w-16 h-16 mx-auto text-green-500" />
                <h3 className="text-2xl font-bold text-gray-800">Training Complete!</h3>
                <button
                  onClick={getSessionResults}
                  className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700"
                >
                  View Results
                </button>
                <button
                  onClick={resetSession}
                  className="ml-3 bg-gray-600 text-white px-6 py-3 rounded-md hover:bg-gray-700"
                >
                  Start New Session
                </button>
              </div>
            )}

            {sessionState === 'answering' && currentQuestion && (
              <div className="space-y-6">
                <div className="bg-yellow-100 p-3 rounded-lg flex justify-between items-center">
                  <span className="font-medium">Time Remaining:</span>
                  <span className={`text-xl font-bold ${
                    timeLeft <= 10 ? 'text-red-600' : 'text-green-600'
                  }`}>{timeLeft}s</span>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-4">{currentQuestion.question_text}</h3>
                  
                  <div className="space-y-3">
                    {currentQuestion.options.map((option) => (
                      <div
                        key={option.option_id}
                        className="flex items-start space-x-3 p-3 border rounded-lg border-gray-200"
                      >
                        <span className="font-medium">{option.option_id}) </span>
                        <span>{option.text}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 space-y-4">
                    <div className="flex gap-4">
                      <button
                        onClick={startRecording}
                        disabled={isRecording || isLoading}
                        className="flex-1 bg-green-600 text-white py-3 rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center justify-center space-x-2"
                      >
                        <MessageCircle className="w-5 h-5" />
                        <span>{isRecording ? 'Listening...' : 'Record Answer'}</span>
                      </button>
                      
                      <button
                        onClick={stopRecording}
                        disabled={!isRecording}
                        className="px-6 bg-red-600 text-white py-3 rounded-md hover:bg-red-700 disabled:opacity-50"
                      >
                        Stop
                      </button>
                    </div>
                    
                    {speechText && (
                      <div className="p-3 bg-gray-100 rounded-lg">
                        <span className="text-sm text-gray-600">You said: </span>
                        <span className="font-medium">"{speechText}"</span>
                      </div>
                    )}
                    
                    <input
                      type="text"
                      value={userAnswer}
                      onChange={(e) => setUserAnswer(e.target.value)}
                      placeholder="Or type your answer (A, B, C, D)"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    />
                    
                    <button
                      onClick={() => submitAnswer()}
                      disabled={!userAnswer || isLoading}
                      className="w-full bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 disabled:opacity-50"
                    >
                      {isLoading ? 'Submitting...' : 'Submit Answer'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {sessionState === 'explaining' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-green-600">
                  Correct! Now explain your reasoning:
                </h3>
                
                <div className="space-y-4">
                  <div className="flex gap-4">
                    <button
                      onClick={startRecording}
                      disabled={isRecording || isLoading}
                      className="flex-1 bg-green-600 text-white py-3 rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center justify-center space-x-2"
                    >
                      <MessageCircle className="w-5 h-5" />
                      <span>{isRecording ? 'Recording...' : 'Record Explanation'}</span>
                    </button>
                    
                    <button
                      onClick={stopRecording}
                      disabled={!isRecording}
                      className="px-6 bg-red-600 text-white py-3 rounded-md hover:bg-red-700 disabled:opacity-50"
                    >
                      Stop
                    </button>
                  </div>
                  
                  {speechText && (
                    <div className="p-3 bg-gray-100 rounded-lg">
                      <span className="text-sm text-gray-600">You said: </span>
                      <span className="font-medium">"{speechText}"</span>
                    </div>
                  )}
                  
                  <textarea
                    value={userExplanation}
                    onChange={(e) => setUserExplanation(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md h-32"
                    placeholder="Or type your explanation..."
                  />
                  
                  <button
                    onClick={() => submitExplanation()}
                    disabled={!userExplanation.trim() || isLoading}
                    className="w-full bg-green-600 text-white py-3 rounded-md hover:bg-green-700 disabled:opacity-50"
                  >
                    {isLoading ? 'Submitting...' : 'Submit Explanation'}
                  </button>
                </div>
              </div>
            )}

            {feedback && (
              <div 
                ref={feedbackRef} 
                className="mt-6 p-4 bg-gray-50 border-l-4 border-blue-500 rounded-r-lg"
              >
                <h4 className="font-semibold mb-2">Feedback:</h4>
                <div className="whitespace-pre-wrap text-gray-700">{feedback}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Question-Based Training System</h1>
          <p className="text-gray-600">Create scenarios and train through AI-powered questions</p>
        </div>

        {/* Navigation */}
        <div className="bg-white rounded-lg shadow-lg mb-6">
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('scenarios')}
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === 'scenarios'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Scenarios ({scenarios.length})
            </button>
            <button
              onClick={() => setActiveTab('create')}
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === 'create'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Create Scenario
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === 'training'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Training
              {currentSession && (
                <span className="ml-2 w-2 h-2 bg-green-500 rounded-full inline-block"></span>
              )}
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === 'scenarios' && renderScenariosTab()}
          {activeTab === 'create' && renderCreateTab()}
          {activeTab === 'training' && renderTrainingTab()}
        </div>

        {/* Loading Overlay */}
        {isLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 flex items-center space-x-3">
              <div className="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full"></div>
              <span className="text-gray-700">Processing...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuestionChatApp;
