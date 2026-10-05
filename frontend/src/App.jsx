import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ChatArea from './components/ChatArea';
import QuickPrompts from './components/QuickPrompts';
import InputBar from './components/InputBar';

const formatTime = () => {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export default function App() {
  // Temporary session ID generated once per browser load in React runtime state
  // On browser refresh, runtime state is cleared and a new UUID is generated
  const [sessionId] = useState(() => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return 'sess_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
  });

  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [voiceState, setVoiceState] = useState('idle');
  const currentAudioRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'ai',
      text: "Hello! Welcome to P.S. Mission Hospital's AI Assistant. I can help you find department information, doctor services, OPD consultation timings, and hospital guidelines based on verified records. How may I assist you today?",
      timestamp: formatTime(),
      sources: [],
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [backendHealth, setBackendHealth] = useState(null);

  // Check backend health status from GET /api/health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('/api/health');
        if (response.ok) {
          const data = await response.json();
          setBackendHealth(data);
        } else {
          setBackendHealth({ status: 'error' });
        }
      } catch (err) {
        // Fallback direct check if proxy is bypassed
        try {
          const directResp = await fetch('http://127.0.0.1:8000/api/health');
          if (directResp.ok) {
            const directData = await directResp.json();
            setBackendHealth(directData);
            return;
          }
        } catch {
          // Backend is offline or starting up
        }
        setBackendHealth({ status: 'offline' });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Stop active audio playback
  const handleStopPlayback = () => {
    if (currentAudioRef.current) {
      try {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
      } catch (err) {
        console.warn("Error pausing audio:", err);
      }
      currentAudioRef.current = null;
    }
    setVoiceState('idle');
  };

  // Convert answer text to speech and play in browser
  const handleSynthesizeAndPlay = async (answerText, languageCode) => {
    try {
      handleStopPlayback();
      setVoiceState('speaking');

      const synthPayload = {
        text: answerText,
        language: languageCode || selectedLanguage || 'en-IN',
      };

      let response = await fetch('/api/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(synthPayload),
      });

      if (!response.ok && (response.status === 404 || response.status === 502)) {
        response = await fetch('http://127.0.0.1:8000/api/voice/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(synthPayload),
        });
      }

      if (!response.ok) {
        console.warn('TTS synthesize returned status:', response.status);
        setVoiceState('idle');
        return;
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.onended = () => {
        setVoiceState('idle');
        currentAudioRef.current = null;
      };

      audio.onerror = (e) => {
        console.error('Audio playback error:', e);
        setVoiceState('idle');
        currentAudioRef.current = null;
      };

      await audio.play();

    } catch (err) {
      console.warn('Voice playback failed:', err);
      setVoiceState('idle');
    }
  };

  const handleSendMessage = async (text, language = selectedLanguage, isVoice = false) => {
    // If speaking, stop playback
    handleStopPlayback();

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text,
      timestamp: formatTime(),
    };

    // Build temporary conversation history payload from active React state
    // Excludes the default greeting (id: 1) and error messages
    const historyPayload = messages
      .filter((m) => (m.sender === 'user' || m.sender === 'ai') && !m.isError && m.id !== 1)
      .slice(-10)
      .map((m) => ({
        role: m.sender === 'user' ? 'user' : 'assistant',
        content: m.text,
      }));

    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    const payload = {
      session_id: sessionId,
      message: text,
      language: language || selectedLanguage || 'en-IN',
      history: historyPayload,
      input_type: isVoice ? 'voice' : 'text',
    };

    try {
      let response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      // Direct fallback if vite proxy is not routed
      if (!response.ok && (response.status === 404 || response.status === 502)) {
        response = await fetch('http://127.0.0.1:8000/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error (${response.status})`);
      }

      const data = await response.json();
      const aiMsg = {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.answer || "No response generated.",
        sources: data.sources || [],
        userQuestion: text,
        timestamp: formatTime(),
      };

      setMessages((prev) => [...prev, aiMsg]);

      // If voice message, trigger TTS voice response
      if (isVoice && data.answer) {
        await handleSynthesizeAndPlay(data.answer, language || selectedLanguage);
      } else {
        setVoiceState('idle');
      }

    } catch (err) {
      console.error("Error communicating with chat API:", err);
      const errorMsg = {
        id: Date.now() + 1,
        sender: 'ai',
        text: err.message || "An error occurred while contacting the hospital assistant. Please try again.",
        sources: [],
        timestamp: formatTime(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
      setVoiceState('idle');
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendVoiceMessage = (text, detectedLanguage) => {
    const activeLang = detectedLanguage || selectedLanguage;
    handleSendMessage(text, activeLang, true);
  };

  const handleSelectPrompt = (promptText) => {
    handleSendMessage(promptText, selectedLanguage, false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Top Hospital Header with Logo, Language Selector and API Status */}
      <Header
        backendHealth={backendHealth}
        selectedLanguage={selectedLanguage}
        onSelectLanguage={setSelectedLanguage}
      />

      {/* Main Chat Messages Container */}
      <ChatArea messages={messages} isTyping={isTyping} />

      {/* Quick Prompt Suggestions */}
      <QuickPrompts onSelectPrompt={handleSelectPrompt} />

      {/* Input Form with Send, Microphone Lifecycle, and Voice Controls */}
      <InputBar
        onSendMessage={(text) => handleSendMessage(text, selectedLanguage, false)}
        onSendVoiceMessage={handleSendVoiceMessage}
        selectedLanguage={selectedLanguage}
        onSelectLanguage={setSelectedLanguage}
        voiceState={voiceState}
        setVoiceState={setVoiceState}
        onStopPlayback={handleStopPlayback}
        disabled={isTyping}
      />
    </div>
  );
}
