import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Square, Volume2, Loader2, AlertCircle, Globe } from 'lucide-react';

const LANGUAGE_LABELS = {
  'auto': 'Auto Detect',
  'en-IN': 'English',
  'ml-IN': 'മലയാളം',
  'hi-IN': 'हिन्दी',
};

export default function InputBar({
  onSendMessage,
  onSendVoiceMessage,
  selectedLanguage = 'auto',
  onSelectLanguage,
  voiceState = 'idle',
  setVoiceState,
  onStopPlayback,
  disabled,
}) {
  const [inputText, setInputText] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // Clean up media streams on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || disabled || voiceState === 'listening') return;
    onSendMessage(inputText.trim(), selectedLanguage);
    setInputText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Start microphone recording
  const startRecording = async () => {
    setErrorMessage('');
    audioChunksRef.current = [];

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage('Audio recording is not supported in this browser.');
      setVoiceState('error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = '';
        }
      }

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        // Stop audio tracks
        stream.getTracks().forEach((track) => track.stop());

        if (audioChunksRef.current.length === 0) {
          setVoiceState('idle');
          return;
        }

        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });

        // Trigger STT transcription
        await handleTranscribe(audioBlob);
      };

      recorder.start(250); // collect in 250ms chunks
      setVoiceState('listening');

    } catch (err) {
      console.error('Microphone access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Microphone access was denied. Please allow microphone permissions in your browser address bar.');
      } else if (err.name === 'NotFoundError') {
        setErrorMessage('No microphone device found on this computer.');
      } else {
        setErrorMessage('Microphone error: ' + (err.message || 'Unable to access audio'));
      }
      setVoiceState('error');
    }
  };

  // Stop microphone recording and finalize
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  // Send audio blob to /api/voice/transcribe
  const handleTranscribe = async (audioBlob) => {
    setVoiceState('transcribing');
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'microphone_input.webm');
      formData.append('language', selectedLanguage);

      let response = await fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData,
      });

      // Direct fallback if vite proxy bypass
      if (!response.ok && (response.status === 404 || response.status === 502)) {
        response = await fetch('http://127.0.0.1:8000/api/voice/transcribe', {
          method: 'POST',
          body: formData,
        });
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Voice transcription failed.');
      }

      const data = await response.json();
      const transcribedText = (data.text || '').trim();

      if (!transcribedText) {
        setErrorMessage('No speech was detected. Please try speaking again.');
        setVoiceState('error');
        return;
      }

      // Voice transcribed successfully: transition to thinking and trigger chat
      setVoiceState('thinking');
      if (onSendVoiceMessage) {
        onSendVoiceMessage(transcribedText, data.language || selectedLanguage);
      } else {
        onSendMessage(transcribedText, data.language || selectedLanguage);
      }

    } catch (err) {
      console.error('STT error:', err);
      setErrorMessage(err.message || 'Speech recognition failed. Please try again.');
      setVoiceState('error');
    }
  };

  const handleMicButtonClick = () => {
    if (voiceState === 'listening') {
      stopRecording();
    } else if (voiceState === 'speaking') {
      if (onStopPlayback) onStopPlayback();
      setVoiceState('idle');
    } else if (voiceState === 'transcribing' || voiceState === 'thinking') {
      // Busy processing, ignore
    } else {
      startRecording();
    }
  };

  return (
    <footer className="bg-white border-t border-slate-200/80 sticky bottom-0 z-20 shadow-lg">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-2.5">
        
        {/* Dynamic Voice Lifecycle Status Banners */}
        {voiceState === 'listening' && (
          <div className="mb-2 px-3.5 py-2 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-semibold flex items-center justify-between shadow-xs animate-pulse">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-600 animate-ping"></span>
              <span>🔴 Listening... Speak now in {LANGUAGE_LABELS[selectedLanguage] || 'your language'}</span>
            </div>
            <button
              type="button"
              onClick={stopRecording}
              className="px-2.5 py-1 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold transition flex items-center gap-1 cursor-pointer"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Done</span>
            </button>
          </div>
        )}

        {voiceState === 'transcribing' && (
          <div className="mb-2 px-3.5 py-2 rounded-xl bg-teal-50 border border-teal-200 text-teal-800 text-xs font-medium flex items-center gap-2 shadow-xs">
            <Loader2 className="w-4 h-4 animate-spin text-teal-600 shrink-0" />
            <span>Transcribing speech...</span>
          </div>
        )}

        {voiceState === 'thinking' && (
          <div className="mb-2 px-3.5 py-2 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-800 text-xs font-medium flex items-center gap-2 shadow-xs">
            <Loader2 className="w-4 h-4 animate-spin text-indigo-600 shrink-0" />
            <span>Finding information...</span>
          </div>
        )}

        {voiceState === 'speaking' && (
          <div className="mb-2 px-3.5 py-2 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-emerald-600 animate-bounce" />
              <span>🔊 Playing answer...</span>
            </div>
            <button
              type="button"
              onClick={() => {
                if (onStopPlayback) onStopPlayback();
                setVoiceState('idle');
              }}
              className="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition flex items-center gap-1 cursor-pointer"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Stop Audio</span>
            </button>
          </div>
        )}

        {voiceState === 'error' && errorMessage && (
          <div className="mb-2 px-3.5 py-2 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => {
                setErrorMessage('');
                setVoiceState('idle');
              }}
              className="text-amber-700 hover:text-amber-900 font-bold ml-2 text-xs cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        {/* Input Form with Microphone & Language Selector */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          
          {/* Connected Microphone Button with Dynamic Voice State */}
          <div className="relative">
            <button
              type="button"
              onClick={handleMicButtonClick}
              title={
                voiceState === 'listening'
                  ? 'Click to finish speaking'
                  : voiceState === 'speaking'
                  ? 'Click to stop playback'
                  : 'Speak your question (Voice Input)'
              }
              className={`p-2.5 rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-teal-500/20 active:scale-95 flex items-center justify-center shrink-0 cursor-pointer ${
                voiceState === 'listening'
                  ? 'bg-rose-600 text-white shadow-md shadow-rose-600/30 animate-pulse'
                  : voiceState === 'speaking'
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30'
                  : voiceState === 'transcribing' || voiceState === 'thinking'
                  ? 'bg-slate-100 text-teal-600 cursor-wait'
                  : 'text-slate-600 hover:text-teal-700 hover:bg-teal-50 border border-slate-200'
              }`}
            >
              {voiceState === 'listening' ? (
                <Square className="w-5 h-5 fill-current" />
              ) : voiceState === 'speaking' ? (
                <Volume2 className="w-5 h-5 animate-pulse" />
              ) : voiceState === 'transcribing' || voiceState === 'thinking' ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Text Input Area */}
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={disabled || voiceState === 'listening'}
              placeholder={
                voiceState === 'listening'
                  ? 'Listening to microphone...'
                  : selectedLanguage === 'ml-IN'
                  ? 'ചോദ്യങ്ങൾ മലയാളത്തിൽ ചോദിക്കൂ...'
                  : selectedLanguage === 'hi-IN'
                  ? 'अस्पताल के बारे में यहाँ पूछें...'
                  : 'Ask about hospital services, doctors, visiting hours...'
              }
              className="w-full pl-4 pr-10 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 transition placeholder-slate-400 text-slate-800 disabled:opacity-60"
            />
          </div>

          {/* Send Button */}
          <button
            type="submit"
            disabled={!inputText.trim() || disabled || voiceState === 'listening'}
            className={`p-2.5 rounded-xl font-medium transition flex items-center justify-center shrink-0 ${
              inputText.trim() && !disabled && voiceState !== 'listening'
                ? 'bg-teal-600 hover:bg-teal-700 text-white shadow-md shadow-teal-600/25 active:scale-95 cursor-pointer'
                : 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
            }`}
          >
            <Send className="w-5 h-5" />
          </button>
        </form>

        {/* Footer Guidance */}
        <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400 px-1">
          <span>AI guidance is based on verified hospital records.</span>
          <div className="flex items-center gap-1 text-slate-500">
            <Globe className="w-3 h-3 text-teal-600" />
            <span>Mode: {LANGUAGE_LABELS[selectedLanguage] || 'Auto Detect'}</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
