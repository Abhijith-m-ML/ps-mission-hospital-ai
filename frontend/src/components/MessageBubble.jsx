import React, { useState, useRef } from 'react';
import { Bot, User, Clock, BookOpen, ExternalLink, AlertCircle, Volume2, Square } from 'lucide-react';
import AssistantResponse from './AssistantResponse';

function formatSourceLabel(src) {
  if (src.doctor_name) {
    const dept = src.department || (src.section !== "General" ? src.section : "");
    return dept ? `${src.doctor_name} — ${dept}` : src.doctor_name;
  }
  if (src.content_type === "department" || (src.section && src.section !== "General" && src.section !== "Hospital")) {
    const dept = src.department || src.section;
    if (src.title && src.title.toLowerCase().includes("department")) {
      return src.title;
    }
    return `${dept} Department — P.S. Mission Hospital`;
  }
  if (src.content_type === "facility") {
    const dept = src.department || src.section;
    const title = src.title ? src.title.replace(/^Facilities:\s*/i, "") : "";
    return title ? `${title} — ${dept}` : `${dept} Facility`;
  }
  if (src.title) {
    return `${src.title} — P.S. Mission Hospital`;
  }
  return "P.S. Mission Hospital";
}

export default function MessageBubble({ message }) {
  const isUser = message.sender === 'user';
  const isError = Boolean(message.isError);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const handleSpeak = async () => {
    if (isPlaying && audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setIsPlaying(false);
      return;
    }

    try {
      setIsPlaying(true);
      const resp = await fetch('/api/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message.text, language: 'auto' }),
      });

      if (!resp.ok) {
        setIsPlaying(false);
        return;
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onended = () => {
        setIsPlaying(false);
        audioRef.current = null;
      };
      audio.onerror = () => {
        setIsPlaying(false);
        audioRef.current = null;
      };

      await audio.play();
    } catch {
      setIsPlaying(false);
    }
  };

  return (
    <div className={`flex w-full mb-4 items-start gap-2.5 sm:gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      
      {/* Avatar */}
      <div 
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-sm ${
          isUser 
            ? 'bg-slate-700 text-white' 
            : isError
            ? 'bg-rose-600 text-white'
            : 'bg-teal-600 text-white shadow-teal-500/20'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : isError ? <AlertCircle className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Bubble Content */}
      <div className={`max-w-[85%] sm:max-w-[75%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        
        {/* Sender Name & Role */}
        <div className="flex items-center gap-1.5 mb-1 px-1">
          <span className="text-xs font-semibold text-slate-700">
            {isUser ? 'You (Patient)' : 'P.S. Mission AI Assistant'}
          </span>
          <span className="text-[10px] text-slate-400 flex items-center gap-0.5">
            <Clock className="w-2.5 h-2.5" />
            {message.timestamp}
          </span>
          {!isUser && !isError && message.text && (
            <button
              type="button"
              onClick={handleSpeak}
              title={isPlaying ? "Stop audio" : "Listen to answer aloud"}
              className="ml-1.5 text-slate-400 hover:text-teal-600 transition p-0.5 rounded cursor-pointer"
            >
              <Volume2 className={`w-3.5 h-3.5 ${isPlaying ? 'text-teal-600 animate-pulse' : ''}`} />
            </button>
          )}
        </div>

        {/* Message Box */}
        <div 
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm transition-all ${
            isUser
              ? 'bg-teal-600 text-white rounded-tr-none'
              : isError
              ? 'bg-rose-50 border border-rose-200 text-rose-800 rounded-tl-none'
              : 'bg-white border border-slate-200/90 text-slate-800 rounded-tl-none'
          }`}
        >
          {isUser || isError ? (
            <p className="whitespace-pre-wrap">{message.text}</p>
          ) : (
            <AssistantResponse text={message.text} sources={message.sources} userQuestion={message.userQuestion} />
          )}

          {/* Grounded Source Citations */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-3 pt-2.5 border-t border-slate-100 flex flex-col gap-2">
              <span className="text-[11px] font-semibold text-slate-500 flex items-center gap-1">
                <BookOpen className="w-3 h-3 text-teal-600" />
                Sources:
              </span>
              <div className="flex flex-wrap gap-2 pt-0.5">
                {message.sources.map((src, idx) => {
                  const label = formatSourceLabel(src);
                  const isDoctor = Boolean(src.doctor_name || src.content_type === 'doctor');
                  if (src.url) {
                    return (
                      <a
                        key={src.chunk_id || idx}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition shadow-sm hover:shadow ${
                          isDoctor
                            ? 'bg-teal-50 hover:bg-teal-100 text-teal-900 border border-teal-200/90'
                            : 'bg-slate-50 hover:bg-slate-100 text-slate-800 border border-slate-200/90'
                        }`}
                        title={`Open official hospital page: ${label}`}
                      >
                        <span className="truncate max-w-[260px] sm:max-w-[340px]">{label}</span>
                        <ExternalLink className="w-3 h-3 text-teal-600 shrink-0" />
                      </a>
                    );
                  }
                  return (
                    <span
                      key={src.chunk_id || idx}
                      className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200"
                    >
                      <span className="truncate max-w-[260px] sm:max-w-[340px]">{label}</span>
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
