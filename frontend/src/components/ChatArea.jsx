import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import { Hospital, Stethoscope, Clock, ShieldAlert } from 'lucide-react';

export default function ChatArea({ messages, isTyping }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 max-w-5xl w-full mx-auto">
      
      {/* Hospital Welcome Banner */}
      <div className="bg-gradient-to-br from-teal-50/80 via-white to-sky-50/60 border border-teal-100 rounded-2xl p-5 sm:p-6 mb-6 shadow-sm">
        <div className="flex items-start sm:items-center gap-3.5 mb-3">
          <div className="p-2.5 bg-teal-600 rounded-xl text-white shadow-sm">
            <Hospital className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900">
              Welcome to P.S. Mission Hospital Assistant
            </h2>
            <p className="text-xs sm:text-sm text-slate-600">
              Your automated guide to hospital facilities, clinical departments, visiting hours, and appointments.
            </p>
          </div>
        </div>

        {/* Feature Badges */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-3 border-t border-teal-100/60 text-xs">
          <div className="flex items-center gap-2 text-slate-700 bg-white/70 px-3 py-2 rounded-lg border border-slate-200/60">
            <Clock className="w-4 h-4 text-teal-600 shrink-0" />
            <span>24/7 Casualty & ICCU Care</span>
          </div>
          <div className="flex items-center gap-2 text-slate-700 bg-white/70 px-3 py-2 rounded-lg border border-slate-200/60">
            <Stethoscope className="w-4 h-4 text-teal-600 shrink-0" />
            <span>Multi-Specialty Care</span>
          </div>
          <div className="flex items-center gap-2 text-slate-700 bg-white/70 px-3 py-2 rounded-lg border border-slate-200/60">
            <ShieldAlert className="w-4 h-4 text-rose-500 shrink-0" />
            <span>Emergency: +91 484 2700543</span>
          </div>
        </div>
      </div>

      {/* Messages Stream */}
      <div className="space-y-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white flex items-center justify-center shrink-0 shadow-sm">
              <Hospital className="w-4 h-4 animate-spin" />
            </div>
            <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-tl-none shadow-sm flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce"></span>
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.2s]"></span>
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.4s]"></span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

    </main>
  );
}
