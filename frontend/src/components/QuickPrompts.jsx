import React from 'react';
import { Sparkles } from 'lucide-react';

const SUGGESTIONS = [
  "Tell me about Cardiology",
  "What are the hospital visiting hours?",
  "Which services are available for children?",
  "How can I contact the hospital?",
];

export default function QuickPrompts({ onSelectPrompt }) {
  return (
    <div className="py-2.5 px-4 bg-slate-50 border-t border-slate-200/80">
      <div className="max-w-5xl mx-auto flex items-center gap-2 overflow-x-auto no-scrollbar py-0.5">
        <span className="text-xs font-semibold text-slate-500 flex items-center gap-1 shrink-0">
          <Sparkles className="w-3.5 h-3.5 text-teal-600" />
          Quick Questions:
        </span>
        {SUGGESTIONS.map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectPrompt(prompt)}
            className="shrink-0 px-3 py-1.5 rounded-full text-xs font-medium bg-white text-slate-700 border border-slate-200 hover:border-teal-400 hover:bg-teal-50 hover:text-teal-800 transition active:scale-95 shadow-2xs"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
