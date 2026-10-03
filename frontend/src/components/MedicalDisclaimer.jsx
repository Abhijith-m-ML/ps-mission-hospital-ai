import React from 'react';
import { Info } from 'lucide-react';

export default function MedicalDisclaimer({ disclaimerText }) {
  if (!disclaimerText) return null;

  // Clean prefix like "Please note that", "Disclaimer:", etc.
  const cleanedText = disclaimerText
    .replace(/^(?:please\s+note(?:\s*that)?\s*[:\-]??|disclaimer\s*[:\-]??|medical\s*disclaimer\s*[:\-]??|important\s*[:\-]??)\s*/i, "")
    .trim();

  return (
    <div className="mt-3 pt-1">
      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80 text-slate-600 shadow-2xs flex items-start gap-2">
        <Info className="w-3.5 h-3.5 text-teal-600 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed">
          <span className="font-semibold text-slate-800 mr-1">Medical Notice:</span>
          <span className="text-slate-600">{cleanedText}</span>
        </div>
      </div>
    </div>
  );
}
