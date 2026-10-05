import React from 'react';
import ReactMarkdown from 'react-markdown';
import DoctorCard from './DoctorCard';
import MedicalDisclaimer from './MedicalDisclaimer';

function cleanDoctorName(name) {
  if (!name) return "";
  let clean = name.replace(/\*\*/g, "").replace(/^[-*•\d\.\s]+/, "").trim();
  clean = clean.replace(/\bJHON\b/gi, "John");
  const words = clean.split(/\s+/);
  return words.map(w => {
    const lw = w.toLowerCase();
    if (lw === "dr." || lw === "dr" || lw === "sr." || lw === "sr") {
      return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
    }
    if (w.length === 1) return w.toUpperCase();
    return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
  }).join(" ");
}

function cleanDays(d) {
  let res = d.replace(/^[:\-\s,|]+/, "").replace(/[:\-\s,|]+$/, "").trim();
  res = res
    .replace(/\bMONDAY\s*(?:to|-|–)\s*FRIDAY\b/i, "Monday – Friday")
    .replace(/\bMonday\s*(?:to|-|–)\s*Friday\b/i, "Monday – Friday")
    .replace(/\bMONDAY\s*(?:to|-|–)\s*SATURDAY\b/i, "Monday – Saturday")
    .replace(/\bMonday\s*(?:to|-|–)\s*Saturday\b/i, "Monday – Saturday")
    .replace(/\bTUESDAY\s*,\s*THURSDAY\b/i, "Tuesday & Thursday")
    .replace(/\bTuesday\s*,\s*Thursday\b/i, "Tuesday & Thursday")
    .replace(/\bDAILY\b/i, "Daily")
    .replace(/\bdaily\b/i, "Daily");
  return res;
}

function cleanTime(t) {
  let s = t.trim().replace(/^[:\-\s,|]+/, "").replace(/[:\-\s,|.]+$/, "");
  s = s.replace(/\bfrom\s+/i, "");
  s = s.replace(/(\d{1,2})\.(\d{2})\s*(AM|PM)/gi, "$1:$2 $3");
  s = s.replace(/(\d)(AM|PM)/gi, "$1 $2");
  s = s.replace(/\s+(?:to|-)\s+/gi, " – ");
  s = s.replace(/\s*-\s*/g, " – ");
  return s.trim();
}

function parseScheduleRows(scheduleText) {
  if (!scheduleText) return [];
  let clean = scheduleText.replace(/\*\*/g, "").trim();
  clean = clean.replace(/^(?:o\.?p\.?\s*(?:consultation)?\s*(?:schedule)?|consultation\s*schedule|schedule)\s*[:\-]??\s*/i, "").trim();
  clean = clean.replace(/^(?:his|her)\s+op\s+consultation\s+schedule\s+(?:is|are)?\s*/i, "");

  const rows = [];

  // Match all day occurrences across the text
  const dayRegex = /\b(monday\s*(?:to|-|–)\s*friday|monday\s*(?:to|-|–)\s*saturday|tuesday\s*,\s*thursday|tuesday\s*&\s*thursday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|daily)\b/gi;
  const matches = [];
  let m;
  while ((m = dayRegex.exec(clean)) !== null) {
    matches.push({ index: m.index, text: m[0] });
  }

  if (matches.length > 0) {
    for (let i = 0; i < matches.length; i++) {
      const start = matches[i].index;
      const end = i + 1 < matches.length ? matches[i + 1].index : clean.length;
      const segment = clean.slice(start, end).trim();

      const dayStr = cleanDays(matches[i].text);
      const afterDay = segment.slice(matches[i].text.length).replace(/^[:\-\s,|]+/, "").trim();

      const rawTimings = afterDay.split(/\s*(?:&|\band\b)\s*/i).map(t => cleanTime(t)).filter(Boolean);
      rows.push({
        day: dayStr,
        timings: rawTimings.length > 0 ? rawTimings : [cleanTime(afterDay)].filter(Boolean)
      });
    }
  } else {
    if (clean.includes("|")) {
      const parts = clean.split("|");
      const day = cleanDays(parts[0]);
      const timings = parts[1].split(/\s*(?:&|\band\b)\s*/i).map(t => cleanTime(t)).filter(Boolean);
      rows.push({ day, timings });
    } else {
      const cleaned = cleanDays(clean);
      if (cleaned) {
        rows.push({ day: cleaned, timings: [] });
      }
    }
  }

  return rows;
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isMedicalSafetyRelevant(userQuestion, text) {
  const q = (userQuestion || "").toLowerCase();

  // Symptoms, diagnosis, treatment, medication, emergency
  const clinicalKeywords = [
    "headache", "headaches", "chest pain", "fever", "stomach ache", "abdominal pain",
    "cough", "cold", "pain", "ache", "rash", "dizziness", "vomiting", "diarrhea",
    "breathless", "breathlessness", "seizure", "seizures", "bleeding", "injury",
    "fracture", "swelling", "itching", "burning", "weakness", "unconscious",
    "numb", "numbness", "trauma", "accident", "wound", "burn", "infection",
    "sick", "unwell", "nausea", "diagnose", "diagnosis", "treatment", "medication",
    "medicine", "drug", "emergency", "urgent", "heart attack", "stroke", "poison"
  ];

  return clinicalKeywords.some(kw => {
    const pattern = new RegExp("\\b" + kw + "\\b", "i");
    return pattern.test(q);
  });
}

function extractDisclaimer(text) {
  if (!text) return { mainText: "", disclaimer: null };
  const regex = /(?:^|\n\n)(?:(?:please\s+note|disclaimer|medical\s+disclaimer|important\s+note)[\s\S]*|if\s+(?:you|your\s+child)\s+(?:experience|are\s+experiencing)[\s\S]*)$/i;
  const match = text.match(regex);
  if (match) {
    return {
      mainText: text.slice(0, match.index).trim(),
      disclaimer: match[0].trim()
    };
  }
  return { mainText: text, disclaimer: null };
}

function extractDoctors(text, sources = []) {
  const doctors = [];
  let remainingText = text;

  // Approach 1: Check verified doctor records from sources
  const doctorSources = sources.filter(s => s.doctor_name || s.content_type === "doctor");
  if (doctorSources.length > 0) {
    for (const s of doctorSources) {
      const cleanName = cleanDoctorName(s.doctor_name);
      const baseName = cleanName.toLowerCase().replace(/^(?:dr\.?|sr\.?)\s*/, "").trim();
      const lastName = baseName.split(/\s+/).pop();

      if (cleanName && (remainingText.toLowerCase().includes(baseName) || remainingText.toLowerCase().includes(lastName))) {
        // Extract qualification from text if present
        let qual = "";
        const qualRegex = new RegExp(
          "(?:dr\\.?|sr\\.?)[^\\n(]*?" + escapeRegExp(lastName) + "[^\\n(]*?\\(([^)]+)\\)",
          "i"
        );
        const qMatch = remainingText.match(qualRegex);
        if (qMatch) {
          qual = qMatch[1].trim();
        }

        // Extract schedule for this doctor
        let schedText = "";
        const schedBlockRegex = /(?:op\s+consultation\s+schedule|consultation\s+schedule|op\s+schedule|schedule)\s*[:\-]??\s*([\s\S]+?)(?:(?=\n\s*\d+\.\s*(?:dr\.?|sr\.?))|$|\n\n(?:please\s+note|sources|disclaimer))/i;
        const sMatch = remainingText.match(schedBlockRegex);
        if (sMatch) {
          schedText = sMatch[1].trim();
        } else {
          // Fallback: check if text has "Monday to Friday from..."
          const dayMatch = remainingText.match(/\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|daily)\b[\s\S]+?(?:(?=\n\n(?:please\s+note|sources|disclaimer))|$)/i);
          if (dayMatch) {
            schedText = dayMatch[0].trim();
          }
        }

        doctors.push({
          doctorName: cleanName,
          qualification: qual,
          department: s.department || "",
          scheduleRows: parseScheduleRows(schedText),
          imageUrl: s.image_url || null,
          url: s.url || null
        });
      }
    }

    if (doctors.length > 0) {
      // Clean up dense prose intro and schedule lines from remainingText
      remainingText = remainingText
        .replace(/^(?:the\s+available\s+doctor[^\n]*?(?:\.\s*|\n))/i, "")
        .replace(/^(?:the\s+following\s+doctors\s+are\s+available[^\n]*?(?:\:\s*|\.\s*|\n))/i, "")
        .replace(/^(?:the\s+doctor\s+available[^\n]*?(?:\.\s*|\n))/i, "")
        .replace(/(?:op\s+consultation\s+schedule|consultation\s+schedule|op\s+schedule|schedule)\s*[:\-]??\s*[\s\S]+?(?:(?=\n\n(?:please\s+note|sources|disclaimer))|$)/i, "")
        .trim();
      return { remainingText, doctors };
    }
  }

  // Fallback: Pattern 1: Bullet list containing doctor info
  const bulletGroupRegex = /(?:^|\n)(?:[ \t]*[*•-][ \t]+(?:\*\*)?(?:Dr\.?|Dr\s|DR\s)[^\n]+(?:\n[ \t]*[*•-][ \t]+[^\n]+)+)/gi;
  let match;

  while ((match = bulletGroupRegex.exec(remainingText)) !== null) {
    const block = match[0];
    const lines = block.split("\n").map(l => l.trim()).filter(Boolean);

    let docName = "";
    let qual = "";
    let dept = "";
    let schedRaw = "";

    for (const line of lines) {
      if (/(?:Dr\.?|Dr\s|DR\s)/i.test(line) && !docName) {
        const parenIdx = line.indexOf("(");
        const lastParen = line.lastIndexOf(")");
        if (parenIdx !== -1) {
          docName = line.slice(0, parenIdx).replace(/[*•-]/g, "").trim();
          qual = lastParen > parenIdx 
            ? line.slice(parenIdx + 1, lastParen).trim() 
            : line.slice(parenIdx + 1).replace(/\)+$/, "").trim();
        } else {
          docName = line.replace(/[*•-]/g, "").trim();
        }
      } else if (/department\s*:/i.test(line)) {
        dept = line.replace(/^[^:]*:\s*/, "").replace(/\*\*/g, "").trim();
      } else if (/(?:schedule|op|consultation|timing)/i.test(line)) {
        schedRaw = line.replace(/^[^:]*:\s*/, "").replace(/\*\*/g, "").trim();
      } else if (/qualification\s*:/i.test(line)) {
        qual = line.replace(/^[^:]*:\s*/, "").replace(/\*\*/g, "").trim();
      }
    }

    if (docName) {
      const cleanName = cleanDoctorName(docName);
      const matchedSource = sources.find(
        s => s.doctor_name && cleanDoctorName(s.doctor_name).toLowerCase() === cleanName.toLowerCase()
      );

      doctors.push({
        doctorName: cleanName,
        qualification: qual,
        department: dept || (matchedSource ? matchedSource.department : ""),
        scheduleRows: parseScheduleRows(schedRaw),
        imageUrl: matchedSource ? matchedSource.image_url : null,
        url: matchedSource ? matchedSource.url : null
      });

      remainingText = remainingText.replace(block, "").trim();
    }
  }

  // Fallback: Pattern 2: Numbered doctors list
  const numberedDoctorRegex = /(?:^|\n)\s*\d+\.\s*(?:\*\*)?(?:Dr\.?|Dr\s|DR\s)[^\n]+(?:\n\s*[-*•]\s*[^\n]+)*/gi;
  if (doctors.length === 0) {
    while ((match = numberedDoctorRegex.exec(remainingText)) !== null) {
      const block = match[0];
      const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
      let docName = "";
      let qual = "";
      let schedRaw = "";

      for (const line of lines) {
        if (/^\d+\./.test(line)) {
          const stripped = line.replace(/^\d+\.\s*/, "").replace(/\*\*/g, "").trim();
          const parenIdx = stripped.indexOf("(");
          const lastParen = stripped.lastIndexOf(")");
          if (parenIdx !== -1) {
            docName = stripped.slice(0, parenIdx).trim();
            qual = lastParen > parenIdx 
              ? stripped.slice(parenIdx + 1, lastParen).trim() 
              : stripped.slice(parenIdx + 1).replace(/\)+$/, "").trim();
          } else {
            docName = stripped;
          }
        } else if (/(?:schedule|op|consultation|timing)/i.test(line)) {
          schedRaw = line.replace(/^[^:]*:\s*/, "").replace(/\*\*/g, "").trim();
        }
      }

      if (docName) {
        const cleanName = cleanDoctorName(docName);
        const matchedSource = sources.find(
          s => s.doctor_name && cleanDoctorName(s.doctor_name).toLowerCase() === cleanName.toLowerCase()
        );

        doctors.push({
          doctorName: cleanName,
          qualification: qual,
          department: matchedSource ? matchedSource.department : "",
          scheduleRows: parseScheduleRows(schedRaw),
          imageUrl: matchedSource ? matchedSource.image_url : null,
          url: matchedSource ? matchedSource.url : null
        });

        remainingText = remainingText.replace(block, "").trim();
      }
    }
  }

  return { remainingText, doctors };
}

const markdownComponents = {
  p: ({ children }) => (
    <p className="text-[14.5px] leading-[1.65] text-slate-800 mb-3 last:mb-0">
      {children}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-900">
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em className="italic text-slate-700">
      {children}
    </em>
  ),
  ul: ({ children }) => (
    <ul className="list-disc pl-5 my-2.5 space-y-2 text-[14.5px] text-slate-700 leading-relaxed">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 my-2.5 space-y-2 text-[14.5px] text-slate-700 leading-relaxed">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="leading-relaxed pl-1 text-slate-700">
      {children}
    </li>
  ),
  h1: ({ children }) => (
    <h1 className="text-lg font-bold text-slate-900 mt-4 mb-2 tracking-tight">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-base font-bold text-slate-900 mt-3.5 mb-2 tracking-tight">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-900 mt-3 mb-1.5 uppercase tracking-wide">
      {children}
    </h3>
  ),
  hr: () => <hr className="border-slate-200 my-3.5" />,
  blockquote: ({ children }) => (
    <blockquote className="border-l-3 border-teal-500 pl-3 my-2.5 text-slate-600 italic text-[14px]">
      {children}
    </blockquote>
  ),
};

export default function AssistantResponse({ text, sources = [], userQuestion = "" }) {
  if (!text) return null;

  // Defensive unpacking: Ensure raw JSON string never renders in the chat UI
  let sanitizedText = String(text).trim();
  if (sanitizedText.startsWith("{") && (sanitizedText.includes('"answer"') || sanitizedText.includes("'answer'"))) {
    try {
      const parsed = JSON.parse(sanitizedText);
      if (parsed && parsed.answer) {
        sanitizedText = String(parsed.answer);
      }
    } catch {
      const match = sanitizedText.match(/["']answer["']\s*:\s*"([\s\S]*?)(?:"\s*,\s*["']source_ids["']|"\s*\})/);
      if (match) {
        sanitizedText = match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n');
      } else {
        const looseMatch = sanitizedText.match(/["']answer["']\s*:\s*"([\s\S]*)/);
        if (looseMatch) {
          sanitizedText = looseMatch[1].replace(/["'}\s]+$/, '').replace(/\\"/g, '"').replace(/\\n/g, '\n');
        }
      }
    }
  }

  // 1. Separate Medical Disclaimer if present
  const { mainText, disclaimer } = extractDisclaimer(sanitizedText);

  // 2. Extract Structured Doctor Cards
  const { remainingText, doctors } = extractDoctors(mainText, sources);

  // 3. Conditional Safety Check: Only show disclaimer if user question involves clinical symptoms/emergency
  const showDisclaimer = disclaimer && isMedicalSafetyRelevant(userQuestion, text);

  return (
    <div className="flex flex-col">
      
      {/* Intro or main textual response */}
      {remainingText && (
        <div className="prose-chat mb-1">
          <ReactMarkdown components={markdownComponents}>
            {remainingText}
          </ReactMarkdown>
        </div>
      )}

      {/* Structured Doctor Information Cards */}
      {doctors.length > 0 && (
        <div className="flex flex-col gap-2.5 my-1">
          {doctors.map((doc, idx) => (
            <DoctorCard key={idx} doctor={doc} />
          ))}
        </div>
      )}

      {/* Conditional Medical Safety Notice: Only shown for symptoms/emergency questions */}
      {showDisclaimer && (
        <MedicalDisclaimer disclaimerText={disclaimer} />
      )}

    </div>
  );
}
