import React from 'react';
import { Calendar, Stethoscope, ExternalLink } from 'lucide-react';

export default function DoctorCard({ doctor }) {
  if (!doctor || !doctor.doctorName) return null;

  const { doctorName, qualification, department, scheduleRows = [], imageUrl, url } = doctor;

  return (
    <div className="my-3 p-4 sm:p-5 rounded-xl bg-white border border-slate-200/90 shadow-xs hover:border-teal-300 hover:shadow-sm transition-all duration-200">
      
      {/* 1. Department Badge/Header */}
      {department && (
        <div className="mb-3.5 pb-2.5 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-teal-800 tracking-wider uppercase">
            <Stethoscope className="w-3.5 h-3.5 text-teal-600 shrink-0" />
            <span>{department}</span>
          </div>
        </div>
      )}

      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0 space-y-3.5">
          
          {/* 2. Doctor Label & Prominent Name (18-20px, font-bold 600/700, no ALL CAPS) */}
          <div>
            <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
              Doctor
            </span>
            <h3 className="text-[18px] sm:text-[19px] font-bold text-slate-900 tracking-tight leading-snug">
              {doctorName}
            </h3>
          </div>

          {/* 3. Qualification Label & Value */}
          {qualification && (
            <div>
              <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                Qualification
              </span>
              <p className="text-[14px] font-normal text-slate-600 leading-relaxed">
                {qualification}
              </p>
            </div>
          )}

          {/* 4. OP Consultation Label & Day-by-Day Rows */}
          {scheduleRows && scheduleRows.length > 0 && (
            <div className="pt-1">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
                <Calendar className="w-3.5 h-3.5 text-teal-600 shrink-0" />
                <span>OP Consultation</span>
              </span>

              <div className="space-y-3 pl-0.5">
                {scheduleRows.map((row, rIdx) => (
                  <div key={rIdx} className="text-xs sm:text-[13px] leading-relaxed">
                    <span className="font-semibold text-slate-800 block mb-1">
                      {row.day}
                    </span>
                    {row.timings && row.timings.length > 0 ? (
                      <div className="flex flex-col gap-1 pl-2.5 border-l-2 border-teal-300">
                        {row.timings.map((t, tIdx) => (
                          <span key={tIdx} className="text-slate-700 font-medium">
                            {t}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Doctor Photo if available */}
        {imageUrl && (
          <img
            src={imageUrl}
            alt={doctorName}
            className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl object-cover border border-slate-200 shadow-xs shrink-0 bg-slate-50"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
        )}
      </div>

      {/* Optional Link to Hospital Doctor Page */}
      {url && (
        <div className="mt-3.5 pt-2.5 border-t border-slate-100 flex justify-end">
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-teal-700 hover:text-teal-800 transition"
            title={`View official hospital page for ${doctorName}`}
          >
            <span>View hospital page</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}

    </div>
  );
}
