import React from 'react';
import { Activity, PhoneCall, ShieldCheck, Wifi, WifiOff } from 'lucide-react';

export default function Header({ backendHealth, selectedLanguage = 'auto', onSelectLanguage }) {
  const isHealthy = backendHealth?.status === 'healthy';

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-3.5 flex items-center justify-between">
        
        {/* Hospital Branding & Logo */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-teal-600 flex items-center justify-center text-white shadow-md shadow-teal-600/20">
            <Activity className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold text-slate-900 leading-tight">
                P.S. Mission Hospital
              </h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-teal-50 text-teal-700 border border-teal-200">
                <ShieldCheck className="w-3 h-3 mr-1 text-teal-600" />
                Verified Care
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">
              Virtual Healthcare Assistant
            </p>
          </div>
        </div>

        {/* Right Info: Language Selector, Status Pill & Emergency Call */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          
          {/* Language Selector Dropdown */}
          <div className="flex items-center">
            <select
              value={selectedLanguage}
              onChange={(e) => onSelectLanguage && onSelectLanguage(e.target.value)}
              className="text-xs font-semibold bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-teal-500/20 cursor-pointer transition shadow-2xs"
              title="Select interaction language"
              aria-label="Language selector"
            >
              <option value="auto">🌐 Auto Detect</option>
              <option value="en-IN">English</option>
              <option value="ml-IN">മലയാളം</option>
              <option value="hi-IN">हिन्दी</option>
            </select>
          </div>
          
          {/* Backend Health Badge */}
          <div 
            title={isHealthy ? `Connected to ${backendHealth.app_name} (${backendHealth.environment})` : 'Connecting to API...'}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all ${
              isHealthy 
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            {isHealthy ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="hidden md:inline">API Online</span>
                <span className="md:hidden">Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                <span className="hidden md:inline">Connecting API...</span>
                <span className="md:hidden">Waiting...</span>
              </>
            )}
          </div>

          {/* Emergency Hotline */}
          <a
            href="tel:+914842700543"
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 text-xs font-semibold transition"
          >
            <PhoneCall className="w-3.5 h-3.5 text-rose-600" />
            <span className="hidden sm:inline">Emergency: +91 484 2700543</span>
            <span className="sm:hidden">Emergency</span>
          </a>

        </div>

      </div>
    </header>
  );
}
