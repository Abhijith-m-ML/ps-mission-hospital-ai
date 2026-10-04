"""Intent recognition and classification for hospital inquiries."""
from enum import Enum
import re
from typing import Dict, List, Optional, Pattern, Tuple


class HospitalIntent(str, Enum):
    """Categorization of hospital inquiries."""
    DOCTOR_SEARCH = "doctor_search"
    DOCTOR_SCHEDULE = "doctor_schedule"
    DEPARTMENT_INFORMATION = "department_information"
    FACILITY_INFORMATION = "facility_information"
    HOSPITAL_HOURS = "hospital_hours"
    OP_HOURS = "op_hours"
    REGISTRATION_HOURS = "registration_hours"
    EMERGENCY_INFORMATION = "emergency_information"
    CONTACT_INFORMATION = "contact_information"
    LOCATION = "location"
    INSURANCE = "insurance"
    SERVICES = "services"
    APPOINTMENT_INFORMATION = "appointment_information"
    SYMPTOM_TO_DEPARTMENT = "symptom_to_department"
    GENERAL_HOSPITAL_INFORMATION = "general_hospital_information"
    GENERAL_QUESTION = "general_question"


class IntentClassifier:
    """Classifies user queries into specific hospital intent categories."""

    def __init__(self):
        # 0. SYMPTOM TO DEPARTMENT / DOCTOR
        self.symptom_patterns = [
            r"\b(?:which\s+doctor|whom\s+to\s+consult|who\s+to\s+see|who\s+to\s+consult|what\s+doct(?:o|e)r|what\s+department|which\s+department)\b.*\b(?:pain|hurting|hurt|ache|fever|cough|sick|problem)\b",
            r"\b(?:pain|hurting|hurt|ache|fever|cough|swelling|bleeding)\b.*\b(?:which\s+doctor|which\s+department|whom\s+to|who\s+to|what\s+department|what\s+doct(?:o|e)r|doctor\s+aar|aar\s+doctor|aare\s+kaanikkanam|aarodu\s+kaanikkanam)\b",
            r"\b(?:enik|enikku|ente|my|i\s+have)\b.*\b(?:pain|vedana|vedhana|hurting|hurt)\b",
            r"(?:വേദന|വേദനയുണ്ട്|വേദനയാണ്).*(?:ഏത്\s+ഡോക്ടർ|ഏത്\s+ഡിപ്പാർട്ട്മെന്റ്|ഏതു\s+ഡോക്ടറെ|ആരെ\s+കാണിക്കണം|ഏത്\s+department)",
            r"(?:കാലിനു|കാലിൽ|നെഞ്ചിൽ|വയറ്റിൽ|മുട്ടിൽ|തലയിൽ|കണ്ണിൽ|ചെവിയിൽ|തൊണ്ടയിൽ).*(?:വേദന|pain)",
            r"\b(?:kaalil|kaalinu|leg|calf|knee|chest|head|stomach|back)\s+pain\b",
            r"\b(?:leg|calf|knee|chest|back|head)\s+(?:is\s+)?hurting\b",
            r"\b(?:kaalil|kaalinu)\s+pain\s+(?:und|aanu|anu|aane)\b",
            r"\bdoctor\s+aar\b|\baar\s+doctor\b|\baarodu\s+kaanikkanam\b|\baare\s+kaanikkanam\b",
            r"(ഏത്\s+ഡോക്ടറെ\s+കാണണം|ഏതു\s+ഡോക്ടറെയാണ്\s+കാണിക്കേണ്ടത്|ആരെയാണ്\s+കാണിക്കേണ്ടത്)",
            r"(കാൽ\s+വേദനയ്ക്ക്|കാലിനു\s+pain\s+und|കാലിലൊരു\s+വേദന)",
        ]

        # 0B. APPOINTMENT INFORMATION
        self.appointment_patterns = [
            r"\b(?:how\s+to\s+)?(?:book|make|take|schedule)\s+(?:an\s+)?appointment\b",
            r"\bappointment\s+(?:booking|process|number|details|procedure)\b",
            r"\bhow\s+to\s+(?:get|take)\s+(?:a\s+)?token\b",
            r"\bonline\s+appointment\b",
            r"(അപ്പോയിന്റ്മെന്റ്|ബുക്കിംഗ്)",
            r"(अपॉइंटमेंट|बुकिंग)",
        ]

        # 0C. GENERAL GREETINGS / CONVERSATION
        self.general_question_patterns = [
            r"^(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
            r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+can\s+you\s+help\s+me)\b",
            r"(ഹലോ|നമസ്കാരം)",
            r"(नमस्ते|नमस्कार)",
        ]
        # 1. EMERGENCY
        self.emergency_patterns = [
            r"\b(emergency|casualty|ambulance|er|trauma|resuscitation|cardiac\s+arrest)\b",
            r"\bemergency\s+(?:desk|room|ward|department)\b",
            r"(അത്യാഹിതം|ആംബുലൻസ്|കാഷ്വാലിറ്റി)",
            r"(आपातकालीन|इमरजेंसी|एम्बुलेंस)",
        ]

        # 2. LOCATION
        self.location_patterns = [
            r"\bwhere\s+is\s+(?:the\s+)?hospital\b",
            r"\bwhere\s+are\s+you\s+located\b",
            r"\bwhere\s+is\s+it\s+located\b",
            r"\bhospital\s+location\b",
            r"\baddress\s+of\s+(?:the\s+)?hospital\b",
            r"\bhow\s+to\s+reach\b",
            r"\bdirections\s+to\s+(?:the\s+)?hospital\b",
            r"\broute\s+to\s+(?:the\s+)?hospital\b",
            r"\bwhere\s+is\s+p\.?s\.?\s*mission\b",
            r"\b(landmark|locate|location|address|directions)\b",
            r"(എവിടെയാണ്|റൂട്ട്|വഴി|സ്ഥലം)",
            r"(कहाँ\s+है|स्थान|पता|दिशा)",
        ]

        # 3. CONTACT
        self.contact_patterns = [
            r"\b(phone|telephone|contact|call|reception|mobile|email|helpline|helpdesk)\b(?:\s+(?:number|details|no))?",
            r"\bhow\s+to\s+(?:call|contact)\b",
            r"(ഫോൺ\s+നമ്പർ|ബന്ധപ്പെടുക)",
            r"(फोन\s+नंबर|संपर्क)",
        ]

        # 4. INSURANCE
        self.insurance_patterns = [
            r"\b(insurance|mediclaim|tpa|cashless|claim|policy|coverage|reimbursement|ayushman|karunya)\b",
            r"(ഇൻഷുറൻസ്|മെഡിക്ലെയിം|ക്യാഷ്‌ലെസ്)",
            r"(बीमा|मेडिक्लेम|कैशलेस)",
        ]

        # 5. REGISTRATION HOURS
        self.registration_patterns = [
            r"\b(registration|token|counter)\s+(timing|timings|hours|time|schedule)\b",
            r"\bwhen\s+does\s+registration\s+(start|close|open)\b",
            r"\b(registration|counter)\s+desk\b",
            r"(രജിസ്ട്രേഷൻ\s+സമയം|ടോക്കൺ)",
            r"(पंजीकरण\s+का\s+समय|रजिस्ट्रेशन)",
        ]

        # 6. OP HOURS (Outpatient doctor consultation hours)
        self.op_hours_patterns = [
            r"\b(op|opd|outpatient)\s+(timing|timings|hours|time|schedule)\b",
            r"\bconsultation\s+(timing|timings|hours|schedule)\b",
            r"\bwhen\s+is\s+opd\b",
            r"\bopd\s+working\s+hours\b",
            r"(ഒപി\s+സമയം|ഒ\.പി\s+സമയം|കൺസൾട്ടേഷൻ\s+സമയം)",
            r"(ओपीडी\s+का\s+समय|परामर्श\s+समय)",
        ]

        # 7. HOSPITAL / VISITING HOURS (In-patient visiting hours, visiting rules, hospital hours)
        self.hospital_hours_patterns = [
            r"\b(visiting|visitor|visit)\s+(hours|timings|timing|time|rules)\b",
            r"\bhospital\s+visiting\s+hours\b",
            r"\bwhen\s+can\s+visitors\s+visit\b",
            r"\bvisiting\s+hours\s+for\s+(?:the\s+)?hospital\b",
            r"\bnormal\s+hospital\s+hours\b",
            r"\bhospital\s+working\s+hours\b",
            r"\bhospital\s+timings\b",
            r"\bhospital\s+open\s+time\b",
            r"(സന്ദർശന\s+സമയം|സന്ദർശകർ)",
            r"(मिलने\s+का\s+समय|विजिटिंग\s+आवर्स)",
        ]

        # 8. DOCTOR SCHEDULE
        self.doctor_schedule_patterns = [
            r"\bwhen\s+is\s+dr\b",
            r"\bwhen\s+can\s+i\s+(?:see|meet|visit|consult)(?:\s+dr)?\b",
            r"\bwhen\s+to\s+(?:visit|see|consult|meet)\b",
            r"\b(schedule|timing|timings|availability|hours)\s+of\s+dr\b",
            r"\bdr\b.+?\b(schedule|timing|timings|availability|available)\b",
            r"\b(appointment|consultation\s+timing)\s+with\s+dr\b",
            r"\bavailable\s+timing\s+for\b",
            r"\bwhat\s+are\s+their\s+timings\b",
            r"\bwhat\s+are\s+(?:his|her)\s+timings\b",
            r"(ഡോക്ടറുടെ\s+സമയം|എപ്പോഴാണ്)",
            r"(डॉक्टर\s+का\s+समय|कब\s+मिलेंगे)",
        ]

        # 9. DOCTOR SEARCH
        self.doctor_search_patterns = [
            r"\b(doctor|doctors|physician|physicians|consultant|consultants|specialist|specialists)\b",
            r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:doctor|doctors|available)\b",
            r"\bwho\s+to\s+see\b",
            r"\bwho\s+to\s+consult\b",
            r"\bwhom\s+to\s+consult\b",
            r"\bwhich\s+doctor\b",
            r"\b(cardiologist|pediatrician|paediatrician|neurologist|orthopedician|gynecologist|surgeon|dermatologist|urologist|nephrologist|radiologist|psychiatrist)\b",
            r"(ഡോക്ടർ|ഡോക്ടർമാർ|ഡോക്ടറെ)",
            r"(डॉक्टर|चिकित्सक)",
        ]

        # 10. FACILITY INFORMATION
        self.facility_patterns = [
            r"\b(facility|facilities|equipment|infrastructure|diagnostic|equipment|amenity|amenities)\b",
            r"\b(x-ray|xray|mri|ct\s+scan|ultrasound|echo|echocardiography|dialysis|nicu|picu|ventilator|operation\s+theatre|ot|lab|laboratory|blood\s+test)\b",
            r"(സൗകര്യം|സൗകര്യങ്ങൾ)",
            r"(सुविधा|सुविधाएं)",
        ]

        # 11. SERVICES
        self.services_patterns = [
            r"\b(service|services|health\s+checkup|preventive\s+checkup|package|packages|health\s+package)\b",
            r"(സേവനങ്ങൾ|സേവനം)",
            r"(सेवाएं|सेवा)",
        ]

        # 12. DEPARTMENT INFORMATION
        self.department_patterns = [
            r"\b(?:tell\s+me\s+about|information\s+about|overview\s+of|details\s+of|what\s+is)\s+(?:the\s+)?\w+\s+department\b",
            r"\bdepartment\s+(?:info|information|overview|details)\b",
            r"(വിഭാഗം|ഡിപ്പാർട്ട്മെന്റ്)",
            r"(विभाग)",
        ]

    def classify(
        self,
        query: str,
        has_doctor: bool = False,
        has_department: bool = False,
        has_facility: bool = False,
        has_symptom: bool = False,
    ) -> Tuple[HospitalIntent, float]:
        """Classifies the query string into a HospitalIntent with confidence score."""
        q_lower = query.lower().strip()

        # 1. Emergency has highest clinical safety priority
        if any(re.search(p, q_lower) for p in self.emergency_patterns):
            return HospitalIntent.EMERGENCY_INFORMATION, 0.98

        # 2. Symptom to Department / Doctor Consultation (User has a symptom or is asking which specialist/department to consult)
        if any(re.search(p, q_lower) for p in self.symptom_patterns) or (has_symptom and not (has_doctor and any(w in q_lower for w in ["timing", "schedule", "when"]))):
            return HospitalIntent.SYMPTOM_TO_DEPARTMENT, 0.95

        # 3. Location & directions
        if any(re.search(p, q_lower) for p in self.location_patterns):
            return HospitalIntent.LOCATION, 0.95

        # 4. Contact & Phone numbers
        if any(re.search(p, q_lower) for p in self.contact_patterns):
            return HospitalIntent.CONTACT_INFORMATION, 0.95

        # 5. Insurance / Mediclaim
        if any(re.search(p, q_lower) for p in self.insurance_patterns):
            return HospitalIntent.INSURANCE, 0.95

        # 6. Doctor Schedule: Doctor + timing / schedule keywords
        if any(re.search(p, q_lower) for p in self.doctor_schedule_patterns):
            return HospitalIntent.DOCTOR_SCHEDULE, 0.95
        if has_doctor and any(w in q_lower for w in ["timing", "timings", "schedule", "when", "time", "available", "visit", "meet", "hours"]):
            return HospitalIntent.DOCTOR_SCHEDULE, 0.95

        # 7. Appointment booking / token inquiries
        if any(re.search(p, q_lower) for p in self.appointment_patterns):
            return HospitalIntent.APPOINTMENT_INFORMATION, 0.95

        # 8. Registration Hours (e.g. registration desk opening/closing)
        if any(re.search(p, q_lower) for p in self.registration_patterns):
            return HospitalIntent.REGISTRATION_HOURS, 0.95

        # 9. OP Hours (Outpatient clinic hours)
        if any(re.search(p, q_lower) for p in self.op_hours_patterns):
            return HospitalIntent.OP_HOURS, 0.95

        # 10. Hospital Visiting Hours (Distinct from OP consultation hours)
        if any(re.search(p, q_lower) for p in self.hospital_hours_patterns):
            return HospitalIntent.HOSPITAL_HOURS, 0.95

        # 11. Doctor Search: User searching for physicians
        if any(re.search(p, q_lower) for p in self.doctor_search_patterns):
            return HospitalIntent.DOCTOR_SEARCH, 0.95
        if has_doctor:
            return HospitalIntent.DOCTOR_SEARCH, 0.90

        # 12. Facility Search
        if any(re.search(p, q_lower) for p in self.facility_patterns) or has_facility:
            return HospitalIntent.FACILITY_INFORMATION, 0.90

        # 13. Services / Packages
        if any(re.search(p, q_lower) for p in self.services_patterns):
            return HospitalIntent.SERVICES, 0.90

        # 14. Department Information
        if any(re.search(p, q_lower) for p in self.department_patterns):
            return HospitalIntent.DEPARTMENT_INFORMATION, 0.90
        if has_department and any(w in q_lower for w in ["about", "overview", "info", "information", "tell me", "what is"]):
            return HospitalIntent.DEPARTMENT_INFORMATION, 0.90

        # 15. General greetings / questions
        if any(re.search(p, q_lower) for p in self.general_question_patterns):
            return HospitalIntent.GENERAL_QUESTION, 0.90

        # 16. General hospital information
        return HospitalIntent.GENERAL_HOSPITAL_INFORMATION, 0.70
