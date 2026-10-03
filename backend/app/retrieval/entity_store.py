import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_config import logger
from app.models.schemas import Department, Doctor, Facility


class HospitalEntityStore:
    """In-memory and file-backed entity catalog for Hospital Departments, Doctors, and Facilities.
    
    Provides fast, deterministic lookups by department, doctor name, and facility name,
    guaranteeing that structured hospital data is immediately available to the retrieval
    layer without depending on arbitrary unstructured embedding chunks.
    """

    DEPARTMENT_SYNONYMS: Dict[str, List[str]] = {
        "Paediatrics & Neonatology": [
            "paediatrics & neonatology", "paediatrics", "pediatrics", "neonatology",
            "paediatric", "pediatric", "neonatal", "nicu", "child", "children",
            "baby", "infant", "kids", "newborn", "paediatrician", "pediatrician",
            "child fever", "baby cough", "immunization", "vaccination",
            "പീഡിയാട്രിക്സ്", "കുട്ടികൾ", "ശിശുരോഗ", "കുട്ടികളുടെ", "നവജാതശിശു", "ശിശുരോഗ വിദഗ്ദ്ധൻ",
            "पीडियाट्रिक्स", "बाल रोग", "बच्चे", "बच्चों", "शिशु रोग"
        ],
        "Cardiology": [
            "cardiology", "cardio", "cardiologist", "heart", "cardiac", "ecg", "echo",
            "chest pain", "palpitation", "palpitations", "hypertension", "heart attack", "cardiac arrest",
            "കാർഡിയോളജി", "ഹൃദയം", "കാർഡിയോളജിസ്റ്റ്", "ഹൃദ്രോഗം", "നെഞ്ചുവേദന",
            "कार्डियोलॉजी", "हृदय", "कार्डियोलॉजिस्ट", "दिल", "हृदय रोग", "सीने में दर्द"
        ],
        "Orthopaedics & Trauma Care": [
            "orthopaedics & trauma care", "orthopaedics", "orthopedics", "ortho",
            "orthopaedic", "orthopedic", "bone", "joint", "fracture", "trauma", "spine",
            "bone pain", "joint pain", "knee pain", "back pain", "sprain", "fractures", "arthritis",
            "ഓർത്തോപീഡിക്സ്", "അസ്ഥി", "ഓർത്തോ", "ഓർത്തോപീഡിക്", "എല്ല്", "സന്ധിവേദന",
            "ऑर्थोपेडिक्स", "हड्डी", "जोड़ों का दर्द", "हड्डी रोग"
        ],
        "Obstetrics & Gynaecology": [
            "obstetrics & gynaecology", "obstetrics", "gynaecology", "gynecology",
            "obgyn", "ob-gyn", "maternity", "pregnancy", "delivery", "pregnant",
            "period", "periods", "menstrual", "bleeding",
            "ഗൈനക്കോളജി", "പ്രസവം", "സ്ത്രീരോഗം", "ഗർഭിണി",
            "गाइनेकोलॉजी", "स्त्री रोग", "प्रसूति", "गर्भावस्था"
        ],
        "General Medicine": [
            "general medicine", "physician", "general physician", "internal medicine",
            "fever", "cold", "cough", "body ache", "weakness", "stomach pain", "acidity",
            "indigestion", "vomiting", "diarrhea",
            "ജനറൽ മെഡിസിൻ", "ഫിസിഷ്യൻ", "പനി", "ചുമ",
            "जनरल मेडिसिन", "फिजिशियन", "बुखार", "खांसी"
        ],
        "General Surgery": [
            "general surgery", "surgery", "surgeon", "surgical", "laparoscopy", "hernia", "appendix",
            "ജനറൽ സർജറി", "സർജൻ", "സർജറി", "ശസ്ത്രക്രിയ",
            "जनरल सर्जरी", "सर्जन", "सर्जरी"
        ],
        "Neurology": [
            "neurology", "neuro", "neurologist", "brain", "stroke", "nervous system",
            "headache", "headaches", "migraine", "migraines", "dizziness", "seizure",
            "seizures", "nerve", "nerves", "neuropathy", "vertigo", "paralysis",
            "ന്യൂറോളജി", "തലച്ചോറ്", "ന്യൂറോളജിസ്റ്റ്", "തലവേദന", "മൈഗ്രെയ്ൻ",
            "न्यूरोलॉजी", "मस्तिष्क", "न्यूरोलॉजिस्ट", "सिरदर्द", "माइग्रेन", "दिमाग"
        ],
        "Nephrology, Toxicology & Dialysis": [
            "nephrology, toxicology & dialysis", "nephrology", "dialysis", "kidney",
            "renal", "toxicology", "kidney failure", "dialysis unit",
            "നെഫ്രോളജി", "ഡയാലിസിസ്", "വൃക്ക", "കിഡ്നി",
            "नेफ्रोलॉजी", "डायलिसिस", "गुर्दा", "किडनी"
        ],
        "Dermatology": [
            "dermatology", "derma", "dermatologist", "skin", "cosmetology", "hair",
            "skin rash", "rash", "rashes", "itching", "acne", "eczema", "psoriasis", "dandruff",
            "ഡെർമറ്റോളജി", "ചർമ്മം", "ഡെർമ", "ചർമ്മരോഗം",
            "डर्मेटोलॉजी", "त्वचा रोग", "चर्म रोग"
        ],
        "ENT (Ear, Nose & Throat)": [
            "ent (ear, nose & throat)", "ent", "ear", "nose", "throat", "audiology",
            "ear pain", "sinus", "sinusitis", "tonsil", "tonsils", "hearing loss", "sore throat",
            "ഇ.എൻ.ടി", "ഇഎൻടി", "ചെവി", "മൂക്ക്", "തൊണ്ട",
            "ईएनटी", "कान नाक गला", "कान", "नाक", "गला"
        ],
        "Ophthalmology": [
            "ophthalmology", "ophthalmologist", "eye", "eyes", "vision", "cataract",
            "eye pain", "blurred vision", "red eye",
            "ഒഫ്താൽമോളജി", "നേത്രരോഗം", "കണ്ണ്", "കാഴ്ച",
            "नेत्र रोग", "आंख", "आँख"
        ],
        "Urology": [
            "urology", "urologist", "urine", "urinary", "bladder", "prostate",
            "urine infection", "kidney stone", "kidney stones", "burning urination",
            "യൂറോളജി", "മൂത്രാശയം", "യൂറോളജിസ്റ്റ്",
            "यूरोलॉजी", "मूत्र रोग"
        ],
        "Pulmonology": [
            "pulmonology", "pulmonologist", "lungs", "chest", "respiratory", "asthma",
            "breathlessness", "breathing problem",
            "പൾമണോളജി", "ശ്വാസകോശം", "ആസ്ത്മ",
            "पल्मोनोलॉजी", "फेफड़े", "दमा"
        ],
        "Anesthesiology & Critical Care": [
            "anesthesiology & critical care", "anesthesiology", "anaesthesiology",
            "critical care", "icu", "anesthesia",
            "അനസ്തെഷ്യോളജി", "ഐസിയു",
            "एनेस्थिसियोलॉजी"
        ],
        "Maxillofacial & Dental Services": [
            "maxillofacial & dental services", "dental", "dentist", "dentistry",
            "teeth", "tooth", "toothache", "gum", "maxillofacial",
            "ഡെന്റൽ", "പല്ല്", "ദന്തൽ",
            "डेंटल", "दांत"
        ],
        "Physical Medicine & Rehabilitation": [
            "physical medicine & rehabilitation", "physiotherapy", "rehabilitation",
            "rehab", "physical medicine",
            "ഫിസിയോതെറാപ്പി", "റീഹാബിലിറ്റേഷൻ",
            "फिजियोथेरेपी"
        ],
        "Radiology": [
            "radiology", "radiologist", "x-ray", "ct scan", "mri", "ultrasound", "scan",
            "റേഡിയോളജി", "സ്കാൻ", "എക്സ്-റേ",
            "रेडियोलॉजी"
        ],
        "Psychiatry & Behavioral Sciences": [
            "psychiatry & behavioral sciences", "psychiatry", "psychiatrist",
            "psychology", "mental health", "behavioral", "counseling", "depression", "anxiety",
            "സൈക്യാട്രി", "മാനസികാരോഗ്യം",
            "मनोचिकित्सा"
        ],
        "Emergency Medicine": [
            "emergency medicine", "emergency", "casualty", "er", "ambulance",
            "എമർജൻസി", "അത്യാഹിത വിഭാഗം", "ആംബുലൻസ്",
            "आपातकालीन", "इमरजेंसी", "एम्बुलेंस"
        ],
        "Geriatric Medicine": [
            "geriatric medicine", "geriatric", "elderly", "senior citizen",
            "ജെറിയാട്രിക്", "മുതിർന്നവർ",
            "जेरियाट्रिक"
        ],
        "Laboratory Medicine": [
            "laboratory medicine", "laboratory", "lab", "pathology", "blood test"
        ],
        "Alternative Medicine & Holistic Health": [
            "alternative medicine & holistic health", "ayurveda", "holistic",
            "alternative medicine", "homeopathy"
        ],
        "Speech Therapy": [
            "speech therapy", "speech"
        ],
        "Orthodontics": [
            "orthodontics", "braces"
        ],
    }

    def __init__(self, data_path: Optional[str] = None):
        self.departments: Dict[str, Department] = {}
        self.doctors: List[Doctor] = []
        self.facilities: List[Facility] = []

        # Indexing caches
        self._doctors_by_dept: Dict[str, List[Doctor]] = {}
        self._facilities_by_dept: Dict[str, List[Facility]] = {}
        self._doctors_by_name: Dict[str, Doctor] = {}

        # Default path
        default_file = Path(__file__).parent.parent / "data" / "hospital_entities.json"
        self.data_path = Path(data_path) if data_path else default_file

        self._load()

    def _load(self):
        """Loads entities from JSON file if present."""
        if not self.data_path.exists():
            logger.info("Entity store file %s does not exist yet.", self.data_path)
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for d_data in data.get("departments", []):
                dept = Department(**d_data)
                self.add_department(dept)

            for doc_data in data.get("doctors", []):
                doc = Doctor(**doc_data)
                self.add_doctor(doc)

            for fac_data in data.get("facilities", []):
                fac = Facility(**fac_data)
                self.add_facility(fac)

            logger.info(
                "HospitalEntityStore loaded %d departments, %d doctors, %d facilities from %s",
                len(self.departments),
                len(self.doctors),
                len(self.facilities),
                self.data_path,
            )
        except Exception as exc:
            logger.exception("Failed to load entity store from %s: %s", self.data_path, exc)

    def save(self, target_path: Optional[str] = None):
        """Saves current state to JSON."""
        dest = Path(target_path) if target_path else self.data_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = {
                "departments": [d.model_dump() for d in self.departments.values()],
                "doctors": [d.model_dump() for d in self.doctors],
                "facilities": [f.model_dump() for f in self.facilities],
            }
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("HospitalEntityStore saved to %s", dest)
        except Exception as exc:
            logger.exception("Failed to save entity store to %s: %s", dest, exc)

    def add_department(self, department: Department):
        """Registers a Department."""
        norm_key = self.normalize_department_name(department.department_name)
        self.departments[norm_key] = department

    def add_doctor(self, doctor: Doctor):
        """Registers a Doctor."""
        norm_name = self._normalize_name(doctor.doctor_name)
        # Avoid duplicate doctors by updating if existing
        if norm_name in self._doctors_by_name:
            existing = self._doctors_by_name[norm_name]
            if not existing.schedule_text and doctor.schedule_text:
                existing.schedule_text = doctor.schedule_text
            if not existing.image_url and doctor.image_url:
                existing.image_url = doctor.image_url
            if doctor.department and (not existing.department or existing.department == "General"):
                existing.department = doctor.department
            return

        self.doctors.append(doctor)
        self._doctors_by_name[norm_name] = doctor

        dept_norm = self.normalize_department_name(doctor.department)
        if dept_norm not in self._doctors_by_dept:
            self._doctors_by_dept[dept_norm] = []
        self._doctors_by_dept[dept_norm].append(doctor)

    def add_facility(self, facility: Facility):
        """Registers a Facility."""
        self.facilities.append(facility)
        dept_norm = self.normalize_department_name(facility.department)
        if dept_norm not in self._facilities_by_dept:
            self._facilities_by_dept[dept_norm] = []
        self._facilities_by_dept[dept_norm].append(facility)

    def get_doctors_by_department(self, department: str) -> List[Doctor]:
        """Retrieves verified doctors belonging to a clinical department."""
        dept_norm = self.normalize_department_name(department)
        return self._doctors_by_dept.get(dept_norm, [])

    def get_doctor_by_name(self, name: str) -> Optional[Doctor]:
        """Finds doctor by name with fuzzy/token matching and spelling tolerance."""
        norm_query = self._normalize_name(name)
        if norm_query in self._doctors_by_name:
            return self._doctors_by_name[norm_query]

        # 1. Direct substring match with known doctors (with spelling tolerance)
        text_lower = name.lower()
        alt_text_lower = re.sub(r"\bjohn\b", "jhon", text_lower) if "john" in text_lower else re.sub(r"\bjhon\b", "john", text_lower)

        for doc in self.doctors:
            doc_lower = doc.doctor_name.lower()
            clean_main = re.sub(r"^(?:dr\.?|sr\.?)\s*", "", doc_lower).strip()
            if (
                doc_lower in text_lower
                or doc_lower in alt_text_lower
                or (len(clean_main) >= 4 and (clean_main in text_lower or clean_main in alt_text_lower))
            ):
                return doc

        # 2. Token overlap matching (excluding stopwords and short tokens)
        stopwords = {"dr", "doctor", "sr", "who", "is", "a", "an", "the", "in", "at", "for", "to", "of", "and"}
        q_tokens = {t for t in re.findall(r"\w+", norm_query) if len(t) > 1 and t not in stopwords}
        if "john" in q_tokens:
            q_tokens.add("jhon")
        elif "jhon" in q_tokens:
            q_tokens.add("john")

        if not q_tokens:
            return None

        best_match = None
        best_overlap = 0

        for doc in self.doctors:
            doc_norm = self._normalize_name(doc.doctor_name)
            d_tokens = {t for t in re.findall(r"\w+", doc_norm) if len(t) > 1 and t not in stopwords}
            if "john" in d_tokens:
                d_tokens.add("jhon")
            elif "jhon" in d_tokens:
                d_tokens.add("john")

            overlap = len(q_tokens.intersection(d_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = doc

        if best_overlap > 0:
            return best_match
        return None


    def get_facilities_by_department(self, department: str) -> List[Facility]:
        """Retrieves clinical facilities and services for a department."""
        dept_norm = self.normalize_department_name(department)
        return self._facilities_by_dept.get(dept_norm, [])

    def get_department(self, department: str) -> Optional[Department]:
        """Retrieves department entity by name or alias."""
        dept_norm = self.normalize_department_name(department)
        return self.departments.get(dept_norm)

    def normalize_department_name(self, text: Optional[str]) -> str:
        """Resolves various user query phrases or department aliases to standard department name."""
        if not text:
            return ""

        clean_text = text.lower().strip()
        clean_text = re.sub(r"[^\w\s&]", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # 1. Exact match with canonical keys
        for canonical, aliases in self.DEPARTMENT_SYNONYMS.items():
            if clean_text == canonical.lower():
                return canonical

        # 2. Check aliases (including singular/plural)
        tokens = set(clean_text.split())
        for canonical, aliases in self.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                # Word boundary match
                pattern = r"\b" + re.escape(alias) + r"(?:s|es)?\b"
                if re.search(pattern, clean_text):
                    return canonical


        # 3. Fallback title case
        return text.strip().title()

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Standardizes doctor name for index keys."""
        cleaned = re.sub(r"\s+", " ", (name or "").lower()).strip()
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        return cleaned
