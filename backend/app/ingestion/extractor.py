import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag

from app.core.logging_config import logger
from app.models.schemas import DoctorItem, FacilityItem


class HospitalMetadataExtractor:
    """Deterministic DOM-based extractor for hospital doctors, qualifications,
    consultation schedules, profile images, and clinical facilities/services.
    
    Operates without LLM calls to ensure deterministic, fast, and 100% faithful
    extraction from source HTML.
    """

    # Common non-content image tokens to discard (social icons, trackers, spacers)
    IMAGE_EXCLUDE_PATTERNS = [
        "data:image",
        "facebook",
        "twitter",
        "instagram",
        "youtube",
        "linkedin",
        "spacer",
        "1x1",
        "pixel",
        "tracker",
        "blank.gif",
    ]

    # Medical qualification recognition keywords
    QUALIFICATION_KEYWORDS = {
        "mbbs", "md", "ms", "dm", "mch", "frcs", "mrcp", "bds", "mds",
        "dnb", "dgo", "dch", "da", "dlo", "ddvl", "dpm", "d-ortho",
        "fellowship", "diploma", "ficc", "fesc", "fage", "fam",
    }

    # Department slug to human-friendly name mapping for psmissionhospital.org
    DEPARTMENT_SLUG_MAP = {
        "cardiology": "Cardiology",
        "emergency-medicine": "Emergency Medicine",
        "paediatrics-neonatology": "Paediatrics & Neonatology",
        "obstetrics-gynaecology": "Obstetrics & Gynaecology",
        "general-medicine": "General Medicine",
        "general-surgery": "General Surgery",
        "orthopaedics-trauma-care": "Orthopaedics & Trauma Care",
        "nephrology-toxicology-dialysis-centre": "Nephrology, Toxicology & Dialysis",
        "neurology": "Neurology",
        "dermatology": "Dermatology",
        "endocrinology": "Endocrinology",
        "gastroenterology": "Gastroenterology",
        "ent": "ENT (Ear, Nose & Throat)",
        "ophthalmology": "Ophthalmology",
        "urology": "Urology",
        "pulmonology": "Pulmonology",
        "psychiatry-and-behavioral-sciences": "Psychiatry & Behavioral Sciences",
        "radiology": "Radiology",
        "anesthesiology-critical-care": "Anesthesiology & Critical Care",
        "maxillofacial-and-dental-services": "Maxillofacial & Dental Services",
        "physical-medicine-and-rehabitation": "Physical Medicine & Rehabilitation",
        "geriatric-medicine": "Geriatric Medicine",
        "laboratory-medicine": "Laboratory Medicine",
        "alternative-medicine-holistic-health": "Alternative Medicine & Holistic Health",
        "speech-therapy": "Speech Therapy",
        "orthodontics": "Orthodontics",
    }

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Normalizes horizontal whitespace and line breaks."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    @classmethod
    def extract_image_url(cls, tag: Optional[Tag], base_url: str) -> str:
        """Extracts the best image URL from an img element, checking lazy-loading
        attributes (data-src, data-lazy-src, data-original, src, srcset) and converting
        relative paths to absolute URLs.
        """
        if not tag or not isinstance(tag, Tag):
            return ""

        raw_url = ""
        # Check attributes in priority order for lazy-loaders
        for attr in ("data-src", "data-lazy-src", "data-original", "data-lazy", "src"):
            val = tag.get(attr)
            if val and isinstance(val, str) and val.strip():
                raw_url = val.strip()
                break

        # Fallback to srcset if single src is not present or is a tiny placeholder
        if (not raw_url or raw_url.startswith("data:image")) and tag.get("srcset"):
            srcset_parts = [p.strip().split()[0] for p in tag["srcset"].split(",") if p.strip()]
            if srcset_parts:
                raw_url = srcset_parts[0]

        if not raw_url:
            return ""

        # Filter out inline data URIs and obvious trackers
        url_lower = raw_url.lower()
        if any(pat in url_lower for pat in cls.IMAGE_EXCLUDE_PATTERNS):
            return ""

        # Resolve relative URL to absolute URL
        absolute_url = urljoin(base_url, raw_url)

        # Standardize HTTPS for psmissionhospital.org assets
        if "psmissionhospital.org" in absolute_url and absolute_url.startswith("http://"):
            absolute_url = "https://" + absolute_url[7:]

        return absolute_url

    @classmethod
    def extract_all_images(cls, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Collects unique content image URLs from the document."""
        images: List[str] = []
        seen: Set[str] = set()

        for img in soup.find_all("img"):
            img_url = cls.extract_image_url(img, base_url)
            if img_url and img_url not in seen:
                # Exclude header logos
                if "logo" not in img_url.lower():
                    seen.add(img_url)
                    images.append(img_url)

        return images

    @classmethod
    def infer_department(cls, soup: BeautifulSoup, url: str) -> str:
        """Infers the clinical department from URL structure, breadcrumbs, or headings."""
        parsed_url = urlparse(url)
        path = parsed_url.path.strip("/")

        # 1. Match from URL category path
        if "category/department/" in path:
            slug = path.split("category/department/")[-1].split("/")[0].lower()
            if slug in cls.DEPARTMENT_SLUG_MAP:
                return cls.DEPARTMENT_SLUG_MAP[slug]
            return slug.replace("-", " ").title()

        # 2. Check breadcrumbs (e.g. Home / Department / Cardiology)
        for bc in soup.find_all(["h3", "div", "nav", "ul"], class_=lambda c: bool(c and any(x in str(c).lower() for x in ["breadcrumb", "main", "header"]))):
            bc_text = bc.get_text()
            if "department" in bc_text.lower():
                parts = [p.strip() for p in re.split(r"[/»\>\|]", bc_text) if p.strip()]
                if len(parts) >= 2:
                    last_part = parts[-1]
                    if len(last_part) < 60 and "home" not in last_part.lower():
                        return cls.clean_text(last_part)

        # 3. Check primary heading if it matches department keywords
        for h in soup.find_all(["h1", "h2"]):
            h_text = cls.clean_text(h.get_text())
            for slug, dept_name in cls.DEPARTMENT_SLUG_MAP.items():
                if slug.replace("-", " ") in h_text.lower() or dept_name.lower() in h_text.lower():
                    return dept_name

        return ""

    @classmethod
    def extract_doctors(
        cls,
        soup: BeautifulSoup,
        base_url: str,
        default_department: Optional[str] = None,
    ) -> List[DoctorItem]:
        """Extracts structured doctor records from known hospital layout cards."""
        doctors: List[DoctorItem] = []
        seen_names: Set[str] = set()

        inferred_dept = default_department or cls.infer_department(soup, base_url)

        # ------------------------------------------------------------------
        # Pattern 1: Department Consultant Cards (div.dstyle)
        # Found on /category/department/* pages
        # ------------------------------------------------------------------
        dstyle_cards = soup.find_all("div", class_=lambda c: bool(c and "dstyle" in str(c).lower()))
        for card in dstyle_cards:
            img = card.find("img")
            image_url = cls.extract_image_url(img, base_url)

            name = ""
            qualification = ""

            # Check text-center paragraph containing name and qualifications
            p_center = card.find("p", class_="text-center")
            if p_center:
                strings = [cls.clean_text(s) for s in p_center.stripped_strings if cls.clean_text(s)]
                if strings:
                    name = strings[0]
                    if len(strings) > 1:
                        qualification = ", ".join(strings[1:])
            else:
                span_name = card.find("span", class_=lambda c: bool(c and "subheading" in str(c).lower()))
                if span_name:
                    name = cls.clean_text(span_name.get_text())

            # Schedule text inside doctor_box_role or paragraphs
            schedule_text = ""
            role_p = card.find("p", class_=lambda c: bool(c and "doctor_box_role" in str(c).lower()))
            if role_p:
                sched_parts = [cls.clean_text(s) for s in role_p.stripped_strings if cls.clean_text(s)]
                schedule_text = " | ".join(sched_parts)

            name_norm = cls._normalize_doctor_name(name)
            if name_norm and name_norm.lower() not in seen_names:
                seen_names.add(name_norm.lower())
                doctors.append(
                    DoctorItem(
                        content_type="doctor",
                        doctor_name=name_norm,
                        qualification=qualification,
                        department=inferred_dept,
                        schedule_text=schedule_text,
                        image_url=image_url,
                        source_url=base_url,
                    )
                )

        # ------------------------------------------------------------------
        # Pattern 2: Doctors Directory Cards (div.doctor_box)
        # Found on /doctors/ directory page
        # ------------------------------------------------------------------
        doctor_boxes = soup.find_all(
            "div",
            class_=lambda c: bool(
                c
                and "doctor_box" in str(c).lower()
                and not any(x in str(c).lower() for x in ["doctor_box_img", "doctor_box_title", "doctor_box_role", "doctor_box_icon"])
            ),
        )
        for box in doctor_boxes:
            img = box.find("img")
            image_url = cls.extract_image_url(img, base_url)

            title_elem = box.find("h3", class_="doctor_box_title") or box.find(["h3", "h4", "h2"])
            name = cls.clean_text(title_elem.get_text()) if title_elem else ""

            qualification = ""
            department = inferred_dept
            schedule_text = ""

            role_elems = box.find_all("p", class_="doctor_box_role")
            role_texts = [cls.clean_text(r.get_text()) for r in role_elems if cls.clean_text(r.get_text())]

            for rt in role_texts:
                rt_lower = rt.lower()
                # Check if it looks like schedule / timings
                if any(k in rt_lower for k in ["o.p", "op:", "am", "pm", "monday", "sunday", "daily", "hrs"]):
                    schedule_text = rt
                # Check if it contains medical degrees
                elif any(q in rt_lower for q in cls.QUALIFICATION_KEYWORDS):
                    qualification = rt
                # Otherwise treat as department
                elif not department or department == inferred_dept:
                    department = rt

            name_norm = cls._normalize_doctor_name(name)
            if name_norm and name_norm.lower() not in seen_names:
                seen_names.add(name_norm.lower())
                doctors.append(
                    DoctorItem(
                        content_type="doctor",
                        doctor_name=name_norm,
                        qualification=qualification,
                        department=department or inferred_dept,
                        schedule_text=schedule_text,
                        image_url=image_url,
                        source_url=base_url,
                    )
                )

        # ------------------------------------------------------------------
        # Pattern 3: Generic Doctor Detection (headings or cards with Dr. / Doctor)
        # ------------------------------------------------------------------
        if not doctors:
            doctor_candidates = soup.find_all(
                lambda tag: tag.name in ["h2", "h3", "h4", "h5", "p", "div"]
                and re.search(r"^(dr\.?|doctor|prof\.?)\s+[a-z]", tag.get_text(strip=True), re.IGNORECASE)
            )
            for cand in doctor_candidates:
                full_text = cls.clean_text(cand.get_text())
                # Extract name portion
                match = re.search(r"^(dr\.?\s+[A-Za-z\.\s]+?)(?:,|\n|-|$)", full_text, re.IGNORECASE)
                if match:
                    raw_cand_name = match.group(1).strip()
                    name_norm = cls._normalize_doctor_name(raw_cand_name)
                    if name_norm and name_norm.lower() not in seen_names and len(name_norm) > 4:
                        # Find closest image in parent
                        parent_box = cand.find_parent(["div", "article", "section", "li"])
                        img_url = ""
                        qual = ""
                        sched = ""
                        if parent_box:
                            p_img = parent_box.find("img")
                            img_url = cls.extract_image_url(p_img, base_url)
                            # Qualification search in siblings
                            box_text = parent_box.get_text()
                            qual_matches = [q for q in cls.QUALIFICATION_KEYWORDS if f" {q}" in box_text.lower() or f",{q}" in box_text.lower()]
                            if qual_matches:
                                qual = ", ".join(m.upper() for m in qual_matches)

                        seen_names.add(name_norm.lower())
                        doctors.append(
                            DoctorItem(
                                content_type="doctor",
                                doctor_name=name_norm,
                                qualification=qual,
                                department=inferred_dept,
                                schedule_text=sched,
                                image_url=img_url,
                                source_url=base_url,
                            )
                        )

        return doctors

    @classmethod
    def extract_facilities(
        cls,
        soup: BeautifulSoup,
        base_url: str,
        default_department: Optional[str] = None,
    ) -> List[FacilityItem]:
        """Extracts department facilities, diagnostic services, and medical offerings."""
        facilities: List[FacilityItem] = []
        seen_names: Set[str] = set()

        inferred_dept = default_department or cls.infer_department(soup, base_url)

        # Detect containers with "Services" or "Facilities" in heading or class
        candidate_containers = soup.find_all(
            "div",
            class_=lambda c: bool(
                c and any(k in str(c).lower() for k in ["green-bg", "pad-large", "inset", "services", "facilities", "features"])
            ),
        )

        for container in candidate_containers:
            heading = container.find(["h1", "h2", "h3", "h4", "h5"])
            if not heading:
                continue

            h_text = cls.clean_text(heading.get_text()).lower()
            if not any(k in h_text for k in ["facilities", "services", "specialities", "diagnostic", "infrastructure"]):
                continue

            # Extract list items
            items = container.find_all("li")
            for li in items:
                fac_name = cls.clean_text(li.get_text())
                if fac_name and len(fac_name) > 2 and fac_name.lower() not in seen_names:
                    seen_names.add(fac_name.lower())
                    facilities.append(
                        FacilityItem(
                            content_type="facility",
                            facility_name=fac_name,
                            description=f"Clinical service/facility offered under {inferred_dept or 'the hospital'}",
                            department=inferred_dept,
                            source_url=base_url,
                        )
                    )

        # Fallback: scan any <h2> or <h3> matching Facilities/Services followed by <ul>
        if not facilities:
            for heading in soup.find_all(["h2", "h3"]):
                h_text = cls.clean_text(heading.get_text()).lower()
                if any(k in h_text for k in ["facilities", "services", "specialities"]):
                    sibling = heading.find_next_sibling(["ul", "ol", "div"])
                    if sibling:
                        for li in sibling.find_all("li"):
                            fac_name = cls.clean_text(li.get_text())
                            if fac_name and len(fac_name) > 2 and fac_name.lower() not in seen_names:
                                seen_names.add(fac_name.lower())
                                facilities.append(
                                    FacilityItem(
                                        content_type="facility",
                                        facility_name=fac_name,
                                        description=f"Clinical offering under {inferred_dept or 'hospital services'}",
                                        department=inferred_dept,
                                        source_url=base_url,
                                    )
                                )

        return facilities

    @classmethod
    def extract_all(cls, html: str, url: str) -> Dict[str, Any]:
        """Executes full entity extraction on raw HTML, returning structured
        doctors, facilities, images, and department metadata.
        """
        if not html or not isinstance(html, str):
            return {
                "title": "",
                "department": "",
                "doctors": [],
                "facilities": [],
                "images": [],
            }

        soup = BeautifulSoup(html, "html.parser")

        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = cls.clean_text(title_tag.string)

        department = cls.infer_department(soup, url)
        doctors = cls.extract_doctors(soup, url, default_department=department)
        facilities = cls.extract_facilities(soup, url, default_department=department)
        images = cls.extract_all_images(soup, url)

        logger.info(
            "Extracted metadata for %s: %d doctors, %d facilities, %d images (dept='%s')",
            url,
            len(doctors),
            len(facilities),
            len(images),
            department,
        )

        return {
            "title": title,
            "department": department,
            "doctors": doctors,
            "facilities": facilities,
            "images": images,
        }

    @staticmethod
    def _normalize_doctor_name(name: str) -> str:
        """Cleans and standardizes doctor title and spacing."""
        cleaned = re.sub(r"\s+", " ", name or "").strip()
        # Normalize Dr . to Dr.
        cleaned = re.sub(r"^dr\s*\.\s*", "Dr. ", cleaned, flags=re.IGNORECASE)
        # Normalize Dr Sr to Dr. Sr.
        cleaned = re.sub(r"^Dr\.\s*Sr\s*\.?\s*", "Dr. Sr. ", cleaned, flags=re.IGNORECASE)
        return cleaned
