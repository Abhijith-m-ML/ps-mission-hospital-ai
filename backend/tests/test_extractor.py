from typing import Any, Dict
import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from app.ingestion.extractor import HospitalMetadataExtractor
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------

HTML_DEPARTMENT_DOCTOR = """
<html>
<head><title>Paediatrist in Kochi | Neonatologist - PS Mission Hospital</title></head>
<body>
<div class="smd2 col gutter">
    <div class="green-bg pad-large inset">
        <h2 class="subheading">Facilities</h2>
        <ul class="list">
            <li>NICU Level III Care</li>
            <li>Phototherapy Unit</li>
            <li>Paediatric Emergency</li>
        </ul>
    </div>
    <div class="dstyle">
        <div>
            <img src="/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg" alt="Dr. Sr. Jaya Joseph" />
        </div>
        <div class="inset">
            <p class="text-center">
                <span class="subheading-small bold text-green">Dr. Sr. Jaya  Joseph</span>
                <br/>
                MBBS,  MD
            </p>
            <p class="doctor_box_role">
                <p>
                    <strong>OP: MONDAY-FRIDAY</strong><br/>
                    9.30AM-1.00PM &amp; 4.00PM-6.00 PM<br/>
                    <strong>SATURDAY</strong> (9.30AM-1.00)<br/>
                    <strong>SUNDAY</strong> (4PM-5.30PM)
                </p>
            </p>
        </div>
    </div>
</div>
</body>
</html>
"""

HTML_DIRECTORY_DOCTORS = """
<html>
<head><title>Doctors Directory</title></head>
<body>
<div class="doctor_box">
    <div class="doctor_box_img">
        <img data-src="https://www.psmissionhospital.org/wp-content/uploads/2019/12/Dr-Siju-A-Joseph-400x400.png" alt="Dr. Siju A Joseph" />
    </div>
    <div class="doctor_box_content">
        <h3 class="doctor_box_title">Dr. Siju  A Joseph</h3>
        <p class="doctor_box_role">MBBS , MD</p>
        <p class="doctor_box_role">Anesthesiology &amp; Critical Care</p>
    </div>
</div>
<div class="doctor_box">
    <div class="doctor_box_img">
        <img src="/uploads/dr-annie.jpg" alt="Dr. Annie" />
    </div>
    <div class="doctor_box_content">
        <h3 class="doctor_box_title">Dr. Sr. Annie Sheela</h3>
        <p class="doctor_box_role">MD, DM, FICC, FESC</p>
        <p class="doctor_box_role">Cardiology</p>
    </div>
</div>
</body>
</html>
"""

HTML_NO_DOCTORS_OR_IMAGES = """
<html>
<head><title>About P.S. Mission Hospital</title></head>
<body>
<div class="main-content">
    <h1>About Our Hospital</h1>
    <p>P.S. Mission Hospital has been serving the community with dedicated healthcare services since 1961.</p>
    <p>Our mission is to heal in love and provide compassionate clinical care to every patient.</p>
</div>
</body>
</html>
"""

HTML_LAZY_IMAGES = """
<html>
<body>
    <img data-src="/images/doctor-lazy1.jpg" alt="Lazy 1" />
    <img data-lazy-src="/images/doctor-lazy2.jpg" alt="Lazy 2" />
    <img data-original="https://cdn.hospital.com/doctor-lazy3.jpg" alt="Lazy 3" />
    <img src="data:image/svg+xml;base64,PHN2Zz..." data-src="/images/doctor-lazy4.jpg" alt="Lazy 4" />
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_image_url_extraction():
    """Verify extracting valid image URLs from standard img src attribute."""
    soup = BeautifulSoup('<img src="https://example.com/assets/dr_smith.jpg" />', "html.parser")
    img_tag = soup.find("img")
    extracted_url = HospitalMetadataExtractor.extract_image_url(img_tag, "https://example.com/dept")
    assert extracted_url == "https://example.com/assets/dr_smith.jpg"


def test_relative_image_url_conversion():
    """Verify converting relative image paths to fully qualified absolute URLs."""
    soup = BeautifulSoup('<img src="/wp-content/uploads/2023/08/dr_jaya.jpg" />', "html.parser")
    img_tag = soup.find("img")
    base = "https://www.psmissionhospital.org/category/department/paediatrics"
    extracted_url = HospitalMetadataExtractor.extract_image_url(img_tag, base)
    assert extracted_url == "https://www.psmissionhospital.org/wp-content/uploads/2023/08/dr_jaya.jpg"


def test_lazy_loaded_image_extraction():
    """Verify that data-src, data-lazy-src, and data-original are detected for lazy loaders."""
    soup = BeautifulSoup(HTML_LAZY_IMAGES, "html.parser")
    images = HospitalMetadataExtractor.extract_all_images(soup, "https://www.psmissionhospital.org/")
    
    assert len(images) == 4
    assert "https://www.psmissionhospital.org/images/doctor-lazy1.jpg" in images
    assert "https://www.psmissionhospital.org/images/doctor-lazy2.jpg" in images
    assert "https://cdn.hospital.com/doctor-lazy3.jpg" in images
    assert "https://www.psmissionhospital.org/images/doctor-lazy4.jpg" in images


def test_doctor_card_extraction_dstyle():
    """Verify extracting doctor card from department consultant cards (div.dstyle)."""
    soup = BeautifulSoup(HTML_DEPARTMENT_DOCTOR, "html.parser")
    url = "https://www.psmissionhospital.org/category/department/paediatrics-neonatology"
    doctors = HospitalMetadataExtractor.extract_doctors(soup, url)

    assert len(doctors) == 1
    doc = doctors[0]
    assert doc.content_type == "doctor"
    assert doc.doctor_name == "Dr. Sr. Jaya Joseph"
    assert "MBBS" in doc.qualification and "MD" in doc.qualification
    assert doc.department == "Paediatrics & Neonatology"
    assert "OP: MONDAY-FRIDAY" in doc.schedule_text
    assert doc.image_url == "https://www.psmissionhospital.org/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg"
    assert doc.source_url == url


def test_doctor_card_extraction_doctor_box():
    """Verify extracting doctor cards from listing directory (div.doctor_box)."""
    soup = BeautifulSoup(HTML_DIRECTORY_DOCTORS, "html.parser")
    url = "https://www.psmissionhospital.org/doctors"
    doctors = HospitalMetadataExtractor.extract_doctors(soup, url)

    assert len(doctors) == 2
    
    doc1 = doctors[0]
    assert doc1.content_type == "doctor"
    assert doc1.doctor_name == "Dr. Siju A Joseph"
    assert doc1.qualification == "MBBS , MD"
    assert "Anesthesiology" in doc1.department
    assert doc1.image_url == "https://www.psmissionhospital.org/wp-content/uploads/2019/12/Dr-Siju-A-Joseph-400x400.png"

    doc2 = doctors[1]
    assert doc2.doctor_name == "Dr. Sr. Annie Sheela"
    assert doc2.qualification == "MD, DM, FICC, FESC"
    assert doc2.department == "Cardiology"
    assert doc2.image_url == "https://www.psmissionhospital.org/uploads/dr-annie.jpg"


def test_facility_extraction():
    """Verify extracting department facilities and clinical services."""
    soup = BeautifulSoup(HTML_DEPARTMENT_DOCTOR, "html.parser")
    url = "https://www.psmissionhospital.org/category/department/paediatrics-neonatology"
    facilities = HospitalMetadataExtractor.extract_facilities(soup, url)

    assert len(facilities) == 3
    names = [f.facility_name for f in facilities]
    assert "NICU Level III Care" in names
    assert "Phototherapy Unit" in names
    assert "Paediatric Emergency" in names
    assert all(f.content_type == "facility" for f in facilities)
    assert all(f.department == "Paediatrics & Neonatology" for f in facilities)


def test_page_without_images():
    """Verify graceful handling when a page contains no images."""
    soup = BeautifulSoup(HTML_NO_DOCTORS_OR_IMAGES, "html.parser")
    images = HospitalMetadataExtractor.extract_all_images(soup, "https://www.psmissionhospital.org/about")
    assert images == []


def test_page_without_doctor_information():
    """Verify that pages without doctor cards return an empty list without false positives."""
    soup = BeautifulSoup(HTML_NO_DOCTORS_OR_IMAGES, "html.parser")
    doctors = HospitalMetadataExtractor.extract_doctors(soup, "https://www.psmissionhospital.org/about")
    assert doctors == []


def test_api_extract_preview_endpoint_with_html():
    """Verify POST /api/extract/preview returns structured metadata when HTML is provided."""
    response = client.post(
        "/api/extract/preview",
        json={
            "url": "https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
            "html": HTML_DEPARTMENT_DOCTOR,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_doctors"] == 1
    assert data["total_facilities"] == 3
    assert data["doctors"][0]["doctor_name"] == "Dr. Sr. Jaya Joseph"
    assert data["doctors"][0]["content_type"] == "doctor"
    assert data["facilities"][0]["facility_name"] == "NICU Level III Care"


def test_api_extract_preview_validation_error():
    """Verify POST /api/extract/preview rejects empty URL."""
    response = client.post(
        "/api/extract/preview",
        json={"url": "  "},
    )
    assert response.status_code == 400
    assert "valid URL string is required" in response.json()["detail"]
