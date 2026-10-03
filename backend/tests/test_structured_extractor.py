from fastapi.testclient import TestClient
import pytest
from bs4 import BeautifulSoup

from app.ingestion.structured_extractor import StructuredExtractor
from app.main import app
from app.models.schemas import Department, Doctor, Facility


SAMPLE_DEPARTMENT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Paediatrist in Kochi | Neonatologist in Kochi - PS Mission Hospital</title>
</head>
<body>
    <h2>Paediatrics & Neonatology</h2>
    <div class="entry-content">
        <p>P S Mission Hospital has a Neonatal I.C.U with 8 beds for high risk children like premature babies. We provide baby health care and vaccination.</p>
    </div>

    <div class="green-bg pad-large inset">
        <h2>Facilities</h2>
        <ul>
            <li>Immunization and Child Vaccines</li>
            <li>Neonatology and Emergency Care</li>
            <li>Phototherapy Unit</li>
        </ul>
    </div>

    <h2>Our Consultants</h2>
    <div class="dstyle">
        <img data-src="/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg" alt="Dr. Sr. Jaya Joseph" />
        <p class="text-center">
            <span class="subheading-small bold text-green">Dr. Sr. Jaya  Joseph</span><br/>
            MBBS, MD
        </p>
        <p class="doctor_box_role">
            OP: MONDAY-FRIDAY | 9.30AM-1.00PM & 4.00PM-6.00 PM | SATURDAY | (9.30AM-1.00)
        </p>
    </div>
</body>
</html>
"""

SAMPLE_DOCTORS_DIRECTORY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Our Doctors - PS Mission Hospital</title>
</head>
<body>
    <div class="doctor_box">
        <div class="doctor_box_img">
            <img src="https://www.psmissionhospital.org/wp-content/uploads/2021/03/i2.jpg" alt="Dr. Sr. Annie Sheela" />
        </div>
        <h3 class="doctor_box_title">Dr. Sr. Annie Sheela</h3>
        <p class="doctor_box_role">MD, DM, FICC, FESC</p>
        <p class="doctor_box_role">Cardiology</p>
    </div>
    <div class="doctor_box">
        <div class="doctor_box_img">
            <img data-lazy-src="https://www.psmissionhospital.org/wp-content/uploads/2023/08/psmissionhospital-dr-1.jpg" alt="Dr. Sudheer" />
        </div>
        <h3 class="doctor_box_title">Dr. Sudheer</h3>
        <p class="doctor_box_role">MBBS, MD, DM</p>
        <p class="doctor_box_role">Cardiology</p>
    </div>
</body>
</html>
"""


class TestStructuredExtractor:
    """Unit tests for deterministic HTML extraction layer."""

    def test_extract_department_entity(self):
        url = "https://www.psmissionhospital.org/category/department/paediatrics-neonatology"
        result = StructuredExtractor.extract(SAMPLE_DEPARTMENT_HTML, url)

        assert result.department is not None
        assert result.department.department_name == "Paediatrics & Neonatology"
        assert "Neonatal I.C.U" in result.department.description
        assert result.department.source_url == url
        assert result.department.type == "department"

    def test_extract_doctor_from_department_page(self):
        url = "https://www.psmissionhospital.org/category/department/paediatrics-neonatology"
        result = StructuredExtractor.extract(SAMPLE_DEPARTMENT_HTML, url)

        assert len(result.doctors) == 1
        doc = result.doctors[0]
        assert doc.type == "doctor"
        assert doc.doctor_name == "Dr. Sr. Jaya Joseph"
        assert "MBBS" in doc.qualification and "MD" in doc.qualification
        assert doc.department == "Paediatrics & Neonatology"
        assert "9.30AM-1.00PM" in doc.schedule_text
        assert doc.image_url == "https://www.psmissionhospital.org/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg"
        assert doc.source_url == url
        assert "Paediatrist in Kochi" in doc.source_title

    def test_extract_facilities_from_department_page(self):
        url = "https://www.psmissionhospital.org/category/department/paediatrics-neonatology"
        result = StructuredExtractor.extract(SAMPLE_DEPARTMENT_HTML, url)

        assert len(result.facilities) == 3
        fac_names = [f.facility_name for f in result.facilities]
        assert "Immunization and Child Vaccines" in fac_names
        assert "Neonatology and Emergency Care" in fac_names
        assert "Phototherapy Unit" in fac_names

        for fac in result.facilities:
            assert fac.type == "facility"
            assert fac.department == "Paediatrics & Neonatology"
            assert fac.source_url == url

    def test_extract_doctors_from_directory(self):
        url = "https://www.psmissionhospital.org/doctors/"
        result = StructuredExtractor.extract(SAMPLE_DOCTORS_DIRECTORY_HTML, url)

        assert len(result.doctors) == 2
        names = [d.doctor_name for d in result.doctors]
        assert "Dr. Sr. Annie Sheela" in names
        assert "Dr. Sudheer" in names

        annie = next(d for d in result.doctors if d.doctor_name == "Dr. Sr. Annie Sheela")
        assert "DM" in annie.qualification
        assert annie.department == "Cardiology"
        assert annie.image_url == "https://www.psmissionhospital.org/wp-content/uploads/2021/03/i2.jpg"

    def test_image_url_normalization_and_lazy_loading(self):
        soup = BeautifulSoup('<img data-src="/images/doc.jpg" />', "html.parser")
        img_url = StructuredExtractor.extract_image_url(soup.find("img"), "https://www.psmissionhospital.org/doctors")
        assert img_url == "https://www.psmissionhospital.org/images/doc.jpg"

        # Tracker exclusion
        soup_tracker = BeautifulSoup('<img src="https://tracker.com/1x1.png" />', "html.parser")
        assert StructuredExtractor.extract_image_url(soup_tracker.find("img"), "https://example.com") == ""


class TestStructuredPreviewAPI:
    """Tests for GET and POST /api/structured/preview."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_preview_post_with_raw_html(self, client):
        payload = {
            "url": "https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
            "html": SAMPLE_DEPARTMENT_HTML,
        }
        res = client.post("/api/structured/preview", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["url"] == payload["url"]
        assert data["total_doctors"] == 1
        assert data["total_facilities"] == 3
        assert data["department"]["department_name"] == "Paediatrics & Neonatology"

        doc = data["doctors"][0]
        assert doc["type"] == "doctor"
        assert doc["doctor_name"] == "Dr. Sr. Jaya Joseph"
        assert doc["department"] == "Paediatrics & Neonatology"
        assert "9.30AM" in doc["schedule_text"]
        assert doc["image_url"].startswith("https://")

        fac = data["facilities"][0]
        assert fac["type"] == "facility"
        assert fac["facility_name"] == "Immunization and Child Vaccines"

    def test_preview_empty_url_returns_400(self, client):
        res = client.post("/api/structured/preview", json={"url": ""})
        assert res.status_code == 400
