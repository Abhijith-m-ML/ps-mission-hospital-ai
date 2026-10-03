import pytest
from app.crawler.url_manager import URLManager


def test_url_normalization_and_fragments():
    manager = URLManager(start_url="https://stjude-hospital.org")

    # Strips fragments
    url_with_fragment = "https://stjude-hospital.org/about#history"
    normalized = manager.normalize_url(url_with_fragment)
    assert normalized == "https://stjude-hospital.org/about"

    # Lowercase scheme and host
    url_mixed_case = "HTTPS://StJude-Hospital.ORG/Departments"
    normalized = manager.normalize_url(url_mixed_case)
    assert normalized == "https://stjude-hospital.org/Departments"

    # Strips default port
    url_with_port = "https://stjude-hospital.org:443/contact"
    normalized = manager.normalize_url(url_with_port)
    assert normalized == "https://stjude-hospital.org/contact"

    # Trailing slash normalization on non-root paths
    assert manager.normalize_url("https://stjude-hospital.org/doctors/") == "https://stjude-hospital.org/doctors"
    # Preserves root slash
    assert manager.normalize_url("https://stjude-hospital.org/") == "https://stjude-hospital.org/"


def test_relative_url_conversion():
    manager = URLManager(start_url="https://stjude-hospital.org/portal")

    # Absolute path relative URL
    rel_url = "/services/cardiology"
    normalized = manager.normalize_url(rel_url, current_page_url="https://stjude-hospital.org/portal")
    assert normalized == "https://stjude-hospital.org/services/cardiology"

    # Relative path without leading slash
    rel_sub = "details"
    normalized = manager.normalize_url(rel_sub, current_page_url="https://stjude-hospital.org/services/")
    assert normalized == "https://stjude-hospital.org/services/details"


def test_unsupported_url_schemes():
    manager = URLManager(start_url="https://stjude-hospital.org")

    assert manager.normalize_url("mailto:info@stjude-hospital.org") is None
    assert manager.normalize_url("tel:+18005552273") is None
    assert manager.normalize_url("javascript:void(0)") is None
    assert manager.normalize_url("ftp://files.stjude-hospital.org/records") is None
    assert manager.normalize_url("data:text/html;base64,PHNjcmlwdD4=") is None


def test_asset_filtering():
    manager = URLManager(start_url="https://stjude-hospital.org")

    assert manager.normalize_url("https://stjude-hospital.org/images/logo.png") is None
    assert manager.normalize_url("https://stjude-hospital.org/styles/main.css") is None
    assert manager.normalize_url("https://stjude-hospital.org/bundle.js") is None
    assert manager.normalize_url("https://stjude-hospital.org/brochure.pdf") is None
    assert manager.normalize_url("https://stjude-hospital.org/video/tour.mp4") is None


def test_same_domain_and_external_rejection():
    manager = URLManager(start_url="https://stjude-hospital.org")

    # Same domain
    assert manager.is_same_domain("https://stjude-hospital.org/about") is True
    assert manager.is_same_domain("http://stjude-hospital.org/departments") is True
    assert manager.is_same_domain("https://www.stjude-hospital.org/about") is True

    # External domain
    assert manager.is_same_domain("https://facebook.com/stjude") is False
    assert manager.is_same_domain("https://twitter.com/stjude") is False
    assert manager.is_same_domain("https://another-hospital.org") is False

    # should_crawl logic
    assert manager.should_crawl("https://stjude-hospital.org/services") is True
    assert manager.should_crawl("https://external-site.com/health") is False


def test_duplicate_detection_and_visited_tracking():
    manager = URLManager(start_url="https://stjude-hospital.org")
    test_url = "https://stjude-hospital.org/doctors"

    # Initially not discovered and not visited
    assert manager.is_discovered(test_url) is False
    assert manager.is_visited(test_url) is False

    # First discovery
    assert manager.mark_discovered(test_url) is True
    assert manager.is_discovered(test_url) is True

    # Duplicate discovery attempt returns False
    assert manager.mark_discovered(test_url) is False

    # Mark as visited
    manager.mark_visited(test_url)
    assert manager.is_visited(test_url) is True

    # should_crawl should reject already visited URLs
    assert manager.should_crawl(test_url) is False
