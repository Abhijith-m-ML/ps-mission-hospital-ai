from app.crawler.parser import HTMLParser
from app.crawler.url_manager import URLManager


def test_html_parser_title_and_clean_text():
    parser = HTMLParser()
    url_manager = URLManager(start_url="https://stjude-hospital.org")

    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>St. Jude Hospital - Cardiology</title>
        <style>body { font-size: 14px; }</style>
        <script>console.log("analytics");</script>
    </head>
    <body>
        <h1>Heart Health Center</h1>
        <p>We provide 24/7 cardiac monitoring and consultation.</p>
        <noscript>Please enable JavaScript</noscript>
    </body>
    </html>
    """

    parsed = parser.parse(sample_html, "https://stjude-hospital.org/cardiology", url_manager)

    assert parsed.title == "St. Jude Hospital - Cardiology"
    assert "Heart Health Center" in parsed.text_content
    assert "We provide 24/7 cardiac monitoring" in parsed.text_content
    # Scripts and styles should be removed
    assert "console.log" not in parsed.text_content
    assert "font-size" not in parsed.text_content
    assert "Please enable JavaScript" not in parsed.text_content


def test_html_parser_link_extraction():
    parser = HTMLParser()
    url_manager = URLManager(start_url="https://stjude-hospital.org")

    sample_html = """
    <html>
    <body>
        <a href="/departments/neurology">Neurology</a>
        <a href="https://stjude-hospital.org/contact#reception">Contact Reception</a>
        <a href="https://external.com/news">External Medical News</a>
        <a href="mailto:er@stjude-hospital.org">Email ER</a>
        <a href="/departments/neurology">Neurology Duplicate</a>
        <a href="/images/header.jpg">Header Image</a>
    </body>
    </html>
    """

    parsed = parser.parse(sample_html, "https://stjude-hospital.org/", url_manager)

    # Valid normalized links extracted (asset and mailto filtered out, duplicate link deduplicated)
    assert "https://stjude-hospital.org/departments/neurology" in parsed.links
    assert "https://stjude-hospital.org/contact" in parsed.links
    # External link is captured by parser, but URLManager.should_crawl will reject it during crawl
    assert "https://external.com/news" in parsed.links

    # Mailto and image should not be in parsed.links
    assert not any("mailto:" in link for link in parsed.links)
    assert not any("header.jpg" in link for link in parsed.links)
