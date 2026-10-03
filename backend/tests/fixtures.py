"""HTML test fixtures representing various real-world and edge-case webpage structures."""

HOSPITAL_PAGE_FIXTURE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>St. Jude Hospital - Cardiology Department</title>
    <meta name="description" content="Comprehensive cardiovascular diagnosis and intensive care at St. Jude Hospital.">
    <link rel="canonical" href="https://stjude-hospital.org/departments/cardiology">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #fff; }
        .hero { color: #0d9488; }
        .hidden-tracker { display: none; }
    </style>
    <script>
        window.analytics = { track: function() { console.log("tracking user event"); } };
    </script>
</head>
<body>
    <!-- Site Navigation & Header -->
    <header class="site-header">
        <nav role="navigation" class="main-navigation">
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/doctors">Doctors</a></li>
                <li><a href="/portal">Patient Portal</a></li>
            </ul>
        </nav>
    </header>

    <!-- Cookie Consent Banner -->
    <div class="cookie-banner">
        <p>We use cookies to improve your healthcare experience. <button>Accept All</button></p>
    </div>

    <!-- Main Content Area -->
    <main>
        <h1>Cardiology Department</h1>
        <p>Our cardiology department provides world-class cardiac care and cardiovascular surgery.</p>

        <h2>Clinical Services</h2>
        <ul>
            <li>Non-invasive diagnostic electrophysiology</li>
            <li>Cardiac catheterization and stent placement</li>
            <li>Post-operative coronary rehabilitation</li>
        </ul>

        <h2>Visiting Hours</h2>
        <p>Monday-Saturday: 9 AM - 5 PM</p>
        <p>Sunday: 10 AM - 3 PM (ICU restricted to immediate family)</p>

        <h2>Weekly Clinic Schedule</h2>
        <table>
            <tr>
                <th>Day</th>
                <th>Clinic Hours</th>
            </tr>
            <tr>
                <td>Mon - Fri</td>
                <td>08:00 - 18:00</td>
            </tr>
            <tr>
                <td>Saturday</td>
                <td>09:00 - 14:00</td>
            </tr>
        </table>
    </main>

    <!-- Sidebar Advertisement -->
    <aside class="advertisement">
        <p>Sponsored health insurance plans available here.</p>
    </aside>

    <!-- Footer -->
    <footer class="site-footer" role="contentinfo">
        <p>© 2026 St. Jude Healthcare Network. All rights reserved.</p>
        <p><a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms of Service</a></p>
    </footer>

    <noscript>
        <p>Please enable JavaScript to access patient records.</p>
    </noscript>
</body>
</html>
"""

SCRIPT_AND_STYLE_FIXTURE = """
<html>
<head>
    <title>Script Test</title>
    <style>
        .test { color: red; }
    </style>
    <script type="text/javascript">
        function alertBox() { alert("malicious alert"); }
    </script>
</head>
<body>
    <script>document.write("inline written code");</script>
    <h1>Clean Title</h1>
    <p>Legitimate patient instructions.</p>
    <script src="/tracking.js"></script>
    <style>p { line-height: 1.5; }</style>
</body>
</html>
"""

MALFORMED_HTML_FIXTURE = """
<h1>Unclosed Heading
<p>First paragraph without closing tag
<div><span>Nested unclosed content
<h2>Second Heading
<p>Second paragraph
"""

EMPTY_HTML_FIXTURE = """
<!DOCTYPE html>
<html>
<head></head>
<body>
    <script>console.log("no content");</script>
</body>
</html>
"""

WHITESPACE_FIXTURE = """
<html>
<body>
    <h1>   Spaced    Heading   </h1>
    <p>
        Line   one    with      irregular       spaces.
        
        
        Line   two   after   many   blanks.
    </p>
</body>
</html>
"""
