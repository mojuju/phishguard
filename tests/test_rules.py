from phishguard import rules


def names(text):
    return {f.rule for f in rules.analyze(text)}


def test_ip_link_is_flagged():
    assert "ip_address_link" in names("Login here: http://192.168.1.10/verify")


def test_shortener_is_flagged():
    assert "url_shortener" in names("Click https://bit.ly/abc123 now")


def test_at_sign_obfuscation():
    assert "obfuscated_link" in names("https://paypal.com@evil.example/login")


def test_risky_attachment():
    assert "risky_attachment" in names("Open the attached invoice.exe to continue")


def test_urgency_and_credentials():
    found = names("URGENT: verify your account and send your password")
    assert {"urgency", "credential_request"} <= found


def test_clean_email_has_no_flags():
    assert rules.analyze("Hi Sara, lunch on Thursday at 1pm? See https://example.com/menu") == []


def test_rule_score_is_bounded():
    text = "urgent password winner http://1.2.3.4 https://bit.ly/x invoice.exe"
    assert 0 < rules.rule_score(rules.analyze(text)) <= 1
