# pyrefly: ignore [missing-import]
import pytest
from app.services.aspect_extraction import AspectExtractor

@pytest.fixture
def extractor():
    return AspectExtractor(use_zero_shot=False)

def test_single_aspect_extraction(extractor):
    text = "The delivery was delayed by three days."
    issues = extractor.extract_aspects(text)
    assert len(issues) >= 1
    assert issues[0]["aspect"] == "delivery"
    assert issues[0]["department"] == "Logistics"

def test_multi_aspect_extraction(extractor):
    text = "The phone display is clear, but the refund was never processed."
    issues = extractor.extract_aspects(text)
    extracted_aspects = [i["aspect"] for i in issues]
    assert "refund_return" in extracted_aspects or "product_quality" in extracted_aspects
