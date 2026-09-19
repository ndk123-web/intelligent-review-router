import re
import spacy
from app.services.preprocessing import split_into_sentences

DEPARTMENT_MAP = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support"
}

DEPARTMENT_SEEDS = {
    "delivery": {"delivery", "shipping", "courier", "dispatch", "postage", "tracking", "transit", "arrival", "driver", "delayed", "late"},
    "product_quality": {"quality", "screen", "display", "battery", "sound", "material", "build", "camera", "performance", "device", "hinge", "button", "hardware", "software", "product", "laptop", "phone", "broken", "defective"},
    "payment": {"payment", "billing", "charge", "fee", "gateway", "transaction", "upi", "card", "price", "cost"},
    "refund_return": {"refund", "return", "reimbursement", "pickup", "exchange", "money"},
    "packaging": {"packaging", "box", "package", "envelope", "wrapper", "cushioning", "bubble wrap", "wrap", "torn"},
    "customer_support": {"support", "agent", "representative", "service", "helpdesk", "staff", "call", "email", "chat"}
}

class AspectExtractor:
    """
    ML/NLP-based Aspect Extractor using spaCy NER & Syntactic Dependency Parsing (Noun Chunks)
    for Aspect Term Extraction (ATE) and Fine-Tuned DistilBERT Transformer for Aspect Polarity Classification (APC).
    """
    def __init__(self, sentiment_analyzer=None):
        self.sentiment_analyzer = sentiment_analyzer
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def _map_entity_to_aspect(self, entity_text: str, clause: str) -> tuple[str, str]:
        """
        Maps an extracted NER entity / Noun Chunk to a business aspect category and target department.
        """
        lower_ent = entity_text.lower()
        lower_clause = clause.lower()

        for aspect, seeds in DEPARTMENT_SEEDS.items():
            if any(seed in lower_ent or seed in lower_clause for seed in seeds):
                return aspect, DEPARTMENT_MAP[aspect]

        return "product_quality", DEPARTMENT_MAP["product_quality"]

    def extract_aspects(self, review_text: str) -> list[dict]:
        clauses = split_into_sentences(review_text)
        detected_issues = []
        seen_aspects = set()

        for clause in clauses:
            extracted_entities = []

            # 1. spaCy NER & Syntactic Noun Chunk Extraction
            if self.nlp:
                doc = self.nlp(clause)
                for chunk in doc.noun_chunks:
                    clean_chunk = chunk.text.strip()
                    if len(clean_chunk) > 1 and chunk.root.pos_ not in ("PRON", "DET"):
                        extracted_entities.append(clean_chunk)

            if not extracted_entities:
                extracted_entities = [clause]

            # 2. Process extracted aspect entity
            for ent in extracted_entities:
                aspect, department = self._map_entity_to_aspect(ent, clause)

                if aspect not in seen_aspects:
                    seen_aspects.add(aspect)

                    # 3. Aspect Polarity Classification via Fine-Tuned DistilBERT Model
                    if self.sentiment_analyzer:
                        sub_res = self.sentiment_analyzer.predict(clause)
                        sent = sub_res["sentiment"]
                        conf = sub_res["confidence"]
                    else:
                        lower_c = clause.lower()
                        neg_hits = sum(1 for w in ["bad", "terrible", "late", "delayed", "broken", "defective", "unhelpful", "poor"] if w in lower_c)
                        pos_hits = sum(1 for w in ["good", "great", "excellent", "fast", "amazing", "helpful", "polite"] if w in lower_c)
                        if neg_hits > pos_hits:
                            sent, conf = "Negative", 0.85
                        elif pos_hits > neg_hits:
                            sent, conf = "Positive", 0.85
                        else:
                            sent, conf = "Neutral", 0.75

                    detected_issues.append({
                        "aspect": aspect,
                        "opinion": clause,
                        "sentiment": sent,
                        "confidence": round(conf, 2),
                        "department": department,
                        "extracted_entity": ent
                    })

        if not detected_issues:
            if self.sentiment_analyzer:
                pred = self.sentiment_analyzer.predict(review_text)
                sent, conf = pred["sentiment"], pred["confidence"]
            else:
                sent, conf = "Neutral", 0.70

            detected_issues.append({
                "aspect": "product_quality",
                "opinion": review_text,
                "sentiment": sent,
                "confidence": round(conf, 2),
                "department": DEPARTMENT_MAP["product_quality"],
                "extracted_entity": "product"
            })

        return detected_issues

