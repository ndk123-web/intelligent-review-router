import re
import spacy
import torch
from transformers import pipeline
from app.services.preprocessing import split_into_sentences

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

CANDIDATE_DEPARTMENTS = [
    "Logistics & Shipping",
    "Product Support & Hardware",
    "Payments & Billing",
    "Refunds & Returns",
    "Packaging & Unboxing",
    "Customer Support & Service",
    "IT & Technical App Bugs"
]

LABEL_TO_ASPECT = {
    "Logistics & Shipping": ("delivery", "Logistics"),
    "Product Support & Hardware": ("product_quality", "Product Support"),
    "Payments & Billing": ("payment", "Payments"),
    "Refunds & Returns": ("refund_return", "Refunds"),
    "Packaging & Unboxing": ("packaging", "Returns / Replacement"),
    "Customer Support & Service": ("customer_support", "Customer Support"),
    "IT & Technical App Bugs": ("technical_issue", "IT Support")
}

DEPARTMENT_MAP = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support",
    "technical_issue": "IT Support"
}

DEPARTMENT_SEEDS = {
    "delivery": {"delivery", "shipping", "courier", "dispatch", "postage", "tracking", "transit", "arrival", "driver", "delayed", "late"},
    "product_quality": {"quality", "screen", "display", "battery", "sound", "material", "build", "camera", "performance", "device", "hinge", "button", "hardware", "software", "product", "laptop", "phone", "broken", "defective"},
    "payment": {"payment", "billing", "charge", "fee", "gateway", "transaction", "upi", "card", "price", "cost"},
    "refund_return": {"refund", "return", "reimbursement", "pickup", "exchange", "money"},
    "packaging": {"packaging", "box", "package", "envelope", "wrapper", "cushioning", "bubble wrap", "wrap", "torn"},
    "customer_support": {"support", "agent", "representative", "service", "helpdesk", "staff", "call", "email", "chat"},
    "technical_issue": {"app", "website", "crash", "bug", "otp", "login", "error", "server", "glitch", "down"}
}

class AspectExtractor:
    """
    ML/NLP-based Aspect Extractor using:
    1. Smart Clause Splitting
    2. Fine-Tuned DistilBERT for Clause Polarity Classification (APC) & Sentiment Gatekeeping
    3. facebook/bart-large-mnli Zero-Shot Classifier for dynamic Department & Aspect mapping
    4. spaCy Syntactic Dependency Parsing for noun entity extraction
    """
    def __init__(self, sentiment_analyzer=None, use_zero_shot: bool = None):
        self.sentiment_analyzer = sentiment_analyzer
        if use_zero_shot is None:
            import os
            self.use_zero_shot = os.getenv("USE_ZERO_SHOT", "true").lower() in ("true", "1", "yes")
        else:
            self.use_zero_shot = use_zero_shot
        self._zero_shot_pipeline = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    @property
    def zero_shot(self):
        """Lazy-loaded Zero-Shot classification pipeline to save startup time and memory."""
        if not self.use_zero_shot:
            return None
        if self._zero_shot_pipeline is None:
            device = 0 if torch.cuda.is_available() else -1
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                # Try loading from local cache first to avoid hanging if network is slow
                self._zero_shot_pipeline = pipeline(
                    "zero-shot-classification",
                    model=ZERO_SHOT_MODEL,
                    device=device,
                    torch_dtype=dtype,
                    model_kwargs={"local_files_only": True}
                )
            except Exception:
                try:
                    self._zero_shot_pipeline = pipeline(
                        "zero-shot-classification",
                        model=ZERO_SHOT_MODEL,
                        device=device,
                        torch_dtype=dtype
                    )
                except Exception as e:
                    print(f"Zero-shot model {ZERO_SHOT_MODEL} not available: {e}")
                    self._zero_shot_pipeline = False
        return self._zero_shot_pipeline if self._zero_shot_pipeline else None

    def _map_entity_to_aspect(self, entity_text: str, clause: str) -> tuple[str, str]:
        """Fallback keyword seed mapping if zero-shot classifier is not available."""
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

            # 1. spaCy Syntactic Noun Chunk Extraction
            if self.nlp:
                doc = self.nlp(clause)
                for chunk in doc.noun_chunks:
                    clean_chunk = chunk.text.strip()
                    if len(clean_chunk) > 1 and chunk.root.pos_ not in ("PRON", "DET"):
                        extracted_entities.append(clean_chunk)

            entity_repr = extracted_entities[0] if extracted_entities else clause

            # 2. Aspect Polarity Classification via Fine-Tuned DistilBERT Model
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

            # 3. Sentiment Gatekeeper:
            # Only actionable (Negative / Neutral) clauses require heavy Zero-Shot classification!
            # Positive praise clauses use fast semantic seed mapping and are logged for analytics.
            is_actionable = (sent in ("Negative", "Neutral"))
            aspect, department = None, None
            top_label = None
            top_score = 0.0

            if is_actionable:
                zs = self.zero_shot
                if zs:
                    try:
                        res = zs(
                            clause,
                            candidate_labels=CANDIDATE_DEPARTMENTS,
                            hypothesis_template="This customer issue is related to {}."
                        )
                        top_label = res["labels"][0]
                        top_score = round(float(res["scores"][0]), 3)
                        aspect, department = LABEL_TO_ASPECT.get(top_label, ("product_quality", "Product Support"))
                    except Exception:
                        aspect, department = self._map_entity_to_aspect(entity_repr, clause)
                else:
                    aspect, department = self._map_entity_to_aspect(entity_repr, clause)
            else:
                # Fast seed mapping for positive feedback
                aspect, department = self._map_entity_to_aspect(entity_repr, clause)
                top_label = department
                top_score = 0.90

            if aspect not in seen_aspects or is_actionable:
                seen_aspects.add(aspect)
                detected_issues.append({
                    "aspect": aspect,
                    "opinion": clause,
                    "sentiment": sent,
                    "confidence": round(conf, 2),
                    "department": department,
                    "extracted_entity": entity_repr,
                    "is_actionable": is_actionable,
                    "zero_shot_label": top_label,
                    "zero_shot_confidence": top_score
                })

        # Fallback if no issues extracted
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
                "extracted_entity": "product",
                "is_actionable": sent in ("Negative", "Neutral"),
                "zero_shot_label": "Product Support & Hardware",
                "zero_shot_confidence": 0.80
            })

        return detected_issues


