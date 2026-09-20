import re

def clean_text(text: str) -> str:
    """
    Cleans raw customer review text by removing excessive whitespace and normalizing characters.
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into individual sentence clauses for aspect-level analysis.
    Uses smart clause splitting:
    - Splits on sentence delimiters (. ! ?)
    - Splits on contrast conjunctions with comma (, but, , however, etc.)
    - Splits on coordinating conjunctions (and, but, or) when introducing a new subject (and my..., and the...)
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    # Smart regex pattern
    pattern = r'(?<=[.!?])\s+|,\s*(?:and|but|or|however|although|yet)\s*|\s+(?:and|but|or)\s+(?=(?:my|the|it|i|they|we|this|our)\b)'
    sentences = re.split(pattern, cleaned, flags=re.IGNORECASE)
    
    cleaned_clauses = []
    for s in sentences:
        s_clean = s.strip()
        # Strip any leading conjunctions or punctuation
        s_clean = re.sub(r'^(?:and|but|or|however|yet|so)\s+', '', s_clean, flags=re.IGNORECASE).strip()
        s_clean = s_clean.strip(',;: ')
        if len(s_clean) > 3:
            cleaned_clauses.append(s_clean)
            
    return cleaned_clauses
