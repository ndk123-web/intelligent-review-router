import os
import json
import random
import pandas as pd

# Categories & Sources matching IberaSoft schema
CATEGORIES = ["Electronics", "Software/SaaS", "Home & Kitchen", "Beauty & Personal Care", "Books & Media", "Clothing & Accessories", "Sports & Outdoors", "Other"]
SOURCES = ["Amazon", "Yelp", "G2", "Capterra", "TrustRadius"]

ASPECT_TAXONOMY = {
    "delivery": {
        "department": "Logistics",
        "opinions": {
            "Negative": ["delivery was extremely late", "delayed by two weeks", "package never arrived", "tracking number not working", "courier was rude"],
            "Neutral": ["delivery took standard 5 days", "arrived on estimated date", "shipped via standard post"],
            "Positive": ["super fast delivery", "arrived earlier than expected", "excellent packaging and quick delivery", "express shipping was prompt"]
        }
    },
    "product_quality": {
        "department": "Product Support",
        "opinions": {
            "Negative": ["screen was broken", "poor material quality", "stopped working after one day", "defective item received", "horrible sound quality"],
            "Neutral": ["quality is average for price", "matches product description", "decent build"],
            "Positive": ["top notch quality", "exceeded my expectations", "feels premium and durable", "works perfectly", "crystal clear display"]
        }
    },
    "payment": {
        "department": "Payments",
        "opinions": {
            "Negative": ["charged twice for single order", "payment failed but money deducted", "unauthorized transaction fee", "payment gateway error"],
            "Neutral": ["paid via credit card", "UPI payment processed normally", "cash on delivery option used"],
            "Positive": ["smooth payment process", "instant checkout", "great discount applied at payment"]
        }
    },
    "refund_return": {
        "department": "Refunds",
        "opinions": {
            "Negative": ["refund has not arrived yet", "return request rejected wrongfully", "waiting 3 weeks for refund", "return pickup not scheduled"],
            "Neutral": ["return policy is 10 days", "initiated return request", "waiting for refund processing"],
            "Positive": ["refund processed within 24 hours", "hassle-free return experience", "quick money back"]
        }
    },
    "packaging": {
        "department": "Returns / Replacement",
        "opinions": {
            "Negative": ["box was crushed and damaged", "poor packaging material", "item was unpacked and torn", "missing protective bubble wrap"],
            "Neutral": ["standard cardboard box packaging", "simple sealed envelope"],
            "Positive": ["beautiful gift packaging", "very secure packaging", "sturdy box with great cushioning"]
        }
    },
    "customer_support": {
        "department": "Customer Support",
        "opinions": {
            "Negative": ["support representative was useless", "no response to customer emails", "call disconnected repeatedly", "horrible customer service"],
            "Neutral": ["contacted support for invoice", "support answered standard query"],
            "Positive": ["customer service was incredibly helpful", "agent solved my issue immediately", "support team is polite and quick"]
        }
    }
}

PRODUCTS = ["smartphone", "laptop", "wireless earbuds", "running shoes", "winter jacket", "smartwatch", "coffee maker", "gaming mouse", "bluetooth speaker", "desk lamp"]

SINGLE_ASPECT_TEMPLATES = [
    "I bought this {product}. {opinion}.",
    "Regarding the {product}, {opinion}.",
    "{opinion}. Overall satisfied with the purchase.",
    "{opinion} for my recent {product} order.",
    "Very disappointed because {opinion}.",
    "Great experience! {opinion}."
]

def determine_priority(aspects_with_sentiments):
    has_urgent = any(a in ["payment", "refund_return"] and s == "Negative" for a, s in aspects_with_sentiments)
    neg_count = sum(1 for _, s in aspects_with_sentiments if s == "Negative")
    
    if has_urgent and neg_count >= 2:
        return "URGENT"
    elif has_urgent or neg_count >= 2:
        return "HIGH"
    elif neg_count == 1:
        return "MEDIUM"
    else:
        return "LOW"

def generate_reviews_dataset(num_samples=20000, random_state=42):
    random.seed(random_state)
    data = []
    
    aspect_keys = list(ASPECT_TAXONOMY.keys())
    
    for i in range(1, num_samples + 1):
        review_id = f"rev_{i:06d}"
        product = random.choice(PRODUCTS)
        category = random.choice(CATEGORIES)
        source = random.choice(SOURCES)
        is_multi = random.random() < 0.40
        
        selected_aspects = []
        issues = []
        
        if not is_multi:
            aspect = random.choice(aspect_keys)
            sentiment = random.choices(["Negative", "Positive", "Neutral"], weights=[0.35, 0.40, 0.25])[0]
            opinion_text = random.choice(ASPECT_TAXONOMY[aspect]["opinions"][sentiment])
            
            review_text = random.choice(SINGLE_ASPECT_TEMPLATES).format(product=product, opinion=opinion_text)
            selected_aspects.append((aspect, sentiment))
            
            issues.append({
                "aspect": aspect,
                "opinion": opinion_text,
                "sentiment": sentiment,
                "confidence": round(random.uniform(0.85, 0.99), 2)
            })
            overall_sentiment = sentiment
            
        else:
            num_aspects = random.choice([2, 3])
            chosen_aspects = random.sample(aspect_keys, num_aspects)
            sentiments_pool = ["Positive", "Negative", "Negative", "Neutral"]
            
            op_texts = []
            for aspect in chosen_aspects:
                sent = random.choice(sentiments_pool)
                op_text = random.choice(ASPECT_TAXONOMY[aspect]["opinions"][sent])
                selected_aspects.append((aspect, sent))
                op_texts.append((aspect, sent, op_text))
                issues.append({
                    "aspect": aspect,
                    "opinion": op_text,
                    "sentiment": sent,
                    "confidence": round(random.uniform(0.82, 0.98), 2)
                })
            
            pos_ops = [op for a, s, op in op_texts if s == "Positive"]
            neg_ops = [op for a, s, op in op_texts if s == "Negative"]
            
            if pos_ops and neg_ops:
                review_text = f"The {product} is {pos_ops[0]}, but {neg_ops[0]}."
            elif neg_ops:
                review_text = f"I am unhappy with the {product}. {neg_ops[0]}."
            elif pos_ops:
                review_text = f"Excellent {product}! {pos_ops[0]}."
            else:
                review_text = f"Bought the {product}. {op_texts[0][2]}."
                
            neg_count = sum(1 for _, s in selected_aspects if s == "Negative")
            pos_count = sum(1 for _, s in selected_aspects if s == "Positive")
            if neg_count > pos_count:
                overall_sentiment = "Negative"
            elif pos_count > neg_count:
                overall_sentiment = "Positive"
            else:
                overall_sentiment = "Neutral"

        # Rating mapping
        if overall_sentiment == "Positive":
            label = 2
            rating = random.choice([4, 5])
        elif overall_sentiment == "Negative":
            label = 0
            rating = random.choice([1, 2])
        else:
            label = 1
            rating = 3

        priority = determine_priority(selected_aspects)
        primary_dept = ASPECT_TAXONOMY[selected_aspects[0][0]]["department"] if selected_aspects else "Customer Support"
        
        data.append({
            "review_id": review_id,
            "text": review_text,
            "review": review_text,
            "label": label,
            "sentiment": overall_sentiment,
            "rating": rating,
            "category": category,
            "source": source,
            "aspects": ",".join([a for a, _ in selected_aspects]),
            "issues_json": json.dumps(issues),
            "department": primary_dept,
            "priority": priority
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Generating 20,000 IberaSoft-schema dataset...")
    df = generate_reviews_dataset(num_samples=20000)
    
    raw_path = "data/raw/ecommerce_reviews.csv"
    proc_path = "data/processed/ecommerce_reviews.csv"
    
    df.to_csv(raw_path, index=False)
    df.to_csv(proc_path, index=False)
    
    # Save train/val/test splits
    train_df = df.iloc[:15000]
    val_df = df.iloc[15000:18000]
    test_df = df.iloc[18000:]
    
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Dataset generated successfully! Total: {len(df)}")
    print(f"Splits saved: Train (15,000), Val (3,000), Test (2,000)")
