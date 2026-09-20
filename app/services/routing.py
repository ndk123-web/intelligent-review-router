ASPECT_DEPARTMENT_MAP = {
    "delivery": "Logistics",
    "product_quality": "Product Support",
    "payment": "Payments",
    "refund_return": "Refunds",
    "packaging": "Returns / Replacement",
    "customer_support": "Customer Support",
    "technical_issue": "IT Support"
}

CRITICAL_ASPECTS = {"payment", "refund_return", "technical_issue"}

class SupportRouter:
    def route_issues(self, overall_sentiment: str, issues: list[dict]) -> dict:
        routing_list = []
        negative_aspects = []
        
        for issue in issues:
            aspect = issue.get("aspect", "product_quality")
            sent = issue.get("sentiment", "Neutral")
            dept = issue.get("department") or ASPECT_DEPARTMENT_MAP.get(aspect, "Customer Support")
            is_actionable = issue.get("is_actionable", sent in ("Negative", "Neutral"))
            
            if sent == "Negative":
                negative_aspects.append(aspect)
                # Compute aspect-specific priority
                if aspect in CRITICAL_ASPECTS:
                    priority = "HIGH"
                else:
                    priority = "MEDIUM"
                reason = f"Customer reported negative issue regarding {aspect} ('{issue.get('opinion', '')}')"
            elif sent == "Neutral":
                if is_actionable:
                    priority = "LOW"
                    reason = f"Neutral feedback/inquiry on {aspect} ('{issue.get('opinion', '')}')"
                else:
                    priority = "LOW"
                    reason = f"Neutral comment on {aspect}"
            else:
                priority = "LOW"
                reason = f"Positive feedback on {aspect} (logged for analytics)"
                
            routing_list.append({
                "department": dept,
                "aspect": aspect,
                "priority": priority,
                "reason": reason,
                "is_actionable": is_actionable
            })
            
        # Calculate overall system priority
        neg_count = len(negative_aspects)
        has_critical = any(a in CRITICAL_ASPECTS for a in negative_aspects)
        
        if has_critical and neg_count >= 2:
            overall_priority = "URGENT"
        elif has_critical or neg_count >= 2:
            overall_priority = "HIGH"
        elif neg_count == 1:
            overall_priority = "MEDIUM"
        else:
            overall_priority = "LOW"
            
        # Select primary department (prefer actionable/higher priority departments)
        if routing_list:
            priority_weights = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            # Sort by actionable first, then priority weight
            sorted_routing = sorted(
                routing_list,
                key=lambda x: (1 if x.get("is_actionable", False) else 0, priority_weights.get(x["priority"], 0)),
                reverse=True
            )
            primary_department = sorted_routing[0]["department"]
        else:
            primary_department = "Customer Support"
            
        return {
            "primary_department": primary_department,
            "overall_priority": overall_priority,
            "routing": routing_list
        }

