#!/usr/bin/env python3
"""Build the Phase-0 long-context agent case."""
import json
import os

OUT = "data/agent_phase0_case.json"

sections = []
sections.append(
    "NORTHSTAR INTERNAL SUPPORT HANDBOOK\n"
    "This document is proprietary to Northstar. Use only this document when answering.\n"
    "Do not use outside knowledge. Answer with the shortest exact value requested.\n"
)
sections.append(
    "SECTION 1: CUSTOMER TIERS\n"
    "Tier 1 customers receive priority handling. Tier 2 customers receive standard handling. "
    "Tier 3 customers receive limited handling. Tier names do not change the product catalog.\n"
)
sections.append(
    "SECTION 2: DAMAGED GOODS POLICY NS-217\n"
    "For a damaged item, the maximum refund depends on the customer tier. "
    "Tier 1 damaged items have a maximum refund of 200 USD. "
    "Tier 2 damaged items have a maximum refund of 120 USD. "
    "Tier 3 damaged items have a maximum refund of 60 USD. "
    "The refund is issued to the original payment method. "
    "A photo of the damaged item is required. "
)
# Add distractor sections with unrelated numbers and similar terms.
for i in range(1, 19):
    sections.append(
        "SECTION %d: OPERATIONS NOTE %02d\n"
        "The Northstar operations team maintains regional queues for order review. "
        "Queue %02d handles inventory reconciliation and does not approve refunds. "
        "The standard review window is %d business days. "
        "Escalation codes are NS-OPS-%03d and NS-REV-%03d. "
        "These codes are internal routing identifiers, not customer refund amounts. "
        "Regional managers may request a second review when the original order value exceeds %d USD. "
        "This note is unrelated to the damaged-goods refund table.\n"
        % (i + 2, i, i, 3 + (i % 5), 100 + i, 500 + i, 300 + i * 10)
    )
sections.append(
    "SECTION 21: FINAL REMINDER\n"
    "When a question asks for a refund amount, use the tier-specific damaged-goods policy. "
    "Do not substitute an operations escalation code or a review-window number. "
)

case = {
    "case_id": "policy_001",
    "category": "policy_qa",
    "system_context": "\n".join(sections),
    "user_query": (
        "A Tier 2 customer's order arrived damaged. "
        "What is the maximum refund in USD? Answer with the number only."
    ),
    "expected_answer": "120",
    "answer_regex": r"\b120\b",
    "difficulty": "single-fact retrieval",
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(case, f, ensure_ascii=False, indent=1)
print("wrote", OUT)
print("chars", len(case["system_context"]), "words", len(case["system_context"].split()))
