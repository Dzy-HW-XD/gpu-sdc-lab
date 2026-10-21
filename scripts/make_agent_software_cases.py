#!/usr/bin/env python3
"""Build 12 long-context agent cases for the software GEMM experiment."""
import json
import os

OUT = "data/agent_software_cases.json"

policy = []
policy.append(
    "NORTHSTAR INTERNAL SUPPORT HANDBOOK\n"
    "This document is proprietary to Northstar. Use only this document. "
    "Do not use outside knowledge. Return the shortest exact value requested.\n"
    "SECTION 1: CUSTOMER TIERS\n"
    "Tier 1 customers receive priority handling. Tier 2 customers receive standard handling. "
    "Tier 3 customers receive limited handling. Tier names do not change the product catalog.\n"
    "SECTION 2: DAMAGED GOODS POLICY NS-217\n"
    "For a damaged item, the maximum refund depends on the customer tier. "
    "Tier 1 damaged items have a maximum refund of 200 USD. "
    "Tier 2 damaged items have a maximum refund of 120 USD. "
    "Tier 3 damaged items have a maximum refund of 60 USD. "
    "A photo of the damaged item is required.\n"
    "SECTION 3: REVIEW WINDOW\n"
    "All damaged-goods claims are reviewed within 5 business days. "
    "The review window is not a refund amount.\n"
)
for i in range(1, 11):
    policy.append(
        "SECTION %d: OPERATIONS NOTE %02d\n"
        "The Northstar operations team maintains regional queues for order review. "
        "Queue %02d handles inventory reconciliation and does not approve refunds. "
        "Escalation codes are NS-OPS-%03d and NS-REV-%03d. "
        "These codes are internal routing identifiers, not customer refund amounts. "
        "Regional managers may request a second review when the original order value exceeds %d USD. "
        "This note is unrelated to the damaged-goods refund table or review window.\n"
        % (i + 3, i, i, 100 + i, 500 + i, 300 + i * 10)
    )
policy_context = "\n".join(policy)

tool = []
tool.append(
    "NORTHSTAR AGENT TOOL CATALOG\n"
    "Use only the tools below. Return the exact tool name requested; do not explain.\n"
    "SECTION 1: TOOL DEFINITIONS\n"
    "reset_mfa(user_id) resets multi-factor authentication for one user.\n"
    "refund_order(order_id, amount) issues a refund for one order.\n"
    "unlock_account(user_id) unlocks a locked customer account.\n"
    "get_invoice(invoice_id) retrieves one invoice record.\n"
    "SECTION 2: USAGE RULES\n"
    "Do not call refund_order for a policy question. Do not call reset_mfa for an invoice lookup. "
    "Tool names are case-sensitive. Return only the tool name, without parentheses or arguments.\n"
)
for i in range(1, 11):
    tool.append(
        "SECTION %d: TOOL ROUTING NOTE %02d\n"
        "The tool gateway records request IDs NS-TOOL-%03d. "
        "Queue %02d is an audit queue, not a callable tool. "
        "Approval codes NS-APP-%03d are not tool names. "
        "An MFA reset is not an account unlock. An invoice lookup is not a refund. "
        "The gateway may retry a failed call, but the tool name does not change.\n"
        % (i + 2, i, 200 + i, i, 700 + i)
    )
tool_context = "\n".join(tool)

api = []
api.append(
    "NORTHSTAR INTERNAL API REFERENCE\n"
    "Use only the endpoints below. Return the exact path requested, including braces. "
    "Do not include the HTTP method unless explicitly asked.\n"
    "SECTION 1: ENDPOINT TABLE\n"
    "PATCH /v2/orders/{id}/status updates one order status.\n"
    "GET /v2/customers/{id} retrieves one customer.\n"
    "POST /v2/refunds creates one refund.\n"
    "GET /v2/invoices lists invoices.\n"
    "SECTION 2: PATH RULES\n"
    "Paths are case-sensitive. Do not add a query string. Do not use the deprecated /v1 paths.\n"
)
for i in range(1, 11):
    api.append(
        "SECTION %d: API OPERATIONS NOTE %02d\n"
        "The API gateway records request IDs NS-API-%03d. "
        "Service NS-SVC-%03d is an internal worker, not an endpoint. "
        "Legacy paths under /v1/%03d are deprecated and must not be used. "
        "The gateway may retry a request, but the endpoint path does not change. "
        "This note is not an endpoint table.\n"
        % (i + 2, i, 300 + i, i, 800 + i)
    )
api_context = "\n".join(api)

cases = []
policy_qs = [
    ("policy_001", "A Tier 2 customer's order arrived damaged. What is the maximum refund in USD? Answer with the number only.", "120"),
    ("policy_002", "A Tier 1 customer's order arrived damaged. What is the maximum refund in USD? Answer with the number only.", "200"),
    ("policy_003", "A Tier 3 customer's order arrived damaged. What is the maximum refund in USD? Answer with the number only.", "60"),
    ("policy_004", "How many business days are allowed for a damaged-goods claim review? Answer with the number only.", "5"),
]
for cid, q, a in policy_qs:
    cases.append({"case_id": cid, "category": "policy_qa", "system_context": policy_context,
                  "user_query": q, "expected_answer": a, "answer_regex": r"\b%s\b" % a,
                  "difficulty": "single-fact retrieval"})

tool_qs = [
    ("tool_001", "Which tool resets MFA for a user? Answer with the tool name only.", "reset_mfa"),
    ("tool_002", "Which tool issues a refund for an order? Answer with the tool name only.", "refund_order"),
    ("tool_003", "Which tool unlocks a locked customer account? Answer with the tool name only.", "unlock_account"),
    ("tool_004", "Which tool retrieves an invoice record? Answer with the tool name only.", "get_invoice"),
]
for cid, q, a in tool_qs:
    cases.append({"case_id": cid, "category": "tool_routing", "system_context": tool_context,
                  "user_query": q, "expected_answer": a, "answer_regex": r"\b%s\b" % a,
                  "difficulty": "tool selection"})

api_qs = [
    ("api_001", "Which endpoint updates an order status? Answer with the path only.", "/v2/orders/{id}/status"),
    ("api_002", "Which endpoint retrieves one customer? Answer with the path only.", "/v2/customers/{id}"),
    ("api_003", "Which endpoint creates a refund? Answer with the path only.", "/v2/refunds"),
    ("api_004", "Which endpoint lists invoices? Answer with the path only.", "/v2/invoices"),
]
for cid, q, a in api_qs:
    cases.append({"case_id": cid, "category": "api_lookup", "system_context": api_context,
                  "user_query": q, "expected_answer": a, "answer_regex": r"\b%s\b" % a,
                  "difficulty": "endpoint lookup"})

assert len(cases) == 12
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(cases, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT, len(cases))
for c in cases:
    print(c["case_id"], len(c["system_context"]), len(c["system_context"].split()))
