"""
Step 3 — Document Generation via Gemini
Reads clean instances from data/instances/*.json, renders them into messy
documents (CSV exports, prose lease text, informal notes, message threads)
using Gemini, and writes results to data/documents/{type}/.

Runs against Gemini specifically — never Claude/GPT — per the circularity
rule: the model disguising the data can't be the same family later tested
on extraction (Step 5-7).

Setup:
    pip install google-genai python-dotenv
    echo "GEMINI_API_KEY=your_key_here" >> .env   (Google AI Studio free tier)

Usage:
    python scripts/generate_documents.py --types csv,lease,notes,messages --limit 3
    (start with --limit 3 per record type and manually check output before
    scaling up — see the checklist at the bottom of
    step3_document_generation_prompts.md)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

INSTANCES_DIR = Path("data/instances")
DOCUMENTS_DIR = Path("data/documents")
MODEL = "gemini-3.1-flash-lite"  # gemini-2.0-flash has 0 free-tier quota as of Aug 2026 — confirmed via aistudio.google.com/rate-limit
SLEEP_BETWEEN_CALLS = 4.5  # seconds — stay under free-tier RPM limits
MAX_RETRIES = 3

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Real schema attribute names Gemini must NOT use verbatim in its output —
# keep this in sync with schema/gold_schema.ttl.
REAL_ATTR_NAMES = {
    "Property": ["address", "squareFootage", "propertyType", "ownerId"],
    "Lease": ["rentAmount", "escalationType", "securityDeposit", "startDate", "endDate"],
    "MaintenanceRequest": ["status", "priority", "estimatedCost", "dateReported", "dateResolved"],
}


def load_instances():
    data = {}
    for f in INSTANCES_DIR.glob("*.json"):
        data[f.stem] = json.loads(f.read_text())
    return data


def index_by_id(records):
    return {r["id"]: r for r in records}


def call_gemini(prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            time.sleep(SLEEP_BETWEEN_CALLS)
            return resp.text.strip()
        except Exception as e:
            wait = SLEEP_BETWEEN_CALLS * attempt * 2
            print(f"  [retry {attempt}/{MAX_RETRIES}] {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries")


# ---------------------------------------------------------------------------
# Prompt builders — mirrors step3_document_generation_prompts.md
# ---------------------------------------------------------------------------

def csv_prompt(entity_type, batch, attr_names):
    ids = [r["id"] for r in batch]
    return f"""You are exporting records from an old, slightly inconsistent property
management CRM system. Given the structured data below, output a realistic
CSV export a property manager might pull for internal use.

Rules:
- Do NOT use these exact field names in your header row: {attr_names}.
  Invent realistic CRM-style column names instead.
- Format currency inconsistently across rows (e.g. "$4,300.00", "4300", "4,300").
- Format dates inconsistently (MM/DD/YYYY, "Jan 2024", YYYY-MM-DD — mix them).
- Leave at least 2 cells blank across the batch (not on ID-equivalent fields).
- Do not include these internal IDs anywhere in the output: {ids}. Refer to
  entities by name/address instead.
- Do not "clean up" or correct any source facts — reproduce them exactly,
  just formatted imperfectly.

Source records:
{json.dumps(batch, indent=2)}

Output only the CSV text, headers first."""


def lease_prompt(lease, owner, tenant, prop):
    return f"""You are drafting a commercial lease agreement document based on the
structured facts below. Write it as realistic prose lease language — the
kind a small commercial landlord's own template would produce, not a
polished law-firm document.

Rules:
- Refer to the parties inconsistently: sometimes by name ("{tenant['name']}"),
  sometimes by role ("Tenant", "the Lessee").
- State the property address and terms in prose, not as a labeled field list.
- Do not use these literal words: {REAL_ATTR_NAMES['Lease']}. Describe these
  facts in ordinary lease language instead.
- Include the escalation and security deposit terms as an actual lease
  clause, not a data field.
- Omit one minor, non-critical detail a real lease might leave implicit.
- Do not include any internal record IDs anywhere in the text.
- Keep the facts accurate — do not alter dates, dollar amounts, or names.

Source records:
Lease: {json.dumps(lease, indent=2)}
Owner: {json.dumps(owner, indent=2)}
Tenant: {json.dumps(tenant, indent=2)}
Property: {json.dumps(prop, indent=2)}

Output only the lease document text."""


def notes_prompt(mr, tenant, prop, agent, vendor):
    return f"""Write a short internal note the way a busy commercial property agent would
actually type it — fast, informal, abbreviated, first-person.

Rules:
- Maximum 3-4 sentences.
- Use realistic abbreviations: first name or initials only, "EOW" for end of
  week, "prop" for property, "req" for request, reference by street name
  not ID.
- Imply at least one relationship rather than stating it outright.
- Do not use these literal words: {REAL_ATTR_NAMES['MaintenanceRequest']}.
  Convey these facts in plain, everyday phrasing instead.
- Do not include any internal record IDs.
- Keep the underlying facts accurate to the source.

Source records:
MaintenanceRequest: {json.dumps(mr, indent=2)}
Tenant: {json.dumps(tenant, indent=2)}
Property: {json.dumps(prop, indent=2)}
Agent: {json.dumps(agent, indent=2)}
Vendor: {json.dumps(vendor, indent=2) if vendor else "null (not yet assigned)"}

Output only the note text."""


def message_prompt(mr, tenant, prop, agent, vendor):
    pairing = random.choice(["tenant-agent", "agent-vendor"])
    return f"""Write a short realistic text-message exchange between two people discussing
the situation described below. Use this pairing: {pairing}
({"the tenant texting the agent about the issue" if pairing == "tenant-agent" else "the agent texting the vendor to coordinate the fix"}).

Rules:
- Format as: [HH:MM AM/PM] FirstName: message text
- Use first names or initials only — never full names, never IDs.
- Keep messages short and natural, 4-8 message exchange, back-and-forth.
- Meaning should be implied, not stated outright.
- Do not use these literal words: {REAL_ATTR_NAMES['MaintenanceRequest']}.
- Keep the underlying facts accurate to the source.

Source records:
MaintenanceRequest: {json.dumps(mr, indent=2)}
Tenant: {json.dumps(tenant, indent=2)}
Property: {json.dumps(prop, indent=2)}
Agent: {json.dumps(agent, indent=2)}
Vendor: {json.dumps(vendor, indent=2) if vendor else "null (not yet assigned)"}

Output only the message exchange."""


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_csv(data, limit):
    out_dir = DOCUMENTS_DIR / "csv_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    for entity_type, key in [("Property", "properties"), ("Lease", "leases"),
                              ("MaintenanceRequest", "maintenance_requests")]:
        batch = data[key][:limit]
        if not batch:
            continue
        print(f"[csv] {entity_type}: {len(batch)} records")
        text = call_gemini(csv_prompt(entity_type, batch, REAL_ATTR_NAMES.get(entity_type, [])))
        (out_dir / f"{key}_export.csv").write_text(text)


def run_lease_texts(data, limit):
    out_dir = DOCUMENTS_DIR / "lease_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    owners = index_by_id(data["owners"])
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    for lease in data["leases"][:limit]:
        print(f"[lease] {lease['id']}")
        text = call_gemini(lease_prompt(
            lease, owners[lease["ownerId"]], tenants[lease["tenantId"]],
            properties[lease["propertyId"]]))
        (out_dir / f"{lease['id']}.txt").write_text(text)


def run_notes(data, limit):
    out_dir = DOCUMENTS_DIR / "notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    agents = index_by_id(data["agents"])
    vendors = index_by_id(data["vendors"])
    for mr in data["maintenance_requests"][:limit]:
        print(f"[notes] {mr['id']}")
        vendor = vendors.get(mr["resolvingVendorId"]) if mr["resolvingVendorId"] else None
        text = call_gemini(notes_prompt(
            mr, tenants[mr["tenantId"]], properties[mr["propertyId"]],
            agents[mr["assigningAgentId"]], vendor))
        (out_dir / f"{mr['id']}.txt").write_text(text)


def run_messages(data, limit):
    out_dir = DOCUMENTS_DIR / "messages"
    out_dir.mkdir(parents=True, exist_ok=True)
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    agents = index_by_id(data["agents"])
    vendors = index_by_id(data["vendors"])
    for mr in data["maintenance_requests"][:limit]:
        print(f"[messages] {mr['id']}")
        vendor = vendors.get(mr["resolvingVendorId"]) if mr["resolvingVendorId"] else None
        text = call_gemini(message_prompt(
            mr, tenants[mr["tenantId"]], properties[mr["propertyId"]],
            agents[mr["assigningAgentId"]], vendor))
        (out_dir / f"{mr['id']}.txt").write_text(text)


RUNNERS = {"csv": run_csv, "lease": run_lease_texts, "notes": run_notes, "messages": run_messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="csv,lease,notes,messages",
                         help="Comma-separated: csv,lease,notes,messages")
    parser.add_argument("--limit", type=int, default=3,
                         help="Records per type — start small (default 3) and eyeball output first")
    args = parser.parse_args()

    data = load_instances()
    for t in args.types.split(","):
        t = t.strip()
        if t not in RUNNERS:
            print(f"Unknown type: {t}, skipping")
            continue
        RUNNERS[t](data, args.limit)

    print("\nDone. Manually check output against the messiness checklist "
          "before scaling up --limit.")


if __name__ == "__main__":
    main()