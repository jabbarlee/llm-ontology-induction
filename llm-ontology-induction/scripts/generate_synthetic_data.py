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
import csv
import io
import json
import os
import random
import re
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
CSV_BATCH_SIZE = 15  # records per CSV file — avoids cramming 45+ rows into one response
CSV_VALIDATION_RETRIES = 3  # re-generate (not just re-prompt) if structure check fails

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


def chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def json_to_csv(text):
    """Parse Gemini's JSON response and serialize it to CSV using Python's
    csv module, which quotes comma-containing fields correctly by
    construction.

    Previously Gemini was asked to hand-write CSV directly, but that put two
    instructions in conflict — 'format messily' vs 'maintain valid CSV
    syntax' — and on rows whose description embeds a comma-heavy address,
    messiness won: fields went unquoted and columns split. Asking for JSON
    removes the conflict entirely; the messiness now lives in the values and
    column names, where it belongs, and delimiter correctness is Python's job.

    Returns (csv_text, None) on success, or (None, reason) on failure."""
    cleaned = text.strip()
    # tolerate markdown fences even though the prompt asks for none
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"response was not valid JSON: {e}"

    if not isinstance(data, dict) or "columns" not in data or "rows" not in data:
        return None, "JSON missing required 'columns' / 'rows' keys"

    columns, rows = data["columns"], data["rows"]
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None, "'columns' and 'rows' must both be lists"
    if not columns:
        return None, "'columns' is empty"

    n = len(columns)
    bad = [i for i, r in enumerate(rows) if not isinstance(r, list) or len(r) != n]
    if bad:
        return None, f"{len(bad)} row(s) have wrong element count (expected {n}): rows {bad[:5]}"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    return buf.getvalue(), None


class Stats:
    """Tracks generated/skipped/failed counts per document type for the
    end-of-run summary — the only way to know a mass run actually succeeded
    without manually counting files."""
    def __init__(self):
        self.counts = {}

    def record(self, doc_type, outcome):
        self.counts.setdefault(doc_type, {"generated": 0, "skipped": 0, "failed": 0})
        self.counts[doc_type][outcome] += 1

    def print_summary(self):
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        total_failed = 0
        for doc_type, c in self.counts.items():
            print(f"  {doc_type:12s} generated={c['generated']:3d}  "
                  f"skipped={c['skipped']:3d}  failed={c['failed']:3d}")
            total_failed += c["failed"]
        if total_failed:
            print(f"\n{total_failed} item(s) failed after {MAX_RETRIES} retries each — "
                  f"re-run the same command to retry just those (already-generated "
                  f"files are skipped automatically unless --force is passed).")
        else:
            print("\nAll items generated successfully.")


stats = Stats()


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

def csv_prompt(entity_type, batch, attr_names, ids):
    return f"""You are simulating an export from an old, slightly inconsistent property
management CRM system. Given the structured data below, produce the rows
that such an export would contain.

Return your answer as JSON with exactly this shape:
{{
  "columns": ["ColName1", "ColName2", ...],
  "rows": [
    ["value1", "value2", ...],
    ...
  ]
}}

Rules:
- Do NOT use these exact field names as column names: {attr_names}.
  Invent realistic CRM-style column names instead.
- Every row array MUST have exactly the same number of elements as the
  "columns" array. Use an empty string "" for a blank cell — never omit an
  element, never add an extra one.
- Format currency inconsistently across rows (e.g. "$4,300.00", "4300", "4,300").
- Format dates inconsistently (MM/DD/YYYY, "Jan 2024", YYYY-MM-DD — mix them).
- Leave at least 2 cells blank (empty string) across the batch.
- CRITICAL: do not include ANY internal record ID anywhere in the output —
  this includes IDs of the records themselves ({ids}) AND any reference ID
  inside a record's fields (e.g. an owner/agent/tenant/vendor/lease
  reference). Every such field has already been resolved to a human-readable
  name for you below — use that name, never the raw ID.
- Do not "clean up" or correct any source facts — reproduce them exactly,
  just formatted imperfectly.
- Values may freely contain commas (addresses, formatted amounts) — you do
  NOT need to escape or quote anything, since this is JSON, not CSV.

Source records (all reference IDs already resolved to names):
{json.dumps(batch, indent=2)}

Output only the JSON object, no markdown fences, no commentary."""


def resolve_property_batch(properties, owners):
    out = []
    for p in properties:
        r = dict(p)
        r["ownerName"] = owners[p["ownerId"]]["name"]
        del r["ownerId"], r["id"]
        out.append(r)
    return out


def resolve_lease_batch(leases, owners, tenants, agents):
    id_to_ref = {l["id"]: l for l in leases}
    out = []
    for l in leases:
        r = dict(l)
        r["ownerName"] = owners[l["ownerId"]]["name"]
        r["tenantName"] = tenants[l["tenantId"]]["name"]
        r["negotiatingAgentName"] = agents[l["negotiatingAgentId"]]["name"]
        if l["amendmentLeaseId"]:
            amendment = id_to_ref.get(l["amendmentLeaseId"])
            r["amendmentSummary"] = (f"renews into a new term starting "
                                      f"{amendment['startDate']}") if amendment else None
        else:
            r["amendmentSummary"] = None
        for k in ("id", "ownerId", "tenantId", "propertyId", "negotiatingAgentId", "amendmentLeaseId"):
            del r[k]
        out.append(r)
    return out


def resolve_mr_batch(mrs, tenants, properties, agents, vendors, owners):
    out = []
    for mr in mrs:
        r = dict(mr)
        r["tenantName"] = tenants[mr["tenantId"]]["name"]
        r["propertyAddress"] = properties[mr["propertyId"]]["address"]
        r["assigningAgentName"] = agents[mr["assigningAgentId"]]["name"]
        r["resolvingVendorName"] = vendors[mr["resolvingVendorId"]]["name"] if mr["resolvingVendorId"] else None
        r["approvalNeededFromOwnerName"] = (
            owners[mr["requiresApprovalOwnerId"]]["name"] if mr["requiresApprovalOwnerId"] else None
        )
        for k in ("id", "tenantId", "propertyId", "leaseId", "assigningAgentId",
                  "resolvingVendorId", "requiresApprovalOwnerId"):
            del r[k]
        out.append(r)
    return out


def lease_prompt(lease, owner, tenant, prop, amendment_summary):
    lease_for_prompt = {k: v for k, v in lease.items() if k != "amendmentLeaseId"}
    if amendment_summary:
        lease_for_prompt["amendmentSummary"] = amendment_summary
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
- If amendmentSummary is present, reference it only via the plain-language
  description given — e.g. "this lease renews into a new term starting
  [date]" — never invent or reference a record number, lease ID, or
  document number of any kind, even one that sounds plausible.
- Omit one minor, non-critical detail a real lease might leave implicit.
- Do not include any internal record IDs anywhere in the text.
- Keep the facts accurate — do not alter dates, dollar amounts, or names.

Source records:
Lease: {json.dumps(lease_for_prompt, indent=2)}
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

def run_csv(data, limit, force):
    out_dir = DOCUMENTS_DIR / "csv_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    owners = index_by_id(data["owners"])
    tenants = index_by_id(data["tenants"])
    agents = index_by_id(data["agents"])
    properties = index_by_id(data["properties"])
    vendors = index_by_id(data["vendors"])

    jobs = [
        ("properties", "Property", data["properties"][:limit],
         lambda batch: resolve_property_batch(batch, owners)),
        ("leases", "Lease", data["leases"][:limit],
         lambda batch: resolve_lease_batch(batch, owners, tenants, agents)),
        ("maintenance_requests", "MaintenanceRequest", data["maintenance_requests"][:limit],
         lambda batch: resolve_mr_batch(batch, tenants, properties, agents, vendors, owners)),
    ]

    for name, entity_type, records, resolver in jobs:
        if not records:
            continue
        batches = list(chunk(records, CSV_BATCH_SIZE))
        for i, batch in enumerate(batches, start=1):
            out_file = out_dir / f"{name}_export_{i:02d}.csv"
            if out_file.exists() and not force:
                print(f"[csv] {out_file.name} already exists, skipping (use --force to regenerate)")
                stats.record("csv", "skipped")
                continue
            print(f"[csv] {entity_type} batch {i}/{len(batches)}: {len(batch)} records -> {out_file.name}")
            try:
                ids = [r["id"] for r in batch]  # from RAW batch, before resolver strips 'id'
                resolved = resolver(batch)
                prompt = csv_prompt(entity_type, resolved, REAL_ATTR_NAMES[entity_type], ids)

                text, reason = None, None
                for attempt in range(1, CSV_VALIDATION_RETRIES + 1):
                    candidate = call_gemini(prompt)
                    converted, reason = json_to_csv(candidate)
                    if converted is not None:
                        text = converted
                        break
                    print(f"  CSV conversion failed (attempt {attempt}/{CSV_VALIDATION_RETRIES}): {reason}")
                if text is None:
                    debug_file = out_dir / f"{out_file.stem}.rejected.txt"
                    debug_file.write_text(candidate)
                    raise RuntimeError(f"CSV never passed conversion after "
                                        f"{CSV_VALIDATION_RETRIES} attempts: {reason} "
                                        f"(last rejected attempt saved to {debug_file.name} for inspection)")

                out_file.write_text(text)
                stats.record("csv", "generated")
            except RuntimeError as e:
                print(f"  FAILED: {out_file.name} — {e}")
                stats.record("csv", "failed")


def run_lease_texts(data, limit, force):
    out_dir = DOCUMENTS_DIR / "lease_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    owners = index_by_id(data["owners"])
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    leases_by_id = index_by_id(data["leases"])
    for lease in data["leases"][:limit]:
        out_file = out_dir / f"{lease['id']}.txt"
        if out_file.exists() and not force:
            print(f"[lease] {lease['id']} already exists, skipping")
            stats.record("lease", "skipped")
            continue
        print(f"[lease] {lease['id']}")
        try:
            amendment_summary = None
            if lease["amendmentLeaseId"]:
                amendment = leases_by_id.get(lease["amendmentLeaseId"])
                if amendment:
                    amendment_summary = f"renews into a new term starting {amendment['startDate']}"
            text = call_gemini(lease_prompt(
                lease, owners[lease["ownerId"]], tenants[lease["tenantId"]],
                properties[lease["propertyId"]], amendment_summary))
            out_file.write_text(text)
            stats.record("lease", "generated")
        except RuntimeError as e:
            print(f"  FAILED: {lease['id']} — {e}")
            stats.record("lease", "failed")


def run_notes(data, limit, force):
    out_dir = DOCUMENTS_DIR / "notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    agents = index_by_id(data["agents"])
    vendors = index_by_id(data["vendors"])
    for mr in data["maintenance_requests"][:limit]:
        out_file = out_dir / f"{mr['id']}.txt"
        if out_file.exists() and not force:
            print(f"[notes] {mr['id']} already exists, skipping")
            stats.record("notes", "skipped")
            continue
        print(f"[notes] {mr['id']}")
        try:
            vendor = vendors.get(mr["resolvingVendorId"]) if mr["resolvingVendorId"] else None
            text = call_gemini(notes_prompt(
                mr, tenants[mr["tenantId"]], properties[mr["propertyId"]],
                agents[mr["assigningAgentId"]], vendor))
            out_file.write_text(text)
            stats.record("notes", "generated")
        except RuntimeError as e:
            print(f"  FAILED: {mr['id']} — {e}")
            stats.record("notes", "failed")


def run_messages(data, limit, force):
    out_dir = DOCUMENTS_DIR / "messages"
    out_dir.mkdir(parents=True, exist_ok=True)
    tenants = index_by_id(data["tenants"])
    properties = index_by_id(data["properties"])
    agents = index_by_id(data["agents"])
    vendors = index_by_id(data["vendors"])
    for mr in data["maintenance_requests"][:limit]:
        out_file = out_dir / f"{mr['id']}.txt"
        if out_file.exists() and not force:
            print(f"[messages] {mr['id']} already exists, skipping")
            stats.record("messages", "skipped")
            continue
        print(f"[messages] {mr['id']}")
        try:
            vendor = vendors.get(mr["resolvingVendorId"]) if mr["resolvingVendorId"] else None
            text = call_gemini(message_prompt(
                mr, tenants[mr["tenantId"]], properties[mr["propertyId"]],
                agents[mr["assigningAgentId"]], vendor))
            out_file.write_text(text)
            stats.record("messages", "generated")
        except RuntimeError as e:
            print(f"  FAILED: {mr['id']} — {e}")
            stats.record("messages", "failed")


RUNNERS = {"csv": run_csv, "lease": run_lease_texts, "notes": run_notes, "messages": run_messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="csv,lease,notes,messages",
                         help="Comma-separated: csv,lease,notes,messages")
    parser.add_argument("--limit", type=int, default=3,
                         help="Max records per type. Start with 3 and eyeball output "
                              "before scaling up. Set higher than your total record "
                              "count (e.g. 9999) to process everything.")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate files even if they already exist. Without "
                              "this flag, already-generated files are skipped — makes "
                              "the script safely re-runnable after a partial failure.")
    args = parser.parse_args()

    data = load_instances()
    for t in args.types.split(","):
        t = t.strip()
        if t not in RUNNERS:
            print(f"Unknown type: {t}, skipping")
            continue
        RUNNERS[t](data, args.limit, args.force)

    stats.print_summary()


if __name__ == "__main__":
    main()