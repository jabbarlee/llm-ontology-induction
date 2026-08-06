"""
Step 2 — Clean Instance Layer Generator
Commercial Real Estate / Property Management Ontology

Generates internally-consistent, CLEAN structured instance data straight from
the gold schema (schema/gold_schema.ttl). No imperfection, no renamed fields,
no missing values — that's Step 3's job (via a different model, per the
circularity rule). This script's only job is: does every reference resolve,
and is every fact consistent across files.

Output: one JSON file per entity type, written to ./data/instances/

Run with: python generate_instances.py
"""

import json
import random
import os
from datetime import date, timedelta
# pyrefly: ignore [missing-import]
from faker import Faker

random.seed(42)      # reproducibility — change/remove if you want fresh runs
fake = Faker()
Faker.seed(42)       # Faker has its own seed, separate from stdlib random

OUT_DIR = "data/instances"

# ---------------------------------------------------------------------------
# Enums / value sets — must exactly match schema/gold_schema.ttl comments
# ---------------------------------------------------------------------------

OWNER_TYPES = ["individual", "LLC", "trust", "corporation"]
INDUSTRY_TYPES = ["retail", "office-professional", "restaurant-food service",
                   "medical", "warehouse-logistics", "other"]
VENDOR_SERVICE_TYPES = ["HVAC", "plumbing", "electrical", "landscaping",
                         "general contractor", "cleaning", "pest control"]
STOREFRONT_TYPES = ["strip mall", "standalone", "mall inline", "street-front"]
FOOT_TRAFFIC = ["low", "medium", "high"]
ESCALATION_TYPES = ["fixed percentage", "CPI-indexed", "none"]
MR_STATUS = ["open", "assigned", "in-progress", "resolved"]
MR_PRIORITY = ["low", "medium", "high", "emergency"]
PROPERTY_TYPES = ["office", "retail", "industrial"]

# Internal-only mapping (not a schema attribute) used to pick a sensible
# vendor for a maintenance request — keeps the data logically consistent
# (an HVAC vendor doesn't get assigned a plumbing issue).
ISSUE_TO_SERVICE_TYPE = {
    "HVAC": ["AC unit not cooling", "heating not turning on", "thermostat malfunction"],
    "plumbing": ["leaking pipe under sink", "toilet not flushing", "water heater failure"],
    "electrical": ["flickering lights", "outlet not working", "breaker tripping repeatedly"],
    "landscaping": ["overgrown parking lot median", "irrigation system broken", "dead trees need removal"],
    "general contractor": ["drywall damage in lobby", "ceiling tile water damage", "door frame needs repair"],
    "cleaning": ["carpet stains in common area", "windows need exterior cleaning"],
    "pest control": ["rodent activity in storage room", "ant infestation near breakroom"],
}

APPROVAL_THRESHOLD = 1000.00  # estimatedCost above this requires Owner sign-off

BUSINESS_SUFFIXES = ["LLC", "Group", "Partners", "Holdings", "Enterprises", "& Associates"]

# Real Houston-area street names, kept alongside Faker's number/city output so
# addresses stay locale-plausible for the domain rather than fully generic.
STREET_NAMES = ["Westheimer Rd", "Richmond Ave", "Post Oak Blvd", "Kirby Dr", "Shepherd Dr",
                "Washington Ave", "Studemont St", "Yale St", "Heights Blvd", "Fannin St",
                "Main St", "Gessner Rd", "Bellaire Blvd", "Beltway 8 Frontage", "Hempstead Rd"]


def pid(prefix, n):
    return f"{prefix}-{n:03d}"


def rand_name():
    return fake.name()


def rand_business_name():
    return f"{fake.last_name()} {random.choice(BUSINESS_SUFFIXES)}"


def rand_email(local_hint, domain_hint):
    return f"{local_hint.lower().replace(' ', '.')}@{domain_hint.lower().replace(' ', '')}.com"


def rand_date(start_year=2022, end_year=2026):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def add_months(d_str, months):
    d = date.fromisoformat(d_str)
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day).isoformat()


# ---------------------------------------------------------------------------
# Generators, in dependency order
# ---------------------------------------------------------------------------

def gen_owners(n=18):
    owners = []
    for i in range(1, n + 1):
        otype = random.choice(OWNER_TYPES)
        name = rand_name() if otype == "individual" else rand_business_name()
        owners.append({
            "id": pid("own", i),
            "name": name,
            "contactInfo": rand_email(name.split()[0], f"owner{i}mail"),
            "ownerType": otype,
            "taxId": f"{random.randint(10,99)}-{random.randint(1000000,9999999)}",
        })
    return owners


def gen_agents(n=14):
    agents = []
    for i in range(1, n + 1):
        name = rand_name()
        agents.append({
            "id": pid("agt", i),
            "name": name,
            "contactInfo": rand_email(name.replace(' ', '.'), "crebrokerage"),
            "licenseNumber": f"TX-{random.randint(100000,999999)}",
            "brokerage": random.choice(["Summit Commercial Realty", "Bayou City Brokers",
                                         "Anchor Point CRE", "Lonestar Property Advisors"]),
        })
    return agents


def gen_vendors(n=12):
    vendors = []
    # ensure every service type has at least one vendor
    service_cycle = VENDOR_SERVICE_TYPES * ((n // len(VENDOR_SERVICE_TYPES)) + 1)
    for i in range(1, n + 1):
        name = f"{fake.last_name()} {random.choice(['Services','Solutions','Co.','Maintenance'])}"
        insured = random.random() > 0.1  # ~90% insured
        vendors.append({
            "id": pid("ven", i),
            "name": name,
            "contactInfo": rand_email(f"dispatch{i}", name.split()[0]),
            "serviceType": service_cycle[i - 1],
            "insured": insured,
            "insuranceExpiryDate": rand_date(2026, 2028) if insured else None,
        })
    return vendors


def gen_properties(owners, n=45):
    properties = []
    type_counts = {"office": n // 3, "retail": n // 3, "industrial": n - 2 * (n // 3)}
    idx = 1
    for ptype, count in type_counts.items():
        for _ in range(count):
            owner = random.choice(owners)
            base = {
                "id": pid("prop", idx),
                "propertyType": ptype,
                "address": f"{random.randint(100,9999)} {random.choice(STREET_NAMES)}, Houston, TX",
                "squareFootage": random.randint(1500, 60000),
                "ownerId": owner["id"],
            }
            if ptype == "office":
                base.update({
                    "numberOfFloors": random.randint(1, 12),
                    "parkingSpaces": random.randint(10, 300),
                })
            elif ptype == "retail":
                base.update({
                    "storefrontType": random.choice(STOREFRONT_TYPES),
                    "footTraffic": random.choice(FOOT_TRAFFIC),
                })
            else:  # industrial
                base.update({
                    "ceilingHeight": round(random.uniform(14.0, 40.0), 1),
                    "loadingDocks": random.randint(1, 20),
                })
            properties.append(base)
            idx += 1
    return properties


def gen_tenants(n=35):
    tenants = []
    for i in range(1, n + 1):
        biz = rand_business_name()
        name = rand_name()
        tenants.append({
            "id": pid("ten", i),
            "name": name,
            "contactInfo": rand_email(f"contact{i}", biz.split()[0]),
            "businessName": biz,
            "industryType": random.choice(INDUSTRY_TYPES),
        })
    return tenants


def gen_leases(owners, tenants, properties, agents):
    """One primary lease per property (roughly), some with an amendment chain."""
    leases = []
    idx = 1
    for prop in properties:
        if random.random() > 0.15:  # ~85% of properties currently leased
            tenant = random.choice(tenants)
            owner = next(o for o in owners if o["id"] == prop["ownerId"])
            agent = random.choice(agents)
            start = rand_date(2022, 2024)
            end = add_months(start, random.choice([12, 24, 36, 60]))
            original = {
                "id": pid("lease", idx),
                "ownerId": owner["id"],
                "tenantId": tenant["id"],
                "propertyId": prop["id"],
                "negotiatingAgentId": agent["id"],
                "startDate": start,
                "endDate": end,
                "rentAmount": round(random.uniform(2000, 45000), 2),
                "escalationType": random.choice(ESCALATION_TYPES),
                "securityDeposit": round(random.uniform(2000, 45000), 2),
                "amendmentLeaseId": None,  # filled below if renewed
            }
            leases.append(original)
            idx += 1

            # ~30% of leases have a renewal/escalation as a second Lease instance
            if random.random() < 0.3:
                renew_start = end
                renew_end = add_months(renew_start, random.choice([12, 24, 36]))
                renewal_rent = round(original["rentAmount"] * random.uniform(1.03, 1.10), 2)
                renewal = {
                    "id": pid("lease", idx),
                    "ownerId": owner["id"],
                    "tenantId": tenant["id"],
                    "propertyId": prop["id"],
                    "negotiatingAgentId": agent["id"],
                    "startDate": renew_start,
                    "endDate": renew_end,
                    "rentAmount": renewal_rent,
                    "escalationType": original["escalationType"],
                    "securityDeposit": original["securityDeposit"],
                    "amendmentLeaseId": None,
                }
                original["amendmentLeaseId"] = renewal["id"]
                leases.append(renewal)
                idx += 1
    return leases


def gen_maintenance_requests(leases, properties, agents, vendors, n=65):
    """
    Built FROM the lease list, not from tenants/properties independently.
    This guarantees every reporting tenant is actually leasing the property
    they're reporting on — a real internal-consistency requirement, not
    just an ID-existence check.
    """
    requests = []
    properties_by_id = {p["id"]: p for p in properties}

    if not leases:
        return requests  # no active leases, nothing to generate against

    for i in range(1, n + 1):
        lease = random.choice(leases)
        prop = properties_by_id[lease["propertyId"]]

        service_type = random.choice(VENDOR_SERVICE_TYPES)
        issue_text = random.choice(ISSUE_TO_SERVICE_TYPE[service_type])
        agent = random.choice(agents)
        status = random.choices(MR_STATUS, weights=[0.15, 0.15, 0.2, 0.5])[0]
        priority = random.choices(MR_PRIORITY, weights=[0.3, 0.35, 0.25, 0.1])[0]
        est_cost = round(random.uniform(50, 6000), 2)

        reported = rand_date(2025, 2026)
        resolved = None
        vendor_id = None
        matching_vendors = [v["id"] for v in vendors if v["serviceType"] == service_type]

        if status in ("assigned", "in-progress", "resolved"):
            vendor_id = random.choice(matching_vendors) if matching_vendors else None
        if status == "resolved":
            # realistic resolution lag, not same-day
            resolved = (date.fromisoformat(reported) + timedelta(days=random.randint(1, 21))).isoformat()

        requires_approval_owner_id = None
        if est_cost > APPROVAL_THRESHOLD:
            requires_approval_owner_id = lease["ownerId"]

        requests.append({
            "id": pid("mr", i),
            "tenantId": lease["tenantId"],
            "propertyId": prop["id"],
            "leaseId": lease["id"],
            "assigningAgentId": agent["id"],
            "resolvingVendorId": vendor_id,
            "requiresApprovalOwnerId": requires_approval_owner_id,
            "status": status,
            "priority": priority,
            "dateReported": reported,
            "dateResolved": resolved,
            "description": f"{issue_text} at {prop['address']}.",
            "estimatedCost": est_cost,
        })
    return requests


# ---------------------------------------------------------------------------
# Verification — the Step 2 "check" from the roadmap
# ---------------------------------------------------------------------------

def verify(owners, agents, vendors, properties, tenants, leases, requests):
    errors = []

    def ids(items):
        return {x["id"] for x in items}

    owner_ids, agent_ids, vendor_ids = ids(owners), ids(agents), ids(vendors)
    property_ids, tenant_ids, lease_ids = ids(properties), ids(tenants), ids(leases)

    # duplicate ID check across all entity sets combined
    all_ids = list(owner_ids) + list(agent_ids) + list(vendor_ids) + \
        list(property_ids) + list(tenant_ids) + list(lease_ids) + \
        [r["id"] for r in requests]
    if len(all_ids) != len(set(all_ids)):
        errors.append("Duplicate IDs found across entity sets.")

    for p in properties:
        if p["ownerId"] not in owner_ids:
            errors.append(f"Property {p['id']} references missing owner {p['ownerId']}")

    for l in leases:
        if l["ownerId"] not in owner_ids:
            errors.append(f"Lease {l['id']} references missing owner {l['ownerId']}")
        if l["tenantId"] not in tenant_ids:
            errors.append(f"Lease {l['id']} references missing tenant {l['tenantId']}")
        if l["propertyId"] not in property_ids:
            errors.append(f"Lease {l['id']} references missing property {l['propertyId']}")
        if l["negotiatingAgentId"] not in agent_ids:
            errors.append(f"Lease {l['id']} references missing agent {l['negotiatingAgentId']}")
        if l["amendmentLeaseId"] and l["amendmentLeaseId"] not in lease_ids:
            errors.append(f"Lease {l['id']} amendment points to missing lease {l['amendmentLeaseId']}")

    leases_by_id = {l["id"]: l for l in leases}
    for r in requests:
        if r["tenantId"] not in tenant_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing tenant {r['tenantId']}")
        if r["propertyId"] not in property_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing property {r['propertyId']}")
        if r["assigningAgentId"] not in agent_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing agent {r['assigningAgentId']}")
        if r["resolvingVendorId"] and r["resolvingVendorId"] not in vendor_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing vendor {r['resolvingVendorId']}")
        if r["requiresApprovalOwnerId"] and r["requiresApprovalOwnerId"] not in owner_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing approval owner {r['requiresApprovalOwnerId']}")
        # logical coherence: the reporting tenant must actually be the tenant on the
        # referenced lease, and the property must match that same lease's property
        if r.get("leaseId") not in lease_ids:
            errors.append(f"MaintenanceRequest {r['id']} references missing lease {r.get('leaseId')}")
        else:
            l = leases_by_id[r["leaseId"]]
            if l["tenantId"] != r["tenantId"]:
                errors.append(f"MaintenanceRequest {r['id']} tenant does not match its lease's tenant")
            if l["propertyId"] != r["propertyId"]:
                errors.append(f"MaintenanceRequest {r['id']} property does not match its lease's property")

    # signal checks: make sure requiresApproval isn't vacuous either direction
    approval_count = sum(1 for r in requests if r["requiresApprovalOwnerId"])
    if approval_count == 0 or approval_count == len(requests):
        errors.append("requiresApproval signal is vacuous (all-or-nothing) — check estimatedCost distribution.")

    return errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    owners = gen_owners()
    agents = gen_agents()
    vendors = gen_vendors()
    properties = gen_properties(owners)
    tenants = gen_tenants()
    leases = gen_leases(owners, tenants, properties, agents)
    requests = gen_maintenance_requests(leases, properties, agents, vendors)

    errors = verify(owners, agents, vendors, properties, tenants, leases, requests)
    if errors:
        print(f"VERIFICATION FAILED — {len(errors)} issue(s):")
        for e in errors[:20]:
            print(" -", e)
        raise SystemExit(1)

    files = {
        "owners.json": owners,
        "agents.json": agents,
        "vendors.json": vendors,
        "properties.json": properties,
        "tenants.json": tenants,
        "leases.json": leases,
        "maintenance_requests.json": requests,
    }
    for fname, data in files.items():
        with open(os.path.join(OUT_DIR, fname), "w") as f:
            json.dump(data, f, indent=2)

    print("Verification passed. Generated:")
    for fname, data in files.items():
        print(f"  {fname}: {len(data)} records")


if __name__ == "__main__":
    main()