# Schema Notes — Gold Schema, Corpus B (Commercial Real Estate)

**Status: first-pass draft.** Per the execution plan, the final justification text should be written/reviewed by you before this goes to Dr. Oncu — this draft captures the reasoning worked out across the design conversation so you're not starting from a blank page.

---

## Classes

**Party** — Superclass over Owner/Tenant/Agent/Vendor. Introduced because all four share attributes (`name`, `contactInfo`) that would otherwise be duplicated four times, and because the `hasParty`/`represents` relations need one general type constraint that every subclass satisfies via inheritance.

**Owner** — Owns Property; the landlord side of a Lease.

**Tenant** — Occupies leased space; reports MaintenanceRequests; the lessee side of a Lease.

**Agent** — Deliberately covers both leasing (lists Property, negotiates Lease) and light property-management coordination (assigns MaintenanceRequest). We did not model a separate PropertyManager class — scope decision to keep the schema lean, not an oversight.

**Vendor** — External maintenance/service provider. Kept distinct from Owner: a Vendor services a Property, it does not own one (this distinction was the first modeling error caught and corrected during design).

**Property** — The physical asset. Split into three subclasses (below) rather than left flat with a `propertyType` attribute, because each subtype carries attributes with no meaning on the others.

**OfficeProperty / RetailProperty / IndustrialProperty** — Each has attributes meaningless on the other two (e.g. `numberOfFloors` vs. `storefrontType` vs. `loadingDocks`). This is the test applied to justify every taxonomic split in this schema: does the subtype need its own structure, or just a different label on the same structure?

**Lease** — The contract/state binding Owner, Tenant, and Property. Renewals/escalations are modeled as a new Lease instance linked via `hasAmendment`, not a separate LeaseAmendment class — reuses the existing Lease structure instead of duplicating it.

**MaintenanceRequest** — A hub entity: connects Tenant (reporter), Property (subject), Agent (coordinator), Vendor (resolver), and optionally Owner (approver). Needs its own attributes (`status`, `priority`, dates) that don't belong on any single party — the signal it's a real class, not just a relation.

### Considered and cut

**LeaseClause** — Considered late in the design process as a way to capture maintenance-responsibility allocation (who pays for what repair). Cut for now: the surface area for "lease clause types" tends to expand quickly (indemnification, assignment, insurance, use restriction...) and risks reopening the exact legal-clause-taxonomy bottleneck the synthetic-corpus approach was chosen to avoid. Revisit only if a later stage specifically needs responsibility-allocation as a first-class fact.

**Institutional/fund-level structure, acquisition/financing pipeline, deep legal clause taxonomy** — Out of scope from the start. Different sub-domain and expertise than day-to-day leasing/operations; doesn't generate the messy-document types (leases, listings, notes, message threads) this corpus needs.

---

## Relations

| Relation | Type | Justification |
|---|---|---|
| `Owner -> owns -> Property` | non-tax. | Core ownership fact. |
| `Lease -> hasParty -> Owner` | non-tax. | Role-specific edge (not collapsed to `hasParty -> Party`) — a Lease structurally requires one of *each* role, which a single general edge can't express. |
| `Lease -> hasParty -> Tenant` | non-tax. | Same reasoning as above. |
| `Lease -> covers -> Property` | non-tax. | Without this, Lease has two parties but no connection to what it's actually about — caught as a gap during design. |
| `Agent -> lists -> Property` | non-tax. | Listing/brokerage role. |
| `Tenant -> reports -> MaintenanceRequest` | non-tax. | Realistic reporting path (tenant occupies the space). |
| `MaintenanceRequest -> concerns -> Property` | non-tax. | Kept general (not split per Property subtype) — no second role is being asserted here, so the superclass range already covers all subtypes via inheritance. Contrast with `hasParty` above. |
| `Agent -> assigns -> MaintenanceRequest` | non-tax. | Coordination step between report and resolution. |
| `Vendor -> resolves -> MaintenanceRequest` | non-tax. | Per-incident resolution. |
| `Agent -> represents -> Owner` | non-tax. | Listing-agent role; split by side (see next row) for the same reason `hasParty` was split. |
| `Agent -> represents -> Tenant` | non-tax. | Tenant-rep broker role — distinct from the listing-agent role above. |
| `Agent -> negotiates -> Lease` | non-tax. | The missing link between "agent lists property" and "lease gets signed." |
| `Owner -> engages -> Vendor` | non-tax. | Standing service relationship (who's on contract) — distinct from the per-incident `resolves` edge. |
| `Lease -> hasAmendment -> Lease` | non-tax., self-referential | One Lease instance points to the instance that renews/supersedes it, rather than introducing a separate class. |
| `MaintenanceRequest -> requiresApproval -> Owner` | non-tax. | Large repairs needing owner sign-off; also a genuinely implicit relationship in source text (e.g. "waiting on landlord to sign off"), useful for testing extraction. |

**Totals:** 11 classes (2 taxonomic splits: Party -> 4 children, Property -> 3 children), 30 attributes, 15 non-taxonomic relations. Within every target range from the execution plan (10-15 classes, 25-40 attributes, 15-25 relations).

*(Corrected 2026-08-07: this line previously read "10 classes ... 27 attributes" — a stale miscount caught while building the Step 4 evaluation harness, whose loader counts classes/attributes directly off `gold_schema.ttl` rather than off this summary. The attribute table above always listed 30 rows; the class list above always named 11 classes (Party, Owner, Tenant, Agent, Vendor, Property, OfficeProperty, RetailProperty, IndustrialProperty, Lease, MaintenanceRequest) — only this totals line was out of sync. See `eval/DECISIONS.md`'s 2026-08-07 "class/attribute count correction" addendum.)*

---

## Attributes — finalized value sets

Every attribute below has a concrete type or enum — no leftover vague `string` fields left in as placeholders.

| Class | Attribute | Type |
|---|---|---|
| Party (shared) | `name` | string |
| Party (shared) | `contactInfo` | string |
| Owner | `ownerType` | enum: individual / LLC / trust / corporation |
| Owner | `taxId` | string |
| Tenant | `businessName` | string |
| Tenant | `industryType` | enum: retail / office-professional / restaurant-food service / medical / warehouse-logistics / other |
| Agent | `licenseNumber` | string |
| Agent | `brokerage` | string |
| Vendor | `serviceType` | enum: HVAC / plumbing / electrical / landscaping / general contractor / cleaning / pest control |
| Vendor | `insured` | boolean |
| Vendor | `insuranceExpiryDate` | date |
| Property (shared) | `address` | string |
| Property (shared) | `squareFootage` | integer |
| OfficeProperty | `numberOfFloors` | integer |
| OfficeProperty | `parkingSpaces` | integer |
| RetailProperty | `storefrontType` | enum: strip mall / standalone / mall inline / street-front |
| RetailProperty | `footTraffic` | enum: low / medium / high |
| IndustrialProperty | `ceilingHeight` | decimal (feet) |
| IndustrialProperty | `loadingDocks` | integer |
| Lease | `startDate` | date |
| Lease | `endDate` | date — original contractual end; current effective end is whatever the latest `hasAmendment` Lease states |
| Lease | `rentAmount` | decimal |
| Lease | `escalationType` | enum: fixed percentage / CPI-indexed / none |
| Lease | `securityDeposit` | decimal |
| MaintenanceRequest | `status` | enum: open / assigned / in-progress / resolved |
| MaintenanceRequest | `priority` | enum: low / medium / high / emergency |
| MaintenanceRequest | `dateReported` | date |
| MaintenanceRequest | `dateResolved` | date |
| MaintenanceRequest | `description` | string |
| MaintenanceRequest | `estimatedCost` | decimal — added specifically to give `requiresApproval` an underlying signal (e.g. above a threshold typically needs owner sign-off) rather than an ungrounded relation |

Considered adding a second attribute layer (`mailingAddress`/`entityState` on Owner; `numberOfEmployees`/`creditRating` on Tenant; `specialty`/`yearsExperience` on Agent; `rating`/`avgResponseTimeHours` on Vendor) but decided against it — the above set was judged sufficiently complete without padding.