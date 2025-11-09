---
date:
    created: 2025-11-09
authors: [xy]
categories: [TIL]
tags: [regulations]
---

# A simple overview to EU banking regulations

<!-- more -->

*courtesy of GPT5*


EU banking rules look messy because different bodies write different layers: global standards, EU legislation, technical rules, and supervisory practice. But once you stack them properly, the architecture becomes predictable.

Here’s the map you want in your head.

---

## 1. Basel Standards: The Global Blueprint

Everything starts with the **Basel Committee**.
Basel sets global rules for capital, liquidity, leverage, and risk management.

Not law. Just the blueprint.

---

## 2. CRR and CRD: The EU Turns Basel Into Law

### **CRR – Capital Requirements Regulation**

* Directly applicable law.
* Covers the “math”: RWAs, capital ratios, leverage, LCR/NSFR, large exposures, Pillar 3.

### **CRD – Capital Requirements Directive**

* Transposed by each member state.
* Covers governance, internal controls, fit & proper, buffers, Pillar 2, supervisory powers.

Together, **CRR + CRD = EU’s implementation of Basel III**.

---

## 3. EBA: Technical Details and Uniformity

CRR/CRD leave details open.
The **EBA** tightens them with:

* **RTS** – binding technical standards
* **ITS** – binding templates/procedures
* **Guidelines** – supervisory expectations (“comply or explain”)

If CRR/CRD are the skeleton, **EBA provides the exact measurement instructions**.

---

## 4. ECB/SSM and NCAs: How the Rules Are Enforced

Supervision happens here:

* **ECB/SSM** for significant institutions
* **NCAs** for smaller banks

Their methodology is documented in:

* SSM Supervisory Manual
* SREP methodology booklets
* ECB ICAAP/ILAAP Guides
* Thematic reviews

These are not laws; they’re the supervisor’s operating manual.

**This is where models get challenged, assumptions get questioned, and Pillar 2 capital gets set.**

---

## 5. Where Specific Models/Risks Sit — Three Concrete Examples

### ✅ **Example 1: IRRBB (Interest Rate Risk in the Banking Book)**

IRRBB is **Pillar 2**.

* **CRD**: defines IRRBB as a Pillar 2 risk
* **EBA Guidelines**: modelling standards, shock scenarios, behavioural assumptions
* **ECB SREP/ICAAP methodology**: how supervisors score, challenge, and set P2R/P2G

**Placement:**
**CRD → EBA Guidelines → ECB/SSM supervision**
(Outside CRR. Not a Pillar 1 capital formula.)

---

### ✅ **Example 2: IRB Models (Internal Ratings-Based credit risk models)**

These are **Pillar 1** internal models.

* **CRR**: legal basis and hard requirements
* **EBA RTS/Guidelines**: PD/LGD estimation, MoC, downturn LGD, validation
* **ECB TRIM + ongoing model reviews**: permissions, inspections, constraints, and monitoring

**Placement:**
**CRR → EBA RTS/Guidelines → ECB TRIM/Supervisory review**

These **directly impact regulatory capital**.

---

### ✅ **Example 3: IFRS 9 Expected Credit Loss Models**

Different lane.
**IFRS 9 comes from accounting, not Basel.**

* **IASB → IFRS 9 standard**: defines how lifetime PDs, scenarios, and ECL must be computed
* **EU endorsement**: makes IFRS 9 legally binding for reporting
* **EBA/ECB expectations**: supervisors check IFRS 9 model quality because it affects CET1, provisioning, stress tests, and ICAAP

These are **not** Pillar 1 or Pillar 2 models in the CRR/CRD sense, but supervisors care because bad accounting leads to bad prudential numbers.

**Placement:**
**IASB → EU endorsement → EBA/ECB supervisory expectations**

Parallel to prudential rules, but tightly connected.

---

## 6. The Whole Stack in One Line

**Basel → CRR/CRD → EBA RTS/ITS/Guidelines → ECB/NCAs (SREP, ICAAP, model reviews)**
and in parallel:

**IASB → EU-endorsed IFRS 9 → supervisory expectations**

Once this structure clicks, everything — IRRBB, IRB, IFRS 9 models, liquidity rules, buffers — falls neatly into place.
