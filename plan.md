

## Job 1 — validate the pipeline

This is what you’ve been doing recently.

You tested:

* quality-gated detector
* shortlist behavior
* MCC=2 carve-out
* scale risk
* false negatives on confirmed cases

That work answers:

**“Can I trust this pipeline enough to use it?”**

That phase is **not** the same as processing all remaining stars.

## Job 2 — run the pipeline on the remaining stars

This is the actual production-style flow.

That answers:

**“Out of all the K2 stars we still haven’t resolved, which ones become candidate outputs for humans to inspect?”**

That is the part you still have “so many to go” on.

---

## So how do you proceed now?

Not by endlessly validating.

You now know enough to move to an **operational funnel**.

The practical flow should be:

### Stage A — freeze the policy you will use

Pick one policy for the main pass.

Right now that probably means:

* keep **default policy** as the main official path
* keep conditional MCC=2 only as experimental
* do **not** keep changing policy every week

Because you cannot mine thousands of stars while the rules keep moving.

### Stage B — define the unresolved population

Make one master input list:

**“These are the K2 EPICs we still need to triage / process / classify.”**

Then split them into:

* already resolved / already reviewed
* known confirmed calibration cases
* unresolved stars needing model + funnel processing

### Stage C — run the production funnel on the unresolved set

For each remaining EPIC:

1. apply your current `.keras` / CNN triage
2. run detector / period shortlist with the chosen official policy
3. classify the EPIC into:

   * likely candidate
   * quarantine / borderline
   * negative / no useful evidence

### Stage D — create the human-review set

This is the real goal.

You do **not** need to “solve” every star scientifically right now.

You need to create:

* a **best candidate list**
* maybe a **borderline review list**
* and a **negative / excluded pool**

That is the citizen-science handoff.

---

## What you should stop doing

Do **not** think:

**“There are thousands left, so we need to fully understand every edge case before proceeding.”**

You don’t.

That will trap you forever.

Instead think:

**“We know enough to run a stable first-pass production funnel, and we will improve recall in later versions.”**

That is how you get moving.

---

## The real next step

You need to switch from:

**validation mindset**
to
**production triage mindset**

Meaning:

**Use the current best stable policy and start classifying the unresolved stars into candidate / borderline / negative buckets.**

---

## In one sentence

You are not trying to prove every remaining star right now.

You are trying to:

**turn the unresolved K2 pool into a ranked candidate funnel that humans can inspect.** 🌌

---

## So what do I want you to do next?

Make one decision:

**Which policy is the official production pass for the unresolved stars?**

My recommendation:

* **default shortlist policy** as official
* conditional MCC=2 kept as side experimental only
* `.keras` model used for candidate scoring / triage
* output a ranked candidate CSV for human review

That gives you forward motion instead of more loops.

If you want, I’ll map your whole next phase as a simple **“remaining stars → candidate CSV → citizen science”** plan.
