import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const decisionPath = path.join(root, "gatevetter_v0_2_deep_review_manual_decisions.csv");
const reportPath = path.join(root, "gatevetter_v0_2_deep_review_reconciliation_report.csv");
const outputCsv = path.join(root, "gatevetter_v0_3_epic_traceability.csv");
const outputSummary = path.join(root, "gatevetter_v0_3_epic_traceability_summary.txt");
const outputDir = path.join(root, "outputs", "gatevetter_v0_3_epic_traceability");
const outputXlsx = path.join(outputDir, "gatevetter_v0_3_epic_traceability.xlsx");
const previewPath = path.join(outputDir, "gatevetter_v0_3_epic_traceability_preview.png");

const allowedClassifications = new Set([
  "no_change_required",
  "recording_change_only",
  "stage_g_logic_change_required",
  "unsupported",
]);

const annotations = {
  EPIC_211624954: {
    decisive_failure_mode: "Untrusted period with a possible near-equal secondary at 2P; remains an uncertain positive hold and must not be promoted.",
    decisive_period_hypothesis: "P is unresolved; 2P = 3.07842 d is the concerning competing hypothesis, not a confirmed EB solution.",
    proposed_v0_3_diagnostic: "Record the 2P competing-secondary result and explicit non-promotion outcome in the rule trace; retain manual hold handling.",
    change_classification: "recording_change_only",
    notes: "Possible 2P EB concern is insufficient to alter the uncertain_hold_positive label.",
  },
  EPIC_211687388: {
    decisive_failure_mode: "Competing-period EB/variable evidence, including a stronger coherent secondary at 2P.",
    decisive_period_hypothesis: "2P = 24.14212 d competing EB hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 blocked on nominal-period odd/even evidence; the decisive manual interpretation came from competing-period review.",
  },
  EPIC_211768304: {
    decisive_failure_mode: "Coherent near-equal secondary structure at 2P with odd/even concern, elevated OOT variability, and an untrusted period.",
    decisive_period_hypothesis: "2P = 19.6963 d weak EB/stellar false-positive hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "The listed validation_summary.json is absent locally; validation_summary_1.json and the EPIC period_comparison.csv provide the source evidence.",
  },
  EPIC_211959909: {
    decisive_failure_mode: "Strong coherent secondary/EB interpretation at 2P.",
    decisive_period_hypothesis: "2P = 9.69822 d competing EB hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 already blocked promotion through high OOT at P; the 2P evidence supplies the decisive failure mode.",
  },
  EPIC_211996306: {
    decisive_failure_mode: "Weak, unreliable event evidence with excessive surrounding variability.",
    decisive_period_hypothesis: "No credible period; P/2, P, and 2P remain weak or variability-dominated.",
    proposed_v0_3_diagnostic: "Retain the existing OOT/depth and weak-primary handling; no new behavioral diagnostic is supported by this review.",
    change_classification: "no_change_required",
    notes: "Although 2P has a concerning secondary metric, the final manual label is noise/artifact and must not be changed to EB.",
  },
  EPIC_211351798: {
    decisive_failure_mode: "Coherent near-equal primary and secondary eclipses at 2P.",
    decisive_period_hypothesis: "2P = 17.76198 d eclipsing-binary hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "The saved P is trusted, but the 2P morphology is decisive for the manual EB label.",
  },
  EPIC_211912465: {
    decisive_failure_mode: "Period-inconsistent and insufficiently reproducible event evidence across P/2, P, and 2P.",
    decisive_period_hypothesis: "No credible period; P/2 = 6.005 d, P = 12.01 d, and 2P = 24.02 d are inconsistent and sparsely supported.",
    proposed_v0_3_diagnostic: "Record cross-period inconsistency, event counts, and insufficient-baseline status as structured trace fields; retain conservative blocking behavior.",
    change_classification: "recording_change_only",
    notes: "A large 2P secondary metric is not treated as a reliable EB decision because the manual review found the competing-event selection implausible.",
  },
  EPIC_211485867: {
    decisive_failure_mode: "Coherently repeated, near-equal primary and secondary eclipses at 2P.",
    decisive_period_hypothesis: "2P = 11.5471039646 d eclipsing-binary hypothesis; nominal P is the half-period fold.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 conservatively held the target but did not encode the decisive 2P EB interpretation.",
  },
  EPIC_211387236: {
    decisive_failure_mode: "Strong coherent, near-equal primary and secondary eclipses at approximately 16.96 d.",
    decisive_period_hypothesis: "2P = 16.9583 d eclipsing-binary hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 conservatively held the target; 2P provides the decisive EB evidence.",
  },
  EPIC_211889082: {
    decisive_failure_mode: "Broad coherent stellar modulation with unresolved period, odd/even concern, and near-equal structure at 2P.",
    decisive_period_hypothesis: "2P = 3.7504047421 d variable/contact-binary hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "The review supports a variable/contact-binary interpretation, not a planet-like promotion.",
  },
  EPIC_211816343: {
    decisive_failure_mode: "Broad periodic modulation, very high OOT/depth, and near-equal structures at 2P.",
    decisive_period_hypothesis: "2P = 1.29128 d EB/stellar-variable hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 correctly blocked promotion via OOT/depth; the explicit 2P veto would encode the decisive interpretation.",
  },
  EPIC_212029934: {
    decisive_failure_mode: "Coherent broad stellar modulation with near-equal primary and secondary features at 2P.",
    decisive_period_hypothesis: "2P = 2.49268 d contact-binary or ellipsoidal-variable hypothesis.",
    proposed_v0_3_diagnostic: "Mandatory 2P competing-EB veto using the secondary assessment, secondary/primary depth ratio, secondary SNR, odd/even, coherence, and OOT context.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 correctly blocked promotion via OOT/depth; 2P evidence supplies the specific stellar-false-positive mode.",
  },
  EPIC_211845034: {
    decisive_failure_mode: "Deep broad EB/variable-like morphology with strong periodic raw variability and non-planet-like event structure.",
    decisive_period_hypothesis: "No single decisive period was assigned by the manual review; P = 1.33061 d remained untrusted.",
    proposed_v0_3_diagnostic: "Retain existing OOT/depth blocking; no additional behavioral change is justified from the recorded manual decision.",
    change_classification: "no_change_required",
    notes: "Earlier batch-10 review was reused with its original provenance; the final label is unchanged.",
  },
  EPIC_211431812: {
    decisive_failure_mode: "Weak period-inconsistent signal with elevated OOT variability, unstable event depths, and no robust EB or planetary pattern.",
    decisive_period_hypothesis: "No credible period; P/2 = 5.277515 d, P = 10.55503 d, and 2P = 21.11006 d remain variability-dominated or weak.",
    proposed_v0_3_diagnostic: "Retain the existing OOT/depth block; no new behavioral diagnostic is supported by this review.",
    change_classification: "no_change_required",
    notes: "Do not reinterpret the concerning 2P secondary metric as an EB label; the manual decision is noise/artifact.",
  },
  EPIC_211696209: {
    decisive_failure_mode: "Numerically suspicious nominal-period SNR plus a stronger 2P secondary, severe odd/even mismatch, high OOT/depth, and coherent stellar/alias evidence.",
    decisive_period_hypothesis: "2P = 13.7505 d EB/stellar-alias hypothesis; nominal P = 6.87525 d contains the suspicious SNR.",
    proposed_v0_3_diagnostic: "Run numerical-integrity checks for absurd, non-finite, or internally inconsistent SNR/depth metrics before gate scoring, then apply the mandatory 2P competing-EB veto.",
    change_classification: "stage_g_logic_change_required",
    notes: "v0.2 recorded absurd_primary_depth_snr_suspicious and blocked on OOT/depth, but the approved design requires explicit integrity gating and 2P veto traceability.",
  },
};

async function csvObjects(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  const wb = await Workbook.fromCSV(text, { sheetName: "Data" });
  const values = wb.worksheets.getItem("Data").getUsedRange(true).values;
  const headers = values[0].map(String);
  return values.slice(1).filter((row) => row.some((value) => value !== null && value !== "")).map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, row[index] === null ? "" : String(row[index])]))
  );
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "NA";
  const n = Number(value);
  return Number.isFinite(n) ? String(Number(n.toFixed(digits))) : "NA";
}

function metricSummary(periods, epicId) {
  const get = (role) => periods.find((row) => row.period_role === role);
  const pHalf = get("P/2");
  const p = get("P");
  const p2 = get("2P");
  if (!pHalf || !p || !p2) throw new Error(`${epicId}: missing P/2, P, or 2P comparison row`);
  if (["EPIC_211996306", "EPIC_211912465", "EPIC_211431812"].includes(epicId)) {
    return `P/2/P/2P primary SNR=${fmt(pHalf.primary_depth_snr)}/${fmt(p.primary_depth_snr)}/${fmt(p2.primary_depth_snr)}; OOT/depth=${fmt(pHalf.oot_to_depth)}/${fmt(p.oot_to_depth)}/${fmt(p2.oot_to_depth)}; event support=${pHalf.event_support_count}/${p.event_support_count}/${p2.event_support_count}; baseline=${pHalf.local_baseline_stability}/${p.local_baseline_stability}/${p2.local_baseline_stability}`;
  }
  if (epicId === "EPIC_211845034") {
    return `P: primary SNR=${fmt(p.primary_depth_snr)}, OOT/depth=${fmt(p.oot_to_depth)}, odd/even=${fmt(p.odd_even_depth_ratio)}; 2P: secondary SNR=${fmt(p2.secondary_depth_snr)}, secondary/primary=${fmt(p2.secondary_to_primary_depth_ratio)}, OOT/depth=${fmt(p2.oot_to_depth)}, coherence=${p2.event_stack_coherence} (${fmt(p2.event_stack_coherence_score)})`;
  }
  return `2P: primary SNR=${fmt(p2.primary_depth_snr)}, secondary SNR=${fmt(p2.secondary_depth_snr)}, secondary/primary=${fmt(p2.secondary_to_primary_depth_ratio)}, odd/even=${fmt(p2.odd_even_depth_ratio)}, OOT/depth=${fmt(p2.oot_to_depth)}, coherence=${p2.event_stack_coherence} (${fmt(p2.event_stack_coherence_score)}), secondary=${p2.secondary_assessment}`;
}

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

const decisions = await csvObjects(decisionPath);
const reconciled = await csvObjects(reportPath);
if (decisions.length !== 15 || new Set(decisions.map((row) => row.epic_id)).size !== 15) {
  throw new Error("Manual decision ledger must contain exactly 15 unique EPICs");
}
if (reconciled.length !== 15 || new Set(reconciled.map((row) => row.epic_id)).size !== 15) {
  throw new Error("Reconciliation report must contain exactly 15 unique EPICs");
}
if (Object.keys(annotations).length !== 15 || new Set(Object.keys(annotations)).size !== 15) {
  throw new Error("Traceability annotations must contain exactly 15 unique EPICs");
}

const rows = [];
for (const decision of decisions) {
  const epicId = decision.epic_id;
  const reconciliation = reconciled.find((row) => row.epic_id === epicId);
  const annotation = annotations[epicId];
  if (!reconciliation || !annotation) throw new Error(`${epicId}: missing reconciliation or annotation`);
  if (reconciliation.final_label !== decision.final_label) throw new Error(`${epicId}: source labels disagree`);
  if (!allowedClassifications.has(annotation.change_classification)) throw new Error(`${epicId}: invalid classification`);

  const epicDir = path.join(root, "plots", "k2_batch", "gatevetter_v0_2_deep_review", epicId);
  let validationPath = path.join(root, decision.validation_summary_json);
  try {
    await fs.access(validationPath);
  } catch {
    const candidates = (await fs.readdir(epicDir)).filter((name) => /^validation_summary.*\.json$/i.test(name));
    if (candidates.length !== 1) throw new Error(`${epicId}: validation summary is missing or ambiguous`);
    validationPath = path.join(epicDir, candidates[0]);
  }
  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));
  if (validation.epic_id !== epicId) throw new Error(`${epicId}: validation summary identity mismatch`);
  const periodPath = path.join(epicDir, "period_comparison.csv");
  const periods = await csvObjects(periodPath);
  if (periods.length !== 3 || periods.some((row) => row.epic_id !== epicId)) {
    throw new Error(`${epicId}: expected three matching period-comparison rows`);
  }
  const source = validation.source_gatevetter;
  const blocking = `${source.gatevetter_v0_2_reason}; primary_gate=${source.primary_gate}; hold_gates=${source.hold_gates_fired}; penalties=${source.penalties_or_missing_evidence}`;
  const handled = annotation.change_classification === "stage_g_logic_change_required"
    ? "yes_for_non_promotion_but_not_for_decisive_2P_failure_mode"
    : "yes_for_non_promotion_and_recorded_v0_2_block";
  rows.push({
    epic_id: epicId,
    final_manual_label: decision.final_label,
    decisive_failure_mode: annotation.decisive_failure_mode,
    decisive_period_hypothesis: annotation.decisive_period_hypothesis,
    key_supporting_metrics: metricSummary(periods, epicId),
    v0_2_blocking_reason: blocking,
    did_v0_2_handle_correctly: handled,
    proposed_v0_3_diagnostic: annotation.proposed_v0_3_diagnostic,
    change_classification: annotation.change_classification,
    notes: annotation.notes,
  });
}

const headers = [
  "epic_id", "final_manual_label", "decisive_failure_mode", "decisive_period_hypothesis",
  "key_supporting_metrics", "v0_2_blocking_reason", "did_v0_2_handle_correctly",
  "proposed_v0_3_diagnostic", "change_classification", "notes",
];
const csvText = [headers.join(","), ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(","))].join("\r\n") + "\r\n";
await fs.writeFile(outputCsv, csvText, "utf8");

const classOrder = ["no_change_required", "recording_change_only", "stage_g_logic_change_required", "unsupported"];
const counts = Object.fromEntries(classOrder.map((classification) => [classification, rows.filter((row) => row.change_classification === classification).length]));
const competingEb = rows.filter((row) => row.change_classification === "stage_g_logic_change_required").map((row) => row.epic_id);
const numericalIntegrity = ["EPIC_211696209"];
const noBehavioralChange = rows.filter((row) => row.change_classification === "no_change_required" || row.change_classification === "recording_change_only").map((row) => row.epic_id);
const strictNoChange = rows.filter((row) => row.change_classification === "no_change_required").map((row) => row.epic_id);

const summary = `GateVetter v0.3 EPIC-level traceability summary
status=design_traceability_only_v0_3_not_implemented
source_review_set=GateVetter_v0.2_deep_review
rows=15
unique_epics=15
labels_changed=0
thresholds_changed=0
new_batches_run=0

change_classification_counts
no_change_required=${counts.no_change_required}
recording_change_only=${counts.recording_change_only}
stage_g_logic_change_required=${counts.stage_g_logic_change_required}
unsupported=${counts.unsupported}

mandatory_2P_competing_EB_veto_support_count=${competingEb.length}
mandatory_2P_competing_EB_veto_supporting_epics=${competingEb.join("|")}

numerical_integrity_check_support_count=${numericalIntegrity.length}
numerical_integrity_check_supporting_epics=${numericalIntegrity.join("|")}

no_behavioral_change_count=${noBehavioralChange.length}
no_behavioral_change_epics=${noBehavioralChange.join("|")}
strict_no_change_required_count=${strictNoChange.length}
strict_no_change_required_epics=${strictNoChange.join("|")}

interpretation=All 15 v0.2 outcomes prevented Stage G promotion. Ten EB/variable decisions support making the 2P competing-EB veto explicit in v0.3 rather than relying on incidental v0.2 hold gates. EPIC_211696209 also supports pre-scoring numerical-integrity checks. Five targets require no behavioral change; two of those need recording changes only.
source_exception=EPIC_211768304 has validation_summary_1.json and deep_review_panel_1.png locally although the ledger records unsuffixed names; the existing suffixed diagnostic was read without renaming or modification.
`;
await fs.writeFile(outputSummary, summary, "utf8");

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("EPIC Traceability");
sheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [headers, ...rows.map((row) => headers.map((header) => row[header]))];
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getRange("A1:J1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
};
sheet.getRange("A2:J16").format = { wrapText: true, verticalAlignment: "top" };
sheet.getRange("A1:J16").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
const widths = [20, 28, 48, 44, 60, 52, 42, 62, 34, 50];
for (let index = 0; index < widths.length; index += 1) sheet.getRangeByIndexes(0, index, rows.length + 1, 1).format.columnWidth = widths[index];
sheet.getRange("A1:J1").format.rowHeight = 36;
sheet.getRange("A2:J16").format.rowHeight = 90;
sheet.tables.add("A1:J16", true, "EpicTraceabilityTable").style = "TableStyleMedium2";

const inspectTop = await workbook.inspect({ kind: "table", range: "EPIC Traceability!A1:J3", include: "values,formulas", tableMaxRows: 3, tableMaxCols: 10, maxChars: 6000 });
const inspectBottom = await workbook.inspect({ kind: "table", range: "EPIC Traceability!A15:J16", include: "values,formulas", tableMaxRows: 2, tableMaxCols: 10, maxChars: 6000 });
if (!inspectTop.ndjson.includes("EPIC_211624954") || !inspectBottom.ndjson.includes("EPIC_211696209")) throw new Error("Workbook inspection failed");
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
if (/"match"/.test(errors.ndjson)) throw new Error("Workbook formula error scan found an error");
await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "EPIC Traceability", range: "A1:J16", scale: 0.75, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsx);

console.log(JSON.stringify({ rows: rows.length, uniqueEpics: new Set(rows.map((row) => row.epic_id)).size, counts, competingEb, numericalIntegrity, noBehavioralChange, outputCsv, outputSummary, outputXlsx, previewPath }, null, 2));
