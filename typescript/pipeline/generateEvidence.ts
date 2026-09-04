import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { gzipSync } from "node:zlib";
import { parseRwl } from "cofecha-js";
import { CofechaEngineEvidenceCache, EVIDENCE_VERSION, normalizeCofechaInputUnits } from "./cofechaEngineEvidence";
import { advanceFrontierTruths, applyFrontierTruth, createPiecewiseLagMixedCase, frontierTruth } from "./scenario";
import type { FrozenCase, RwlTreeData, TargetFile, Truth } from "./types";

const args = process.argv.slice(2);
const option = (name: string, fallback?: string): string => {
  const index = args.indexOf(name);
  const value = index >= 0 ? args[index + 1] : fallback;
  if (value === undefined) throw new Error(`missing ${name}`);
  return value;
};
const sha256 = (value: string | Buffer): string => createHash("sha256").update(value).digest("hex");
const role = option("--role");
if (role !== "development" && role !== "calibration") throw new Error("role must be development or calibration");
const casesPath = resolve(option("--cases", `scenarios/v5-${role}-cases.json`));
const targetsPath = resolve(option("--targets", `splits/${role}-targets.json`));
const rwlRoot = resolve(option("--rwl-root", "datasets/rwl"));
const output = resolve(option("--output", `work/index/${role}.jsonl`));
const evidenceRoot = resolve(option("--evidence-root", "work/evidence"));
const limit = Number(option("--limit", "0"));

const casePayload = JSON.parse(readFileSync(casesPath, "utf8")) as { role: string; cases: FrozenCase[] };
const targetPayload = JSON.parse(readFileSync(targetsPath, "utf8")) as { files: TargetFile[] };
if (casePayload.role !== role) throw new Error("case role mismatch");
const files = new Map(targetPayload.files.map((file) => [file.fileId, file]));
const selected = limit > 0 ? casePayload.cases.slice(0, limit) : casePayload.cases;
const engine = new CofechaEngineEvidenceCache();
const rows: Record<string, unknown>[] = [];
let currentFile = "";
let site = new Map<string, RwlTreeData>();

const operation = (truth: Truth | null): "none" | "whole" | "partial" | "missing" | "false" => {
  if (!truth) return "none";
  return truth.eventType === "wholeSeriesMove" ? "whole"
    : truth.eventType === "partialMove" ? "partial"
      : truth.eventType === "missingRing" ? "missing" : "false";
};
const rBin = (value: number): string => value < 0.7 ? "0.60-0.70" : value < 0.8 ? "0.70-0.80" : "0.80+";
const distanceBand = (distance: number | null): string => distance === null ? "none"
  : distance === 0 ? "0(endpoint)" : distance <= 5 ? "1-5" : distance <= 10 ? "6-10"
    : distance <= 15 ? "11-15" : distance <= 19 ? "16-19" : distance <= 24 ? "20-24"
      : distance <= 29 ? "25-29" : distance <= 34 ? "30-34" : distance <= 69 ? "35-69" : "70+";

for (const item of selected) {
  const file = files.get(item.fileId);
  if (!file) throw new Error(`file absent from target manifest: ${item.fileId}`);
  if (currentFile !== file.fileId) {
    const path = resolve(rwlRoot, file.relativePath);
    const bytes = readFileSync(path);
    if (sha256(bytes) !== file.sourceSha256) throw new Error(`source hash mismatch: ${file.relativePath}`);
    site = normalizeCofechaInputUnits(parseRwl(bytes.toString("utf8")).data);
    currentFile = file.fileId;
  }
  const original = site.get(item.targetId);
  if (!original) throw new Error(`target absent from ${file.relativePath}: ${item.targetId}`);
  const observed = new Map([...original].flatMap(([year, value]) => (
    value !== null && value !== -9999 ? [[year, value] as const] : []
  )));
  const years = [...observed.keys()];
  const source = {
    id: item.targetId,
    valuesByYear: observed,
    startYear: Math.min(...years),
    endYear: Math.max(...years),
  };
  let remaining = item.truths.map((truth) => ({ ...truth }));
  let tree = createPiecewiseLagMixedCase(source, remaining);
  let stageIndex = 0;
  do {
    const truth = frontierTruth(remaining);
    const stateHash = sha256(JSON.stringify({
      file: file.sourceSha256,
      target: item.targetId,
      values: [...tree].sort(([left], [right]) => left - right),
      inputUnits: "0.001mm-explicit-markers-v2",
    }));
    const evidenceKey = sha256(`${EVIDENCE_VERSION}\0${stateHash}`);
    const expected = item.expectedStages[stageIndex];
    if (!expected || expected.stateHash !== stateHash || expected.evidenceKey !== evidenceKey) {
      throw new Error(`frozen stage identity mismatch: ${item.caseId}:${stageIndex}`);
    }
    const evidencePath = resolve(evidenceRoot, evidenceKey.slice(0, 2), `${evidenceKey}.json.gz`);
    if (!existsSync(evidencePath)) {
      const state = new Map(site);
      state.set(item.targetId, tree);
      const sample = engine.build(state, item.targetId);
      if (!sample) throw new Error(`cofecha evidence unavailable: ${item.caseId}:${stageIndex}`);
      mkdirSync(dirname(evidencePath), { recursive: true });
      writeFileSync(evidencePath, gzipSync(Buffer.from(JSON.stringify({
        schemaVersion: 1,
        evidenceVersion: EVIDENCE_VERSION,
        stateHash,
        sourceStopMarker: -9999,
        referenceIds: sample.referenceIds,
        targetYears: [...sample.target.keys()],
        targetValues: [...sample.target.values()],
        referenceYears: [...sample.master.keys()],
        referenceValues: [...sample.master.values()],
        individualReferences: sample.references.map((reference) => ({
          years: [...reference.keys()], values: [...reference.values()],
        })),
      })), { level: 6 }));
    }
    const rawTargetPath = resolve(evidenceRoot, "raw-targets-v1", stateHash.slice(0, 2), `${stateHash}.json.gz`);
    if (!existsSync(rawTargetPath)) {
      mkdirSync(dirname(rawTargetPath), { recursive: true });
      writeFileSync(rawTargetPath, gzipSync(Buffer.from(JSON.stringify({
        version: "raw-counterfactual-v1", stateHash, targetId: item.targetId, entries: [...tree],
      })), { level: 6 }));
    }
    const distance = truth?.year == null ? null : Math.max(...tree.keys()) - truth.year;
    rows.push({
      role,
      protocol: "real-edit-frontier-explicit-units-v2",
      stateHash,
      evidenceKey,
      evidencePath,
      scenarioGeneratorVersion: item.scenarioGeneratorVersion,
      primaryABCD: item.primaryABCD,
      caseId: item.caseId,
      stageIndex,
      relativePath: file.relativePath,
      fileContentHash: file.sourceSha256,
      targetId: item.targetId,
      fileIntercorrelation: file.seriesIntercorrelation,
      targetMasterCorrelation: item.masterCorrelation,
      category: item.family,
      binName: rBin(file.seriesIntercorrelation),
      distanceBand: distanceBand(distance),
      placement: distance === null ? "none" : distance <= 34 ? "near" : "middle",
      truth: { kind: operation(truth), shift: truth?.shiftYears ?? 0,
        year: truth?.year ?? null, barkDistanceYears: distance },
      remainingTruths: remaining.map((other) => ({
        kind: operation(other), shift: other.shiftYears, year: other.year,
      })),
      remainingEventCount: remaining.length,
      remainingBand: remaining.length <= 1 ? "1" : remaining.length <= 4 ? "2-4" : "5+",
      rawTargetPath,
    });
    if (!truth) break;
    ({ tree, remaining } = applyFrontierTruth(tree, remaining));
    stageIndex += 1;
  } while (remaining.length);
}

mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
writeFileSync(`${output}.summary.json`, `${JSON.stringify({
  schemaVersion: 1,
  role,
  cases: selected.length,
  states: rows.length,
  uniqueStates: new Set(rows.map((row) => row.stateHash)).size,
  evidenceVersion: EVIDENCE_VERSION,
  sourceManifest: targetsPath,
  casesPath,
  coreCache: engine.stats,
  indexSha256: sha256(readFileSync(output)),
}, null, 2)}\n`, "utf8");
process.stdout.write(readFileSync(`${output}.summary.json`, "utf8"));
