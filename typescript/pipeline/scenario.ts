import type { RwlTreeData, Truth } from "./types";

export interface SourceSeries {
  id: string;
  startYear: number;
  endYear: number;
  valuesByYear: Map<number, number>;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function falseValueAt(correct: Map<number, number>, sourceYear: number): number {
  const neighborhood: number[] = [];
  for (let year = sourceYear - 3; year <= sourceYear + 3; year += 1) {
    const value = correct.get(year);
    if (value !== undefined && value > 0) neighborhood.push(value);
  }
  return Math.max(1, Math.round(median(neighborhood)));
}

/** Exact immutable-calendar corruption used by the frozen v5 scenario builders. */
export function createPiecewiseLagMixedCase(
  series: SourceSeries,
  truths: readonly Truth[],
): RwlTreeData {
  const whole = truths.find((truth) => truth.eventType === "wholeSeriesMove");
  const wholeLag = whole?.shiftYears ?? 0;
  const local = truths.filter((truth) => truth.year !== null)
    .sort((left, right) => right.year! - left.year!);
  const olderSideLag = wholeLag + local.reduce((sum, truth) => sum + truth.shiftYears, 0);
  const displayedStart = series.startYear - olderSideLag;
  const displayedEnd = series.endYear - wholeLag;
  const corrupted: RwlTreeData = new Map();
  for (let year = displayedStart; year <= displayedEnd; year += 1) {
    const active = local.filter((truth) => truth.eventType === "partialMove"
      ? year < truth.year! : year <= truth.year!);
    const lag = wholeLag + active.reduce((sum, truth) => sum + truth.shiftYears, 0);
    const sourceYear = year + lag;
    const falseEvent = active.find((truth) => truth.eventType === "falseRing" && truth.year === year);
    if (falseEvent) {
      corrupted.set(year, falseValueAt(series.valuesByYear, sourceYear));
      continue;
    }
    const value = series.valuesByYear.get(sourceYear);
    if (value !== undefined) corrupted.set(year, value);
  }
  return corrupted;
}

export function frontierTruth(truths: readonly Truth[]): Truth | null {
  return truths.find((truth) => truth.eventType === "wholeSeriesMove")
    ?? [...truths].filter((truth) => truth.year !== null)
      .sort((left, right) => right.year! - left.year!)[0]
    ?? null;
}

export function advanceFrontierTruths(truths: readonly Truth[], resolvedId: string): Truth[] {
  const truth = frontierTruth(truths);
  if (!truth || truth.truthId !== resolvedId) throw new Error("only the current frontier may be resolved");
  return truths.filter((other) => other.truthId !== resolvedId).map((other) => ({
    ...other,
    year: other.year === null ? null : other.year + truth.shiftYears,
  }));
}

function moveRange(tree: RwlTreeData, first: number, last: number, offset: number): RwlTreeData {
  const output: RwlTreeData = new Map();
  for (const [year, value] of tree) if (year < first || year > last) output.set(year, value);
  for (const [year, value] of tree) if (year >= first && year <= last) output.set(year + offset, value);
  return new Map([...output].sort(([left], [right]) => left - right));
}

export function applyFrontierTruth(tree: RwlTreeData, truths: readonly Truth[]): {
  tree: RwlTreeData;
  remaining: Truth[];
} {
  const truth = frontierTruth(truths);
  if (!truth) throw new Error("no frontier truth");
  const years = [...tree.keys()];
  if (years.length === 0) throw new Error("empty target");
  let next: RwlTreeData;
  if (truth.eventType === "wholeSeriesMove") {
    next = moveRange(tree, Math.min(...years), Math.max(...years), truth.shiftYears);
  } else if (truth.eventType === "partialMove") {
    next = moveRange(tree, Math.min(...years), truth.year! - 1, truth.shiftYears);
  } else if (truth.eventType === "missingRing") {
    next = new Map([...tree].filter(([, value]) => value !== -9999)
      .map(([year, value]) => [year <= truth.year! ? year - 1 : year, value]));
    next.set(truth.year!, 0);
    next = new Map([...next].sort(([left], [right]) => left - right));
  } else {
    next = new Map();
    for (const [year, value] of tree) {
      if (year === truth.year) continue;
      next.set(year < truth.year! ? year + 1 : year, value);
    }
  }
  return { tree: next, remaining: advanceFrontierTruths(truths, truth.truthId) };
}
