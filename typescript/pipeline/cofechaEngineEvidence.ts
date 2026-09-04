import {
  prepareCofecha606SeriesForReport,
  splitCofecha606SeriesSegments,
  type Cofecha606PreparedSeries,
  type Cofecha606SeriesAnalysisOptions,
} from "cofecha-js";
import type { RwlSiteData, RwlTreeData } from "./types";

export const EVIDENCE_VERSION = "cofecha-js-0.2.0-explicit-units-testing-values-v2";
const OPTIONS: Cofecha606SeriesAnalysisOptions = {
  splineRigidityYears: 32,
  splineFrequencyResponse: 0.5,
  segmentLength: 50,
  segmentLag: 25,
  useAutoregressiveModel: true,
  useLogTransform: true,
  useFirstDifference: false,
  segmentGridStartYear: -10000,
  segmentGridEndYear: 10000,
  analysisStartYear: -10000,
  analysisEndYear: 10000,
};

function zScore(source: Map<number, number>): Map<number, number> {
  if (!source.size) return new Map();
  const values = [...source.values()];
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const sd = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length) || 1;
  return new Map([...source].map(([year, value]) => [year, (value - mean) / sd]));
}

function testingMap(prepared: Cofecha606PreparedSeries[]): Map<number, number> {
  return new Map(prepared.flatMap((part) => part.years.flatMap((year, index) => (
    part.rawValues[index] > 0 && Number.isFinite(part.testingValues[index])
      ? [[year, part.testingValues[index]] as const] : []
  ))));
}

export function normalizeCofechaInputUnits(source: RwlSiteData): RwlSiteData {
  const result: RwlSiteData = new Map();
  for (const segment of splitCofecha606SeriesSegments(source)) {
    const tree = result.get(segment.seriesId) ?? new Map<number, number | null>();
    const scale = segment.stopMarkerValue === 999 ? 10 : 1;
    for (const [year, width] of segment.data) if (width !== null) tree.set(year, width * scale);
    const end = Math.max(...segment.data.keys());
    if (Number.isFinite(end)) tree.set(end + 1, -9999);
    result.set(segment.seriesId, tree);
  }
  return result;
}

export class CofechaEngineEvidenceCache {
  private readonly cores = new Map<string, Cofecha606PreparedSeries[]>();
  readonly stats = { coreHits: 0, coreMisses: 0 };

  private prepare(id: string, tree: RwlTreeData): Cofecha606PreparedSeries[] {
    const entries = [...tree].sort(([left], [right]) => left - right);
    if (!entries.length) return [];
    const last = entries[entries.length - 1];
    if (last[1] !== -9999) entries.push([last[0] + 1, -9999]);
    for (let index = entries.length - 2; index >= 0; index -= 1) {
      if (entries[index][1] !== null && entries[index][1] !== -9999
          && entries[index + 1][0] > entries[index][0] + 1) {
        entries.splice(index + 1, 0, [entries[index][0] + 1, -9999]);
      }
    }
    const key = `${EVIDENCE_VERSION}:${id}:-9999:${JSON.stringify(entries)}`;
    const cached = this.cores.get(key);
    if (cached) {
      this.stats.coreHits += 1;
      return cached;
    }
    this.stats.coreMisses += 1;
    const prepared = prepareCofecha606SeriesForReport(new Map([[id, new Map(entries)]]), OPTIONS);
    this.cores.set(key, prepared);
    return prepared;
  }

  build(site: RwlSiteData, targetId: string) {
    const series = [...site].map(([id, tree]) => ({ id, prepared: this.prepare(id, tree) }));
    const target = testingMap(series.find((row) => row.id === targetId)?.prepared ?? []);
    if (target.size < 40) return null;
    const references = series.filter((row) => row.id !== targetId)
      .map((row) => ({ id: row.id, values: testingMap(row.prepared) }))
      .filter((row) => row.values.size > 0);
    if (!references.length) return null;
    const annual = new Map<number, { sum: number; count: number }>();
    references.forEach((reference) => reference.values.forEach((value, year) => {
      const current = annual.get(year) ?? { sum: 0, count: 0 };
      current.sum = Math.fround(current.sum + Math.fround(value));
      current.count += 1;
      annual.set(year, current);
    }));
    const master = zScore(new Map([...annual].sort(([left], [right]) => left - right)
      .map(([year, value]) => [year, Math.fround(value.sum / Math.fround(value.count))])));
    return {
      target: zScore(target),
      master,
      references: references.map((row) => row.values),
      referenceIds: references.map((row) => row.id),
    };
  }
}
