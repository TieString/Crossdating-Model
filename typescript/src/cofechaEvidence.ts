import {
  COFECHA_REFERENCE_DEFAULT_OPTIONS,
  buildCofecha606MasterFromPreparedSeries,
  parseRwl,
  prepareCofecha606SeriesForReport,
  type Cofecha606PreparedSeries,
  type RwlSiteData,
} from "cofecha-js";

export interface EvidenceSeries {
  sequence: number;
  seriesId: string;
  years: number[];
  testingValues: number[];
  arResidualValues: number[];
}

export interface CofechaEvidenceState {
  schemaVersion: "crossdating-evidence/1";
  cofechaJsVersion: "0.2.0";
  target: EvidenceSeries;
  references: EvidenceSeries[];
  leaveTargetOutMaster: { years: number[]; values: number[]; sampleDepth: number[] };
}

function publicSeries(series: Cofecha606PreparedSeries): EvidenceSeries {
  return {
    sequence: series.sequence,
    seriesId: series.segment.seriesId,
    years: [...series.years],
    testingValues: [...series.testingValues],
    arResidualValues: [...series.arResidualValues],
  };
}

export function buildEvidenceFromSiteData(
  siteData: RwlSiteData,
  targetSeriesId: string,
  targetSequence?: number,
): CofechaEvidenceState {
  const years = [...siteData.values()].flatMap((series) => [...series.keys()]);
  if (years.length === 0) throw new Error("RWL contains no dated observations");
  const first = Math.min(...years);
  const last = Math.max(...years);
  const prepared = prepareCofecha606SeriesForReport(siteData, {
    splineRigidityYears: 32,
    splineFrequencyResponse: 0.5,
    segmentLength: 50,
    segmentLag: 25,
    useAutoregressiveModel: true,
    useLogTransform: true,
    useFirstDifference: false,
    segmentGridStartYear: first,
    segmentGridEndYear: last,
    analysisStartYear: first,
    analysisEndYear: last,
  });
  const matches = prepared.filter((series) => series.segment.seriesId === targetSeriesId
    && (targetSequence === undefined || series.sequence === targetSequence));
  if (matches.length !== 1) {
    throw new Error(`target must identify exactly one prepared sequence; matched ${matches.length}`);
  }
  const target = matches[0];
  const references = prepared.filter((series) => series.sequence !== target.sequence);
  if (references.length === 0) throw new Error("target-excluded reference set is empty");
  const master = buildCofecha606MasterFromPreparedSeries(references, COFECHA_REFERENCE_DEFAULT_OPTIONS);
  if (master === null) throw new Error("target-excluded master could not be constructed");
  const masterYears = [...master.data.keys()].sort((left, right) => left - right);
  return {
    schemaVersion: "crossdating-evidence/1",
    cofechaJsVersion: "0.2.0",
    target: publicSeries(target),
    references: references.map(publicSeries),
    leaveTargetOutMaster: {
      years: masterYears,
      values: masterYears.map((year) => master.data.get(year) as number),
      sampleDepth: masterYears.map((year) => master.sampleDepth.get(year) ?? 0),
    },
  };
}

export function buildEvidenceFromRwl(
  rwlText: string,
  targetSeriesId: string,
  targetSequence?: number,
): CofechaEvidenceState {
  return buildEvidenceFromSiteData(parseRwl(rwlText).data, targetSeriesId, targetSequence);
}
