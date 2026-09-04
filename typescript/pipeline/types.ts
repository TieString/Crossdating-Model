export type RwlTreeData = Map<number, number | null>;
export type RwlSiteData = Map<string, RwlTreeData>;
export type EventKind = "missingRing" | "falseRing" | "partialMove" | "wholeSeriesMove";

export interface Truth {
  truthId: string;
  eventType: EventKind;
  year: number | null;
  shiftYears: number;
}

export interface FrozenCase {
  index: number;
  caseId: string;
  family: "Clean" | "A" | "B" | "C" | "D";
  fileId: string;
  relativePath: string;
  targetId: string;
  targetEndYear: number;
  masterCorrelation: number;
  truths: Truth[];
  scenarioGeneratorVersion: 10 | 11;
  primaryABCD: boolean;
  expectedStages: { stageIndex: number; stateHash: string; evidenceKey: string; truth: unknown }[];
}

export interface TargetFile {
  fileId: string;
  relativePath: string;
  sourceSha256: string;
  seriesIntercorrelation: number;
}
