import { describe, expect, test } from "vitest";
import { buildEvidenceFromSiteData } from "../src/cofechaEvidence";

function series(multiplier: number, phase: number): Map<number, number> {
  return new Map(Array.from({ length: 90 }, (_, index) => {
    const value = 220 + 60 * Math.sin((index + phase) / 4) + 15 * Math.cos(index / 9);
    return [1900 + index, Math.max(1, Math.round(value * multiplier))];
  }));
}

describe("cofecha-js evidence boundary", () => {
  test("builds testing values and excludes the current target from references", () => {
    const site = new Map([
      ["TARGET", series(1, 0)],
      ["REF001", series(1.05, 1)],
      ["REF002", series(0.95, -1)],
    ]);
    const state = buildEvidenceFromSiteData(site, "TARGET");
    expect(state.cofechaJsVersion).toBe("0.2.0");
    expect(state.target.seriesId).toBe("TARGET");
    expect(state.target.testingValues).toHaveLength(90);
    expect(state.references.map((item) => item.seriesId)).toEqual(["REF001", "REF002"]);
    expect(state.leaveTargetOutMaster.years.length).toBeGreaterThan(0);
    expect(state.leaveTargetOutMaster.sampleDepth.every((depth) => depth <= 2)).toBe(true);
  });
});
