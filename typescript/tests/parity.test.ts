import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { scoreCandidates, type UnifiedV5Model } from "../src/lightgbm";

interface GoldState {
  stateHash: string;
  features: (number | null)[][];
  scores: number[];
}

interface GoldFile {
  featureCount: number;
  states: GoldState[];
}

const modelPath = process.env.CROSSDATING_MODEL_JSON
  ?? resolve("model-releases/v5.0.0/unifiedV5Model.json");
const goldPath = process.env.CROSSDATING_GOLD_JSON
  ?? resolve("model-releases/v5.0.0/golden-fixtures/python-scores.json");

describe("Python / TypeScript LightGBM parity", () => {
  test("scores every frozen candidate within numerical tolerance", () => {
    const model = JSON.parse(readFileSync(modelPath, "utf8")) as UnifiedV5Model;
    const gold = JSON.parse(readFileSync(goldPath, "utf8")) as GoldFile;
    expect(model.columns).toHaveLength(gold.featureCount);
    expect(gold.states.length).toBeGreaterThan(0);
    let maximumError = 0;
    for (const state of gold.states) {
      const features = state.features.map((row) => row.map((value) => value ?? Number.NaN));
      const actual = scoreCandidates(model, features);
      expect(actual).toHaveLength(state.scores.length);
      actual.forEach((score, index) => {
        maximumError = Math.max(maximumError, Math.abs(score - state.scores[index]));
      });
    }
    expect(maximumError).toBeLessThanOrEqual(1e-12);
  });
});
