// Input feature engineering transformations.
// Allows users to transform raw 2D (x, y) coordinates into non-linear features.

export type FeatureId = "x1" | "x2" | "x1_sq" | "x2_sq" | "x1_x2" | "sin_x1" | "sin_x2";

export interface FeatureInfo {
  id: FeatureId;
  label: string;
  mathLabel: string;
  fn: (x: number, y: number) => number;
}

export const ALL_FEATURES: FeatureInfo[] = [
  { id: "x1", label: "X₁", mathLabel: "X₁", fn: (x) => x },
  { id: "x2", label: "X₂", mathLabel: "X₂", fn: (_, y) => y },
  { id: "x1_sq", label: "X₁²", mathLabel: "X₁²", fn: (x) => x * x },
  { id: "x2_sq", label: "X₂²", mathLabel: "X₂²", fn: (_, y) => y * y },
  { id: "x1_x2", label: "X₁X₂", mathLabel: "X₁X₂", fn: (x, y) => x * y },
  { id: "sin_x1", label: "sin(X₁)", mathLabel: "sin(X₁)", fn: (x) => Math.sin(x * Math.PI) },
  { id: "sin_x2", label: "sin(X₂)", mathLabel: "sin(X₂)", fn: (_, y) => Math.sin(y * Math.PI) },
];

export const DEFAULT_FEATURES: FeatureId[] = ["x1", "x2"];

export function extractFeatures(x: number, y: number, activeIds: FeatureId[]): number[] {
  return activeIds.map((id) => {
    const feat = ALL_FEATURES.find((f) => f.id === id);
    return feat ? feat.fn(x, y) : 0;
  });
}
