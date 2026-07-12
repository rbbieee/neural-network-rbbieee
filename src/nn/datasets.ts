// Classic 2D toy datasets for binary classification.
// All points live in the range [-1, 1] on both axes so the
// decision boundary canvas can use a fixed coordinate system.

import type { TrainingSample } from "./network";

export type DatasetName = "xor" | "circles" | "spiral" | "blobs";

// Same seeded RNG idea as the network, so datasets are reproducible
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const POINTS = 200;

export const datasetLabels: Record<DatasetName, string> = {
  xor: "XOR quadrants",
  circles: "Concentric circles",
  spiral: "Two spirals",
  blobs: "Gaussian blobs",
};

export function generateDataset(
  name: DatasetName,
  seed = 42
): TrainingSample[] {
  const rand = mulberry32(seed);
  const noise = () => (rand() - 0.5) * 0.15;
  const samples: TrainingSample[] = [];

  switch (name) {
    case "xor": {
      // Opposite quadrants share a label, the classic non linear problem
      for (let i = 0; i < POINTS; i++) {
        const x = rand() * 2 - 1;
        const y = rand() * 2 - 1;
        const label = x * y > 0 ? 1 : 0;
        samples.push({ x: x + noise(), y: y + noise(), label });
      }
      break;
    }
    case "circles": {
      // Inner disk is one class, outer ring is the other
      for (let i = 0; i < POINTS; i++) {
        const inner = i % 2 === 0;
        const r = inner ? rand() * 0.35 : 0.6 + rand() * 0.3;
        const t = rand() * Math.PI * 2;
        samples.push({
          x: r * Math.cos(t) + noise(),
          y: r * Math.sin(t) + noise(),
          label: inner ? 1 : 0,
        });
      }
      break;
    }
    case "spiral": {
      // Two interleaved spirals, the hardest of the four
      for (let i = 0; i < POINTS; i++) {
        const label = (i % 2) as 0 | 1;
        const t = (i / POINTS) * 3 * Math.PI + (label ? Math.PI : 0);
        const r = 0.05 + (i / POINTS) * 0.85;
        samples.push({
          x: r * Math.cos(t) + noise() * 0.5,
          y: r * Math.sin(t) + noise() * 0.5,
          label,
        });
      }
      break;
    }
    case "blobs": {
      // Two clusters, linearly separable, good as a warm up
      for (let i = 0; i < POINTS; i++) {
        const label = (i % 2) as 0 | 1;
        const cx = label ? 0.5 : -0.5;
        const cy = label ? 0.4 : -0.4;
        samples.push({
          x: cx + (rand() - 0.5) * 0.7,
          y: cy + (rand() - 0.5) * 0.7,
          label,
        });
      }
      break;
    }
  }
  return samples;
}
