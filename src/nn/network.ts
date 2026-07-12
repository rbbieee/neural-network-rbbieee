// A small feedforward neural network written from scratch.
// No libraries, just arrays and math, so every step of learning
// is visible and easy to inspect from the UI.
//
// The network solves binary classification on 2D points, which is
// perfect for visualization because we can paint the decision
// boundary directly onto a canvas.

import { activations, sigmoid, type ActivationName } from "./activations";

export interface NetworkConfig {
  // Sizes of hidden layers, for example [4, 3] means two hidden layers
  hiddenLayers: number[];
  activation: ActivationName;
  learningRate: number;
  seed: number;
}

export interface TrainingSample {
  x: number; // first input feature
  y: number; // second input feature
  label: 0 | 1;
}

// Weights are stored per layer as weights[layer][to][from].
// Biases are stored per layer as biases[layer][to].
export interface NetworkState {
  layerSizes: number[]; // includes input (2) and output (1)
  weights: number[][][];
  biases: number[][];
}

// Simple deterministic random number generator (mulberry32).
// A seeded RNG means the same seed always gives the same starting
// weights, which makes training runs reproducible and comparable.
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

export class NeuralNetwork {
  state: NetworkState;
  config: NetworkConfig;
  // Activations from the latest forward pass, kept for visualization
  lastActivations: number[][] = [];

  constructor(config: NetworkConfig) {
    this.config = config;
    this.state = NeuralNetwork.initialize(config);
  }

  static initialize(config: NetworkConfig): NetworkState {
    const layerSizes = [2, ...config.hiddenLayers, 1];
    const rand = mulberry32(config.seed);
    const weights: number[][][] = [];
    const biases: number[][] = [];

    for (let l = 1; l < layerSizes.length; l++) {
      const fanIn = layerSizes[l - 1];
      // Xavier style scaling keeps early signals in a healthy range
      const scale = Math.sqrt(1 / fanIn);
      const layerW: number[][] = [];
      const layerB: number[] = [];
      for (let to = 0; to < layerSizes[l]; to++) {
        const row: number[] = [];
        for (let from = 0; from < fanIn; from++) {
          row.push((rand() * 2 - 1) * scale);
        }
        layerW.push(row);
        layerB.push(0);
      }
      weights.push(layerW);
      biases.push(layerB);
    }
    return { layerSizes, weights, biases };
  }

  reset() {
    this.state = NeuralNetwork.initialize(this.config);
    this.lastActivations = [];
  }

  // Forward pass. Returns activations for every layer,
  // where index 0 is the input itself.
  forward(x: number, y: number): number[][] {
    const { weights, biases, layerSizes } = this.state;
    const act = activations[this.config.activation];
    const all: number[][] = [[x, y]];

    for (let l = 0; l < weights.length; l++) {
      const prev = all[l];
      const isOutput = l === weights.length - 1;
      const layer: number[] = [];
      for (let to = 0; to < layerSizes[l + 1]; to++) {
        let sum = biases[l][to];
        for (let from = 0; from < prev.length; from++) {
          sum += weights[l][to][from] * prev[from];
        }
        // Hidden layers use the chosen activation,
        // the output uses sigmoid so it reads as a probability
        layer.push(isOutput ? sigmoid(sum) : act.fn(sum));
      }
      all.push(layer);
    }
    this.lastActivations = all;
    return all;
  }

  predict(x: number, y: number): number {
    const all = this.forward(x, y);
    return all[all.length - 1][0];
  }

  // One step of mini batch gradient descent with backpropagation.
  // Returns the mean binary cross entropy loss over the batch.
  trainBatch(batch: TrainingSample[]): number {
    const { weights, biases, layerSizes } = this.state;
    const act = activations[this.config.activation];
    const lr = this.config.learningRate;

    // Accumulate gradients over the whole batch before applying them
    const gradW = weights.map((lw) => lw.map((row) => row.map(() => 0)));
    const gradB = biases.map((lb) => lb.map(() => 0));
    let totalLoss = 0;

    for (const sample of batch) {
      const all = this.forward(sample.x, sample.y);
      const out = all[all.length - 1][0];

      // Binary cross entropy, clamped to avoid log(0)
      const p = Math.min(Math.max(out, 1e-7), 1 - 1e-7);
      totalLoss -=
        sample.label * Math.log(p) + (1 - sample.label) * Math.log(1 - p);

      // Backward pass. With sigmoid output and cross entropy loss the
      // output delta simplifies to prediction minus target
      let deltas: number[] = [out - sample.label];

      for (let l = weights.length - 1; l >= 0; l--) {
        const prevAct = all[l];
        // Add this layer's contribution to the gradients
        for (let to = 0; to < layerSizes[l + 1]; to++) {
          gradB[l][to] += deltas[to];
          for (let from = 0; from < prevAct.length; from++) {
            gradW[l][to][from] += deltas[to] * prevAct[from];
          }
        }
        // Propagate deltas to the previous layer, skip for the input
        if (l > 0) {
          const nextDeltas: number[] = [];
          for (let from = 0; from < layerSizes[l]; from++) {
            let sum = 0;
            for (let to = 0; to < layerSizes[l + 1]; to++) {
              sum += weights[l][to][from] * deltas[to];
            }
            nextDeltas.push(sum * act.dFromOutput(prevAct[from]));
          }
          deltas = nextDeltas;
        }
      }
    }

    // Apply averaged gradients
    const n = batch.length;
    for (let l = 0; l < weights.length; l++) {
      for (let to = 0; to < weights[l].length; to++) {
        biases[l][to] -= (lr * gradB[l][to]) / n;
        for (let from = 0; from < weights[l][to].length; from++) {
          weights[l][to][from] -= (lr * gradW[l][to][from]) / n;
        }
      }
    }
    return totalLoss / n;
  }

  // Classification accuracy over a dataset, used for the stats readout
  accuracy(data: TrainingSample[]): number {
    let correct = 0;
    for (const s of data) {
      const p = this.predict(s.x, s.y);
      if ((p >= 0.5 ? 1 : 0) === s.label) correct++;
    }
    return correct / data.length;
  }
}
