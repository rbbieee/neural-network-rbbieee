// A small feedforward neural network written from scratch.
// No libraries, just arrays and math, so every step of learning
// is visible and easy to inspect from the UI.
//
// The network solves binary classification on 2D points, which is
// perfect for visualization because we can paint the decision
// boundary directly onto a canvas.

import { activations, sigmoid, type ActivationName } from "./activations";

export type OptimizerName = "sgd" | "momentum" | "adam";
export type RegularizationName = "none" | "l1" | "l2";

export interface NetworkConfig {
  hiddenLayers: number[];
  activation: ActivationName;
  learningRate: number;
  seed: number;
  optimizer: OptimizerName;
  regularization: RegularizationName;
  regularizationRate: number;
  batchSize: number;
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
  
  // Optimizer state buffers
  mW: number[][][];
  mB: number[][];
  vW: number[][][];
  vB: number[][];
  t: number; // Adam timestep
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
  // Gradients from the latest backward pass, kept for visualization
  lastGradients: number[][][] = [];

  constructor(config: NetworkConfig) {
    this.config = config;
    this.state = NeuralNetwork.initialize(config);
  }

  static initialize(config: NetworkConfig): NetworkState {
    const layerSizes = [2, ...config.hiddenLayers, 1];
    const rand = mulberry32(config.seed);
    const weights: number[][][] = [];
    const biases: number[][] = [];
    const mW: number[][][] = [];
    const mB: number[][] = [];
    const vW: number[][][] = [];
    const vB: number[][] = [];

    for (let l = 1; l < layerSizes.length; l++) {
      const fanIn = layerSizes[l - 1];
      // Xavier style scaling keeps early signals in a healthy range
      const scale = Math.sqrt(1 / fanIn);
      const layerW: number[][] = [];
      const layerB: number[] = [];
      const layerMW: number[][] = [];
      const layerMB: number[] = [];
      const layerVW: number[][] = [];
      const layerVB: number[] = [];

      for (let to = 0; to < layerSizes[l]; to++) {
        const rowW: number[] = [];
        const rowMW: number[] = [];
        const rowVW: number[] = [];
        for (let from = 0; from < fanIn; from++) {
          rowW.push((rand() * 2 - 1) * scale);
          rowMW.push(0);
          rowVW.push(0);
        }
        layerW.push(rowW);
        layerB.push(0);
        layerMW.push(rowMW);
        layerMB.push(0);
        layerVW.push(rowVW);
        layerVB.push(0);
      }
      weights.push(layerW);
      biases.push(layerB);
      mW.push(layerMW);
      mB.push(layerMB);
      vW.push(layerVW);
      vB.push(layerVB);
    }
    return { layerSizes, weights, biases, mW, mB, vW, vB, t: 0 };
  }

  reset() {
    this.state = NeuralNetwork.initialize(this.config);
    this.lastActivations = [];
    this.lastGradients = [];
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
    const { weights, biases, layerSizes, mW, mB, vW, vB } = this.state;
    const act = activations[this.config.activation];
    const { learningRate: lr, optimizer, regularization, regularizationRate: lambda } = this.config;

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

      // Backward pass
      let deltas: number[] = [out - sample.label];

      for (let l = weights.length - 1; l >= 0; l--) {
        const prevAct = all[l];
        for (let to = 0; to < layerSizes[l + 1]; to++) {
          gradB[l][to] += deltas[to];
          for (let from = 0; from < prevAct.length; from++) {
            gradW[l][to][from] += deltas[to] * prevAct[from];
          }
        }
        // Propagate deltas
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

    const n = batch.length;
    let regLoss = 0;
    
    // Optimizers
    this.state.t += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    for (let l = 0; l < weights.length; l++) {
      for (let to = 0; to < weights[l].length; to++) {
        // Average gradient for bias
        let gb = gradB[l][to] / n;
        
        for (let from = 0; from < weights[l][to].length; from++) {
          let gw = gradW[l][to][from] / n;
          const w = weights[l][to][from];

          // Regularization
          if (regularization === "l2") {
            regLoss += 0.5 * lambda * w * w;
            gw += lambda * w;
          } else if (regularization === "l1") {
            regLoss += lambda * Math.abs(w);
            gw += lambda * (w > 0 ? 1 : -1);
          }

          // Optimizer step for weight
          if (optimizer === "adam") {
            mW[l][to][from] = beta1 * mW[l][to][from] + (1 - beta1) * gw;
            vW[l][to][from] = beta2 * vW[l][to][from] + (1 - beta2) * gw * gw;
            const mHat = mW[l][to][from] / (1 - Math.pow(beta1, this.state.t));
            const vHat = vW[l][to][from] / (1 - Math.pow(beta2, this.state.t));
            weights[l][to][from] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
          } else if (optimizer === "momentum") {
            mW[l][to][from] = beta1 * mW[l][to][from] - lr * gw;
            weights[l][to][from] -= lr * gw - mW[l][to][from];
          } else { // SGD
            weights[l][to][from] -= lr * gw;
          }
        }

        // Optimizer step for bias (no regularization on biases usually)
        if (optimizer === "adam") {
          mB[l][to] = beta1 * mB[l][to] + (1 - beta1) * gb;
          vB[l][to] = beta2 * vB[l][to] + (1 - beta2) * gb * gb;
          const mHat = mB[l][to] / (1 - Math.pow(beta1, this.state.t));
          const vHat = vB[l][to] / (1 - Math.pow(beta2, this.state.t));
          biases[l][to] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        } else if (optimizer === "momentum") {
          mB[l][to] = beta1 * mB[l][to] - lr * gb;
          biases[l][to] -= lr * gb - mB[l][to];
        } else { // SGD
          biases[l][to] -= lr * gb;
        }
      }
    }
    
    // Store gradients for visualization (averaging by batch size)
    this.lastGradients = gradW.map(layer => layer.map(node => node.map(g => g / n)));

    return (totalLoss / n) + regLoss;
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
