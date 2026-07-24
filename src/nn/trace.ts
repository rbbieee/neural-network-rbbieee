// Step-by-step math trace for a single training sample.
// Runs a full forward pass, loss computation, backward pass, and
// simulated weight update on one sample without mutating the network.
// The returned StepTrace object contains every intermediate number
// so the UI can render the math in human-readable form.

import { activations, sigmoid } from "./activations";
import { ALL_FEATURES, extractFeatures, DEFAULT_FEATURES } from "./features";
import type { NeuralNetwork, TrainingSample } from "./network";

// A single weight contribution inside a neuron's weighted sum
export interface WeightTerm {
  fromLabel: string;
  weight: number;
  input: number;
  product: number;
}

// Full computation record for one neuron
export interface NeuronTrace {
  label: string;
  terms: WeightTerm[];
  bias: number;
  weightedSum: number;
  activationName: string;
  activation: number;
}

// One layer of neuron traces
export interface LayerTrace {
  name: string;
  neurons: NeuronTrace[];
}

// Loss computation breakdown
export interface LossTrace {
  prediction: number;
  label: number;
  clampedP: number;
  bce: number;
}

// Backward pass record for one neuron
export interface NeuronBackward {
  label: string;
  delta: number;
  formula: string;
}

// Backward pass record for one layer
export interface LayerBackward {
  name: string;
  neurons: NeuronBackward[];
}

// A single weight update record
export interface WeightUpdate {
  layer: string;
  fromLabel: string;
  toLabel: string;
  oldWeight: number;
  gradient: number;
  learningRate: number;
  newWeight: number;
}

// The complete trace object returned by traceStep
export interface StepTrace {
  sample: { x: number; y: number; label: 0 | 1 };
  inputFeatures: { name: string; value: number }[];
  forwardLayers: LayerTrace[];
  loss: LossTrace;
  backward: LayerBackward[];
  updates: WeightUpdate[];
}

// Build a human-readable label for a neuron
function neuronLabel(layerIdx: number, neuronIdx: number, totalLayers: number): string {
  if (layerIdx === 0) return `Input ${neuronIdx + 1}`;
  if (layerIdx === totalLayers - 1) return "Output";
  return `H${layerIdx}N${neuronIdx + 1}`;
}

function layerName(layerIdx: number, totalLayers: number): string {
  if (layerIdx === 0) return "Input";
  if (layerIdx === totalLayers - 1) return "Output";
  return `Hidden ${layerIdx}`;
}

// Run a traced forward + backward pass on a single sample.
// This does NOT mutate the network state.
export function traceStep(net: NeuralNetwork, sample: TrainingSample): StepTrace {
  const { weights, biases, layerSizes } = net.state;
  const act = activations[net.config.activation];
  const activeInputs = (net.config.inputs && net.config.inputs.length > 0)
    ? net.config.inputs
    : DEFAULT_FEATURES;
  const lr = net.config.learningRate;
  const totalLayers = layerSizes.length;

  // --- Extract input features ---
  const feats = extractFeatures(sample.x, sample.y, activeInputs);
  const inputFeatures = activeInputs.map((id, i) => {
    const info = ALL_FEATURES.find(f => f.id === id);
    return { name: info ? info.label : id, value: feats[i] };
  });

  // --- Forward pass with full trace ---
  const allActivations: number[][] = [feats];
  const preSums: number[][] = [feats]; // pre-activation sums (for input layer, same as feats)
  const forwardLayers: LayerTrace[] = [];

  // Input layer trace (no computation, just feature values)
  forwardLayers.push({
    name: "Input",
    neurons: feats.map((v, i) => ({
      label: inputFeatures[i].name,
      terms: [],
      bias: 0,
      weightedSum: v,
      activationName: "identity",
      activation: v,
    })),
  });

  for (let l = 0; l < weights.length; l++) {
    const prev = allActivations[l];
    const isOutput = l === weights.length - 1;
    const layerActivations: number[] = [];
    const layerSums: number[] = [];
    const neurons: NeuronTrace[] = [];

    for (let to = 0; to < layerSizes[l + 1]; to++) {
      const terms: WeightTerm[] = [];
      let sum = biases[l][to];

      for (let from = 0; from < prev.length; from++) {
        const w = weights[l][to][from];
        const inp = prev[from];
        const product = w * inp;
        sum += product;

        // Label for the source neuron
        const fromLbl = l === 0
          ? inputFeatures[from].name
          : neuronLabel(l, from, totalLayers);

        terms.push({ fromLabel: fromLbl, weight: w, input: inp, product });
      }

      const actName = isOutput ? "sigmoid" : net.config.activation;
      const actVal = isOutput ? sigmoid(sum) : act.fn(sum);

      layerSums.push(sum);
      layerActivations.push(actVal);

      neurons.push({
        label: neuronLabel(l + 1, to, totalLayers),
        terms,
        bias: biases[l][to],
        weightedSum: sum,
        activationName: actName,
        activation: actVal,
      });
    }

    allActivations.push(layerActivations);
    preSums.push(layerSums);
    forwardLayers.push({
      name: layerName(l + 1, totalLayers),
      neurons,
    });
  }

  // --- Loss computation ---
  const prediction = allActivations[allActivations.length - 1][0];
  const clampedP = Math.min(Math.max(prediction, 1e-7), 1 - 1e-7);
  const bce = -(sample.label * Math.log(clampedP) + (1 - sample.label) * Math.log(1 - clampedP));
  const loss: LossTrace = { prediction, label: sample.label, clampedP, bce };

  // --- Backward pass ---
  const backward: LayerBackward[] = [];
  let deltas: number[] = [prediction - sample.label];

  // Output layer backward
  backward.push({
    name: "Output",
    neurons: [{
      label: "Output",
      delta: deltas[0],
      formula: `prediction - label = ${prediction.toFixed(4)} - ${sample.label} = ${deltas[0].toFixed(4)}`,
    }],
  });

  // Compute per-layer gradients for update display
  const sampleGradW: number[][][] = weights.map(lw => lw.map(row => row.map(() => 0)));

  // Record gradients for output layer
  {
    const l = weights.length - 1;
    const prevAct = allActivations[l];
    for (let to = 0; to < layerSizes[l + 1]; to++) {
      for (let from = 0; from < prevAct.length; from++) {
        sampleGradW[l][to][from] = deltas[to] * prevAct[from];
      }
    }
  }

  for (let l = weights.length - 1; l > 0; l--) {
    const prevAct = allActivations[l];
    const nextDeltas: number[] = [];
    const neurons: NeuronBackward[] = [];

    for (let from = 0; from < layerSizes[l]; from++) {
      let sum = 0;
      const parts: string[] = [];
      for (let to = 0; to < layerSizes[l + 1]; to++) {
        const contrib = weights[l][to][from] * deltas[to];
        sum += contrib;
        parts.push(`w*d = ${weights[l][to][from].toFixed(4)}*${deltas[to].toFixed(4)}`);
      }
      const dAct = act.dFromOutput(prevAct[from]);
      const delta = sum * dAct;
      nextDeltas.push(delta);

      const lbl = neuronLabel(l, from, totalLayers);
      neurons.push({
        label: lbl,
        delta,
        formula: `(${parts.join(" + ")}) * ${net.config.activation}'(${prevAct[from].toFixed(4)}) = ${delta.toFixed(4)}`,
      });
    }

    deltas = nextDeltas;
    backward.push({
      name: layerName(l, totalLayers),
      neurons,
    });

    // Record gradients for this layer
    const prevPrevAct = allActivations[l - 1];
    for (let to = 0; to < layerSizes[l]; to++) {
      for (let from = 0; from < prevPrevAct.length; from++) {
        sampleGradW[l - 1][to][from] = deltas[to] * prevPrevAct[from];
      }
    }
  }

  // Reverse backward so it reads input-to-output
  backward.reverse();

  // --- Weight updates (show first 8 representative weights) ---
  const updates: WeightUpdate[] = [];
  let count = 0;
  const MAX_UPDATES = 8;

  for (let l = 0; l < weights.length && count < MAX_UPDATES; l++) {
    for (let to = 0; to < weights[l].length && count < MAX_UPDATES; to++) {
      for (let from = 0; from < weights[l][to].length && count < MAX_UPDATES; from++) {
        const oldW = weights[l][to][from];
        const grad = sampleGradW[l][to][from];
        const newW = oldW - lr * grad;

        const fromLbl = l === 0
          ? inputFeatures[from]?.name || `In${from + 1}`
          : neuronLabel(l, from, totalLayers);
        const toLbl = neuronLabel(l + 1, to, totalLayers);

        updates.push({
          layer: layerName(l + 1, totalLayers),
          fromLabel: fromLbl,
          toLabel: toLbl,
          oldWeight: oldW,
          gradient: grad,
          learningRate: lr,
          newWeight: newW,
        });
        count++;
      }
    }
  }

  return {
    sample: { x: sample.x, y: sample.y, label: sample.label },
    inputFeatures,
    forwardLayers,
    loss,
    backward,
    updates,
  };
}
