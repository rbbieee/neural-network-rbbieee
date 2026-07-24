// Educational Preset Labs.
// Pre-configured experiments designed to demonstrate key machine learning phenomena.

import type { NetworkConfig } from "./network";
import type { DatasetName } from "./datasets";

export interface PresetLab {
  id: string;
  title: string;
  badge: string;
  summary: string;
  explanation: string;
  dataset: DatasetName;
  config: NetworkConfig;
}

export const PRESET_LABS: PresetLab[] = [
  {
    id: "circle_features",
    title: "Feature Engineering Miracle",
    badge: "Geometry",
    summary: "Solve non-linear Concentric Circles with 0 hidden layers using X1^2 + X2^2",
    explanation:
      "A linear model (0 hidden layers) cannot separate concentric circles in raw (X1, X2) space. Providing transformed features X1^2 and X2^2 renders the problem linearly separable.",
    dataset: "circles",
    config: {
      inputs: ["x1", "x2", "x1_sq", "x2_sq"],
      hiddenLayers: [],
      activation: "tanh",
      learningRate: 0.1,
      seed: 42,
      optimizer: "adam",
      regularization: "none",
      regularizationRate: 0,
      batchSize: 20,
    },
  },
  {
    id: "xor_shortcut",
    title: "The XOR Solution",
    badge: "Logic Gate",
    summary: "Solve the XOR problem with 0 hidden layers using X1X2",
    explanation:
      "Single-layer perceptrons historically failed on XOR. Adding the product feature X1X2 transforms XOR into a simple 1D decision boundary.",
    dataset: "xor",
    config: {
      inputs: ["x1", "x2", "x1_x2"],
      hiddenLayers: [],
      activation: "tanh",
      learningRate: 0.1,
      seed: 1337,
      optimizer: "adam",
      regularization: "none",
      regularizationRate: 0,
      batchSize: 20,
    },
  },
  {
    id: "spiral_adam",
    title: "Deep Spirals and Adam",
    badge: "Challenge",
    summary: "Watch Adam optimizer solve the complex 2-Spirals dataset",
    explanation:
      "Interleaved spirals require non-linear decision boundaries. Three hidden layers combined with Adam's momentum find the spiral geometry smoothly.",
    dataset: "spiral",
    config: {
      inputs: ["x1", "x2"],
      hiddenLayers: [8, 8, 4],
      activation: "tanh",
      learningRate: 0.03,
      seed: 777,
      optimizer: "adam",
      regularization: "none",
      regularizationRate: 0,
      batchSize: 20,
    },
  },
  {
    id: "vanishing_gradient",
    title: "Vanishing Gradient Problem",
    badge: "Deep Learning",
    summary: "Compare Sigmoid vs ReLU across 5 deep hidden layers",
    explanation:
      "Using Sigmoid across 5 deep layers causes derivatives to saturate near zero, halting early layer updates. Switching activation to ReLU restores gradient flow.",
    dataset: "spiral",
    config: {
      inputs: ["x1", "x2"],
      hiddenLayers: [6, 6, 6, 6, 6],
      activation: "sigmoid",
      learningRate: 0.05,
      seed: 101,
      optimizer: "sgd",
      regularization: "none",
      regularizationRate: 0,
      batchSize: 20,
    },
  },
  {
    id: "overfitting_l2",
    title: "Overfitting and L2 Regularization",
    badge: "Generalization",
    summary: "Prevent complex model overfitting on noisy Gaussian Blobs",
    explanation:
      "An oversized network trained on noisy data will overfit. Notice Test Loss rising while Train Loss decreases. L2 Regularization smooths the decision boundary.",
    dataset: "blobs",
    config: {
      inputs: ["x1", "x2", "sin_x1", "sin_x2"],
      hiddenLayers: [8, 8, 6],
      activation: "relu",
      learningRate: 0.05,
      seed: 999,
      optimizer: "adam",
      regularization: "l2",
      regularizationRate: 0.01,
      batchSize: 10,
    },
  },
];

