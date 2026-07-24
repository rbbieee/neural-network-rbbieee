// Tutorial step definitions.
// Each step targets a UI element by its data-tour attribute
// and shows a short explanation in plain English.

export interface TutorialStep {
  id: string;
  targetSelector: string;
  title: string;
  body: string;
  position: "top" | "bottom" | "left" | "right";
}

export const tutorialSteps: TutorialStep[] = [
  {
    id: "welcome",
    targetSelector: '[data-tour="header"]',
    title: "Welcome to Neural Network Visualizer",
    body:
      "This tool lets you watch a neural network learn in real time. " +
      "There is no black box library doing the work behind the scenes. " +
      "Every weight, every neuron, and every prediction is rendered live.",
    position: "bottom",
  },
  {
    id: "dataset",
    targetSelector: '[data-tour="dataset"]',
    title: "Pick a Dataset",
    body:
      "These are 2D datasets made of colored dots. " +
      "Each dataset gives the network a different challenge. " +
      "XOR and spirals are difficult because you cannot separate classes with a single straight line.",
    position: "right",
  },
  {
    id: "controls",
    targetSelector: '[data-tour="run-controls"]',
    title: "Start Training",
    body:
      "Click Run to start training. The network will keep learning until you click Stop. " +
      "Use Step to run one epoch at a time to observe changes closely. " +
      "Reset restores the initial weight values.",
    position: "right",
  },
  {
    id: "decision-boundary",
    targetSelector: '[data-tour="decision-boundary"]',
    title: "The Decision Boundary",
    body:
      "This map shows the network predictions across 2D space. " +
      "Red areas indicate prediction for class 1. Dark gray areas indicate class 0. " +
      "Color intensity shows prediction confidence.",
    position: "left",
  },
  {
    id: "data-points",
    targetSelector: '[data-tour="stats"]',
    title: "Training Stats",
    body:
      "These stats update after every training step. " +
      "Epoch counts full passes through the training data. " +
      "Loss measures prediction error, so lower is better. " +
      "Accuracy shows the percentage of correctly classified points.",
    position: "top",
  },
  {
    id: "network-graph",
    targetSelector: '[data-tour="network-graph"]',
    title: "The Network Graph",
    body:
      "This diagram shows neurons and connections. " +
      "Circles represent neurons. Lines represent weights. " +
      "Data flows from left to right: inputs enter on the left and produce a prediction on the right.",
    position: "left",
  },
  {
    id: "weights",
    targetSelector: '[data-tour="network-graph"]',
    title: "Reading the Weights",
    body:
      "Each connection line has a color and thickness. " +
      "Red lines carry positive weights. Dark gray lines carry negative weights. " +
      "Thicker lines have a larger impact on predictions.",
    position: "left",
  },
  {
    id: "probe",
    targetSelector: '[data-tour="decision-boundary"]',
    title: "Probe a Point",
    body:
      "Click anywhere on the decision boundary. " +
      "The network will run a forward pass on that point, and the network graph will highlight the activations for that input.",
    position: "left",
  },
  {
    id: "loss-chart",
    targetSelector: '[data-tour="loss-chart"]',
    title: "Training and Test Loss",
    body:
      "This chart plots loss over time. " +
      "The solid red line is Train Loss and the dashed orange line is Test Loss. " +
      "A healthy run shows both lines decreasing smoothly.",
    position: "left",
  },
  {
    id: "architecture",
    targetSelector: '[data-tour="architecture"]',
    title: "Configure Architecture",
    body:
      "You can select input features, add or remove hidden layers, and pick activation functions. " +
      "Experiment with different settings to see how model capacity affects learning.",
    position: "right",
  },
];

export const sectionInfo: Record<string, { title: string; body: string }> = {
  "decision-boundary": {
    title: "Decision Boundary",
    body:
      "This heatmap visualizes network predictions across 2D space. " +
      "Red indicates class 1 and dark gray indicates class 0. " +
      "Color intensity shows confidence.",
  },
  "network-graph": {
    title: "Weights and Activations",
    body:
      "A diagram of internal structure. " +
      "Circles are neurons and lines are weighted connections. " +
      "Red lines are positive weights and dark gray lines are negative weights.",
  },
  "loss-chart": {
    title: "Training and Test Loss",
    body:
      "A plot of cross-entropy loss over time. " +
      "Lower loss indicates better predictions.",
  },
};

