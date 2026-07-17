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
    title: "Welcome to the Neural Network Visualizer",
    body:
      "This tool lets you watch a small neural network learn in real time. " +
      "There is no hidden math library doing the work behind the scenes. " +
      "Every weight, every neuron, and every prediction is rendered live so you can see exactly what is happening. " +
      "Let's walk through each part of the interface.",
    position: "bottom",
  },
  {
    id: "dataset",
    targetSelector: '[data-tour="dataset"]',
    title: "Pick a Dataset",
    body:
      "These are toy datasets made of colored dots on a 2D plane. " +
      "Each dataset gives the network a different challenge. " +
      "XOR and spirals are hard because you cannot separate the classes with a straight line. " +
      "Blobs are the easiest since the two groups barely overlap.",
    position: "right",
  },
  {
    id: "controls",
    targetSelector: '[data-tour="run-controls"]',
    title: "Start Training",
    body:
      "Hit Run to start training. The network will keep learning until you press Stop. " +
      "Use Step to run just one epoch at a time so you can watch changes closely. " +
      "Reset brings everything back to the starting weights.",
    position: "right",
  },
  {
    id: "decision-boundary",
    targetSelector: '[data-tour="decision-boundary"]',
    title: "The Decision Boundary",
    body:
      "This colored map shows the network's opinion about every point in 2D space. " +
      "Blue areas are where it predicts class 1. Orange areas are where it predicts class 0. " +
      "Strong color means high confidence. Faded color means the network is unsure. " +
      "As training goes on, you will see the colors shift and sharpen.",
    position: "left",
  },
  {
    id: "data-points",
    targetSelector: '[data-tour="stats"]',
    title: "Training Stats",
    body:
      "These numbers update after every training step. " +
      "Epoch counts how many full passes the network has made through the data. " +
      "Loss measures how wrong the predictions are, so lower is better. " +
      "Accuracy tells you what percentage of points the network classifies correctly.",
    position: "top",
  },
  {
    id: "network-graph",
    targetSelector: '[data-tour="network-graph"]',
    title: "The Network Graph",
    body:
      "This diagram shows every neuron and every connection inside the network. " +
      "The circles are neurons. The lines between them are weights. " +
      "Data flows from left to right: input goes in, passes through the hidden layers, and comes out as a prediction on the right side.",
    position: "left",
  },
  {
    id: "weights",
    targetSelector: '[data-tour="network-graph"]',
    title: "Reading the Weights",
    body:
      "Each line has a color and a thickness. " +
      "Blue lines carry positive weights, which means they amplify the signal. " +
      "Orange lines carry negative weights, which means they flip or suppress the signal. " +
      "Thicker lines have a bigger effect on the output. You can watch them grow and shrink during training.",
    position: "left",
  },
  {
    id: "probe",
    targetSelector: '[data-tour="decision-boundary"]',
    title: "Probe a Point",
    body:
      "Click anywhere on the decision boundary map. " +
      "The network will run a forward pass on that exact point, and the graph on the right will light up to show the activation at each neuron. " +
      "This lets you trace how a single input travels through the network and turns into a prediction.",
    position: "left",
  },
  {
    id: "loss-chart",
    targetSelector: '[data-tour="loss-chart"]',
    title: "Training Loss",
    body:
      "This sparkline charts the loss over time. " +
      "A healthy training run shows the line dropping quickly at first and then flattening out near zero. " +
      "If the line bounces around wildly, the learning rate might be too high. " +
      "If it barely moves, the network might not have enough neurons to solve the problem.",
    position: "left",
  },
  {
    id: "architecture",
    targetSelector: '[data-tour="architecture"]',
    title: "Change the Architecture",
    body:
      "You can add or remove hidden layers and change how many neurons each one has. " +
      "More neurons give the network more capacity to learn complex shapes. " +
      "The activation function controls how neurons transform their inputs. " +
      "Try switching between tanh, ReLU, and sigmoid to see how the decision boundary changes shape. " +
      "The learning rate controls how big each update step is. Too high and training gets unstable. Too low and it takes forever.",
    position: "right",
  },
];

// Shorter explanations used by the info tooltips next to section headers.
// These are always available without running the full tutorial.
export const sectionInfo: Record<string, { title: string; body: string }> = {
  "decision-boundary": {
    title: "Decision Boundary",
    body:
      "This heatmap visualizes the network's predictions across all of 2D space. " +
      "Blue means the network predicts class 1, orange means class 0. " +
      "Color intensity shows confidence. Click anywhere to probe that point and see the activations light up in the network graph.",
  },
  "network-graph": {
    title: "Weights & Activations",
    body:
      "A diagram of the network's internal structure. " +
      "Circles are neurons, lines are weighted connections. " +
      "Blue lines are positive weights (they amplify), orange lines are negative (they suppress). " +
      "Thicker lines have larger magnitude. When you probe a point, each neuron shows its activation value.",
  },
  "loss-chart": {
    title: "Training Loss",
    body:
      "A plot of the binary cross-entropy loss over time. " +
      "Loss tells you how wrong the network's predictions are. " +
      "You want this number to go down. If it flattens out at a high value, the network might need more capacity. " +
      "If it oscillates, try lowering the learning rate.",
  },
};
