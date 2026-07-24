# neuroscope.ai

A neural network visualizer that runs entirely in your browser. The network is written from scratch in TypeScript with zero math libraries. Every weight, gradient, and activation is rendered live as the model learns.

## Features

- Train on 5 datasets: XOR, circles, spirals, Gaussian blobs, or draw your own
- Watch the decision boundary reshape in real time
- 7 input features including squared terms and sine transforms
- 3 optimizers: SGD, Momentum, Adam
- L1 and L2 regularization with adjustable strength
- Train/test split with dual loss curves to spot overfitting
- Click the boundary to probe any point and trace activations through the graph
- Step-by-step math mode showing forward pass, loss, gradients, and weight updates with real numbers
- 5 educational preset labs covering feature engineering, XOR, spirals, vanishing gradients, and overfitting
- Built-in guided tutorial

## Running locally

Node.js 18 or newer.

```bash
git clone https://github.com/rbbieee/neural-network-rbbieee.git
cd neural-network-rbbieee
npm install
npm run dev
```

Open http://localhost:5173 and start training.

## Project structure

```
src/
  nn/
    network.ts         feedforward network, backprop, optimizers
    activations.ts     tanh, ReLU, sigmoid and their derivatives
    datasets.ts        dataset generators
    features.ts        input feature transformations
    labs.ts            educational preset configurations
    trace.ts           step-by-step math trace engine
  hooks/
    useTraining.ts     training loop on requestAnimationFrame
  components/
    DecisionBoundary   canvas heatmap with custom paint tools
    NetworkGraph       SVG weight and activation diagram
    LossChart          dual train/test loss curves
    ControlPanel       architecture and hyperparameter controls
    MathPanel          collapsible step-by-step math breakdown
    PresetLabsBar      educational lab selector
    TutorialOverlay    guided walkthrough
    InfoTooltip        section info popovers
  utils/
    audio.ts           Web Audio API sound effects
```

## Keyboard shortcuts

- **Space** toggle training
- **S** step one epoch
- **R** reset

## License

MIT
