# neural-network-rbbieee

An interactive neural network visualizer that runs entirely in your browser. The network itself is written from scratch in TypeScript, no TensorFlow, no PyTorch, no math libraries. Every forward pass, every gradient, every weight update is plain code you can read in an afternoon.

The goal is simple: make backpropagation something you can watch instead of something you take on faith.

## What you can do with it

- Train a small feedforward network on four classic 2D datasets: XOR, concentric circles, two spirals, and Gaussian blobs
- Watch the decision boundary reshape itself in real time as the loss goes down
- See every weight in the network drawn as an edge, where thickness means magnitude and color means sign (blue positive, orange negative)
- Click anywhere on the map to probe the network with that point and watch the activations light up layer by layer
- Change the architecture on the fly: add or remove hidden layers, resize them, switch between tanh, ReLU, and sigmoid
- Tune the learning rate mid training and see the loss curve react immediately
- Step through training one epoch at a time when you want to look closely

## Running it locally

You need Node.js 18 or newer.

```bash
git clone https://github.com/rbbieee/neural-network-rbbieee.git
cd neural-network-rbbieee
npm install
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173) and start training.

To build for production:

```bash
npm run build
npm run preview
```

## How the network works

The model is a fully connected feedforward network for binary classification. Inputs are 2D points, the output is a single sigmoid neuron read as the probability of class 1.

**Forward pass.** Each neuron computes a weighted sum of the previous layer plus a bias, then applies the activation function. Hidden layers use whichever activation you pick in the UI. The output layer always uses sigmoid so the result stays between 0 and 1.

**Loss.** Binary cross entropy. It punishes confident wrong answers much harder than hesitant ones, which is exactly the pressure you want on a classifier.

**Backward pass.** Classic backpropagation. Because the output combines sigmoid with cross entropy, the output delta collapses to just `prediction - target`, which is one of the tidier results in machine learning. Deltas then flow backward through the layers using the chain rule, and each weight accumulates a gradient equal to its delta times the activation feeding into it.

**Optimization.** Mini batch gradient descent with a batch size of 20 and shuffling every epoch. Weights start with Xavier style initialization from a seeded random generator, so resetting always gives you the exact same starting point. That makes experiments reproducible: change one setting, reset, and you are comparing apples to apples.

Everything above lives in `src/nn/`, around 250 lines total. Start with `network.ts` if you want to read the math.

## Project structure

```
src/
  nn/
    activations.ts     activation functions and their derivatives
    network.ts         the network, forward pass, backprop, training
    datasets.ts        generators for the four toy datasets
  hooks/
    useTraining.ts     training loop on requestAnimationFrame
  components/
    NetworkGraph.tsx      SVG graph of weights and activations
    DecisionBoundary.tsx  canvas heatmap of predictions over 2D space
    LossChart.tsx         loss sparkline
    ControlPanel.tsx      all the controls
  App.tsx              layout and state wiring
```

## Things worth trying

- Train on the spiral with one hidden layer of 2 neurons. It will fail, and the boundary shows you why: the network simply does not have enough capacity to bend that far.
- Switch to ReLU on the circles dataset and watch the boundary become polygonal. ReLU networks build decisions out of straight creases, and you can see it.
- Crank the learning rate to 1.0 and watch the loss curve oscillate or explode. Then bring it back down and watch it settle.
- Probe points near the boundary. The activations will be indecisive across the whole network, not just at the output.

## Design notes

- Zero runtime dependencies beyond React itself. Fewer packages means a smaller attack surface, faster installs, and nothing to audit but your own code.
- The training loop runs on requestAnimationFrame instead of setInterval, so it pauses when the tab is hidden and stays in sync with rendering.
- The network mutates its weights in place for speed, and a tick counter tells React when to repaint. For a 60x60 heatmap that means 3600 forward passes per frame, which plain arrays handle comfortably.

## License

MIT. See [LICENSE](LICENSE).
