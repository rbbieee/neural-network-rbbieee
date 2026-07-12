// Activation functions and their derivatives.
// Each derivative takes the activated output, not the raw input.
// This works because tanh and sigmoid derivatives can be written
// in terms of their own output, which saves us from storing raw sums.

export type ActivationName = "tanh" | "relu" | "sigmoid";

export interface Activation {
  fn: (x: number) => number;
  // derivative expressed in terms of the activated value a = fn(x)
  dFromOutput: (a: number) => number;
}

export const activations: Record<ActivationName, Activation> = {
  tanh: {
    fn: (x) => Math.tanh(x),
    // d/dx tanh(x) = 1 - tanh(x)^2
    dFromOutput: (a) => 1 - a * a,
  },
  relu: {
    fn: (x) => (x > 0 ? x : 0),
    // ReLU output is positive exactly when the input was positive
    dFromOutput: (a) => (a > 0 ? 1 : 0),
  },
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-x)),
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    dFromOutput: (a) => a * (1 - a),
  },
};

// The output layer always uses sigmoid so we can read it as a probability
export const sigmoid = activations.sigmoid.fn;
