// Renders the network as an SVG graph.
// Edge thickness shows weight magnitude, color shows sign.
// Neuron fill shows the activation from the most recent forward pass,
// so you can literally watch a data point flow through the network.

import { useMemo } from "react";
import type { NeuralNetwork } from "../nn/network";

interface Props {
  network: NeuralNetwork;
  probeActivations: number[][] | null;
}

const WIDTH = 560;
const HEIGHT = 360;
const PAD_X = 60;
const PAD_Y = 40;

function neuronPositions(layerSizes: number[]) {
  return layerSizes.map((size, l) => {
    const x = PAD_X + (l / (layerSizes.length - 1)) * (WIDTH - PAD_X * 2);
    return Array.from({ length: size }, (_, i) => {
      const y =
        size === 1
          ? HEIGHT / 2
          : PAD_Y + (i / (size - 1)) * (HEIGHT - PAD_Y * 2);
      return { x, y };
    });
  });
}

export function NetworkGraph({ network, probeActivations }: Props) {
  const { layerSizes, weights } = network.state;
  const positions = useMemo(() => neuronPositions(layerSizes), [layerSizes]);

  // Normalize edge widths against the largest weight in the network
  let maxW = 0.1;
  for (const lw of weights)
    for (const row of lw)
      for (const w of row) maxW = Math.max(maxW, Math.abs(w));

  let maxG = 0.000001;
  if (network.lastGradients && network.lastGradients.length > 0) {
    for (const lg of network.lastGradients)
      for (const row of lg)
        for (const g of row) maxG = Math.max(maxG, Math.abs(g));
  }

  // Only show live numbers when the user has explicitly probed a point.
  // network.lastActivations gets overwritten on every heatmap pixel pass,
  // so falling back to it here would show noise from wherever the last
  // grid cell happened to be, not a real prediction the user asked for.
  const acts = probeActivations;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="network-graph"
      role="img"
      aria-label="Neural network structure with weighted connections"
    >
      {/* Edges first so neurons draw on top */}
      {weights.map((layerW, l) =>
        layerW.map((row, to) =>
          row.map((w, from) => {
            const a = positions[l][from];
            const b = positions[l + 1][to];
            const strength = Math.abs(w) / maxW;
            const grad = network.lastGradients && network.lastGradients.length > 0
              ? Math.abs(network.lastGradients[l][to][from])
              : 0;
            const gradStrength = grad / maxG;
            // Only animate gradients that are relatively active (top 10%)
            const showGradient = gradStrength > 0.1;

            return (
              <g key={`e-${l}-${to}-${from}`}>
                <line
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={w >= 0 ? "var(--pos)" : "var(--neg)"}
                  strokeWidth={0.5 + strength * 4}
                  strokeOpacity={0.25 + strength * 0.65}
                />
                {showGradient && (
                  <circle r={2 + gradStrength * 2} fill="#ffdd00" opacity={0.6 + gradStrength * 0.4}>
                    <animateMotion
                      dur={`${Math.max(0.2, 1.5 - gradStrength)}s`}
                      repeatCount="indefinite"
                      path={`M${b.x},${b.y} L${a.x},${a.y}`}
                    />
                  </circle>
                )}
              </g>
            );
          })
        )
      )}
      {/* Neurons */}
      {positions.map((layer, l) =>
        layer.map((p, i) => {
          const a = acts?.[l]?.[i];
          const glow = a === undefined ? 0 : Math.min(Math.abs(a), 1);
          return (
            <g key={`n-${l}-${i}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={14}
                className="neuron"
                style={{ fillOpacity: 0.25 + glow * 0.75 }}
              />
              {a !== undefined && (
                <text x={p.x} y={p.y + 3.5} className="neuron-value">
                  {a.toFixed(1)}
                </text>
              )}
            </g>
          );
        })
      )}
      {/* Layer captions */}
      {positions.map((layer, l) => (
        <text
          key={`cap-${l}`}
          x={layer[0].x}
          y={HEIGHT - 8}
          className="layer-caption"
        >
          {l === 0 ? "input" : l === positions.length - 1 ? "output" : `hidden ${l}`}
        </text>
      ))}
    </svg>
  );
}
