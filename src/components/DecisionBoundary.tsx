// Paints the decision boundary as a heatmap on a canvas.
// Every pixel block is a forward pass over the grid position,
// so the picture is the network's honest opinion about all of 2D space.
// Clicking anywhere probes the network with that point and lights up
// the network graph with the resulting activations.

import { useEffect, useRef, type MouseEvent } from "react";
import type { NeuralNetwork, TrainingSample } from "../nn/network";

import type { DatasetName } from "../nn/datasets";

interface Props {
  network: NeuralNetwork;
  data: TrainingSample[];
  epoch: number;
  datasetName: DatasetName;
  onProbe: (x: number, y: number) => void;
  onAddPoint?: (x: number, y: number, label: 0 | 1) => void;
}

const SIZE = 360;
const GRID = 60; // resolution of the heatmap, 60x60 forward passes

// Map from data space [-1, 1] to canvas pixels
const toPx = (v: number) => ((v + 1) / 2) * SIZE;
const toData = (px: number) => (px / SIZE) * 2 - 1;

export function DecisionBoundary({ network, data, epoch, datasetName, onProbe, onAddPoint }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cell = SIZE / GRID;
    // Clear first. Without this, every re-render (reset, dataset change,
    // each training epoch) painted on top of whatever was already there,
    // so semi-transparent pixels kept darkening and old points from a
    // previous dataset never went away.
    ctx.clearRect(0, 0, SIZE, SIZE);

    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const x = toData((gx + 0.5) * cell);
        const y = toData((gy + 0.5) * cell);
        // Pick one color per pixel, blue for class 1, orange for class 0.
        // Confidence controls opacity instead of blending the two hues,
        // so an undecided network fades toward white rather than
        // painting a muddy third color everywhere.
        const t = network.predict(x, y);
        const confidence = Math.abs(t - 0.5) * 2;
        // Get styles from CSS variables so they update with theme
        const style = getComputedStyle(document.documentElement);
        const colorPos = style.getPropertyValue("--pos").trim() || "#0071e3";
        const colorNeg = style.getPropertyValue("--neg").trim() || "#ff9500";
        
        ctx.fillStyle = t >= 0.5 ? colorPos : colorNeg;
        ctx.globalAlpha = 0.06 + confidence * 0.5;
        ctx.fillRect(gx * cell, gy * cell, cell + 1, cell + 1);
      }
    }
    ctx.globalAlpha = 1;

    // Draw the training points on top
    const style = getComputedStyle(document.documentElement);
    const colorPos = style.getPropertyValue("--pos").trim() || "#0071e3";
    const colorNeg = style.getPropertyValue("--neg").trim() || "#ff9500";
    const colorWindow = style.getPropertyValue("--window").trim() || "#ffffff";

    for (const s of data) {
      ctx.beginPath();
      ctx.arc(toPx(s.x), toPx(s.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = s.label === 1 ? colorPos : colorNeg;
      ctx.fill();
      ctx.strokeStyle = colorWindow;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [network, data, epoch]);

  const handleClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = SIZE / rect.width;
    const x = toData((e.clientX - rect.left) * scale);
    const y = toData((e.clientY - rect.top) * scale);
    
    if (datasetName === "custom" && onAddPoint) {
      // Left click = blue (1), Shift+click or right click = orange (0)
      const label = e.shiftKey || e.button === 2 ? 0 : 1;
      onAddPoint(x, y, label);
    } else {
      onProbe(x, y);
    }
  };

  const handleContextMenu = (e: MouseEvent<HTMLCanvasElement>) => {
    if (datasetName === "custom") {
      e.preventDefault(); // Prevent context menu
      handleClick(e);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      className="boundary-canvas"
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      aria-label="Decision boundary heatmap, click to probe a point or add points in custom mode"
    />
  );
}
