// Custom hook that owns the network instance and the training loop.
// Training runs on requestAnimationFrame so the UI stays responsive
// and the visuals update in sync with the browser's paint cycle.

import { useCallback, useEffect, useRef, useState } from "react";
import { NeuralNetwork, type NetworkConfig, type TrainingSample } from "../nn/network";
import { generateDataset, type DatasetName } from "../nn/datasets";

const EPOCHS_PER_FRAME = 4;
const MAX_LOSS_POINTS = 300;

export interface TrainingStats {
  epoch: number;
  loss: number;
  accuracy: number;
  lossHistory: number[];
}

export function useTraining(initialConfig: NetworkConfig, initialDataset: DatasetName) {
  const [config, setConfig] = useState(initialConfig);
  const [datasetName, setDatasetName] = useState<DatasetName>(initialDataset);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<TrainingStats>({
    epoch: 0,
    loss: 0,
    accuracy: 0,
    lossHistory: [],
  });
  // A counter that forces re renders after each training frame,
  // since the network object itself is mutated in place for speed
  const [, setTick] = useState(0);

  const networkRef = useRef(new NeuralNetwork(initialConfig));
  const dataRef = useRef<TrainingSample[]>(generateDataset(initialDataset));
  const rafRef = useRef(0);

  // Rebuild the network whenever architecture settings change.
  // Learning rate changes alone do not need a rebuild.
  const rebuild = useCallback((next: NetworkConfig) => {
    networkRef.current = new NeuralNetwork(next);
    setStats({ epoch: 0, loss: 0, accuracy: 0, lossHistory: [] });
    setTick((t) => t + 1);
  }, []);

  const updateConfig = useCallback(
    (patch: Partial<NetworkConfig>) => {
      setConfig((prev) => {
        const next = { ...prev, ...patch };
        const structural =
          patch.hiddenLayers !== undefined ||
          patch.activation !== undefined ||
          patch.seed !== undefined;
        if (structural) {
          setRunning(false);
          rebuild(next);
        } else {
          networkRef.current.config = next;
        }
        return next;
      });
    },
    [rebuild]
  );

  const changeDataset = useCallback(
    (name: DatasetName) => {
      setDatasetName(name);
      dataRef.current = generateDataset(name);
      setRunning(false);
      rebuild(networkRef.current.config);
    },
    [rebuild]
  );

  // Run a few epochs of mini batch training
  const runEpochs = useCallback((count: number) => {
    const net = networkRef.current;
    const data = dataRef.current;
    let loss = 0;
    for (let e = 0; e < count; e++) {
      // Shuffle indices each epoch for better convergence
      const idx = data.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let b = 0; b < idx.length; b += net.config.batchSize) {
        const batch = idx.slice(b, b + net.config.batchSize).map((i) => data[i]);
        loss = net.trainBatch(batch);
      }
    }
    setStats((prev) => {
      const history = [...prev.lossHistory, loss].slice(-MAX_LOSS_POINTS);
      return {
        epoch: prev.epoch + count,
        loss,
        accuracy: net.accuracy(data),
        lossHistory: history,
      };
    });
    setTick((t) => t + 1);
  }, []);

  // The animation loop, active only while running is true
  useEffect(() => {
    if (!running) return;
    const loop = () => {
      runEpochs(EPOCHS_PER_FRAME);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, runEpochs]);

  const reset = useCallback(() => {
    setRunning(false);
    rebuild(networkRef.current.config);
  }, [rebuild]);

  const addSample = useCallback((x: number, y: number, label: 0 | 1) => {
    dataRef.current.push({ x, y, label });
    // If we're not running, force a re-render so the new point shows up
    if (!running) setTick((t) => t + 1);
  }, [running]);

  return {
    network: networkRef.current,
    data: dataRef.current,
    config,
    datasetName,
    running,
    stats,
    setRunning,
    updateConfig,
    changeDataset,
    step: () => runEpochs(1),
    reset,
    addSample,
  };
}
