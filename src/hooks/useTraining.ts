// Custom hook that owns the network instance and the training loop.
// Training runs on requestAnimationFrame so the UI stays responsive
// and the visuals update in sync with the browser's paint cycle.

import { useCallback, useEffect, useRef, useState } from "react";
import { NeuralNetwork, type NetworkConfig, type TrainingSample } from "../nn/network";
import { generateDataset, type DatasetName } from "../nn/datasets";
import { traceStep, type StepTrace } from "../nn/trace";

const MAX_LOSS_POINTS = 300;

export interface TrainingStats {
  epoch: number;
  loss: number;
  testLoss: number;
  accuracy: number;
  testAccuracy: number;
  lossHistory: number[];
  testLossHistory: number[];
  confusion: { tp: number; fp: number; fn: number; tn: number };
  precision: number;
  recall: number;
  f1: number;
}

export function useTraining(initialConfig: NetworkConfig, initialDataset: DatasetName) {
  const [config, setConfig] = useState(initialConfig);
  const [datasetName, setDatasetName] = useState<DatasetName>(initialDataset);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<TrainingStats>({
    epoch: 0,
    loss: 0,
    testLoss: 0,
    accuracy: 0,
    testAccuracy: 0,
    lossHistory: [],
    testLossHistory: [],
    confusion: { tp: 0, fp: 0, fn: 0, tn: 0 },
    precision: 0,
    recall: 0,
    f1: 0,
  });
  // A counter that forces re renders after each training frame,
  // since the network object itself is mutated in place for speed
  const [, setTick] = useState(0);
  const [lastTrace, setLastTrace] = useState<StepTrace | null>(null);
  const [noiseLevel, setNoiseLevel] = useState(0.5);
  const [epochsPerFrame, setEpochsPerFrame] = useState(4);

  const networkRef = useRef(new NeuralNetwork(initialConfig));
  const dataRef = useRef<TrainingSample[]>(generateDataset(initialDataset, config.seed, noiseLevel));
  const rafRef = useRef(0);

  // Rebuild the network whenever architecture settings change.
  // Learning rate changes alone do not need a rebuild.
  const rebuild = useCallback((next: NetworkConfig) => {
    networkRef.current = new NeuralNetwork(next);
    setStats({
      epoch: 0,
      loss: 0,
      testLoss: 0,
      accuracy: 0,
      testAccuracy: 0,
      lossHistory: [],
      testLossHistory: [],
      confusion: { tp: 0, fp: 0, fn: 0, tn: 0 },
      precision: 0,
      recall: 0,
      f1: 0,
    });
    setTick((t) => t + 1);
  }, []);

  const updateConfig = useCallback(
    (patch: Partial<NetworkConfig>) => {
      setConfig((prev) => {
        const next = { ...prev, ...patch };
        const structural =
          patch.hiddenLayers !== undefined ||
          patch.inputs !== undefined ||
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
      dataRef.current = generateDataset(name, networkRef.current.config.seed, noiseLevel);
      setRunning(false);
      rebuild(networkRef.current.config);
    },
    [rebuild, noiseLevel]
  );

  const changeNoise = useCallback(
    (level: number) => {
      setNoiseLevel(level);
      dataRef.current = generateDataset(datasetName, networkRef.current.config.seed, level);
      setRunning(false);
      rebuild(networkRef.current.config);
    },
    [datasetName, rebuild]
  );

  // Run a few epochs of mini batch training
  const runEpochs = useCallback((count: number) => {
    const net = networkRef.current;
    const fullData = dataRef.current;
    if (fullData.length === 0) return;

    // 80% Train, 20% Test split (interleaved so spatial distribution is uniform)
    const trainData = fullData.filter((_, i) => i % 5 !== 0);
    const testData = fullData.filter((_, i) => i % 5 === 0);

    let loss = 0;
    for (let e = 0; e < count; e++) {
      const idx = trainData.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let b = 0; b < idx.length; b += net.config.batchSize) {
        const batch = idx.slice(b, b + net.config.batchSize).map((i) => trainData[i]);
        loss = net.trainBatch(batch);
      }
    }

    // Evaluate test loss & accuracy
    let evalTestLoss = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;
    
    if (testData.length > 0) {
      for (const s of testData) {
        const out = net.predict(s.x, s.y);
        const p = Math.min(Math.max(out, 1e-7), 1 - 1e-7);
        evalTestLoss -= s.label * Math.log(p) + (1 - s.label) * Math.log(1 - p);
        
        const predLabel = out >= 0.5 ? 1 : 0;
        if (s.label === 1 && predLabel === 1) tp++;
        else if (s.label === 0 && predLabel === 1) fp++;
        else if (s.label === 1 && predLabel === 0) fn++;
        else if (s.label === 0 && predLabel === 0) tn++;
      }
      evalTestLoss /= testData.length;
    } else {
      evalTestLoss = loss;
    }
    
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    setStats((prev) => {
      const history = [...prev.lossHistory, loss].slice(-MAX_LOSS_POINTS);
      const testHistory = [...prev.testLossHistory, evalTestLoss].slice(-MAX_LOSS_POINTS);
      return {
        epoch: prev.epoch + count,
        loss,
        testLoss: evalTestLoss,
        accuracy: net.accuracy(trainData),
        testAccuracy: testData.length > 0 ? net.accuracy(testData) : net.accuracy(trainData),
        lossHistory: history,
        testLossHistory: testHistory,
        confusion: { tp, fp, fn, tn },
        precision,
        recall,
        f1,
      };
    });
    setTick((t) => t + 1);
  }, []);

  // The animation loop, active only while running is true
  useEffect(() => {
    if (!running) return;
    const loop = () => {
      runEpochs(epochsPerFrame);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, runEpochs, epochsPerFrame]);

  const reset = useCallback(() => {
    setRunning(false);
    setLastTrace(null);
    if (datasetName === "custom") {
      dataRef.current = [];
    }
    rebuild(networkRef.current.config);
  }, [rebuild, datasetName]);

  const addSample = useCallback((x: number, y: number, label: 0 | 1) => {
    // Reassign with a new array so React detects the reference change
    dataRef.current = [...dataRef.current, { x, y, label }];
    // Force a re-render so the new point shows up immediately
    setTick((t) => t + 1);
  }, []);

  // Run a single step with trace capture
  const stepWithTrace = useCallback(() => {
    const net = networkRef.current;
    const fullData = dataRef.current;
    if (fullData.length === 0) return;

    // Pick a training sample to trace (first one in the shuffled data)
    const trainData = fullData.filter((_, i) => i % 5 !== 0);
    if (trainData.length > 0) {
      const traceSample = trainData[Math.floor(Math.random() * trainData.length)];
      setLastTrace(traceStep(net, traceSample));
    }

    runEpochs(1);
  }, [runEpochs]);

  return {
    network: networkRef.current,
    data: dataRef.current,
    config,
    datasetName,
    running,
    stats,
    lastTrace,
    noiseLevel,
    epochsPerFrame,
    setRunning,
    setEpochsPerFrame,
    updateConfig,
    changeDataset,
    changeNoise,
    step: stepWithTrace,
    reset,
    addSample,
  };
}
