// All the knobs and switches.
// Changing architecture or activation rebuilds the network,
// changing the learning rate applies immediately.

import type { NetworkConfig } from "../nn/network";
import type { OptimizerName, RegularizationName, WeightInitName } from "../nn/network";
import type { ActivationName } from "../nn/activations";
import { datasetLabels, type DatasetName } from "../nn/datasets";
import { ALL_FEATURES, type FeatureId } from "../nn/features";

interface Props {
  config: NetworkConfig;
  datasetName: DatasetName;
  running: boolean;
  noiseLevel: number;
  epochsPerFrame: number;
  onToggleRun: () => void;
  onStep: () => void;
  onReset: () => void;
  onConfigChange: (patch: Partial<NetworkConfig>) => void;
  onDatasetChange: (name: DatasetName) => void;
  onChangeNoise: (level: number) => void;
  onChangeSpeed: (speed: number) => void;
}

const MAX_LAYERS = 6;
const MAX_NEURONS = 8;

export function ControlPanel({
  config,
  datasetName,
  running,
  noiseLevel,
  epochsPerFrame,
  onToggleRun,
  onStep,
  onReset,
  onConfigChange,
  onDatasetChange,
  onChangeNoise,
  onChangeSpeed,
}: Props) {
  const layers = config.hiddenLayers;

  const setNeurons = (i: number, value: number) => {
    const next = [...layers];
    next[i] = value;
    onConfigChange({ hiddenLayers: next });
  };

  const addLayer = () => {
    if (layers.length < MAX_LAYERS)
      onConfigChange({ hiddenLayers: [...layers, 4] });
  };

  const removeLayer = () => {
    if (layers.length > 0)
      onConfigChange({ hiddenLayers: layers.slice(0, -1) });
  };

  const toggleFeature = (featId: FeatureId) => {
    const current = config.inputs || ["x1", "x2"];
    let next: FeatureId[];
    if (current.includes(featId)) {
      if (current.length <= 1) return; // Must have at least 1 feature
      next = current.filter((f) => f !== featId);
    } else {
      next = [...current, featId];
    }
    onConfigChange({ inputs: next });
  };

  return (
    <div className="panel">
      <div className="panel-row" data-tour="run-controls">
        <button
          className={`btn primary${running ? " live" : ""}`}
          onClick={onToggleRun}
        >
          {running ? "Stop" : "Run"}
        </button>
        <button className="btn" onClick={onStep} disabled={running}>
          Step
        </button>
        <button className="btn" onClick={onReset}>
          Reset
        </button>
        <select
          className="speed-select"
          value={epochsPerFrame}
          onChange={(e) => onChangeSpeed(Number(e.target.value))}
        >
          <option value={1}>1x Speed</option>
          <option value={2}>2x Speed</option>
          <option value={4}>4x Speed</option>
          <option value={8}>8x Speed</option>
          <option value={16}>16x Speed</option>
        </select>
      </div>

      <label className="field" data-tour="dataset">
        <span>Dataset</span>
        <select
          value={datasetName}
          onChange={(e) => onDatasetChange(e.target.value as DatasetName)}
        >
          {Object.entries(datasetLabels).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>
          Noise <code>{noiseLevel.toFixed(2)}</code>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={noiseLevel}
          onChange={(e) => onChangeNoise(Number(e.target.value))}
        />
      </label>

      <div className="field" data-tour="features">
        <span>Features (Inputs)</span>
        <div className="features-grid">
          {ALL_FEATURES.map((feat) => {
            const active = (config.inputs || ["x1", "x2"]).includes(feat.id);
            return (
              <button
                key={feat.id}
                type="button"
                className={`feature-chip ${active ? "active" : ""}`}
                onClick={() => toggleFeature(feat.id)}
              >
                {feat.label}
              </button>
            );
          })}
        </div>
      </div>

      <div data-tour="architecture">
        <label className="field">
          <span>Activation</span>
          <select
            value={config.activation}
            onChange={(e) =>
              onConfigChange({ activation: e.target.value as ActivationName })
            }
          >
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">sigmoid</option>
          </select>
        </label>

        <label className="field">
          <span>Weight Init</span>
          <select
            value={config.weightInit || "xavier"}
            onChange={(e) =>
              onConfigChange({ weightInit: e.target.value as WeightInitName })
            }
          >
            <option value="xavier">Xavier</option>
            <option value="he">He</option>
            <option value="zero">Zero</option>
          </select>
        </label>

        <label className="field">
          <span>Optimizer</span>
          <select
            value={config.optimizer}
            onChange={(e) =>
              onConfigChange({ optimizer: e.target.value as OptimizerName })
            }
          >
            <option value="sgd">SGD</option>
            <option value="momentum">Momentum</option>
            <option value="adam">Adam</option>
          </select>
        </label>

        <label className="field">
          <span>
            Learning rate <code>{config.learningRate.toFixed(3)}</code>
          </span>
          <input
            type="range"
            min={0.001}
            max={1}
            step={0.001}
            value={config.learningRate}
            onChange={(e) =>
              onConfigChange({ learningRate: Number(e.target.value) })
            }
          />
        </label>

        <label className="field">
          <span>Regularization</span>
          <select
            value={config.regularization}
            onChange={(e) =>
              onConfigChange({ regularization: e.target.value as RegularizationName })
            }
          >
            <option value="none">None</option>
            <option value="l1">L1</option>
            <option value="l2">L2</option>
          </select>
        </label>

        {config.regularization !== "none" && (
          <label className="field">
            <span>
              Reg. rate <code>{config.regularizationRate.toFixed(4)}</code>
            </span>
            <input
              type="range"
              min={0.0001}
              max={0.1}
              step={0.0001}
              value={config.regularizationRate}
              onChange={(e) =>
                onConfigChange({ regularizationRate: Number(e.target.value) })
              }
            />
          </label>
        )}

        {layers.length > 0 && (
          <label className="field">
            <span>
              Dropout <code>{(config.dropoutRate || 0).toFixed(2)}</code>
            </span>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.05}
              value={config.dropoutRate || 0}
              onChange={(e) =>
                onConfigChange({ dropoutRate: Number(e.target.value) })
              }
            />
          </label>
        )}

        <label className="field">
          <span>
            Batch size <code>{config.batchSize}</code>
          </span>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={config.batchSize}
            onChange={(e) =>
              onConfigChange({ batchSize: Number(e.target.value) })
            }
          />
        </label>

        <div className="field">
          <span>
            Hidden layers ({layers.length})
            <button className="btn tiny" onClick={removeLayer} disabled={layers.length <= 0}>
              -
            </button>
            <button className="btn tiny" onClick={addLayer} disabled={layers.length >= MAX_LAYERS}>
              +
            </button>
          </span>
          {layers.map((n, i) => (
            <label key={i} className="layer-slider">
              <code>layer {i + 1}: {n} neurons</code>
              <input
                type="range"
                min={1}
                max={MAX_NEURONS}
                step={1}
                value={n}
                onChange={(e) => setNeurons(i, Number(e.target.value))}
              />
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
