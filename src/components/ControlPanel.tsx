// All the knobs and switches.
// Changing architecture or activation rebuilds the network,
// changing the learning rate applies immediately.

import type { NetworkConfig } from "../nn/network";
import type { OptimizerName, RegularizationName } from "../nn/network";
import type { ActivationName } from "../nn/activations";
import { datasetLabels, type DatasetName } from "../nn/datasets";

interface Props {
  config: NetworkConfig;
  datasetName: DatasetName;
  running: boolean;
  onToggleRun: () => void;
  onStep: () => void;
  onReset: () => void;
  onConfigChange: (patch: Partial<NetworkConfig>) => void;
  onDatasetChange: (name: DatasetName) => void;
}

const MAX_LAYERS = 6;
const MAX_NEURONS = 8;

export function ControlPanel({
  config,
  datasetName,
  running,
  onToggleRun,
  onStep,
  onReset,
  onConfigChange,
  onDatasetChange,
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
    if (layers.length > 1)
      onConfigChange({ hiddenLayers: layers.slice(0, -1) });
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
            Hidden layers
            <button className="btn tiny" onClick={removeLayer} disabled={layers.length <= 1}>
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
