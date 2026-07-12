// All the knobs and switches.
// Changing architecture or activation rebuilds the network,
// changing the learning rate applies immediately.

import type { NetworkConfig } from "../nn/network";
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

const MAX_LAYERS = 4;
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
      <div className="panel-row">
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

      <label className="field">
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
        <span>
          Learning rate <code>{config.learningRate.toFixed(2)}</code>
        </span>
        <input
          type="range"
          min={0.01}
          max={1}
          step={0.01}
          value={config.learningRate}
          onChange={(e) =>
            onConfigChange({ learningRate: Number(e.target.value) })
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
  );
}
