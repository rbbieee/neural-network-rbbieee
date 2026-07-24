import { useState } from "react";
import { PRESET_LABS, type PresetLab } from "../nn/labs";
import type { NetworkConfig } from "../nn/network";
import type { DatasetName } from "../nn/datasets";

interface Props {
  onSelectLab: (config: NetworkConfig, dataset: DatasetName) => void;
}

export function PresetLabsBar({ onSelectLab }: Props) {
  const [activeLab, setActiveLab] = useState<PresetLab | null>(null);

  const handleSelect = (lab: PresetLab) => {
    setActiveLab(lab);
    onSelectLab(lab.config, lab.dataset);
  };

  return (
    <div className="preset-labs-container">
      <div className="preset-labs-header">
        <span className="preset-labs-title">Educational Labs:</span>
        <div className="preset-labs-list">
          {PRESET_LABS.map((lab) => (
            <button
              key={lab.id}
              className={`preset-lab-btn ${activeLab?.id === lab.id ? "active" : ""}`}
              onClick={() => handleSelect(lab)}
              title={lab.summary}
            >
              <span className="lab-badge">{lab.badge}</span>
              <span className="lab-title">{lab.title}</span>
            </button>
          ))}
        </div>
      </div>

      {activeLab && (
        <div className="lab-banner">
          <div className="lab-banner-content">
            <strong>{activeLab.title}:</strong> {activeLab.explanation}
          </div>
          <button
            className="lab-banner-close"
            onClick={() => setActiveLab(null)}
            title="Dismiss banner"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
