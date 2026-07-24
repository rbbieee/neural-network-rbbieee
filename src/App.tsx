// neural-network-rbbieee root component.
// Rendered as a macOS style window: controls in the sidebar,
// the decision boundary as the main view, network state and
// loss trace in the panel on the right.

import { useState, useEffect } from "react";
import { useTraining } from "./hooks/useTraining";
import { NetworkGraph } from "./components/NetworkGraph";
import { DecisionBoundary } from "./components/DecisionBoundary";
import { LossChart } from "./components/LossChart";
import { ControlPanel } from "./components/ControlPanel";
import { TutorialOverlay } from "./components/TutorialOverlay";
import { InfoTooltip } from "./components/InfoTooltip";
import { PresetLabsBar } from "./components/PresetLabsBar";
import { MathPanel } from "./components/MathPanel";
import { soundManager } from "./utils/audio";
import { type NetworkConfig } from "./nn/network";

const INITIAL_CONFIG: NetworkConfig = {
  inputs: ["x1", "x2"],
  hiddenLayers: [4, 4],
  activation: "tanh",
  learningRate: 0.3,
  seed: 1337,
  optimizer: "sgd",
  regularization: "none",
  regularizationRate: 0.001,
  batchSize: 20,
};

export default function App() {
  const t = useTraining(INITIAL_CONFIG, "circles");
  const [probe, setProbe] = useState<number[][] | null>(null);
  const [showTutorial, setShowTutorial] = useState(false);

  // Set dark obsidian theme by default
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "dark");
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      
      if (e.code === "Space") {
        e.preventDefault();
        t.setRunning(!t.running);
      } else if (e.code === "KeyS" && !t.running) {
        t.step();
      } else if (e.code === "KeyR") {
        setProbe(null);
        t.reset();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [t.running, t.setRunning, t.step, t.reset]);

  const handleProbe = (x: number, y: number) => {
    setProbe(t.network.forward(x, y));
  };

  return (
    <>
      <main className="app">
        <div className="titlebar">
          <div className="traffic-lights">
            <span className="tl-red" />
            <span className="tl-yellow" />
            <span className="tl-green" />
          </div>
          <span className="titlebar-title">neuroscope.ai</span>
        </div>

        <header className="header" data-tour="header">
          <div className="header-top">
            <h1>
              neuroscope.<span className="accent">ai</span>
            </h1>
            <button
              className="btn guide-btn"
              onClick={() => setShowTutorial(true)}
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                <path d="M8 5.5V8.5M8 11.5V12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              Guide
            </button>
          </div>
          <p className="tagline">
            Real-time readout of neural backpropagation. Every <span className="unit">weight</span>{" "}
            and <span className="unit">activation</span> rendered live as it learns.
          </p>
        </header>

        <PresetLabsBar
          onSelectLab={(config, dataset) => {
            setProbe(null);
            t.changeDataset(dataset);
            t.updateConfig(config);
            soundManager.playClick();
          }}
        />

        <div className="layout">
          <ControlPanel
            config={t.config}
            datasetName={t.datasetName}
            running={t.running}
            onToggleRun={() => {
              soundManager.playClick();
              t.setRunning(!t.running);
            }}
            onStep={() => {
              soundManager.playClick();
              t.step();
            }}
            onReset={() => {
              soundManager.playReset();
              setProbe(null);
              t.reset();
            }}
            onConfigChange={(patch) => {
              setProbe(null);
              t.updateConfig(patch);
            }}
            onDatasetChange={(name) => {
              soundManager.playClick();
              setProbe(null);
              t.changeDataset(name);
            }}
          />

          <section className="card" data-tour="decision-boundary">
            <h2 className="channel-label">
              Decision boundary
              <InfoTooltip sectionId="decision-boundary" />
            </h2>
            <DecisionBoundary
              network={t.network}
              data={t.data}
              epoch={t.stats.epoch}
              datasetName={t.datasetName}
              onProbe={handleProbe}
              onAddPoint={t.addSample}
            />
            <div className="stats" data-tour="stats">
              <span>
                epoch <code>{t.stats.epoch}</code>
              </span>
              <span>
                train loss <code>{t.stats.loss.toFixed(4)}</code>
              </span>
              <span>
                test loss <code>{t.stats.testLoss.toFixed(4)}</code>
              </span>
              <span>
                train acc <code>{(t.stats.accuracy * 100).toFixed(1)}%</code>
              </span>
              <span>
                test acc <code>{(t.stats.testAccuracy * 100).toFixed(1)}%</code>
              </span>
            </div>
          </section>

          <section className="card">
            <h2 className="channel-label" data-tour="network-graph">
              Weights and activations
              <InfoTooltip sectionId="network-graph" />
            </h2>
            <NetworkGraph network={t.network} probeActivations={probe} />
            <h2 className="channel-label" data-tour="loss-chart">
              Training & Test Loss
              <InfoTooltip sectionId="loss-chart" />
            </h2>
            <LossChart history={t.stats.lossHistory} testHistory={t.stats.testLossHistory} />
          </section>
        </div>

        <MathPanel trace={t.lastTrace} />

        <footer className="footer">
          <p>
            <span className="pos-word">Red</span> edges carry positive
            weights, <span className="neg-word">dark gray</span> edges carry
            negative ones. Line weight tracks magnitude. Click the decision
            boundary to probe a point.
          </p>
        </footer>
      </main>

      {showTutorial && (
        <TutorialOverlay onClose={() => setShowTutorial(false)} />
      )}
    </>
  );
}
