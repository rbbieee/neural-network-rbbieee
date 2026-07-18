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

const INITIAL_CONFIG = {
  hiddenLayers: [4, 4],
  activation: "tanh" as const,
  learningRate: 0.3,
  seed: 1337,
};

export default function App() {
  const t = useTraining(INITIAL_CONFIG, "circles");
  const [probe, setProbe] = useState<number[][] | null>(null);
  const [showTutorial, setShowTutorial] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }, [theme]);

  const handleProbe = (x: number, y: number) => {
    // Run a forward pass on the clicked point and keep the
    // activations so the graph can display them
    setProbe(t.network.forward(x, y));
  };

  return (
    <main className="app">
      <div className="titlebar">
        <div className="traffic-lights">
          <span className="tl-red" />
          <span className="tl-yellow" />
          <span className="tl-green" />
        </div>
        <span className="titlebar-title">neural-network-rbbieee</span>
      </div>

      <header className="header" data-tour="header">
        <div className="header-top">
          <h1>
            neural-network-<span className="accent">rbbieee</span>
          </h1>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle theme"
            title="Toggle theme"
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          <button
            className="btn guide-btn"
            onClick={() => setShowTutorial(true)}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
              <text
                x="8"
                y="12"
                textAnchor="middle"
                fill="currentColor"
                fontSize="10"
                fontWeight="700"
                fontFamily="inherit"
              >
                ?
              </text>
            </svg>
            Guide
          </button>
        </div>
        <p className="tagline">
          A live readout of backpropagation. No black box, no math library
          between you and the gradient, every <span className="unit">weight</span>{" "}
          and <span className="unit">activation</span> rendered as it updates.
        </p>
      </header>

      <div className="layout">
        <ControlPanel
          config={t.config}
          datasetName={t.datasetName}
          running={t.running}
          onToggleRun={() => t.setRunning(!t.running)}
          onStep={t.step}
          onReset={() => {
            setProbe(null);
            t.reset();
          }}
          onConfigChange={(patch) => {
            setProbe(null);
            t.updateConfig(patch);
          }}
          onDatasetChange={(name) => {
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
            onProbe={handleProbe}
          />
          <div className="stats" data-tour="stats">
            <span>
              epoch <code>{t.stats.epoch}</code>
            </span>
            <span>
              loss <code>{t.stats.loss.toFixed(4)}</code>
            </span>
            <span>
              accuracy <code>{(t.stats.accuracy * 100).toFixed(1)}%</code>
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
            Training loss
            <InfoTooltip sectionId="loss-chart" />
          </h2>
          <LossChart history={t.stats.lossHistory} />
        </section>
      </div>

      <footer className="footer">
        <p>
          <span className="pos-word">Blue</span> edges carry positive
          weights, <span className="neg-word">orange</span> edges carry
          negative ones. Line weight tracks magnitude. Click the decision
          boundary to probe a point and trace it through every layer.
        </p>
      </footer>

      {showTutorial && (
        <TutorialOverlay onClose={() => setShowTutorial(false)} />
      )}
    </main>
  );
}
