// A minimal sparkline for the training loss.
// Drawn as a single SVG path, no charting library needed.

interface Props {
  history: number[];
  testHistory?: number[];
}

const WIDTH = 560;
const HEIGHT = 90;

export function LossChart({ history, testHistory = [] }: Props) {
  if (history.length < 2) {
    return (
      <div className="loss-empty">
        Loss curves will appear here once training starts
      </div>
    );
  }

  const allVals = [...history, ...testHistory];
  // Scale with a minimum upper bound of 1.0 so BCE loss (~0.69) sits comfortably in the middle
  const max = Math.max(...allVals, 1.0);

  const trainPoints = history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * WIDTH;
      const y = HEIGHT - (v / max) * (HEIGHT - 16) - 8;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const testPoints = testHistory.length >= 2
    ? testHistory
        .map((v, i) => {
          const x = (i / (testHistory.length - 1)) * WIDTH;
          const y = HEIGHT - (v / max) * (HEIGHT - 16) - 8;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ")
    : "";

  return (
    <div className="loss-chart-wrapper">
      <div className="loss-legend-bar">
        <span className="legend-item pos">Train Loss</span>
        <span className="legend-item test">Test Loss</span>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="loss-chart"
        role="img"
        aria-label="Training and test loss over time"
      >
        <polyline points={trainPoints} className="loss-line loss-line-train" fill="none" />
        {testPoints && (
          <polyline
            points={testPoints}
            className="loss-line loss-line-test"
            fill="none"
          />
        )}
      </svg>
    </div>
  );
}
