// A minimal sparkline for the training loss.
// Drawn as a single SVG path, no charting library needed.

interface Props {
  history: number[];
}

const WIDTH = 560;
const HEIGHT = 90;

export function LossChart({ history }: Props) {
  if (history.length < 2) {
    return (
      <div className="loss-empty">
        Loss will appear here once training starts
      </div>
    );
  }

  const max = Math.max(...history, 0.01);
  const points = history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * WIDTH;
      const y = HEIGHT - (v / max) * (HEIGHT - 8) - 4;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="loss-chart"
      role="img"
      aria-label="Training loss over time"
    >
      <polyline points={points} className="loss-line" fill="none" />
    </svg>
  );
}
