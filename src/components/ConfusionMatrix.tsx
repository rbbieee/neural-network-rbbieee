import type { TrainingStats } from "../hooks/useTraining";

interface Props {
  stats: TrainingStats;
}

export function ConfusionMatrix({ stats }: Props) {
  const { tp, fp, fn, tn } = stats.confusion;
  const total = tp + fp + fn + tn;
  
  if (total === 0) return null;

  return (
    <div className="confusion-matrix-panel">
      <div className="cm-title">Test Set Confusion Matrix</div>
      <div className="cm-grid">
        <div className="cm-cell header"></div>
        <div className="cm-cell header">Pred Pos (1)</div>
        <div className="cm-cell header">Pred Neg (0)</div>
        
        <div className="cm-cell header">Actual Pos (1)</div>
        <div className="cm-cell pos-cell" title="True Positive">
          <div className="cm-val">{tp}</div>
          <div className="cm-lbl">TP</div>
        </div>
        <div className="cm-cell err-cell" title="False Negative">
          <div className="cm-val">{fn}</div>
          <div className="cm-lbl">FN</div>
        </div>
        
        <div className="cm-cell header">Actual Neg (0)</div>
        <div className="cm-cell err-cell" title="False Positive">
          <div className="cm-val">{fp}</div>
          <div className="cm-lbl">FP</div>
        </div>
        <div className="cm-cell neg-cell" title="True Negative">
          <div className="cm-val">{tn}</div>
          <div className="cm-lbl">TN</div>
        </div>
      </div>
      
      <div className="cm-metrics">
        <div className="cm-metric">
          <span className="cm-mlbl">Precision:</span>
          <span className="cm-mval">{(stats.precision * 100).toFixed(1)}%</span>
        </div>
        <div className="cm-metric">
          <span className="cm-mlbl">Recall:</span>
          <span className="cm-mval">{(stats.recall * 100).toFixed(1)}%</span>
        </div>
        <div className="cm-metric">
          <span className="cm-mlbl">F1 Score:</span>
          <span className="cm-mval">{(stats.f1 * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
