// Collapsible panel that renders the step-by-step math trace.
// Shows forward pass, loss, backward pass, and weight updates
// with real numbers from the actual network computation.

import { useState } from "react";
import type { StepTrace } from "../nn/trace";

interface Props {
  trace: StepTrace | null;
}

function fmt(n: number): string {
  return n.toFixed(4);
}

function ValSpan({ v }: { v: number }) {
  const cls = v >= 0 ? "math-val pos" : "math-val neg";
  return <span className={cls}>{fmt(v)}</span>;
}

function Section({ title, children, defaultOpen = false }: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="math-section">
      <button
        className="math-section-toggle"
        onClick={() => setOpen(!open)}
      >
        <span className="math-section-arrow">{open ? "\u25BC" : "\u25B6"}</span>
        {title}
      </button>
      {open && <div className="math-section-body">{children}</div>}
    </div>
  );
}

export function MathPanel({ trace }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (!trace) return null;

  return (
    <div className="math-panel">
      <button
        className="btn math-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? "Hide Math" : "Show Math"}
      </button>

      {expanded && (
        <div className="math-content">

          {/* Input */}
          <Section title="Input Sample" defaultOpen={true}>
            <div className="math-formula">
              <span className="math-label">Point:</span>{" "}
              x = <ValSpan v={trace.sample.x} />, y = <ValSpan v={trace.sample.y} />
              {" "} | label = <span className="math-val">{trace.sample.label}</span>
            </div>
            <div className="math-formula">
              <span className="math-label">Features:</span>{" "}
              {trace.inputFeatures.map((f, i) => (
                <span key={i}>
                  {i > 0 && ", "}
                  {f.name} = <ValSpan v={f.value} />
                </span>
              ))}
            </div>
          </Section>

          {/* Forward Pass */}
          <Section title="Forward Pass" defaultOpen={true}>
            {trace.forwardLayers.slice(1).map((layer, li) => (
              <div key={li} className="math-layer-block">
                <div className="math-layer-title">{layer.name}</div>
                {layer.neurons.map((neuron, ni) => (
                  <div key={ni} className="math-neuron-block">
                    <div className="math-neuron-label">{neuron.label}</div>
                    <div className="math-formula">
                      <span className="math-label">z</span> ={" "}
                      {neuron.terms.map((t, ti) => (
                        <span key={ti}>
                          {ti > 0 && " + "}
                          (<ValSpan v={t.weight} />)(<ValSpan v={t.input} />)
                        </span>
                      ))}
                      {" "}+ <ValSpan v={neuron.bias} />
                      {" "}= <strong><ValSpan v={neuron.weightedSum} /></strong>
                    </div>
                    <div className="math-formula">
                      <span className="math-label">a</span> ={" "}
                      {neuron.activationName}(<ValSpan v={neuron.weightedSum} />)
                      {" "}= <strong><ValSpan v={neuron.activation} /></strong>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </Section>

          {/* Loss */}
          <Section title="Loss (Binary Cross-Entropy)" defaultOpen={true}>
            <div className="math-formula">
              <span className="math-label">prediction</span> = <ValSpan v={trace.loss.prediction} />
            </div>
            <div className="math-formula">
              L = -[y * log(p) + (1 - y) * log(1 - p)]
            </div>
            <div className="math-formula">
              L = -[{trace.loss.label} * log(<ValSpan v={trace.loss.clampedP} />)
              {" "}+ {1 - trace.loss.label} * log({fmt(1 - trace.loss.clampedP)})]
            </div>
            <div className="math-formula">
              L = <strong><ValSpan v={trace.loss.bce} /></strong>
            </div>
          </Section>

          {/* Backward Pass */}
          <Section title="Backward Pass (Gradients)">
            {trace.backward.map((layer, li) => (
              <div key={li} className="math-layer-block">
                <div className="math-layer-title">{layer.name}</div>
                {layer.neurons.map((neuron, ni) => (
                  <div key={ni} className="math-neuron-block">
                    <div className="math-neuron-label">{neuron.label}</div>
                    <div className="math-formula">
                      delta = {neuron.formula}
                    </div>
                    <div className="math-formula">
                      delta = <strong><ValSpan v={neuron.delta} /></strong>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </Section>

          {/* Weight Updates */}
          <Section title="Weight Updates (first 8)">
            <table className="math-table">
              <thead>
                <tr>
                  <th>Connection</th>
                  <th>Old Weight</th>
                  <th>Gradient</th>
                  <th>LR</th>
                  <th>New Weight</th>
                </tr>
              </thead>
              <tbody>
                {trace.updates.map((u, i) => (
                  <tr key={i}>
                    <td className="math-conn">{u.fromLabel} to {u.toLabel}</td>
                    <td><ValSpan v={u.oldWeight} /></td>
                    <td><ValSpan v={u.gradient} /></td>
                    <td>{fmt(u.learningRate)}</td>
                    <td><strong><ValSpan v={u.newWeight} /></strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

        </div>
      )}
    </div>
  );
}
