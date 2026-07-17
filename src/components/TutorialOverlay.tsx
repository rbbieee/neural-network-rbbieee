// Full-screen tutorial overlay with spotlight highlighting.
// Walks the user through each part of the visualizer step by step.
// The tutorial dims the background and cuts out a bright window
// around the element being explained.

import { useCallback, useEffect, useRef, useState } from "react";
import { tutorialSteps, type TutorialStep } from "../tutorial/steps";

interface Props {
  onClose: () => void;
}

// Returns the bounding rect of the element matching the selector,
// or null if nothing matches.
function getTargetRect(selector: string): DOMRect | null {
  const el = document.querySelector(selector);
  if (!el) return null;
  return el.getBoundingClientRect();
}

// Figure out where to place the tooltip card relative to the
// highlighted element so it does not go off screen.
function computeTooltipStyle(
  rect: DOMRect,
  position: TutorialStep["position"]
): React.CSSProperties {
  const gap = 16;
  const cardWidth = 340;
  const style: React.CSSProperties = { position: "fixed", width: cardWidth };

  switch (position) {
    case "right":
      style.left = rect.right + gap;
      style.top = rect.top + rect.height / 2;
      style.transform = "translateY(-50%)";
      // If it would overflow the right edge, flip to left
      if (rect.right + gap + cardWidth > window.innerWidth - 16) {
        style.left = rect.left - gap - cardWidth;
      }
      break;
    case "left":
      style.left = rect.left - gap - cardWidth;
      style.top = rect.top + rect.height / 2;
      style.transform = "translateY(-50%)";
      // If it would overflow the left edge, flip to right
      if (rect.left - gap - cardWidth < 16) {
        style.left = rect.right + gap;
      }
      break;
    case "bottom":
      style.left = rect.left + rect.width / 2 - cardWidth / 2;
      style.top = rect.bottom + gap;
      break;
    case "top":
      style.left = rect.left + rect.width / 2 - cardWidth / 2;
      style.top = rect.top - gap;
      style.transform = "translateY(-100%)";
      break;
  }

  // Clamp horizontal position
  const leftNum =
    typeof style.left === "number" ? style.left : parseFloat(style.left as string) || 0;
  if (leftNum < 16) style.left = 16;
  if (leftNum + cardWidth > window.innerWidth - 16) {
    style.left = window.innerWidth - 16 - cardWidth;
  }

  return style;
}

export function TutorialOverlay({ onClose }: Props) {
  const [step, setStep] = useState(0);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const [animating, setAnimating] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);

  const current = tutorialSteps[step];
  const total = tutorialSteps.length;

  // Measure the target element whenever the step changes or the
  // window resizes. We poll briefly to wait for layout shifts.
  const measure = useCallback(() => {
    if (!current) return;
    const rect = getTargetRect(current.targetSelector);
    setTargetRect(rect);
  }, [current]);

  useEffect(() => {
    measure();
    // Re-measure on resize and scroll
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [measure]);

  // Animate transitions between steps
  const goTo = useCallback(
    (next: number) => {
      if (next < 0 || next >= total) return;
      setAnimating(true);
      setTimeout(() => {
        setStep(next);
        setAnimating(false);
      }, 200);
    },
    [total]
  );

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" || e.key === "Enter") {
        if (step < total - 1) goTo(step + 1);
        else onClose();
      }
      if (e.key === "ArrowLeft") goTo(step - 1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [step, total, goTo, onClose]);

  // Scroll target element into view
  useEffect(() => {
    if (!current) return;
    const el = document.querySelector(current.targetSelector);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      // Re-measure after scroll settles
      const t = setTimeout(measure, 350);
      return () => clearTimeout(t);
    }
  }, [current, measure]);

  // Build the spotlight clip path. We cut a rounded rectangle out of
  // the dark overlay so the target element shows through.
  const spotlightPad = 10;
  let clipPath = "none";
  if (targetRect) {
    const x = targetRect.left - spotlightPad;
    const y = targetRect.top - spotlightPad;
    const w = targetRect.width + spotlightPad * 2;
    const h = targetRect.height + spotlightPad * 2;
    const r = 14;
    // SVG-style clip path with a rounded rect cutout
    clipPath = `polygon(
      0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%,
      ${x + r}px ${y}px,
      ${x + w - r}px ${y}px,
      ${x + w}px ${y + r}px,
      ${x + w}px ${y + h - r}px,
      ${x + w - r}px ${y + h}px,
      ${x + r}px ${y + h}px,
      ${x}px ${y + h - r}px,
      ${x}px ${y + r}px,
      ${x + r}px ${y}px
    )`;
  }

  const tooltipStyle = targetRect
    ? computeTooltipStyle(targetRect, current.position)
    : {
        position: "fixed" as const,
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: 340,
      };

  return (
    <div
      className={`tutorial-overlay${animating ? " tutorial-animating" : ""}`}
      ref={overlayRef}
    >
      {/* Dark backdrop with spotlight cutout */}
      <div
        className="tutorial-backdrop"
        style={{ clipPath }}
        onClick={onClose}
      />

      {/* Spotlight ring glow */}
      {targetRect && (
        <div
          className="tutorial-spotlight-ring"
          style={{
            left: targetRect.left - spotlightPad,
            top: targetRect.top - spotlightPad,
            width: targetRect.width + spotlightPad * 2,
            height: targetRect.height + spotlightPad * 2,
          }}
        />
      )}

      {/* Tooltip card */}
      <div className="tutorial-card" style={tooltipStyle}>
        <div className="tutorial-card-step">
          {step + 1} of {total}
        </div>
        <h3 className="tutorial-card-title">{current.title}</h3>
        <p className="tutorial-card-body">{current.body}</p>

        {/* Progress dots */}
        <div className="tutorial-dots">
          {tutorialSteps.map((_, i) => (
            <span
              key={i}
              className={`tutorial-dot${i === step ? " active" : ""}${
                i < step ? " done" : ""
              }`}
            />
          ))}
        </div>

        {/* Navigation */}
        <div className="tutorial-nav">
          <button
            className="btn tutorial-skip"
            onClick={onClose}
          >
            Skip
          </button>
          <div className="tutorial-nav-right">
            {step > 0 && (
              <button
                className="btn"
                onClick={() => goTo(step - 1)}
              >
                Back
              </button>
            )}
            <button
              className="btn primary"
              onClick={() => {
                if (step < total - 1) goTo(step + 1);
                else onClose();
              }}
            >
              {step < total - 1 ? "Next" : "Done"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
