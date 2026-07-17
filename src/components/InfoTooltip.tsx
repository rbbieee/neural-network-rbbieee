// A small info icon that shows a popover with an explanation
// when clicked. Placed next to section headers so users can
// read about any part of the UI at any time.

import { useCallback, useEffect, useRef, useState } from "react";
import { sectionInfo } from "../tutorial/steps";

interface Props {
  sectionId: string;
}

export function InfoTooltip({ sectionId }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const info = sectionInfo[sectionId];

  // Close on click outside
  const handleClickOutside = useCallback(
    (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    },
    []
  );

  useEffect(() => {
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [open, handleClickOutside]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open]);

  if (!info) return null;

  return (
    <span className="info-tooltip-wrapper" ref={ref}>
      <button
        className="info-tooltip-trigger"
        onClick={() => setOpen(!open)}
        aria-label={`Learn about ${info.title}`}
        title={`Learn about ${info.title}`}
      >
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <text
            x="8"
            y="12"
            textAnchor="middle"
            fill="currentColor"
            fontSize="10"
            fontWeight="600"
            fontFamily="inherit"
          >
            i
          </text>
        </svg>
      </button>
      {open && (
        <div className="info-tooltip-popover">
          <h4 className="info-tooltip-title">{info.title}</h4>
          <p className="info-tooltip-body">{info.body}</p>
        </div>
      )}
    </span>
  );
}
