import { useEffect } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function ShortcutsOverlay({ isOpen, onClose }: Props) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-modal" onClick={(e) => e.stopPropagation()}>
        <h2>Keyboard Shortcuts</h2>
        <table className="shortcuts-table">
          <tbody>
            <tr>
              <td><kbd>Space</kbd></td>
              <td>Toggle training Run / Stop</td>
            </tr>
            <tr>
              <td><kbd>S</kbd></td>
              <td>Step one epoch forward</td>
            </tr>
            <tr>
              <td><kbd>R</kbd></td>
              <td>Reset network to initial state</td>
            </tr>
            <tr>
              <td><kbd>?</kbd></td>
              <td>Show / hide this shortcuts menu</td>
            </tr>
            <tr>
              <td><kbd>Esc</kbd></td>
              <td>Close modals</td>
            </tr>
            <tr>
              <td><kbd>Shift</kbd> + Click</td>
              <td>Force paint negative (0) class in custom dataset mode</td>
            </tr>
          </tbody>
        </table>
        <button className="btn primary" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
