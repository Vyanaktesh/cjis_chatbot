import { Component, type ErrorInfo, type ReactNode } from "react";

/** Last line of defence: if rendering throws, show a short message with a
 * reload button instead of a blank page. */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Chat widget crashed", error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div
        role="alert"
        className="m-4 rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-4 text-sm text-[var(--cc-text)]"
      >
        <p>Something went wrong. Please reload the page and try again.</p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-3 rounded-full bg-[var(--cc-accent)] px-4 py-1.5 text-xs font-semibold text-[var(--cc-accent-ink)]"
        >
          Reload
        </button>
      </div>
    );
  }
}
