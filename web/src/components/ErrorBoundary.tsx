import { Component, type ReactNode } from "react";

/** If anything throws while rendering, show a calm message with a Reload button instead of a blank page. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="wrap" style={{ paddingTop: 80, maxWidth: 560 }}>
        <h1 style={{ fontSize: "1.8rem" }}>Something went wrong on this page.</h1>
        <p className="muted" style={{ marginTop: 12 }}>
          Reloading usually fixes it, for example after the site was updated. Your current check is kept in this tab.
        </p>
        <button className="btn primary" style={{ marginTop: 20 }} onClick={() => location.reload()}>
          Reload
        </button>
      </div>
    );
  }
}
