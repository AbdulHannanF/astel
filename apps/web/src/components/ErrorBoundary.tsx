import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * App-level error boundary. A render-time exception anywhere in the tree
 * previously unmounted everything to a black screen (e.g. the Truth Meter
 * calling `.toFixed()` on a null metric). This catches it, logs it, and shows a
 * recoverable fallback so a single bad component never takes down the whole UI.
 */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface to the console for diagnosis; a real deploy would ship this to
    // an error tracker.
    console.error("Uncaught UI error:", error, info.componentStack);
  }

  private reset = (): void => this.setState({ error: null });

  override render(): ReactNode {
    const { error } = this.state;
    if (error === null) return this.props.children;

    return (
      <div className="app-error" role="alert">
        <div className="app-error__card">
          <p className="app-error__eyebrow mono">Something went wrong</p>
          <h1 className="app-error__title">The view hit an unexpected error.</h1>
          <p className="app-error__body">
            This shouldn&rsquo;t happen — the error has been logged to the
            console. You can retry the view or reload the page.
          </p>
          <pre className="app-error__detail mono">{error.message}</pre>
          <div className="app-error__actions">
            <button type="button" className="cta-link cta-link--primary" onClick={this.reset}>
              Retry
            </button>
            <button
              type="button"
              className="cta-link cta-link--ghost"
              onClick={() => window.location.reload()}
            >
              Reload page
            </button>
          </div>
        </div>
      </div>
    );
  }
}
