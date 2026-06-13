import { useEffect, useState } from "react";

type Health = "checking" | "ok" | "down";

/** Polls /healthz once on mount so the operator sees gateway liveness. */
function useHealth(): Health {
  const [health, setHealth] = useState<Health>("checking");
  useEffect(() => {
    const ctrl = new AbortController();
    fetch("/healthz", { signal: ctrl.signal })
      .then((res) => setHealth(res.ok ? "ok" : "down"))
      .catch(() => {
        if (!ctrl.signal.aborted) setHealth("down");
      });
    return () => ctrl.abort();
  }, []);
  return health;
}

const HEALTH_LABEL: Record<Health, string> = {
  checking: "linking",
  ok: "gateway online",
  down: "gateway offline",
};

export function TopBar(): React.JSX.Element {
  const health = useHealth();
  return (
    <header className="topbar">
      <div className="brand">
        <img className="brand__mark" src="/favicon.svg" alt="" aria-hidden />
        <span className="brand__word">Astel</span>
        <span className="brand__tag">Splat Studio</span>
      </div>
      <div className="topbar__spacer" />
      <span
        className={
          "health " +
          (health === "ok"
            ? "health--ok"
            : health === "down"
              ? "health--down"
              : "")
        }
      >
        <span className="health__dot" />
        {HEALTH_LABEL[health]}
      </span>
    </header>
  );
}
