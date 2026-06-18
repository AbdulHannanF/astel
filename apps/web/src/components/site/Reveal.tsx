import { useEffect, useRef, useState } from "react";

interface RevealProps {
  children: React.ReactNode;
  className?: string;
  /** Delay before the reveal transition starts, in ms (default 0) — for stagger. */
  delay?: number;
}

/**
 * Returns true when the content should skip the scroll-reveal and show
 * immediately: the user prefers reduced motion, or IntersectionObserver is
 * unavailable (SSR / jsdom). In those cases content must never start hidden.
 */
function shouldRevealImmediately(): boolean {
  if (typeof IntersectionObserver === "undefined") return true;
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function Reveal({
  children,
  className,
  delay = 0,
}: RevealProps): React.JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  // Lazy initialiser keeps the reduced-motion / no-observer paths from needing
  // a synchronous setState inside the effect (react-hooks/set-state-in-effect).
  const [visible, setVisible] = useState(shouldRevealImmediately);

  useEffect(() => {
    if (visible) return; // already shown — nothing to observe
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [visible]);

  return (
    <div
      ref={ref}
      className={`reveal${visible ? " reveal--in" : ""}${className ? ` ${className}` : ""}`}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  );
}
