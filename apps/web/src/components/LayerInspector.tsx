import { useMemo, useState } from "react";

import type { LayerDef, LayerId } from "../lib/layers.ts";
import "./LayerInspector.css";

interface LayerInspectorProps {
  layers: readonly LayerDef[];
  visible: ReadonlySet<LayerId>;
  onToggle: (id: LayerId) => void;
}

export function LayerInspector({
  layers,
  visible,
  onToggle,
}: LayerInspectorProps): React.JSX.Element {
  // Scrub position indexes into the layer stack (film-timeline affordance).
  // Defaults to the highest available layer (the refined hero).
  const lastAvailable = useMemo(() => {
    const idx = layers.map((l) => l.availability).lastIndexOf("available");
    return idx === -1 ? 0 : idx;
  }, [layers]);
  const [scrub, setScrub] = useState(lastAvailable);

  const availableCount = layers.filter(
    (l) => l.availability === "available",
  ).length;
  const fill = layers.length > 1 ? scrub / (layers.length - 1) : 0;
  const scrubbed = layers[Math.min(scrub, layers.length - 1)];

  return (
    <section className="panel">
      <div className="panel__head">
        <h2 className="panel__title">Layer Stack</h2>
        <span className="panel__meta mono">
          {availableCount}/{layers.length} ready
        </span>
      </div>

      <div className="layers">
        {layers.map((layer) => {
          const isVisible = visible.has(layer.id);
          const isActive = layer.availability === "available" && isVisible;
          return (
            <div
              key={layer.id}
              className={
                "layer " +
                `layer--${layer.availability} ` +
                (isActive ? "layer--active" : "")
              }
            >
              <div className="layer__spine">
                <span className="layer__node" />
              </div>
              <div className="layer__body">
                <div className="layer__top">
                  <span className="layer__id mono">{layer.id}</span>
                  <span className="layer__name">{layer.name}</span>
                  <span className="layer__kind">{layer.kind}</span>
                  <span className="layer__spacer" />
                  <LayerControl
                    layer={layer}
                    isVisible={isVisible}
                    onToggle={onToggle}
                  />
                </div>
                <p className="layer__blurb">{layer.blurb}</p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="layer-scrub">
        <div className="layer-scrub__labels">
          <span>scrub layers</span>
          <span>L0 → L7</span>
        </div>
        <input
          type="range"
          min={0}
          max={layers.length - 1}
          step={1}
          value={scrub}
          aria-label="Scrub through layer stack"
          style={{ ["--fill" as string]: String(fill) }}
          onChange={(e) => setScrub(Number(e.currentTarget.value))}
        />
        <div className="layer-scrub__current mono">
          inspecting <b>{scrubbed?.id}</b> · {scrubbed?.name}
          {scrubbed?.availability !== "available" && " (preview pending)"}
        </div>
      </div>
    </section>
  );
}

function LayerControl({
  layer,
  isVisible,
  onToggle,
}: {
  layer: LayerDef;
  isVisible: boolean;
  onToggle: (id: LayerId) => void;
}): React.JSX.Element {
  if (layer.availability === "available") {
    return (
      <button
        type="button"
        className={
          "layer__toggle " +
          (isVisible ? "layer__toggle--on" : "layer__toggle--off")
        }
        aria-pressed={isVisible}
        onClick={() => onToggle(layer.id)}
      >
        {isVisible ? "shown" : "hidden"}
      </button>
    );
  }
  if (layer.availability === "pending") {
    return (
      <span className="layer__state" title="Produced by a later stage">
        <ClockIcon /> pending
      </span>
    );
  }
  return (
    <span className="layer__state" title="Not applicable to this asset">
      <LockIcon /> n/a
    </span>
  );
}

function ClockIcon(): React.JSX.Element {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden>
      <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.1" />
      <path
        d="M6 3.6V6l1.7 1"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinecap="round"
      />
    </svg>
  );
}

function LockIcon(): React.JSX.Element {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden>
      <rect
        x="2.5"
        y="5.2"
        width="7"
        height="5"
        rx="1"
        stroke="currentColor"
        strokeWidth="1.1"
      />
      <path
        d="M4 5.2V4a2 2 0 0 1 4 0v1.2"
        stroke="currentColor"
        strokeWidth="1.1"
      />
    </svg>
  );
}
