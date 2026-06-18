interface SectionProps {
  eyebrow?: string;
  title?: string;
  lede?: string;
  className?: string;
  children?: React.ReactNode;
}

export function Section({
  eyebrow,
  title,
  lede,
  className,
  children,
}: SectionProps): React.JSX.Element {
  return (
    <section className={`site-section${className ? ` ${className}` : ""}`}>
      <div className="site-section__inner">
        {(eyebrow || title || lede) && (
          <div className="site-section__head">
            {eyebrow && (
              <p className="site-section__eyebrow mono">{eyebrow}</p>
            )}
            {title && <h2 className="site-section__title">{title}</h2>}
            {lede && <p className="site-section__lede">{lede}</p>}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}
