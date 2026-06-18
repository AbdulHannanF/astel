interface PageHeaderProps {
  eyebrow?: string;
  title: string;
  lede?: string;
}

export function PageHeader({ eyebrow, title, lede }: PageHeaderProps): React.JSX.Element {
  return (
    <div className="page-header">
      {eyebrow && (
        <p className="page-header__eyebrow mono">{eyebrow}</p>
      )}
      <h1 className="page-header__title">{title}</h1>
      {lede && <p className="page-header__lede">{lede}</p>}
    </div>
  );
}
