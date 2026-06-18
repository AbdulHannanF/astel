import { Link } from "react-router-dom";

interface CtaLinkProps {
  to: string;
  variant?: "primary" | "ghost";
  children: React.ReactNode;
  className?: string;
}

export function CtaLink({
  to,
  variant = "primary",
  children,
  className,
}: CtaLinkProps): React.JSX.Element {
  const base = "cta-link";
  const mod = `cta-link--${variant}`;
  return (
    <Link to={to} className={`${base} ${mod}${className ? ` ${className}` : ""}`}>
      {children}
    </Link>
  );
}
