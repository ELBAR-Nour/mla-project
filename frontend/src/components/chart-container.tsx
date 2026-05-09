import { cn } from "@/lib/utils";

interface ChartContainerProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

export function ChartContainer({
  title,
  description,
  actions,
  className,
  children,
}: ChartContainerProps) {
  return (
    <div className={cn("glass rounded-2xl p-5", className)}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-display text-base font-semibold">{title}</h3>
          {description && (
            <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </div>
  );
}
