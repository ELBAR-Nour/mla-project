import { cn } from "@/lib/utils";

interface BudgetBarProps {
  used: number;
  total: number;
  className?: string;
}

export function BudgetBar({ used, total, className }: BudgetBarProps) {
  const pct = Math.min(100, (used / total) * 100);
  const tone =
    pct < 60 ? "text-success" : pct < 85 ? "text-warning" : "text-destructive";
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Annotation Budget
        </span>
        <span className={cn("font-mono-num text-sm font-semibold", tone)}>
          {used} / {total}
        </span>
      </div>
      <div className="relative h-2.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background:
              "linear-gradient(90deg, oklch(0.68 0.16 155), oklch(0.74 0.16 60) 70%, oklch(0.6 0.22 25))",
          }}
        />
      </div>
    </div>
  );
}
