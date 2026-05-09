import { motion } from "framer-motion";
import { ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string;
  delta?: number;
  hint?: string;
  tone?: "default" | "success" | "warning" | "danger";
  icon?: React.ReactNode;
}

const toneClass: Record<NonNullable<MetricCardProps["tone"]>, string> = {
  default: "text-primary",
  success: "text-success",
  warning: "text-warning",
  danger: "text-destructive",
};

export function MetricCard({ label, value, delta, hint, tone = "default", icon }: MetricCardProps) {
  const positive = (delta ?? 0) >= 0;
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass relative overflow-hidden rounded-2xl p-5"
    >
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</p>
          <p className={cn("mt-2 font-mono-num text-3xl font-semibold tracking-tight", toneClass[tone])}>
            {value}
          </p>
          {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
        </div>
        {icon && (
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-primary/10 text-primary">
            {icon}
          </div>
        )}
      </div>
      {typeof delta === "number" && (
        <div
          className={cn(
            "mt-3 inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
            positive ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive",
          )}
        >
          {positive ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
          {Math.abs(delta).toFixed(1)}%
        </div>
      )}
    </motion.div>
  );
}
