import { motion } from "framer-motion";
import { CheckCircle2, Zap, Compass } from "lucide-react";
import type { StepRecord } from "@/lib/store";
import { cn } from "@/lib/utils";

export function BudgetTimeline({ history, showReward = true }: { history: StepRecord[]; showReward?: boolean }) {
  const items = history.slice(-12).reverse();
  return (
    <div className="glass rounded-2xl p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Decision Timeline
        </h3>
        <span className="font-mono-num text-xs text-muted-foreground">
          {history.length} steps
        </span>
      </div>
      {items.length === 0 ? (
        <div className="grid place-items-center rounded-xl border border-dashed border-border/60 px-3 py-8 text-xs text-muted-foreground">
          No actions yet — start the simulation.
        </div>
      ) : (
        <ol className="relative space-y-2 border-l border-border/60 pl-4">
          {items.map((s) => {
            const isLabel = s.action === "label";
            return (
              <motion.li
                key={s.step}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                className="relative"
              >
                <span
                  className={cn(
                    "absolute -left-[21px] top-1.5 grid h-4 w-4 place-items-center rounded-full ring-2 ring-background",
                    isLabel ? "bg-primary" : "bg-warning",
                  )}
                >
                  {isLabel ? (
                    <CheckCircle2 className="h-2.5 w-2.5 text-primary-foreground" />
                  ) : (
                    <Zap className="h-2.5 w-2.5 text-warning-foreground" />
                  )}
                </span>
                <div className="flex items-center justify-between rounded-lg border border-border/50 bg-card/50 px-2.5 py-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-mono-num text-muted-foreground">#{s.step}</span>
                    <span className="font-mono-num">{s.sampleId}</span>
                    {s.exploration && <Compass className="h-3 w-3 text-warning" />}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={cn("font-medium", isLabel ? "text-primary" : "text-warning")}>
                      {isLabel ? "Label" : "Predict"}
                    </span>
                    {showReward && (
                      <span className={cn("font-mono-num", s.reward >= 0 ? "text-success" : "text-destructive")}>
                        {s.reward >= 0 ? "+" : ""}{s.reward.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
              </motion.li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
