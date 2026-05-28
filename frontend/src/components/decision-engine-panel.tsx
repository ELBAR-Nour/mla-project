import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Compass, Target, Info, ArrowRight } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Sample, StepRecord } from "@/lib/store";

interface Props {
  sample: Sample | null;
  preview: { action: "label" | "predict"; expectedReward: number; policyConfidence: number; exploration: boolean } | null;
  lastStep: StepRecord | null;
  strategyLabel?: string;
  showReward?: boolean;
}

export function DecisionEnginePanel({ sample, preview, lastStep, strategyLabel = "Strategy", showReward = true }: Props) {
  const classes = sample?.classLabels?.length ? sample.classLabels : ["Negative", "Positive"];
  return (
    <div className="space-y-4">
      <div className="glass rounded-2xl p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Model Prediction
          </h3>
          <Tooltip>
            <TooltipTrigger>
              <Info className="h-3.5 w-3.5 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>Class probabilities from the selected model.</TooltipContent>
          </Tooltip>
        </div>
        {sample?.liveLoaded ? (
          <div className="space-y-3">
            {classes.map((c, i) => (
              <div key={c}>
                <div className="mb-1 flex justify-between text-sm">
                  <span>{c}</span>
                  <span className="font-mono-num text-muted-foreground">
                    {(sample.probs[i] * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={sample.probs[i] * 100} />
              </div>
            ))}
            <div className="mt-4 grid grid-cols-2 gap-3 border-t border-border/60 pt-3 text-xs">
              <div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  Entropy
                  <Tooltip>
                    <TooltipTrigger><Info className="h-3 w-3" /></TooltipTrigger>
                    <TooltipContent>Higher = more uncertain. Drives query value.</TooltipContent>
                  </Tooltip>
                </div>
                <div className="font-mono-num text-base font-semibold text-warning">
                  {sample.entropy.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">Confidence</div>
                <div className="font-mono-num text-base font-semibold">
                  {(sample.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        ) : (
          <EmptyHint label={sample ? "Loading model output..." : "No active sample. Start the experiment."} />
        )}
      </div>

      <motion.div
        layout
        className="glass relative overflow-hidden rounded-2xl p-5"
      >
        <div className="absolute -right-8 -top-8 h-28 w-28 rounded-full bg-primary/15 blur-2xl" />
        <div className="mb-2 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <h3 className="font-display text-sm font-semibold uppercase tracking-wider">
            {showReward ? "RL Policy Decision" : `${strategyLabel} Selection`}
          </h3>
          {showReward && preview?.exploration && (
            <Badge variant="outline" className="ml-auto border-warning/40 text-warning">
              <Compass className="mr-1 h-3 w-3" /> Exploring
            </Badge>
          )}
          {showReward && preview && !preview.exploration && (
            <Badge variant="outline" className="ml-auto border-primary/40 text-primary">
              <Target className="mr-1 h-3 w-3" /> Exploiting
            </Badge>
          )}
        </div>

        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key={preview.action}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              className="rounded-xl border border-primary/30 bg-primary/5 p-3"
            >
            <div className="text-xs text-muted-foreground">{showReward ? "Recommended action" : "Acquisition action"}</div>
              <div className="mt-1 flex items-center gap-2 font-display text-lg font-semibold text-gradient">
                {preview.action === "label" ? "Request Expert Label" : "Predict Automatically"}
                <ArrowRight className="h-4 w-4 text-primary" />
              </div>
            </motion.div>
          ) : (
            <EmptyHint label="Awaiting decision…" />
          )}
        </AnimatePresence>

        {preview && (
          <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
            <div>
              <div className="text-muted-foreground">{showReward ? "Expected reward" : sample?.acquisitionLabel ?? "Acquisition score"}</div>
              <div className="font-mono-num text-base font-semibold text-success">
                {showReward && preview.expectedReward >= 0 ? "+" : ""}
                {preview.expectedReward.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">{showReward ? "Policy confidence" : "Model confidence"}</div>
              <div className="font-mono-num text-base font-semibold">
                {(preview.policyConfidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        )}

        {lastStep && (
          <div className="mt-4 rounded-lg border border-border/60 bg-muted/40 p-2.5 text-[11px]">
            <div className="text-muted-foreground">Last step #{lastStep.step}</div>
            <div className="mt-0.5 flex items-center justify-between font-mono-num">
              <span>{lastStep.sampleId}</span>
              {showReward ? (
                <span className={lastStep.reward >= 0 ? "text-success" : "text-destructive"}>
                  reward {lastStep.reward >= 0 ? "+" : ""}{lastStep.reward.toFixed(2)}
                </span>
              ) : (
                <span className="text-muted-foreground">
                  {strategyLabel} {lastStep.action === "label" ? "queried" : "predicted"}
                </span>
              )}
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}

function EmptyHint({ label }: { label: string }) {
  return (
    <div className="grid place-items-center rounded-xl border border-dashed border-border/60 px-3 py-6 text-xs text-muted-foreground">
      {label}
    </div>
  );
}
