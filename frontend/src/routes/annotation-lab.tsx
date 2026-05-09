import { useEffect, useMemo } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Play, Pause, StepForward, RotateCcw, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { ImageStream } from "@/components/image-stream";
import { DecisionEnginePanel } from "@/components/decision-engine-panel";
import { BudgetTimeline } from "@/components/budget-timeline";
import { BudgetBar } from "@/components/budget-bar";
import { useApp } from "@/lib/store";
import { toast } from "sonner";

export const Route = createFileRoute("/annotation-lab")({
  head: () => ({
    meta: [
      { title: "Live Simulation — MedAL" },
      { name: "description", content: "Step through the RL agent's annotation decisions in real time." },
    ],
  }),
  component: AnnotationLab,
});

function AnnotationLab() {
  const {
    samples,
    currentIndex,
    history,
    running,
    speedMs,
    experimentStarted,
    budget,
    strategy,
    rl,
    stepOnce,
    setRunning,
    setSpeed,
    resetExperiment,
    startExperiment,
  } = useApp();

  const current = samples[currentIndex] ?? null;
  const upcoming = samples.slice(currentIndex + 1, currentIndex + 7);
  const used = history.filter((h) => h.action === "label").length;
  const lastStep = history[history.length - 1] ?? null;

  // Compute live preview of agent decision for current sample
  const preview = useMemo(() => {
    if (!current) return null;
    const isExplore = strategy === "rl" && Math.random() < rl.epsilon * 0.3;
    if (strategy === "random") {
      return { action: "label" as const, expectedReward: 0, policyConfidence: 0.5, exploration: true };
    }
    if (strategy === "entropy") {
      const action = current.entropy > 0.5 ? ("label" as const) : ("predict" as const);
      return { action, expectedReward: current.entropy - 0.5, policyConfidence: Math.abs(current.entropy - 0.5) + 0.5, exploration: false };
    }
    const utility = current.entropy + (1 - current.confidence) * 0.5;
    const action = utility > 0.55 ? ("label" as const) : ("predict" as const);
    return {
      action,
      expectedReward: +(utility * rl.rewardScale).toFixed(3),
      policyConfidence: +Math.min(0.99, 0.55 + utility * 0.4).toFixed(3),
      exploration: isExplore,
    };
  }, [current, strategy, rl]);

  useEffect(() => {
    if (!running) return;
    if (currentIndex >= samples.length) {
      setRunning(false);
      toast("Simulation complete", { description: `Used ${used}/${budget} budget across ${history.length} steps.` });
      return;
    }
    const t = setTimeout(stepOnce, speedMs);
    return () => clearTimeout(t);
  }, [running, currentIndex, samples.length, speedMs, stepOnce, setRunning, used, budget, history.length]);

  if (!experimentStarted) {
    return <NotStarted onStart={startExperiment} />;
  }

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold md:text-3xl">
            Live <span className="text-gradient">Simulation</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Watch the agent decide between requesting expert labels and predicting automatically.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="font-mono-num">step {history.length}/{samples.length}</Badge>
          <Badge variant="outline" className="border-primary/40 text-primary uppercase">{strategy}</Badge>
          <Button variant="outline" size="sm" onClick={resetExperiment}>
            <RotateCcw className="mr-1 h-4 w-4" /> Reset
          </Button>
          <Button variant="outline" size="sm" onClick={stepOnce} disabled={running || currentIndex >= samples.length}>
            <StepForward className="mr-1 h-4 w-4" /> Step
          </Button>
          <Button
            size="sm"
            className="bg-gradient-primary text-primary-foreground shadow-elegant"
            onClick={() => setRunning(!running)}
            disabled={currentIndex >= samples.length}
          >
            {running ? <><Pause className="mr-1 h-4 w-4" /> Pause</> : <><Play className="mr-1 h-4 w-4" /> Auto-Run</>}
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr_1fr]">
        <ImageStream current={current} upcoming={upcoming} />
        <DecisionEnginePanel sample={current} preview={preview} lastStep={lastStep} />
        <div className="space-y-4">
          <div className="glass rounded-2xl p-5 space-y-4">
            <BudgetBar used={used} total={budget} />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Auto-run speed</span>
              <span className="font-mono-num">{speedMs} ms / step</span>
            </div>
            <Slider value={[speedMs]} min={120} max={1500} step={60} onValueChange={(v) => setSpeed(v[0])} />
          </div>
          <BudgetTimeline history={history} />
        </div>
      </div>
    </motion.div>
  );
}

function NotStarted({ onStart }: { onStart: () => void }) {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <div className="glass max-w-md rounded-3xl p-8 text-center">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-primary text-primary-foreground shadow-elegant">
          <FlaskConical className="h-6 w-6" />
        </div>
        <h2 className="mt-4 font-display text-xl font-semibold">No experiment running</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Configure your experiment first or start a quick run with current defaults.
        </p>
        <div className="mt-5 flex justify-center gap-2">
          <Link
            to="/experiment-setup"
            className="inline-flex h-9 items-center rounded-md border border-border/60 px-4 text-sm font-medium hover:bg-accent"
          >
            Open Setup
          </Link>
          <Button onClick={onStart} className="bg-gradient-primary text-primary-foreground">
            <Play className="mr-1 h-4 w-4" /> Quick Start
          </Button>
        </div>
      </div>
    </div>
  );
}
