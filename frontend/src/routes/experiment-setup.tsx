import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { FlaskConical, Play } from "lucide-react";
import { ChartContainer } from "@/components/chart-container";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApp } from "@/lib/store";
import { isRLStrategy, strategyDefinitions } from "@/lib/strategies";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/experiment-setup")({
  head: () => ({
    meta: [
      { title: "Experiment Setup — MedAL" },
      { name: "description", content: "Configure dataset, budget, sampling strategy and RL hyperparameters before running a simulation." },
    ],
  }),
  component: ExperimentSetup,
});

const strategies = strategyDefinitions;
/*
  { id: "random", label: "Random", icon: <Shuffle className="h-4 w-4" />, desc: "Uniform random sampling — baseline." },
  { id: "entropy", label: "Entropy", icon: <Gauge className="h-4 w-4" />, desc: "Query the most uncertain samples." },
  { id: "rl", label: "RL Agent", icon: <Brain className="h-4 w-4" />, desc: "Learns when to query vs. predict." },

*/
function ExperimentSetup() {
  const { dataset, setDataset, budget, setBudget, strategy, setStrategy, rl, setRL, startExperiment } = useApp();
  const navigate = useNavigate();

  const handleStart = () => {
    startExperiment();
    toast.success("Experiment initialized", { description: `${dataset} · ${strategy.toUpperCase()} · budget ${budget}` });
    navigate({ to: "/annotation-lab" });
  };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold md:text-3xl">
            Experiment <span className="text-gradient">Setup</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Configure dataset, annotation budget, and sampling policy. Then run the live simulation.
          </p>
        </div>
        <Badge variant="outline" className="gap-1.5 border-primary/30 text-primary">
          <FlaskConical className="h-3 w-3" /> Step 1 of 5
        </Badge>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartContainer title="Dataset" description="Source of medical samples">
          <div className="space-y-3">
            <Label>Active dataset</Label>
            <Select value={dataset} onValueChange={(v) => setDataset(v as never)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="PneumoniaMNIST">PneumoniaMNIST · Chest X-rays</SelectItem>
                <SelectItem value="BreastMNIST">BreastMNIST · Ultrasound</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Pool size loads up to 80 samples from the ML service.
            </p>
          </div>
        </ChartContainer>

        <ChartContainer title="Annotation Budget" description="Maximum expert queries">
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <Label>Budget</Label>
              <span className="font-mono-num text-2xl font-semibold text-gradient">{budget}</span>
            </div>
            <Slider value={[budget]} min={10} max={80} step={5} onValueChange={(v) => setBudget(v[0])} />
            <div className="flex justify-between text-xs text-muted-foreground font-mono-num">
              <span>10</span><span>80</span>
            </div>
          </div>
        </ChartContainer>

        <ChartContainer className="lg:col-span-2" title="Sampling Strategy" description="Who decides which samples to label">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {strategies.map((s) => {
              const active = strategy === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => setStrategy(s.id)}
                  className={cn(
                    "group flex flex-col items-start gap-2 rounded-2xl border p-4 text-left transition-all",
                    active
                      ? "border-primary/60 bg-primary/5 shadow-elegant"
                      : "border-border/60 hover:border-primary/40 hover:bg-accent/40",
                  )}
                >
                  <div className="flex w-full items-center justify-between">
                    <span className={cn("grid h-9 w-9 place-items-center rounded-xl",
                      active ? "bg-gradient-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>
                      {s.icon}
                    </span>
                    {active && <Badge className="bg-gradient-primary text-primary-foreground">Selected</Badge>}
                  </div>
                  <div className="font-display text-base font-semibold">{s.label}</div>
                  <div className="text-xs text-muted-foreground">{s.desc}</div>
                </button>
              );
            })}
          </div>
        </ChartContainer>

        {isRLStrategy(strategy) && (
        <ChartContainer
          className="lg:col-span-2"
          title="RL Configuration"
          description="Hyperparameters used when the RL agent strategy is active"
        >
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            <RLSlider label="ε-greedy" value={rl.epsilon} min={0} max={0.5} step={0.01} fmt={(v) => v.toFixed(2)} onChange={(v) => setRL("epsilon", v)} hint="Exploration rate" />
            <RLSlider label="Reward scale" value={rl.rewardScale} min={0.2} max={2} step={0.1} fmt={(v) => v.toFixed(1)} onChange={(v) => setRL("rewardScale", v)} hint="Scales reward signal" />
            <RLSlider label="Batch size" value={rl.batchSize} min={4} max={128} step={4} fmt={(v) => String(v)} onChange={(v) => setRL("batchSize", v)} hint="Replay batch" />
            <RLSlider label="Learning rate" value={rl.learningRate * 10000} min={1} max={50} step={1} fmt={(v) => `${(v / 10).toFixed(1)}e-4`} onChange={(v) => setRL("learningRate", v / 10000)} hint="Policy step size" />
          </div>
        </ChartContainer>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-primary/30 bg-primary/5 p-4">
        <div className="text-sm">
          <div className="font-display font-semibold">Ready to run</div>
          <div className="text-xs text-muted-foreground">
            {dataset} · strategy <b className="text-foreground">{strategy.toUpperCase()}</b> · budget {budget}
          </div>
        </div>
        <Button size="lg" onClick={handleStart} className="bg-gradient-primary text-primary-foreground shadow-elegant transition-transform hover:scale-[1.02]">
          <Play className="mr-2 h-4 w-4" /> Start Experiment
        </Button>
      </div>
    </motion.div>
  );
}

function RLSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt: (v: number) => string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-border/60 p-3">
      <div className="flex items-baseline justify-between">
        <Label className="text-xs">{label}</Label>
        <span className="font-mono-num text-sm font-semibold">{fmt(value)}</span>
      </div>
      <Slider className="mt-3" value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
      {hint && <p className="mt-2 text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
