import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ChartContainer } from "@/components/chart-container";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { useApp } from "@/lib/store";
import { toast } from "sonner";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — MedAL" },
      { name: "description", content: "Configure dataset, annotation budget and model parameters." },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const { dataset, setDataset, budget, setBudget, theme, toggleTheme } = useApp();

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold md:text-3xl">
          <span className="text-gradient">Settings</span>
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure the experiment, dataset, and model behavior.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartContainer title="Dataset" description="Choose the medical imaging dataset">
          <div className="space-y-3">
            <Label>Active dataset</Label>
            <Select value={dataset} onValueChange={(v) => setDataset(v as never)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="PneumoniaMNIST">PneumoniaMNIST</SelectItem>
                <SelectItem value="BreastMNIST">BreastMNIST</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </ChartContainer>

        <ChartContainer title="Annotation Budget" description="Maximum expert queries per run">
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <Label>Budget</Label>
              <span className="font-mono-num text-2xl font-semibold text-gradient">{budget}</span>
            </div>
            <Slider
              value={[budget]}
              min={10}
              max={80}
              step={5}
              onValueChange={(v) => setBudget(v[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground font-mono-num">
              <span>10</span><span>80</span>
            </div>
          </div>
        </ChartContainer>

        <ChartContainer title="Appearance" description="Theme preferences">
          <div className="flex items-center justify-between">
            <div>
              <Label>Dark mode</Label>
              <p className="text-xs text-muted-foreground">Easier on the eyes for long sessions.</p>
            </div>
            <Switch checked={theme === "dark"} onCheckedChange={toggleTheme} />
          </div>
        </ChartContainer>

        <ChartContainer title="Model Parameters" description="Advanced (RL agent)">
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: "Learning rate", value: "3e-4" },
              { label: "Discount γ", value: "0.95" },
              { label: "ε-greedy", value: "0.10" },
              { label: "Replay buffer", value: "10k" },
            ].map((p) => (
              <div key={p.label} className="rounded-xl border border-border/60 p-3">
                <div className="text-xs text-muted-foreground">{p.label}</div>
                <div className="font-mono-num text-base font-semibold">{p.value}</div>
              </div>
            ))}
          </div>
          <Button
            className="mt-4 w-full bg-gradient-primary text-primary-foreground"
            onClick={() => toast.success("Configuration saved")}
          >
            Save Configuration
          </Button>
        </ChartContainer>
      </div>
    </motion.div>
  );
}
