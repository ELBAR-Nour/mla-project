import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Coins, Gauge, Target, Play, FlaskConical } from "lucide-react";
import { MetricCard } from "@/components/metric-card";
import { ChartContainer } from "@/components/chart-container";
import { BudgetBar } from "@/components/budget-bar";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import { learningCurve } from "@/lib/mock-data";
import { useApp } from "@/lib/store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — MedAL" },
      { name: "description", content: "Experiment overview and learning efficiency for the RL active learning agent." },
    ],
  }),
  component: Dashboard,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "12px",
  fontSize: "12px",
};

function Dashboard() {
  const { visibleStrategies, toggleStrategy, budget, history, experimentStarted } = useApp();
  const used = history.filter((h) => h.action === "label").length;
  const last = history[history.length - 1];
  const accuracy = last ? `${(last.accuracy * 100).toFixed(1)}%` : "—";
  const auc = last ? last.auc.toFixed(3) : "—";

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight md:text-3xl">
            Experiment <span className="text-gradient">Overview</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Live performance of the RL annotation agent across sampling strategies.
          </p>
        </div>
        <div className="flex gap-2">
          <Link to="/experiment-setup" className="inline-flex h-9 items-center rounded-md border border-border/60 px-4 text-sm font-medium hover:bg-accent">
            <FlaskConical className="mr-1.5 h-4 w-4" /> Setup
          </Link>
          <Link to="/annotation-lab">
            <Button className="bg-gradient-primary text-primary-foreground shadow-elegant">
              <Play className="mr-1.5 h-4 w-4" /> Open Simulation
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Accuracy" value={accuracy} hint={experimentStarted ? "live" : "no run yet"} tone="success" icon={<Target className="h-5 w-5" />} />
        <MetricCard label="AUC Score" value={auc} hint="latest step" icon={<Gauge className="h-5 w-5" />} />
        <MetricCard label="Remaining Budget" value={`${budget - used}`} hint={`of ${budget} queries`} tone="warning" icon={<Coins className="h-5 w-5" />} />
        <MetricCard label="Queries Used" value={`${used}`} hint="this run" icon={<Activity className="h-5 w-5" />} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartContainer
          className="lg:col-span-2"
          title="Learning Efficiency Curve"
          description="AUC achieved per number of expert annotations queried"
          actions={
            <div className="flex flex-wrap items-center gap-1.5">
              {(["rl", "entropy", "random"] as const).map((s) => (
                <Toggle
                  key={s}
                  size="sm"
                  pressed={visibleStrategies[s]}
                  onPressedChange={() => toggleStrategy(s)}
                  className="h-7 gap-1.5 text-xs"
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ background: `var(--color-chart-${s === "rl" ? 1 : s === "entropy" ? 2 : 3})` }}
                  />
                  {s === "rl" ? "RL Agent" : s === "entropy" ? "Entropy" : "Random"}
                </Toggle>
              ))}
            </div>
          }
        >
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={learningCurve} margin={{ left: -10, right: 8, top: 8 }}>
              <defs>
                <linearGradient id="grad-rl" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-chart-1)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="var(--color-chart-1)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} domain={[0.5, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              {visibleStrategies.random && <Line type="monotone" dataKey="random" stroke="var(--color-chart-3)" strokeWidth={2} dot={false} name="Random" />}
              {visibleStrategies.entropy && <Line type="monotone" dataKey="entropy" stroke="var(--color-chart-2)" strokeWidth={2} dot={false} name="Entropy" />}
              {visibleStrategies.rl && <Area type="monotone" dataKey="rl" stroke="var(--color-chart-1)" strokeWidth={3} fill="url(#grad-rl)" name="RL Agent" />}
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>

        <div className="space-y-4">
          <ChartContainer title="Budget Status" description="Remaining annotation budget">
            <BudgetBar used={used} total={budget} />
            <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-lg bg-success/10 p-2 text-success">
                <div className="font-mono-num text-base font-semibold">{budget - used}</div>
                <div>Left</div>
              </div>
              <div className="rounded-lg bg-warning/10 p-2 text-warning">
                <div className="font-mono-num text-base font-semibold">{used}</div>
                <div>Used</div>
              </div>
              <div className="rounded-lg bg-primary/10 p-2 text-primary">
                <div className="font-mono-num text-base font-semibold">
                  {budget > 0 ? Math.round((used / budget) * 100) : 0}%
                </div>
                <div>Spent</div>
              </div>
            </div>
          </ChartContainer>

          <ChartContainer title="Recent Activity" description="Last 10 queries">
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={history.length ? history.slice(-10) : learningCurve.slice(-10)}>
                <Line
                  type="monotone"
                  dataKey={history.length ? "auc" : "rl"}
                  stroke="var(--color-chart-1)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Tooltip contentStyle={tooltipStyle} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </div>
    </motion.div>
  );
}
