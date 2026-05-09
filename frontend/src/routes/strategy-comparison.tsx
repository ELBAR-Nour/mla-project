import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import { ArrowUpDown, Sparkles, Trophy } from "lucide-react";
import { ChartContainer } from "@/components/chart-container";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { learningCurve, strategyTable } from "@/lib/mock-data";

export const Route = createFileRoute("/strategy-comparison")({
  head: () => ({
    meta: [
      { title: "Strategy Arena — MedAL" },
      { name: "description", content: "Race RL, entropy and random sampling head-to-head on efficiency and AUC." },
    ],
  }),
  component: StrategyArena,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "12px",
  fontSize: "12px",
};

type Key = "queries" | "accuracy" | "auc" | "efficiency";

function StrategyArena() {
  const [sortKey, setSortKey] = useState<Key>("efficiency");
  const [desc, setDesc] = useState(true);
  const sorted = [...strategyTable].sort((a, b) => (desc ? b[sortKey] - a[sortKey] : a[sortKey] - b[sortKey]));
  const sortBy = (k: Key) => { if (k === sortKey) setDesc((d) => !d); else { setSortKey(k); setDesc(true); } };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold md:text-3xl">
            Strategy <span className="text-gradient">Arena</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Head-to-head race between sampling strategies. Efficiency = AUC ÷ queries used.
          </p>
        </div>
        <Badge variant="outline" className="gap-1.5 border-primary/30 text-primary">
          <Trophy className="h-3 w-3" /> RL leads at +35% efficiency
        </Badge>
      </div>

      <ChartContainer title="Race to High AUC" description="Animated AUC trajectory across query budget">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={learningCurve} margin={{ left: -10, right: 8 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0.5, 1]} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="rl" stroke="var(--color-chart-1)" strokeWidth={3} dot={{ r: 2 }} animationDuration={1400} name="RL Agent" />
            <Line type="monotone" dataKey="entropy" stroke="var(--color-chart-2)" strokeWidth={2} dot={false} animationDuration={1400} name="Entropy" />
            <Line type="monotone" dataKey="random" stroke="var(--color-chart-3)" strokeWidth={2} dot={false} animationDuration={1400} name="Random" />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartContainer className="lg:col-span-2" title="Performance Table" description="Click columns to sort">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Strategy</TableHead>
                  {(["queries", "accuracy", "auc", "efficiency"] as Key[]).map((k) => (
                    <TableHead key={k} className="text-right">
                      <Button variant="ghost" size="sm" onClick={() => sortBy(k)} className="h-7 capitalize">
                        {k} <ArrowUpDown className="ml-1 h-3 w-3" />
                      </Button>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((row) => {
                  const isRL = row.strategy === "RL Agent";
                  return (
                    <TableRow key={row.strategy} className={isRL ? "bg-primary/5" : ""}>
                      <TableCell className="font-medium">
                        {row.strategy}
                        {isRL && <Badge className="ml-2 bg-gradient-primary text-primary-foreground">Best</Badge>}
                      </TableCell>
                      <TableCell className="text-right font-mono-num">{row.queries}</TableCell>
                      <TableCell className="text-right font-mono-num">{(row.accuracy * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-right font-mono-num">{row.auc.toFixed(3)}</TableCell>
                      <TableCell className="text-right font-mono-num font-semibold text-success">
                        {row.efficiency.toFixed(3)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </ChartContainer>

        <ChartContainer title="Insights" description="Auto-generated from the run">
          <ul className="space-y-3 text-sm">
            {[
              "RL agent reaches 0.90 AUC with 35% fewer expert queries.",
              "Entropy sampling plateaus near 0.91 AUC after 240 queries.",
              "RL agent becomes more selective after the 40-query mark.",
              "Random sampling shows highest variance and lowest efficiency.",
            ].map((t) => (
              <li key={t} className="flex gap-2 rounded-xl border border-border/60 bg-card/50 p-3">
                <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </ChartContainer>
      </div>

      <ChartContainer title="Final Performance" description="AUC and efficiency by strategy">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={strategyTable} margin={{ left: -10, right: 8 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="strategy" stroke="var(--muted-foreground)" fontSize={11} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="auc" fill="var(--color-chart-1)" radius={[8, 8, 0, 0]} name="AUC" />
            <Bar dataKey="efficiency" fill="var(--color-chart-2)" radius={[8, 8, 0, 0]} name="Efficiency" />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </motion.div>
  );
}
