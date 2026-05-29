import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
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
import { getExperimentSummary, type ExperimentSummary } from "@/lib/api";
import { strategyDefinitions, strategyForNotebookName } from "@/lib/strategies";

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

type Key = "queries" | "f1" | "auc" | "alc" | "efficiency";

function StrategyArena() {
  const [sortKey, setSortKey] = useState<Key>("efficiency");
  const [desc, setDesc] = useState(true);
  const [summary, setSummary] = useState<ExperimentSummary | null>(null);
  useEffect(() => {
    void getExperimentSummary().then(setSummary).catch(() => setSummary(null));
  }, []);

  const strategyTable = useMemo(() => {
    if (!summary) return [];
    return summary.main_results.map((result) => {
      const metric = summary.clinical_metrics.find((item) => item.strategy.toLowerCase() === result.strategy.toLowerCase());
      const id = strategyForNotebookName(result.strategy);
      const definition = strategyDefinitions.find((item) => item.id === id);
      return {
        strategy: definition?.label ?? result.strategy,
        queries: result.queries,
        f1: metric?.f1 ?? 0,
        auc: result.final_test_auc,
        alc: result.alc,
        efficiency: metric?.efficiency ?? result.alc / Math.max(result.queries, 1),
      };
    });
  }, [summary]);

  const learningCurve = useMemo(() => {
    if (!summary) return [];
    const rows = new Map<number, Record<string, number>>();
    for (const point of summary.learning_curves) {
      const id = strategyForNotebookName(point.strategy);
      const existing = rows.get(point.queries) ?? { queries: point.queries };
      existing[id] = point.test_auc;
      rows.set(point.queries, existing);
    }
    return [...rows.values()].sort((a, b) => a.queries - b.queries);
  }, [summary]);

  const sorted = [...strategyTable].sort((a, b) => (desc ? b[sortKey] - a[sortKey] : a[sortKey] - b[sortKey]));
  const sortBy = (k: Key) => { if (k === sortKey) setDesc((d) => !d); else { setSortKey(k); setDesc(true); } };
  const leading = sorted[0];

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
        {leading && (
          <Badge variant="outline" className="gap-1.5 border-primary/30 text-primary">
            <Trophy className="h-3 w-3" /> {leading.strategy} leads on {sortKey === "f1" ? "F1" : sortKey}
          </Badge>
        )}
      </div>

      <ChartContainer title="Race to High AUC" description="Animated AUC trajectory across query budget">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={learningCurve} margin={{ left: -10, right: 8 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0.5, 1]} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {strategyDefinitions.map((definition, index) => (
              <Line
                key={definition.id}
                type="monotone"
                dataKey={definition.id}
                stroke={`var(--color-chart-${(index % 5) + 1})`}
                strokeWidth={definition.kind === "rl" ? 3 : 2}
                dot={definition.kind === "rl" ? { r: 2 } : false}
                animationDuration={1400}
                name={definition.label}
              />
            ))}
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
                  {(["queries", "f1", "auc", "alc", "efficiency"] as Key[]).map((k) => (
                    <TableHead key={k} className="text-right">
                      <Button variant="ghost" size="sm" onClick={() => sortBy(k)} className="h-7 capitalize">
                        {k === "f1" || k === "alc" ? k.toUpperCase() : k} <ArrowUpDown className="ml-1 h-3 w-3" />
                      </Button>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((row) => {
                  const isRL = row.strategy.includes("DQN");
                  const isLeading = row.strategy === leading?.strategy;
                  return (
                    <TableRow key={row.strategy} className={isRL ? "bg-primary/5" : ""}>
                      <TableCell className="font-medium">
                        {row.strategy}
                        {isLeading && <Badge className="ml-2 bg-gradient-primary text-primary-foreground">Top</Badge>}
                      </TableCell>
                      <TableCell className="text-right font-mono-num">{row.queries}</TableCell>
                      <TableCell className="text-right font-mono-num">{row.f1.toFixed(3)}</TableCell>
                      <TableCell className="text-right font-mono-num">{row.auc.toFixed(3)}</TableCell>
                      <TableCell className="text-right font-mono-num">{row.alc.toFixed(3)}</TableCell>
                      <TableCell className="text-right font-mono-num font-semibold text-success">
                        {row.efficiency.toFixed(6)}
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
              "All rows come from the exported notebook experiment summary.",
              "DQN-family policies are compared alongside Random, Margin, Entropy, BALD, and BADGE.",
              "The curve uses test AUC from the long learning-curve CSV.",
              "Efficiency is read from the notebook clinical metrics export.",
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
