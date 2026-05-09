import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartContainer } from "@/components/chart-container";
import {
  confusionMatrix,
  entropyHistogram,
  rocCurve,
  trainingCurves,
} from "@/lib/mock-data";

export const Route = createFileRoute("/model-insights")({
  head: () => ({
    meta: [
      { title: "Model Insights — MedAL" },
      { name: "description", content: "Confusion matrix, ROC curves, and training analytics." },
    ],
  }),
  component: ModelInsights,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "12px",
  fontSize: "12px",
};

function ConfusionHeatmap() {
  const labels = ["Normal", "Pneumonia"];
  const max = Math.max(...confusionMatrix.flat());
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-[80px_1fr_1fr] gap-1 text-xs text-muted-foreground">
        <div />
        <div className="text-center">Pred Normal</div>
        <div className="text-center">Pred Pneumonia</div>
      </div>
      {confusionMatrix.map((row, i) => (
        <div key={i} className="grid grid-cols-[80px_1fr_1fr] items-center gap-1">
          <div className="text-right text-xs text-muted-foreground">{labels[i]}</div>
          {row.map((v, j) => {
            const intensity = v / max;
            const correct = i === j;
            return (
              <div
                key={j}
                className="grid h-20 place-items-center rounded-xl border border-border/60 font-mono-num text-lg font-semibold transition-transform hover:scale-[1.03]"
                style={{
                  background: correct
                    ? `color-mix(in oklab, var(--success) ${intensity * 60}%, transparent)`
                    : `color-mix(in oklab, var(--destructive) ${intensity * 60}%, transparent)`,
                  color: correct ? "var(--success)" : "var(--destructive)",
                }}
              >
                {v}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function ModelInsights() {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold md:text-3xl">
          Model <span className="text-gradient">Insights</span>
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Diagnostics and uncertainty analysis for the current model.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartContainer title="Confusion Matrix" description="True vs predicted on hold-out set">
          <ConfusionHeatmap />
        </ChartContainer>

        <ChartContainer title="ROC Curve" description="Multi-strategy comparison">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocCurve} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="fpr" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="rl" stroke="var(--color-chart-1)" strokeWidth={3} dot={false} name="RL" />
              <Line type="monotone" dataKey="entropy" stroke="var(--color-chart-2)" strokeWidth={2} dot={false} name="Entropy" />
              <Line type="monotone" dataKey="random" stroke="var(--color-chart-3)" strokeWidth={2} dot={false} name="Random" />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="Training Loss" description="Per epoch">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trainingCurves} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="loss" stroke="var(--color-chart-5)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="Training Accuracy" description="Per epoch">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trainingCurves} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0.5, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="accuracy" stroke="var(--color-chart-2)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer
          className="lg:col-span-2"
          title="Uncertainty Analysis"
          description="Entropy distribution across the unlabeled pool — highlighted bars are samples selected by the RL agent"
        >
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={entropyHistogram} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="bin" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {entropyHistogram.map((d, i) => (
                  <Cell
                    key={i}
                    fill={d.selected > 5 ? "var(--color-chart-1)" : "var(--color-chart-3)"}
                    fillOpacity={d.selected > 5 ? 1 : 0.5}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>
    </motion.div>
  );
}
