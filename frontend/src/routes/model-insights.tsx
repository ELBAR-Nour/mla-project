import { useCallback, useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Brain, Loader2, RefreshCw, ScanSearch } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartContainer } from "@/components/chart-container";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getModelArtifacts,
  getConfusionMatrix,
  getExperimentSummary,
  getROCCurve,
  runModelInference,
  runRLDecision,
  type ConfusionMatrixResponse,
  type ExperimentSummary,
  type InferenceResult,
  type ModelArtifact,
  type ModelArtifactsResponse,
  type ROCResponse,
  type RLDecisionResult,
} from "@/lib/api";
import { strategyDefinitions, strategyForNotebookName } from "@/lib/strategies";
import { toast } from "sonner";

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

const datasets = ["PneumoniaMNIST", "BreastMNIST"] as const;

function artifactLabel(artifact: ModelArtifact) {
  const detail = artifact.strategy ?? artifact.source ?? artifact.architecture ?? "artifact";
  return `${artifact.name} · ${detail}`;
}

function pickClassifier(artifacts: ModelArtifact[]) {
  return (
    artifacts.find((artifact) => artifact.strategy === "entropy") ??
    artifacts.find((artifact) => !artifact.error)
  );
}

function pickCheckpoint(artifacts: ModelArtifact[]) {
  return (
    artifacts.find((artifact) => artifact.name === "dueling_dqn_sota.pt") ??
    artifacts.find((artifact) => artifact.name.includes("dueling")) ??
    artifacts.find((artifact) => !artifact.error)
  );
}

function InferenceWorkbench({ onEvaluationRefresh }: { onEvaluationRefresh?: () => Promise<void> }) {
  const [artifacts, setArtifacts] = useState<ModelArtifactsResponse | null>(null);
  const [modelName, setModelName] = useState("");
  const [checkpointName, setCheckpointName] = useState("");
  const [dataset, setDataset] = useState<(typeof datasets)[number]>("PneumoniaMNIST");
  const [imageId, setImageId] = useState(0);
  const [budgetRemaining, setBudgetRemaining] = useState(120);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [decision, setDecision] = useState<RLDecisionResult | null>(null);

  const classifiers = useMemo(
    () => artifacts?.classifiers.filter((artifact) => !artifact.error) ?? [],
    [artifacts],
  );
  const checkpoints = useMemo(
    () => artifacts?.checkpoints.filter((artifact) => !artifact.error) ?? [],
    [artifacts],
  );

  const refreshArtifacts = async () => {
    setLoadingArtifacts(true);
    setError(null);
    try {
      const result = await getModelArtifacts();
      setArtifacts(result);
      const nextModel = result.active_model ?? pickClassifier(result.classifiers)?.name ?? "";
      const nextCheckpoint = result.active_checkpoint ?? pickCheckpoint(result.checkpoints)?.name ?? "";
      setModelName((current) => current || nextModel);
      setCheckpointName((current) => current || nextCheckpoint);
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : "Unable to reach backend";
      setError(message);
      toast.error("Could not load model artifacts", { description: message });
    } finally {
      setLoadingArtifacts(false);
    }
  };

  useEffect(() => {
    void refreshArtifacts();
  }, []);

  const handleRun = async () => {
    if (!modelName) {
      toast.error("Select a classifier model");
      return;
    }

    setRunning(true);
    setError(null);
    try {
      const safeImageId = Math.max(0, Number(imageId) || 0);
      const prediction = await runModelInference({
        model_name: modelName,
        image_id: safeImageId,
        dataset_name: dataset,
        split: "train",
      });
      setInference(prediction);

      if (checkpointName) {
        try {
          const policy = await runRLDecision({
            checkpoint_name: checkpointName,
            model_name: modelName,
            image_id: safeImageId,
            dataset_name: dataset,
            split: "train",
            budget_remaining: budgetRemaining,
            max_budget: 200,
          });
          setDecision(policy);
        } catch (exc) {
          const message = exc instanceof Error ? exc.message : "RL policy decision failed";
          setDecision(null);
          setError(message);
          toast.warning("Classifier inference completed", { description: message });
        }
      } else {
        setDecision(null);
      }

      try {
        await onEvaluationRefresh?.();
      } catch (exc) {
        const message = exc instanceof Error ? exc.message : "Could not refresh model diagnostics";
        toast.warning("Diagnostics refresh failed", { description: message });
      }

      toast.success("Inference complete", {
        description: `${prediction.predicted_label_name} · ${(prediction.confidence * 100).toFixed(1)}% confidence`,
      });
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : "Inference failed";
      setError(message);
      toast.error("Inference failed", { description: message });
    } finally {
      setRunning(false);
    }
  };

  const sampleImage = inference?.image_base64 ?? decision?.image_base64 ?? null;
  const prediction = inference ?? decision?.prediction ?? null;

  return (
    <ChartContainer
      className="lg:col-span-2"
      title="Live Inference"
      description="Saved PyTorch classifier and RL policy artifacts"
      actions={
        <Button variant="outline" size="sm" onClick={refreshArtifacts} disabled={loadingArtifacts}>
          {loadingArtifacts ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-1 h-4 w-4" />}
          Refresh
        </Button>
      }
    >
      <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={dataset} onValueChange={(value) => setDataset(value as never)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {datasets.map((item) => (
                    <SelectItem key={item} value={item}>{item}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Image ID</Label>
              <Input
                type="number"
                min={0}
                value={imageId}
                onChange={(event) => setImageId(Number(event.target.value))}
              />
            </div>

            <div className="space-y-2 sm:col-span-2 xl:col-span-1">
              <Label>Classifier</Label>
              <Select value={modelName} onValueChange={setModelName} disabled={classifiers.length === 0}>
                <SelectTrigger><SelectValue placeholder="No classifier artifacts" /></SelectTrigger>
                <SelectContent>
                  {classifiers.map((artifact) => (
                    <SelectItem key={artifact.name} value={artifact.name}>{artifactLabel(artifact)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 sm:col-span-2 xl:col-span-1">
              <Label>RL Checkpoint</Label>
              <Select value={checkpointName} onValueChange={setCheckpointName} disabled={checkpoints.length === 0}>
                <SelectTrigger><SelectValue placeholder="No RL checkpoints" /></SelectTrigger>
                <SelectContent>
                  {checkpoints.map((artifact) => (
                    <SelectItem key={`${artifact.source}-${artifact.name}`} value={artifact.name}>{artifactLabel(artifact)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Budget remaining</Label>
              <Input
                type="number"
                min={0}
                max={200}
                value={budgetRemaining}
                onChange={(event) => setBudgetRemaining(Number(event.target.value))}
              />
            </div>
          </div>

          <Button
            className="w-full bg-gradient-primary text-primary-foreground"
            onClick={handleRun}
            disabled={running || !modelName}
          >
            {running ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ScanSearch className="mr-2 h-4 w-4" />}
            Run Inference
          </Button>

          {error && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}
        </div>

        <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
          <div className="overflow-hidden rounded-xl border border-border/60 bg-black">
            <div className="grid aspect-square place-items-center">
              {sampleImage ? (
                <img
                  src={`data:image/png;base64,${sampleImage}`}
                  alt="Inference sample"
                  className="h-full w-full object-contain"
                  draggable={false}
                />
              ) : (
                <div className="px-4 text-center text-xs text-muted-foreground">
                  Run inference to preview the sample.
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-border/60 p-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="font-display text-sm font-semibold">Classifier Output</div>
                {prediction && (
                  <Badge variant="outline" className="border-primary/40 text-primary">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </Badge>
                )}
              </div>
              {prediction ? (
                <div className="space-y-3">
                  {prediction.probabilities.map((probability, index) => (
                    <div key={prediction.class_labels[index] ?? index}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span>{prediction.class_labels[index] ?? `Class ${index}`}</span>
                        <span className="font-mono-num text-muted-foreground">
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={probability * 100} />
                    </div>
                  ))}
                  <div className="grid grid-cols-2 gap-3 border-t border-border/60 pt-3 text-xs">
                    <div>
                      <div className="text-muted-foreground">Prediction</div>
                      <div className="font-semibold">{prediction.predicted_label_name}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Entropy</div>
                      <div className="font-mono-num font-semibold">{prediction.entropy.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">True label</div>
                      <div className="font-semibold">{prediction.true_label_name ?? "Unknown"}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Correct</div>
                      <div className="font-semibold">{prediction.correct === null ? "Unknown" : prediction.correct ? "Yes" : "No"}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
                  Awaiting classifier result.
                </div>
              )}
            </div>

            <div className="rounded-xl border border-border/60 p-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2 font-display text-sm font-semibold">
                  <Brain className="h-4 w-4 text-primary" />
                  RL Decision
                </div>
                {decision && (
                  <Badge variant="outline" className="border-success/40 text-success">
                    {(decision.policy_confidence * 100).toFixed(1)}%
                  </Badge>
                )}
              </div>
              {decision ? (
                <div className="space-y-3">
                  <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                    <div className="text-xs text-muted-foreground">Recommended action</div>
                    <div className="mt-1 font-display text-base font-semibold text-gradient">
                      {decision.recommended_action === "request_label" ? "Request Expert Label" : "Predict Automatically"}
                    </div>
                  </div>
                  {decision.q_values.map((value, index) => (
                    <div key={index}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span>{index === 1 ? "Request label" : "Predict"}</span>
                        <span className="font-mono-num text-muted-foreground">{value.toFixed(3)}</span>
                      </div>
                      <Progress value={decision.action_probabilities[index] * 100} />
                    </div>
                  ))}
                  <div className="grid grid-cols-2 gap-2 border-t border-border/60 pt-3 text-[11px]">
                    {decision.state_used.slice(0, 6).map((value, index) => (
                      <div key={decision.state_features[index] ?? index}>
                        <div className="text-muted-foreground">{decision.state_features[index] ?? `f${index + 1}`}</div>
                        <div className="font-mono-num font-semibold">{value.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
                  Awaiting policy decision.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </ChartContainer>
  );
}

function ConfusionHeatmap({ data }: { data: ConfusionMatrixResponse | null }) {
  if (!data?.matrix.length) {
    return (
      <div className="grid min-h-52 place-items-center rounded-lg border border-dashed border-border/60 px-3 text-center text-xs text-muted-foreground">
        Run inference to load model evaluation.
      </div>
    );
  }

  const labels = data.labels.map(String);
  const max = Math.max(...data.matrix.flat()) || 1;
  return (
    <div className="space-y-2">
      <div
        className="grid gap-1 text-xs text-muted-foreground"
        style={{ gridTemplateColumns: `80px repeat(${labels.length}, minmax(0, 1fr))` }}
      >
        <div />
        {labels.map((label) => (
          <div key={label} className="text-center">Pred {label}</div>
        ))}
      </div>
      {data.matrix.map((row, i) => (
        <div
          key={labels[i] ?? i}
          className="grid items-center gap-1"
          style={{ gridTemplateColumns: `80px repeat(${row.length}, minmax(0, 1fr))` }}
        >
          <div className="text-right text-xs text-muted-foreground">{labels[i] ?? i}</div>
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
  const [confusion, setConfusion] = useState<ConfusionMatrixResponse | null>(null);
  const [roc, setRoc] = useState<ROCResponse | null>(null);
  const [summary, setSummary] = useState<ExperimentSummary | null>(null);

  const refreshEvaluation = useCallback(async () => {
    const [nextConfusion, nextRoc] = await Promise.all([
      getConfusionMatrix(),
      getROCCurve(),
    ]);
    setConfusion(nextConfusion);
    setRoc(nextRoc);
  }, []);

  useEffect(() => {
    void refreshEvaluation().catch(() => {
      setConfusion(null);
      setRoc(null);
    });
    void getExperimentSummary().then(setSummary).catch(() => setSummary(null));
  }, [refreshEvaluation]);

  const rocChartData = useMemo(
    () => roc?.fpr.map((fpr, index) => ({
      fpr: +fpr.toFixed(3),
      tpr: roc.tpr[index] ?? 0,
    })) ?? [],
    [roc],
  );

  const experimentCurves = useMemo(() => {
    if (!summary) return [];
    const rows = new Map<number, Record<string, number>>();
    for (const point of summary.learning_curves) {
      const id = strategyForNotebookName(point.strategy);
      const row = rows.get(point.queries) ?? { queries: point.queries };
      row[`${id}_val`] = point.val_auc;
      row[`${id}_test`] = point.test_auc;
      rows.set(point.queries, row);
    }
    return [...rows.values()].sort((a, b) => a.queries - b.queries);
  }, [summary]);

  const finalMetrics = useMemo(() => {
    if (!summary) return [];
    return summary.main_results.map((result) => {
      const id = strategyForNotebookName(result.strategy);
      const definition = strategyDefinitions.find((item) => item.id === id);
      return {
        strategy: definition?.shortLabel ?? result.strategy,
        val_auc: result.final_val_auc,
        test_auc: result.final_test_auc,
        alc: result.alc,
      };
    });
  }, [summary]);

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
        <InferenceWorkbench onEvaluationRefresh={refreshEvaluation} />

        <ChartContainer title="Confusion Matrix" description="True vs predicted on hold-out set">
          <ConfusionHeatmap data={confusion} />
        </ChartContainer>

        <ChartContainer title="ROC Curve" description="Current classifier on hold-out set">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocChartData} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="fpr" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="tpr" stroke="var(--color-chart-1)" strokeWidth={3} dot={false} name="Current model" />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="Notebook Validation AUC" description="All exported strategies">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={experimentCurves} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              {strategyDefinitions.map((definition, index) => (
                <Line key={definition.id} type="monotone" dataKey={`${definition.id}_val`} stroke={`var(--color-chart-${(index % 5) + 1})`} strokeWidth={definition.kind === "rl" ? 3 : 2} dot={false} name={definition.label} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="Notebook Test AUC" description="All exported strategies">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={experimentCurves} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0.75, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              {strategyDefinitions.map((definition, index) => (
                <Line key={definition.id} type="monotone" dataKey={`${definition.id}_test`} stroke={`var(--color-chart-${(index % 5) + 1})`} strokeWidth={definition.kind === "rl" ? 3 : 2} dot={false} name={definition.label} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer
          className="lg:col-span-2"
          title="Final Notebook Metrics"
          description="Validation AUC, test AUC, and area under the learning curve from the notebook exports"
        >
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={finalMetrics} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="strategy" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="val_auc" fill="var(--color-chart-1)" radius={[6, 6, 0, 0]} name="Validation AUC" />
              <Bar dataKey="test_auc" fill="var(--color-chart-2)" radius={[6, 6, 0, 0]} name="Test AUC" />
              <Bar dataKey="alc" fill="var(--color-chart-3)" radius={[6, 6, 0, 0]} name="ALC" />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>
    </motion.div>
  );
}
