import { motion } from "framer-motion";
import xray from "@/assets/xray-sample.jpg";
import type { Sample } from "@/lib/store";
import { cn } from "@/lib/utils";

interface Props {
  current: Sample | null;
  upcoming: Sample[];
}

export function ImageStream({ current, upcoming }: Props) {
  return (
    <div className="glass overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono-num text-xs text-muted-foreground">CURRENT</span>
          <span className="font-mono-num text-sm font-semibold">
            {current?.id ?? "—"}
          </span>
        </div>
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
          Stream
        </span>
      </div>

      <div className="relative grid aspect-square place-items-center overflow-hidden bg-black">
        {current ? (
          <motion.img
            key={current.id}
            initial={{ opacity: 0, scale: 1.04 }}
            animate={{ opacity: 1, scale: 1 }}
            src={xray}
            alt={`Sample ${current.id}`}
            className="h-full w-full select-none object-contain"
            draggable={false}
          />
        ) : (
          <div className="text-xs text-muted-foreground">No active sample</div>
        )}
        {current && (
          <>
            <div className="pointer-events-none absolute inset-0 ring-2 ring-inset ring-primary/40" />
            <div className="absolute left-3 top-3 rounded-md bg-black/60 px-2 py-0.5 font-mono-num text-[10px] text-white">
              entropy {current.entropy.toFixed(2)}
            </div>
            <div className="absolute right-3 top-3 rounded-md bg-black/60 px-2 py-0.5 font-mono-num text-[10px] text-white">
              conf {(current.confidence * 100).toFixed(0)}%
            </div>
          </>
        )}
      </div>

      <div className="border-t border-border/60 p-3">
        <div className="mb-2 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Up next
        </div>
        <div className="flex gap-2 overflow-x-auto">
          {upcoming.length === 0 && (
            <div className="text-xs text-muted-foreground">Queue empty</div>
          )}
          {upcoming.map((s, i) => (
            <div
              key={s.id}
              className={cn(
                "flex h-12 w-16 shrink-0 flex-col items-center justify-center rounded-lg border bg-card/60 px-1 text-[10px]",
                i === 0 ? "border-primary/50" : "border-border/60",
              )}
              title={s.id}
            >
              <span className="font-mono-num">{s.id.slice(-4)}</span>
              <span className="font-mono-num text-muted-foreground">
                e{s.entropy.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
