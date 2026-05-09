import { useState } from "react";
import { ZoomIn, ZoomOut, Grid3x3, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import xray from "@/assets/xray-sample.jpg";

interface ImageViewerProps {
  imageId: string;
}

export function ImageViewer({ imageId }: ImageViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [grid, setGrid] = useState(false);
  return (
    <div className="glass relative overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono-num text-xs text-muted-foreground">ID</span>
          <span className="font-mono-num text-sm font-semibold">{imageId}</span>
        </div>
        <div className="flex items-center gap-1">
          <Toggle
            size="sm"
            pressed={grid}
            onPressedChange={setGrid}
            aria-label="Toggle pixel grid"
          >
            <Grid3x3 className="h-4 w-4" />
          </Toggle>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button size="icon" variant="ghost" onClick={() => setZoom(1)}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="relative grid aspect-square place-items-center overflow-hidden bg-black">
        <img
          src={xray}
          alt="Medical sample"
          className="select-none object-contain transition-transform duration-300"
          style={{ transform: `scale(${zoom})` }}
          draggable={false}
        />
        {grid && (
          <div
            className="pointer-events-none absolute inset-0 opacity-40"
            style={{
              backgroundImage:
                "linear-gradient(to right, rgba(0,255,200,0.25) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,255,200,0.25) 1px, transparent 1px)",
              backgroundSize: "32px 32px",
            }}
          />
        )}
        <div className="absolute bottom-2 right-3 rounded-md bg-black/60 px-2 py-0.5 font-mono-num text-[10px] text-white">
          {(zoom * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}
