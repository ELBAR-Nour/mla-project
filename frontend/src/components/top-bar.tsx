import { Moon, Sun, Database } from "lucide-react";
import { SidebarTrigger } from "@/components/ui/sidebar";
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

export function TopBar() {
  const { dataset, setDataset, theme, toggleTheme, budget, remaining } = useApp();
  const budgetRemaining = remaining();
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border/60 glass-strong px-3 md:px-5">
      <SidebarTrigger />
      <div className="hidden items-center gap-2 md:flex">
        <Database className="h-4 w-4 text-muted-foreground" />
        <Select value={dataset} onValueChange={(v) => setDataset(v as never)}>
          <SelectTrigger className="h-8 w-[180px] border-border/60">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="PneumoniaMNIST">PneumoniaMNIST</SelectItem>
            <SelectItem value="BreastMNIST">BreastMNIST</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <Badge variant="outline" className="hidden gap-1.5 border-success/30 text-success md:inline-flex">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
          Live
        </Badge>
        <Badge variant="outline" className="font-mono-num">
          Budget {budgetRemaining}/{budget}
        </Badge>
        <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === "light" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
