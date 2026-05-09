import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Microscope,
  LineChart,
  GitCompare,
  Settings,
  Activity,
  FlaskConical,
  Brain,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const flow = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Experiment Setup", url: "/experiment-setup", icon: FlaskConical },
  { title: "Live Simulation", url: "/annotation-lab", icon: Microscope },
  { title: "Learning Evolution", url: "/learning-evolution", icon: Brain },
];
const analysis = [
  { title: "Strategy Arena", url: "/strategy-comparison", icon: GitCompare },
  { title: "Model Intelligence", url: "/model-insights", icon: LineChart },
  { title: "Settings", url: "/settings", icon: Settings },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const path = useRouterState({ select: (r) => r.location.pathname });

  const renderItems = (items: typeof flow) =>
    items.map((item) => {
      const active = path === item.url;
      return (
        <SidebarMenuItem key={item.title}>
          <SidebarMenuButton asChild isActive={active} tooltip={item.title}>
            <Link to={item.url} className="flex items-center gap-2">
              <item.icon className="h-4 w-4" />
              {!collapsed && <span>{item.title}</span>}
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      );
    });

  return (
    <Sidebar collapsible="icon" className="border-r border-sidebar-border">
      <SidebarHeader className="border-b border-sidebar-border">
        <Link to="/" className="flex items-center gap-2 px-2 py-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-primary shadow-elegant">
            <Activity className="h-5 w-5 text-primary-foreground" />
          </div>
          {!collapsed && (
            <div className="flex flex-col leading-tight">
              <span className="font-display text-sm font-semibold tracking-wide">MedAL</span>
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                RL · Active Learning
              </span>
            </div>
          )}
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Experiment Flow</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>{renderItems(flow)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Analysis</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>{renderItems(analysis)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
