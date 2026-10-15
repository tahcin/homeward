import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Icon } from "./components/Icon.tsx";
import { getConfig, type Config } from "./lib/api.ts";
import { Evidence } from "./pages/Evidence.tsx";
import { Flow } from "./pages/Flow.tsx";
import { Home } from "./pages/Home.tsx";
import { How } from "./pages/How.tsx";

// ---------------------------------------------------------------- tiny router

const RouterCtx = createContext<{ path: string; go: (p: string) => void }>({ path: "/", go: () => {} });
export const useRouter = () => useContext(RouterCtx);

export function Link({ to, children, className, ...rest }: { to: string; children: React.ReactNode; className?: string } & React.AnchorHTMLAttributes<HTMLAnchorElement>) {
  const { go, path } = useRouter();
  return (
    <a
      href={to}
      className={className}
      aria-current={path === to ? "page" : undefined}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        e.preventDefault();
        go(to);
      }}
      {...rest}
    >
      {children}
    </a>
  );
}

const ConfigCtx = createContext<Config | null>(null);
export const useConfig = () => useContext(ConfigCtx);

const ChromeCtx = createContext<(hidden: boolean) => void>(() => {});
/** Patient mode hides the site header so the screen belongs to the patient. */
export const useHideChrome = () => useContext(ChromeCtx);

export function App() {
  const [path, setPath] = useState(() => location.pathname);
  const [config, setConfig] = useState<Config | null>(null);
  const [chromeHidden, setChromeHidden] = useState(false);

  useEffect(() => {
    const on = () => setPath(location.pathname);
    window.addEventListener("popstate", on);
    getConfig().then(setConfig);
    return () => window.removeEventListener("popstate", on);
  }, []);

  const go = useCallback((p: string) => {
    if (p !== location.pathname) history.pushState(null, "", p);
    setPath(p);
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, []);

  useEffect(() => {
    document.title = path === "/how" ? "How Homeward works" : path === "/evidence" ? "Evidence · Homeward" : path === "/check" ? "Discharge check · Homeward" : "Homeward";
  }, [path]);

  const page = path === "/how" ? <How /> : path === "/evidence" ? <Evidence /> : path === "/check" ? <Flow /> : <Home />;

  return (
    <RouterCtx.Provider value={{ path, go }}>
      <ConfigCtx.Provider value={config}>
        <ChromeCtx.Provider value={setChromeHidden}>
          {!chromeHidden && (
            <header className="top">
              <div className="wrap">
                <Link to="/" className="brand">
                  <Icon name="home" size={22} />
                  Homeward
                </Link>
                <nav className="nav" aria-label="Main">
                  <Link to="/check" className="hide-sm">
                    Discharge check
                  </Link>
                  <Link to="/how">How it works</Link>
                  <Link to="/evidence">Evidence</Link>
                </nav>
              </div>
            </header>
          )}
          <main>{page}</main>
          {!chromeHidden && (
            <footer className="foot">
              <div className="wrap">
                <span>Hosted demo: synthetic data only, model calls run on AMD GPUs through AMD Token Factory. Not medical advice.</span>
              </div>
            </footer>
          )}
        </ChromeCtx.Provider>
      </ConfigCtx.Provider>
    </RouterCtx.Provider>
  );
}
