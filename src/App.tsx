import { useEffect, useMemo, useState } from "react";
import Upload from "./Upload";
import Dashboard from "./Dashboard";

export type ReportItem = {
  id: string;
  filename: string;
  createdAt: string; // ISO
  status: "uploaded" | "processed" | "failed";
};

const LS_KEY = "md_reports_v1";

function loadReports(): ReportItem[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveReports(items: ReportItem[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(items));
}

export default function App() {
  const [reports, setReports] = useState<ReportItem[]>(() => loadReports());

  useEffect(() => {
    saveReports(reports);
  }, [reports]);

  const sorted = useMemo(() => {
    return [...reports].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }, [reports]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 600px at 20% 10%, rgba(59,130,246,0.22), transparent 55%), radial-gradient(900px 500px at 90% 30%, rgba(147,51,234,0.18), transparent 55%), #070A12",
        color: "#E5E7EB",
      }}
    >
      <div style={{ maxWidth: 980, margin: "0 auto", padding: "38px 18px 60px" }}>
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginBottom: 18,
          }}
        >
          <img
            src="/logo.png"
            alt="MedDecode AI"
            style={{ width: 44, height: 44, borderRadius: 12 }}
            onError={(e) => {
              // if logo isn't in public, just hide it
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
          <div>
            <div style={{ fontSize: 28, fontWeight: 900, letterSpacing: 0.2 }}>
              MedDecode AI
            </div>
            <div style={{ color: "#9CA3AF", fontSize: 13, marginTop: 2 }}>
              Upload an X-ray PDF → get a clear, structured summary PDF.
            </div>
          </div>
        </div>

        {/* Hero + Steps */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.25fr 0.75fr",
            gap: 16,
            alignItems: "stretch",
          }}
        >
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(255,255,255,0.04)",
              borderRadius: 18,
              padding: 16,
            }}
          >
            <Upload
              onNewReport={(item) => {
                setReports((prev) => [item, ...prev]);
              }}
              onUpdateReport={(id, patch) => {
                setReports((prev) =>
                  prev.map((r) => (r.id === id ? { ...r, ...patch } : r))
                );
              }}
            />
          </div>

          <div
            style={{
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(255,255,255,0.03)",
              borderRadius: 18,
              padding: 16,
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 10 }}>
              What you’ll get
            </div>

            <div style={{ display: "grid", gap: 10, color: "#D1D5DB", fontSize: 13 }}>
              <div>
                ✅ <b>Possible diagnosis</b> (not definitive)
              </div>
              <div>
                ✅ <b>Supporting visual cues</b> (why it thinks so)
              </div>
              <div>
                ✅ <b>What would confirm</b> (views/tests)
              </div>
              <div>
                ✅ <b>Limits + next step</b>
              </div>

              <div
                style={{
                  marginTop: 10,
                  padding: "10px 12px",
                  borderRadius: 14,
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  color: "#9CA3AF",
                  fontSize: 12,
                  lineHeight: 1.4,
                }}
              >
                Not medical advice. Use it to ask better questions and confirm with a clinician.
              </div>
            </div>
          </div>
        </div>

        {/* Reports */}
        <div
          style={{
            marginTop: 18,
            border: "1px solid rgba(255,255,255,0.10)",
            background: "rgba(255,255,255,0.03)",
            borderRadius: 18,
            padding: 16,
          }}
        >
          <Dashboard
            reports={sorted}
            onClear={() => setReports([])}
            onRemove={(id) => setReports((prev) => prev.filter((r) => r.id !== id))}
          />
        </div>
      </div>
    </div>
  );
}