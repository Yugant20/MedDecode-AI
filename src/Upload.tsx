import { useState } from "react";
import { processReport, uploadReport } from "./api";
import type { ReportItem } from "./App";

export default function Upload(props: {
  onNewReport: (item: ReportItem) => void;
  onUpdateReport: (id: string, patch: Partial<ReportItem>) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string>("");

  const btnBase: React.CSSProperties = {
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.08)",
    color: "#e5e7eb",
    padding: "10px 14px",
    borderRadius: 12,
    cursor: "pointer",
    fontWeight: 800,
  };

  const btnPrimary: React.CSSProperties = {
    ...btnBase,
    background:
      "linear-gradient(135deg, rgba(59,130,246,0.90), rgba(147,51,234,0.80))",
    border: "1px solid rgba(255,255,255,0.16)",
  };

  async function onUploadAnalyze() {
    if (!file) return;

    setBusy(true);
    setMsg("");

    try {
      // 1) upload
      const up = await uploadReport(file);

      const newItem: ReportItem = {
        id: up.report_id,
        filename: file.name,
        createdAt: new Date().toISOString(),
        status: "uploaded",
      };
      props.onNewReport(newItem);

      // 2) process
      const pr = await processReport(up.report_id);

      if (pr.status === "processed") {
        props.onUpdateReport(up.report_id, { status: "processed" });
        setMsg("✅ Uploaded + analyzed. Your PDF is ready in ‘My reports’.");
      } else {
        props.onUpdateReport(up.report_id, { status: "failed" });
        setMsg(`⚠️ Process finished with status: ${pr.status}`);
      }

      // reset input
      setFile(null);
      const input = document.getElementById("fileInput") as HTMLInputElement | null;
      if (input) input.value = "";
    } catch (e: any) {
      setMsg(`❌ ${e?.message ?? "Upload failed"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, marginBottom: 4 }}>Upload radiology PDF</div>
          <div style={{ color: "#9ca3af", fontSize: 13, lineHeight: 1.4 }}>
            Best results if the PDF contains clear X-ray images (not blurry screenshots).
          </div>
        </div>
        <div style={{ color: "#9ca3af", fontSize: 12 }}>PDF only • Local dev</div>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap", alignItems: "center" }}>
        <div
          style={{
            flex: 1,
            minWidth: 280,
            border: "1px dashed rgba(255,255,255,0.20)",
            background: "rgba(255,255,255,0.04)",
            borderRadius: 14,
            padding: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div style={{ display: "grid", gap: 4 }}>
            <div style={{ fontWeight: 800, fontSize: 13 }}>
              {file ? file.name : "Choose a PDF to analyze"}
            </div>
            <div style={{ color: "#9ca3af", fontSize: 12 }}>
              {file ? `${Math.round(file.size / 1024)} KB` : "No file selected"}
            </div>
          </div>

          <label style={btnBase}>
            Browse
            <input
              id="fileInput"
              type="file"
              accept="application/pdf"
              style={{ display: "none" }}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </div>

        <button
          style={{
            ...btnPrimary,
            opacity: !file || busy ? 0.55 : 1,
            cursor: !file || busy ? "not-allowed" : "pointer",
          }}
          disabled={!file || busy}
          onClick={onUploadAnalyze}
        >
          {busy ? "Analyzing…" : "Upload + Analyze"}
        </button>
      </div>

      {busy && (
        <div style={{ marginTop: 10, color: "#93c5fd", fontSize: 13 }}>
          ⏳ Rendering pages • Analyzing the report • Building summary PDF…
        </div>
      )}

      {msg ? (
        <div
          style={{
            marginTop: 10,
            display: "inline-flex",
            padding: "7px 10px",
            borderRadius: 999,
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.06)",
            color: "#d1d5db",
            fontSize: 12,
            maxWidth: "100%",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {msg}
        </div>
      ) : null}
    </div>
  );
}