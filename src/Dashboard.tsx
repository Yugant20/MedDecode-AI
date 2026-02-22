import { pdfLink } from "./api";
import type { ReportItem } from "./App";

export default function Dashboard(props: {
  reports: ReportItem[];
  onClear: () => void;
  onRemove: (id: string) => void;
}) {
  const btn: React.CSSProperties = {
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.08)",
    color: "#e5e7eb",
    padding: "10px 14px",
    borderRadius: 12,
    cursor: "pointer",
    fontWeight: 800,
  };

  const th: React.CSSProperties = {
    textAlign: "left",
    fontSize: 12,
    color: "#9ca3af",
    padding: "10px 12px",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.04)",
  };

  const td: React.CSSProperties = {
    padding: "10px 12px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    fontSize: 13,
    color: "#e5e7eb",
    verticalAlign: "middle",
  };

  const pill = (text: string) => (
    <span
      style={{
        display: "inline-flex",
        padding: "4px 10px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(255,255,255,0.06)",
        color: "#d1d5db",
        fontSize: 12,
      }}
    >
      {text}
    </span>
  );

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, marginBottom: 4 }}>My reports</div>
          <div style={{ color: "#9ca3af", fontSize: 13 }}>
            Stored locally in your browser (for now). Each report has a downloadable summary PDF.
          </div>
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <button style={btn} onClick={props.onClear}>
            Clear
          </button>
        </div>
      </div>

      <div style={{ marginTop: 14, border: "1px solid rgba(255,255,255,0.10)", borderRadius: 14, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={th}>File</th>
              <th style={th}>Status</th>
              <th style={th}>Uploaded</th>
              <th style={th}>Actions</th>
            </tr>
          </thead>

          <tbody>
            {props.reports.length === 0 ? (
              <tr>
                <td style={td} colSpan={4}>
                  <span style={{ color: "#9ca3af" }}>No reports yet. Upload a PDF above.</span>
                </td>
              </tr>
            ) : (
              props.reports.map((r) => (
                <tr key={r.id}>
                  <td style={td}>{r.filename}</td>
                  <td style={td}>{pill(r.status)}</td>
                  <td style={td}>{new Date(r.createdAt).toLocaleString()}</td>
                  <td style={td}>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                      <a
                        href={pdfLink(r.id)}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          ...btn,
                          textDecoration: "none",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 8,
                          padding: "8px 12px",
                        }}
                      >
                        Open PDF
                      </a>

                      <button
                        style={{ ...btn, padding: "8px 12px" }}
                        onClick={() => props.onRemove(r.id)}
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 10, color: "#9ca3af", fontSize: 12 }}>
        Tip: If you restart the backend, old report IDs won’t exist on the server anymore (local-only history).
      </div>
    </div>
  );
}