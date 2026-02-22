const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type UploadResp = {
  report_id: string;
  filename?: string;
  status?: string;
};

export type ProcessResp = {
  status: string;
  pdf_url?: string;
  pages_used?: number;
  page_count_in_pdf?: number;
  model?: string;
  extract_method?: string;
  extracted_chars?: number;
  debug_error?: string;
};

async function readError(res: Response) {
  const txt = await res.text();
  try {
    const j = JSON.parse(txt);
    return j?.detail ?? txt;
  } catch {
    return txt || `HTTP ${res.status}`;
  }
}

export async function uploadReport(file: File): Promise<UploadResp> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function processReport(reportId: string): Promise<ProcessResp> {
  const res = await fetch(`${API_BASE}/process/${reportId}`, {
    method: "POST",
  });

  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function pdfLink(reportId: string) {
  return `${API_BASE}/pdf/${reportId}`;
}