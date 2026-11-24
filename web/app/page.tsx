"use client";

import { useState } from "react";
import type { Run, TraceEvent } from "@/lib/types";

const box: React.CSSProperties = { background: "#fff", border: "1px solid #e2e2dd", borderRadius: 8, padding: 16, marginTop: 16 };

export default function Page() {
  const [q, setQ] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function call(path: string, body: unknown): Promise<void> {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`/api/${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? res.statusText);
      setRun(data as Run);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 920, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 22 }}>Injection device safety agent</h1>
      <p style={{ color: "#5c5c57", marginTop: 0 }}>
        Answers from FDA recall, 510(k) and MAUDE records. Every claim cites a retrieved record; briefs need human approval.
      </p>
      <form onSubmit={(e) => { e.preventDefault(); if (q.trim()) void call("ask", { question: q }); }} style={{ display: "flex", gap: 8 }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. Why were Jiangsu Shenli syringes recalled?"
          style={{ flex: 1, padding: 10, fontSize: 15, border: "1px solid #ccc", borderRadius: 6 }} />
        <button disabled={busy} style={{ padding: "10px 18px", fontSize: 15 }}>{busy ? "Working..." : "Ask"}</button>
      </form>
      {err && <div style={{ ...box, borderColor: "#d9a3a3", color: "#8a1f1f" }}>{err}</div>}
      {run && <RunView run={run} busy={busy} onDecide={(approved) => call(`runs/${run.id}/approve`, { approved, reviewer: "ui-user" })} />}
    </main>
  );
}

function RunView({ run, busy, onDecide }: { run: Run; busy: boolean; onDecide: (approved: boolean) => void }) {
  const a = run.answer;
  return (
    <>
      <section style={box}>
        <div style={{ fontSize: 12, color: "#777" }}>status: {run.status} · tokens: {run.usage.prompt + run.usage.completion}</div>
        {a ? (
          <>
            <p style={{ fontSize: 16 }}>{a.answer}</p>
            {a.insufficient_evidence && <p style={{ color: "#8a5a00" }}>Insufficient cited evidence.</p>}
            <ol>
              {a.claims.map((c, i) => (
                <li key={i} style={{ marginBottom: 6 }}>
                  {c.text} {c.citations.map((id) => <code key={id} style={{ background: "#eef", marginLeft: 4, padding: "1px 4px" }}>{id}</code>)}
                </li>
              ))}
            </ol>
            {a.dropped_claims.length > 0 && (
              <p style={{ fontSize: 13, color: "#8a1f1f" }}>
                {a.dropped_claims.length} claim(s) removed for citing records that were not retrieved.
              </p>
            )}
          </>
        ) : <p>No answer.</p>}
      </section>
      {run.pending_action && (
        <section style={{ ...box, borderColor: "#e0c36b" }}>
          <strong>Draft safety brief: {run.pending_action.args.title}</strong>
          <p>{run.pending_action.args.summary}</p>
          <p style={{ fontSize: 13 }}>Citations: {run.pending_action.args.citations.join(", ")}</p>
          {run.pending_action.invalid_citations.length > 0 && (
            <p style={{ fontSize: 13, color: "#8a1f1f" }}>Not retrieved in this run: {run.pending_action.invalid_citations.join(", ")}</p>
          )}
          {run.pending_action.approved === null ? (
            <div style={{ display: "flex", gap: 8 }}>
              <button disabled={busy} onClick={() => onDecide(true)}>Approve</button>
              <button disabled={busy} onClick={() => onDecide(false)}>Reject</button>
            </div>
          ) : <p>Decision: {run.pending_action.approved ? "approved" : "rejected"} by {run.pending_action.reviewer}</p>}
        </section>
      )}
      <section style={box}>
        <strong>Trace</strong>
        <ul style={{ fontSize: 13, fontFamily: "ui-monospace, monospace" }}>
          {run.trace.map((t, i) => <li key={i}>{describe(t)}</li>)}
        </ul>
      </section>
    </>
  );
}

function describe(t: TraceEvent): string {
  switch (t.kind) {
    case "tool":
      return `tool ${String(t.name)} ${JSON.stringify(t.args)} -> ${t.error ? `error: ${String(t.error)}` : `${(t.result_ids as string[] | null)?.length ?? 0} records`} (${String(t.ms)} ms)`;
    case "model":
      return `model step ${String(t.step)}: ${(t.tool_calls as string[]).join(", ") || "final"}`;
    case "guard_redaction":
      return `guard redacted instruction-like text in ${String(t.doc)}`;
    case "validation":
      return `citation check: ${String(t.dropped_claims)} claim(s) dropped`;
    default:
      return `${t.kind}: ${JSON.stringify(t)}`;
  }
}
