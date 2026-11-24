// Server-side proxy to the FastAPI backend, so the browser never needs the backend URL or CORS.
import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";
const ALLOWED = [/^ask$/, /^runs\/[a-f0-9]{12}$/, /^runs\/[a-f0-9]{12}\/approve$/];

async function forward(req: NextRequest, path: string[]): Promise<NextResponse> {
  const target = path.join("/");
  if (!ALLOWED.some((re) => re.test(target))) {
    return NextResponse.json({ detail: "not found" }, { status: 404 });
  }
  const init: RequestInit = { method: req.method, headers: { "content-type": "application/json" } };
  if (req.method === "POST") init.body = await req.text();
  try {
    const res = await fetch(`${BACKEND}/${target}`, init);
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (err) {
    return NextResponse.json({ detail: `backend unreachable: ${(err as Error).message}` }, { status: 502 });
  }
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
