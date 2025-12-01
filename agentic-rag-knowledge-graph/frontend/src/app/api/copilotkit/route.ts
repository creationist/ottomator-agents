import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// Use the empty adapter since we're only using one agent
const serviceAdapter = new ExperimentalEmptyAdapter();

// Create the CopilotRuntime instance connecting to the AG-UI Python backend
const runtime = new CopilotRuntime({
  agents: {
    // RAG agent with knowledge graph support on port 8001
    "rag_agent": new HttpAgent({ url: "http://localhost:8001/" }),
  }
});

// Build a Next.js API route that handles the CopilotKit runtime requests
export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};



