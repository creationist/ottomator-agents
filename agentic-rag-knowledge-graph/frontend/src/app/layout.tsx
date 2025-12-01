import type { Metadata } from "next";

import { CopilotKit } from "@copilotkit/react-core";
import "./globals.css";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "RAG Knowledge Graph Explorer",
  description: "AI-powered RAG agent with vector search and knowledge graph capabilities",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <CopilotKit runtimeUrl="/api/copilotkit" agent="rag_agent">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}



