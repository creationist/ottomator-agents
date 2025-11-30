"use client";

import { useCoAgent, useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState, useEffect } from "react";

// RAG Agent State Types - must match the backend RAGState
type RAGState = {
  retrieved_chunks: RetrievedChunk[];
  graph_results: GraphResult[];
  current_query: SearchQuery | null;
  search_history: SearchQuery[];
  selected_chunk_id: string | null;
  total_chunks_in_kb: number;
  knowledge_base_status: string;
  graph_status: string;
};

type RetrievedChunk = {
  chunk_id: string;
  document_id: string;
  content: string;
  similarity: number;
  metadata: Record<string, unknown>;
  document_title: string;
  document_source: string;
  highlight?: string;
};

type GraphResult = {
  fact: string;
  uuid: string;
  valid_at: string | null;
  invalid_at: string | null;
  source_node_uuid: string | null;
};

type SearchQuery = {
  query: string;
  timestamp: string;
  match_count: number;
  search_type: string;
};

export default function RAGKnowledgeGraphPage() {
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  const [searchFilter, setSearchFilter] = useState("");
  const [activeTab, setActiveTab] = useState<"chunks" | "graph">("chunks");
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering CopilotKit after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  // Connect to the RAG agent's shared state
  const { state, setState } = useCoAgent<RAGState>({
    name: "rag_agent",
    initialState: {
      retrieved_chunks: [],
      graph_results: [],
      current_query: null,
      search_history: [],
      selected_chunk_id: null,
      total_chunks_in_kb: 0,
      knowledge_base_status: "initializing",
      graph_status: "initializing",
    },
  });

  // Frontend action for highlighting specific chunks
  useCopilotAction({
    name: "highlightChunk",
    description: "Highlight a specific chunk in the UI",
    parameters: [{
      name: "chunkId",
      type: "string",
      description: "The ID of the chunk to highlight",
      required: true,
    }],
    handler({ chunkId }) {
      setState({
        ...state,
        selected_chunk_id: chunkId,
      });
      setExpandedChunks(prev => new Set(prev).add(chunkId));
      setActiveTab("chunks");
    },
  });

  // Filter chunks based on search
  const filteredChunks = (state?.retrieved_chunks || []).filter(chunk =>
    searchFilter === "" ||
    chunk?.content?.toLowerCase().includes(searchFilter.toLowerCase()) ||
    chunk?.document_title?.toLowerCase().includes(searchFilter.toLowerCase())
  );

  // Filter graph results based on search
  const filteredGraphResults = (state?.graph_results || []).filter(result =>
    searchFilter === "" ||
    result?.fact?.toLowerCase().includes(searchFilter.toLowerCase())
  );

  const hasChunks = (state?.retrieved_chunks?.length || 0) > 0;
  const hasGraphResults = (state?.graph_results?.length || 0) > 0;

  return (
    <main
      style={{ "--copilot-kit-primary-color": "#10b981" } as CopilotKitCSSProperties}
    >
      <div className="flex h-screen">
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="bg-gradient-to-r from-emerald-900 to-teal-900 p-6 shadow-lg">
            <h1 className="text-2xl font-bold text-white mb-2">
              Knowledge Graph Explorer
            </h1>
            <div className="flex items-center justify-between">
              <p className="text-emerald-200 text-sm">
                {hasChunks || hasGraphResults
                  ? `${state?.retrieved_chunks?.length || 0} chunks, ${state?.graph_results?.length || 0} graph facts`
                  : "No results yet - ask the assistant to search"}
              </p>
              <div className="flex gap-2 text-xs">
                <span className={`px-3 py-1 rounded-full ${
                  state?.knowledge_base_status === "ready" 
                    ? "bg-emerald-500/20 text-emerald-300" 
                    : "bg-yellow-500/20 text-yellow-300"
                }`}>
                  KB: {state?.knowledge_base_status || "initializing"}
                </span>
                <span className={`px-3 py-1 rounded-full ${
                  state?.graph_status === "ready"
                    ? "bg-emerald-500/20 text-emerald-300"
                    : "bg-yellow-500/20 text-yellow-300"
                }`}>
                  Graph: {state?.graph_status || "initializing"}
                </span>
              </div>
            </div>

            {/* Current Query Display */}
            {state?.current_query && (
              <div className="mt-4 bg-black/20 p-3 rounded-lg">
                <p className="text-xs uppercase tracking-wide text-emerald-300/70 mb-1">
                  Current Query
                </p>
                <p className="font-medium text-white">{state.current_query.query}</p>
                <p className="text-xs text-emerald-200/70 mt-1">
                  {state.current_query.search_type} search
                  {state.current_query.match_count > 0 && ` • ${state.current_query.match_count} results requested`}
                </p>
              </div>
            )}
          </header>

          {/* Tab Navigation */}
          {(hasChunks || hasGraphResults) && (
            <div className="bg-[var(--surface)] border-b border-[var(--border)] px-4">
              <div className="flex gap-1">
                <button
                  onClick={() => setActiveTab("chunks")}
                  className={`px-4 py-3 text-sm font-medium transition-colors relative ${
                    activeTab === "chunks"
                      ? "text-emerald-400"
                      : "text-gray-400 hover:text-gray-200"
                  }`}
                >
                  Vector Results ({state?.retrieved_chunks?.length || 0})
                  {activeTab === "chunks" && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400" />
                  )}
                </button>
                <button
                  onClick={() => setActiveTab("graph")}
                  className={`px-4 py-3 text-sm font-medium transition-colors relative ${
                    activeTab === "graph"
                      ? "text-emerald-400"
                      : "text-gray-400 hover:text-gray-200"
                  }`}
                >
                  Graph Facts ({state?.graph_results?.length || 0})
                  {activeTab === "graph" && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400" />
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Search Filter */}
          {(hasChunks || hasGraphResults) && (
            <div className="p-4 bg-[var(--surface)] border-b border-[var(--border)]">
              <input
                type="text"
                placeholder="Filter results..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="w-full px-4 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg 
                         text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
          )}

          {/* Results Display */}
          <div className="flex-1 overflow-y-auto p-4 bg-[var(--background)]">
            {!hasChunks && !hasGraphResults ? (
              <EmptyState />
            ) : activeTab === "chunks" ? (
              <ChunksView
                chunks={filteredChunks}
                selectedChunkId={state?.selected_chunk_id}
                expandedChunks={expandedChunks}
                setExpandedChunks={setExpandedChunks}
              />
            ) : (
              <GraphView results={filteredGraphResults} />
            )}
          </div>

          {/* Search History Footer */}
          {state?.search_history && state.search_history.length > 0 && (
            <div className="p-4 bg-[var(--surface)] border-t border-[var(--border)]">
              <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">
                Recent Searches
              </p>
              <div className="flex flex-wrap gap-2">
                {state.search_history.slice(-5).map((query, idx) => (
                  <span
                    key={idx}
                    className="text-xs bg-[var(--background)] text-gray-300 px-2 py-1 rounded-full"
                    title={new Date(query.timestamp).toLocaleString()}
                  >
                    {query.query.length > 30 ? query.query.slice(0, 30) + "..." : query.query}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {mounted && (
          <CopilotSidebar
            clickOutsideToClose={false}
            defaultOpen={true}
            labels={{
              title: "RAG + Graph Assistant",
              initial: `Welcome to the RAG Knowledge Graph Explorer!

I can search both a vector database and a knowledge graph to answer your questions about big tech companies and AI.

Try asking:
- "What information do you have about OpenAI?"
- "How does Microsoft relate to OpenAI?"
- "Search for funding information about AI companies"
- "What's the timeline of events for Anthropic?"

KB Status: ${state?.knowledge_base_status || "initializing"}
Graph Status: ${state?.graph_status || "initializing"}`,
            }}
          />
        )}
      </div>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-gray-500">
      <svg
        className="w-16 h-16 mb-4 opacity-25"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      <p className="text-lg font-medium mb-2">No results yet</p>
      <p className="text-sm text-center max-w-md">
        Ask the assistant to search for information about tech companies, AI initiatives,
        or explore relationships in the knowledge graph.
      </p>
    </div>
  );
}

function ChunksView({
  chunks,
  selectedChunkId,
  expandedChunks,
  setExpandedChunks,
}: {
  chunks: RetrievedChunk[];
  selectedChunkId: string | null;
  expandedChunks: Set<string>;
  setExpandedChunks: React.Dispatch<React.SetStateAction<Set<string>>>;
}) {
  if (chunks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <p className="text-lg font-medium mb-2">No matching chunks</p>
        <p className="text-sm">Try adjusting your filter criteria</p>
      </div>
    );
  }

  const toggleExpansion = (chunkId: string) => {
    setExpandedChunks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(chunkId)) {
        newSet.delete(chunkId);
      } else {
        newSet.add(chunkId);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-4">
      {chunks.map((chunk, index) => {
        const isExpanded = expandedChunks.has(chunk.chunk_id);
        const isSelected = selectedChunkId === chunk.chunk_id;
        const relevanceClass =
          chunk.similarity > 0.8
            ? "bg-emerald-500/20 text-emerald-300"
            : chunk.similarity > 0.6
            ? "bg-yellow-500/20 text-yellow-300"
            : "bg-gray-500/20 text-gray-300";

        return (
          <div
            key={chunk.chunk_id}
            className={`bg-[var(--surface)] rounded-lg overflow-hidden transition-all duration-200
              ${isSelected ? "ring-2 ring-emerald-500 ring-offset-2 ring-offset-[var(--background)]" : ""}
              ${isExpanded ? "shadow-lg" : "hover:shadow-lg"}`}
          >
            <div
              className="p-4 cursor-pointer select-none"
              onClick={() => toggleExpansion(chunk.chunk_id)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-gray-500">#{index + 1}</span>
                    <span className={`text-xs px-2 py-1 rounded-full ${relevanceClass}`}>
                      {(chunk.similarity * 100).toFixed(1)}% match
                    </span>
                    {isSelected && (
                      <span className="text-xs px-2 py-1 rounded-full bg-emerald-500 text-white">
                        Selected
                      </span>
                    )}
                  </div>
                  <h3 className="font-medium text-white">{chunk.document_title}</h3>
                  <p className="text-xs text-gray-500 mt-1">{chunk.document_source}</p>
                </div>
                <ChevronIcon expanded={isExpanded} />
              </div>

              <p className="text-sm text-gray-300 line-clamp-2">
                {chunk.highlight || chunk.content.substring(0, 150)}
                {chunk.content.length > 150 && !isExpanded && "..."}
              </p>
            </div>

            {isExpanded && (
              <div className="px-4 pb-4 border-t border-[var(--border)]">
                <div className="mt-4">
                  <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                    Full Content
                  </h4>
                  <div className="bg-[var(--background)] p-3 rounded-lg">
                    <p className="text-sm text-gray-300 whitespace-pre-wrap">{chunk.content}</p>
                  </div>
                </div>

                {Object.keys(chunk.metadata).length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                      Metadata
                    </h4>
                    <div className="bg-[var(--background)] p-3 rounded-lg">
                      {Object.entries(chunk.metadata).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-xs mb-1">
                          <span className="font-medium text-gray-400">{key}:</span>
                          <span className="text-gray-300">
                            {typeof value === "object" ? JSON.stringify(value) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-4 flex gap-4 text-xs text-gray-500">
                  <span>Chunk ID: {chunk.chunk_id}</span>
                  <span>Document ID: {chunk.document_id}</span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function GraphView({ results }: { results: GraphResult[] }) {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <p className="text-lg font-medium mb-2">No graph results</p>
        <p className="text-sm">Try searching for entity relationships or facts</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {results.map((result, index) => (
        <div
          key={result.uuid}
          className="bg-[var(--surface)] rounded-lg p-4 hover:shadow-lg transition-shadow"
        >
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 bg-teal-500/20 text-teal-300 rounded-full flex items-center justify-center text-sm font-medium">
              {index + 1}
            </div>
            <div className="flex-1">
              <p className="text-white font-medium mb-2">{result.fact}</p>
              <div className="flex flex-wrap gap-2 text-xs">
                {result.valid_at && (
                  <span className="bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded">
                    Valid: {result.valid_at}
                  </span>
                )}
                {result.invalid_at && (
                  <span className="bg-red-500/20 text-red-300 px-2 py-1 rounded">
                    Invalid: {result.invalid_at}
                  </span>
                )}
                <span className="bg-gray-500/20 text-gray-400 px-2 py-1 rounded">
                  ID: {result.uuid.slice(0, 8)}...
                </span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={`w-5 h-5 text-gray-400 transform transition-transform ${
        expanded ? "rotate-180" : ""
      }`}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M19 9l-7 7-7-7"
      />
    </svg>
  );
}

