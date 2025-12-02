-- =============================================================================
-- Migration 00003: Row Level Security Policies
-- Ensures users can only access their own data
-- =============================================================================

-- =============================================================================
-- Enable RLS on all user-specific tables
-- =============================================================================
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Documents and chunks are public (knowledge base)
-- No RLS needed - anyone can read

-- =============================================================================
-- User Profiles Policies
-- =============================================================================

-- Users can view their own profile
CREATE POLICY "Users can view own profile"
ON user_profiles FOR SELECT
USING (
    auth.uid()::text = user_id 
    OR auth.uid() = auth_user_id
);

-- Users can insert their own profile
CREATE POLICY "Users can create own profile"
ON user_profiles FOR INSERT
WITH CHECK (
    auth.uid()::text = user_id 
    OR auth.uid() = auth_user_id
);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
ON user_profiles FOR UPDATE
USING (
    auth.uid()::text = user_id 
    OR auth.uid() = auth_user_id
);

-- Service role can do anything (for batch processing)
CREATE POLICY "Service role full access to profiles"
ON user_profiles FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- =============================================================================
-- Generated Content Policies
-- =============================================================================

-- Users can view their own content AND general content (user_id IS NULL)
CREATE POLICY "Users can view own and general content"
ON generated_content FOR SELECT
USING (
    user_id IS NULL  -- General content visible to all
    OR auth.uid()::text = user_id
);

-- Service role can insert/update any content (for batch generation)
CREATE POLICY "Service role full access to content"
ON generated_content FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- =============================================================================
-- Sessions Policies
-- =============================================================================

-- Users can view their own sessions
CREATE POLICY "Users can view own sessions"
ON sessions FOR SELECT
USING (auth.uid()::text = user_id);

-- Users can create sessions
CREATE POLICY "Users can create sessions"
ON sessions FOR INSERT
WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

-- Users can update their own sessions
CREATE POLICY "Users can update own sessions"
ON sessions FOR UPDATE
USING (auth.uid()::text = user_id);

-- Service role full access
CREATE POLICY "Service role full access to sessions"
ON sessions FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- =============================================================================
-- Messages Policies
-- =============================================================================

-- Users can view messages in their sessions
CREATE POLICY "Users can view own messages"
ON messages FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM sessions s 
        WHERE s.id = messages.session_id 
        AND s.user_id = auth.uid()::text
    )
);

-- Users can create messages in their sessions
CREATE POLICY "Users can create messages"
ON messages FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM sessions s 
        WHERE s.id = messages.session_id 
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    )
);

-- Service role full access
CREATE POLICY "Service role full access to messages"
ON messages FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- =============================================================================
-- Batch Jobs Policies
-- Only service role can access batch jobs
-- =============================================================================
ALTER TABLE batch_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role only for batch jobs"
ON batch_jobs FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- =============================================================================
-- Public access policies for knowledge base (documents, chunks)
-- =============================================================================

-- Documents are readable by anyone (authenticated or not for public knowledge)
CREATE POLICY "Anyone can read documents"
ON documents FOR SELECT
USING (true);

-- Only service role can modify documents
CREATE POLICY "Service role can modify documents"
ON documents FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- Chunks are readable by anyone
CREATE POLICY "Anyone can read chunks"
ON chunks FOR SELECT
USING (true);

-- Only service role can modify chunks
CREATE POLICY "Service role can modify chunks"
ON chunks FOR ALL
USING (auth.jwt() ->> 'role' = 'service_role');

-- Enable RLS on documents and chunks
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;


