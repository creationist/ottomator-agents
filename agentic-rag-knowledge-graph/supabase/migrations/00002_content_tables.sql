-- =============================================================================
-- Migration 00002: Content Generation Tables
-- Tables for personalized astrology content generation
-- =============================================================================

-- =============================================================================
-- User Astrological Profiles
-- Stores birth data and computed chart information
-- Links to auth.users via auth_user_id
-- =============================================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    auth_user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    -- Birth data (required for chart calculation)
    birth_datetime TIMESTAMP WITH TIME ZONE,
    birth_latitude DOUBLE PRECISION,
    birth_longitude DOUBLE PRECISION,
    birth_location_name TEXT,
    -- Computed chart data (cached after calculation)
    sun_sign TEXT,
    moon_sign TEXT,
    rising_sign TEXT,
    natal_positions JSONB DEFAULT '{}',
    natal_aspects JSONB DEFAULT '[]',
    house_cusps JSONB DEFAULT '{}',
    chart_computed_at TIMESTAMP WITH TIME ZONE,
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles (user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_auth_user_id ON user_profiles (auth_user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_sun_sign ON user_profiles (sun_sign);

CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Generated Content Cache
-- Stores pre-generated personalized content with validity periods
-- =============================================================================
CREATE TABLE IF NOT EXISTS generated_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_type TEXT NOT NULL,
    user_id TEXT,  -- NULL for general content (same for all users)
    content TEXT NOT NULL,
    -- Validity period for caching
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_generated_content_lookup 
    ON generated_content (content_type, user_id, valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_generated_content_type ON generated_content (content_type);
CREATE INDEX IF NOT EXISTS idx_generated_content_user ON generated_content (user_id) WHERE user_id IS NOT NULL;

-- =============================================================================
-- Batch Processing Jobs
-- Tracks batch content generation jobs for monitoring
-- =============================================================================
CREATE TABLE IF NOT EXISTS batch_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_users INTEGER,
    processed_users INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs (status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs (created_at DESC);

-- =============================================================================
-- Helper function to get valid cached content
-- =============================================================================
CREATE OR REPLACE FUNCTION get_cached_content(
    p_content_type TEXT,
    p_user_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    valid_until TIMESTAMP WITH TIME ZONE,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        gc.id,
        gc.content,
        gc.valid_until,
        gc.metadata
    FROM generated_content gc
    WHERE gc.content_type = p_content_type
      AND (gc.user_id = p_user_id OR (gc.user_id IS NULL AND p_user_id IS NULL))
      AND CURRENT_TIMESTAMP BETWEEN gc.valid_from AND gc.valid_until
    ORDER BY gc.created_at DESC
    LIMIT 1;
END;
$$;

-- =============================================================================
-- Trigger to auto-create profile stub on user signup
-- =============================================================================
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (user_id, auth_user_id)
    VALUES (NEW.id::text, NEW.id)
    ON CONFLICT (user_id) DO UPDATE SET auth_user_id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger on auth.users (only works when auth schema exists)
DO $$
BEGIN
    -- Drop existing trigger if it exists
    DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
    
    -- Create the trigger
    CREATE TRIGGER on_auth_user_created
        AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION handle_new_user();
EXCEPTION
    WHEN undefined_table THEN
        -- auth.users doesn't exist yet (will be created by Supabase)
        RAISE NOTICE 'auth.users table not found, trigger will be created when auth is ready';
END;
$$;


