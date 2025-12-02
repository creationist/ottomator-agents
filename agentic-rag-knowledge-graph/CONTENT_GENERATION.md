# Personalized Content Generation

This module generates personalized astrology content based on user birth data and current celestial transits.

## Overview

The content generation system produces three types of content:

| Content Type | Description | Personalized | Cache Duration |
|-------------|-------------|--------------|----------------|
| **Monthly General** | Overview of the month's major transits, retrogrades, and cosmic energies | No (same for all) | 30 days |
| **Monthly Personal** | 3-5 personalized highlights based on transits to user's natal chart | Yes | 30 days |
| **Moon Reflection** | 3 reflection questions based on current Moon + user's natal Moon | Yes | ~2.5 days |

## Architecture

```
content/
├── content_types.py      # ContentType enum + prompt templates
├── user_profile.py       # User birth chart calculations (pyswisseph)
├── transit_service.py    # Current celestial positions
├── context_assembler.py  # Combines user + transit + ontology data
├── generator.py          # LLM generation with caching
├── batch.py              # Batch processing for multiple users
└── prompts/
    ├── monthly_general.py
    ├── monthly_personal.py
    └── moon_reflection.py
```

## Setup

### 1. Install Dependencies

```bash
pip install pyswisseph
```

### 2. Apply Database Schema

```bash
docker exec -i agentic-rag-postgres psql -U postgres -d agentic_rag < sql/schema.sql
```

This creates three new tables:
- `user_profiles` - Stores birth data and computed chart
- `generated_content` - Caches generated content
- `batch_jobs` - Tracks batch generation jobs

### 3. Start the API

```bash
python -m agent.api
```

## API Endpoints

### User Profile Management

#### Create/Update User Profile

```bash
POST /users/{user_id}/profile
```

Creates a user profile with birth data and computes the birth chart.

**Request:**
```json
{
  "birth_data": {
    "birth_datetime": "1990-06-15T14:30:00+02:00",
    "latitude": 52.52,
    "longitude": 13.405,
    "location_name": "Berlin"
  }
}
```

**Response:**
```json
{
  "user_id": "user123",
  "sun_sign": "Zwillinge",
  "moon_sign": "Skorpion",
  "rising_sign": "Waage",
  "birth_datetime": "1990-06-15T14:30:00+02:00",
  "birth_location": "Berlin",
  "natal_positions": {
    "sonne": {"sign": "Zwillinge", "degree": 24.5, "house": 9},
    "mond": {"sign": "Skorpion", "degree": 12.3, "house": 2},
    ...
  },
  "chart_computed_at": "2024-12-01T12:00:00Z"
}
```

#### Get User Profile

```bash
GET /users/{user_id}/profile
```

### Content Generation

#### Generate Any Content Type

```bash
POST /content/generate
```

**Request:**
```json
{
  "content_type": "monthly_personal",
  "user_id": "user123",
  "year": 2024,
  "month": 12,
  "force_refresh": false
}
```

**Content Types:**
- `monthly_general` - No user_id required
- `monthly_personal` - Requires user_id
- `moon_reflection` - Requires user_id

**Response:**
```json
{
  "content_type": "monthly_personal",
  "content": "### 🌟 Highlight 1: Transformative Energien...",
  "user_id": "user123",
  "valid_from": "2024-12-01T00:00:00Z",
  "valid_until": "2025-01-01T00:00:00Z",
  "from_cache": false,
  "metadata": {
    "sun_sign": "Zwillinge",
    "transit_count": 5
  }
}
```

#### Get All Monthly Content for User

```bash
GET /content/{user_id}/monthly?year=2024&month=12
```

Returns both general and personal monthly content.

#### Get Moon Reflection Questions

```bash
GET /content/{user_id}/moon-reflection
```

### Transit Information

#### Current Planetary Positions

```bash
GET /transits/current
```

**Response:**
```json
{
  "timestamp": "2024-12-01T12:00:00Z",
  "positions": {
    "sonne": {"sign": "Schütze", "degree": 9.5, "retrograde": false},
    "mond": {"sign": "Widder", "degree": 15.2, "retrograde": false},
    ...
  },
  "moon_sign": "Widder",
  "moon_phase": "zunehmender_mond",
  "moon_phase_percent": 65.5,
  "retrograde_planets": ["merkur"],
  "active_aspects": [...]
}
```

#### Monthly Transit Events

```bash
GET /transits/monthly?year=2024&month=12
```

### Batch Processing

#### Generate Content for Multiple Users

```bash
POST /content/batch/generate
```

**Request:**
```json
{
  "content_type": "monthly_personal",
  "user_ids": ["user1", "user2"],  // null = all users
  "year": 2024,
  "month": 12
}
```

#### Generate All Monthly Content

```bash
POST /content/batch/monthly?year=2024&month=12
```

Generates both general monthly content (once) and personal monthly content for all users.

#### Check Batch Job Status

```bash
GET /content/batch/status/{job_id}
```

## Cron Jobs

Use the provided script for scheduled generation:

```bash
# Make executable
chmod +x scripts/batch_cron.sh

# Generate all monthly content
./scripts/batch_cron.sh monthly

# Generate moon reflections
./scripts/batch_cron.sh moon

# Generate everything
./scripts/batch_cron.sh all
```

### Recommended Cron Schedule

```cron
# Monthly content - 1st of each month at midnight
0 0 1 * * /path/to/scripts/batch_cron.sh monthly

# Moon reflections - every 12 hours (Moon changes sign ~every 2.5 days)
0 */12 * * * /path/to/scripts/batch_cron.sh moon
```

### Environment Variables

```bash
API_BASE_URL=http://localhost:8058  # API URL
LOG_FILE=/var/log/astro-batch.log   # Log file path
```

## Content Templates

### Monthly General

Describes the month's cosmic weather:
- Major transits and their timing
- Retrograde planets
- Moon phases (new/full moons)
- General recommendations

**Output:** ~500 words

### Monthly Personal

Personalized highlights based on:
- User's natal chart (Sun, Moon, Rising, all planets)
- Current transits aspecting natal planets
- Relevant moon phases for user's signs

**Output:** 3-5 highlights with specific dates and recommendations

### Moon Reflection

Reflection questions based on:
- Current Moon sign and phase
- User's natal Moon sign and house
- Interaction between transit and natal Moon

**Output:** 3 deep reflection questions with context

## Caching Strategy

| Content | Cache Duration | Invalidation |
|---------|---------------|--------------|
| Monthly General | Entire month | 1st of next month |
| Monthly Personal | Entire month | 1st of next month |
| Moon Reflection | ~60 hours | Moon sign change |
| Transit Data | 2 hours | Automatic |

The system automatically checks cache before generating. Use `force_refresh=true` to bypass.

## Database Schema

### user_profiles

```sql
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    birth_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    birth_latitude DOUBLE PRECISION NOT NULL,
    birth_longitude DOUBLE PRECISION NOT NULL,
    birth_location_name TEXT,
    sun_sign TEXT,
    moon_sign TEXT,
    rising_sign TEXT,
    natal_positions JSONB,
    natal_aspects JSONB,
    house_cusps JSONB,
    chart_computed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

### generated_content

```sql
CREATE TABLE generated_content (
    id UUID PRIMARY KEY,
    content_type TEXT NOT NULL,
    user_id TEXT,  -- NULL for general content
    content TEXT NOT NULL,
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
);
```

### batch_jobs

```sql
CREATE TABLE batch_jobs (
    id UUID PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,  -- pending, running, completed, failed
    total_users INTEGER,
    processed_users INTEGER,
    errors JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE
);
```

## Example Usage

### Python Client

```python
import httpx
from datetime import datetime

async def create_user_and_generate_content():
    async with httpx.AsyncClient() as client:
        # Create user profile
        profile = await client.post(
            "http://localhost:8058/users/user123/profile",
            json={
                "birth_data": {
                    "birth_datetime": "1990-06-15T14:30:00+02:00",
                    "latitude": 52.52,
                    "longitude": 13.405,
                    "location_name": "Berlin"
                }
            }
        )
        print(f"Created profile: {profile.json()['sun_sign']} Sun")
        
        # Get moon reflection
        reflection = await client.get(
            "http://localhost:8058/content/user123/moon-reflection"
        )
        print(f"Moon Reflection:\n{reflection.json()['content']}")
        
        # Get monthly content
        monthly = await client.get(
            "http://localhost:8058/content/user123/monthly"
        )
        print(f"Monthly Personal:\n{monthly.json()['personal']['content']}")
```

### curl Examples

```bash
# Create profile
curl -X POST http://localhost:8058/users/test/profile \
  -H "Content-Type: application/json" \
  -d '{"birth_data":{"birth_datetime":"1990-06-15T14:30:00+02:00","latitude":52.52,"longitude":13.405}}'

# Get current transits
curl http://localhost:8058/transits/current

# Generate monthly personal content
curl -X POST http://localhost:8058/content/generate \
  -H "Content-Type: application/json" \
  -d '{"content_type":"monthly_personal","user_id":"test"}'

# Get moon reflection
curl http://localhost:8058/content/test/moon-reflection

# Batch generate for all users
curl -X POST http://localhost:8058/content/batch/monthly
```

## Troubleshooting

### "pyswisseph not installed"

```bash
pip install pyswisseph
```

The system falls back to approximate calculations if pyswisseph is unavailable.

### "Profile not found"

Create a profile first with birth data before requesting personalized content.

### "Content generation failed"

- Check that the LLM client is configured (OpenAI API key)
- Verify database connection
- Check logs for specific errors

### Batch job stuck in "running"

Check `/content/batch/status/{job_id}` for error details. Jobs may fail for individual users while completing for others.


