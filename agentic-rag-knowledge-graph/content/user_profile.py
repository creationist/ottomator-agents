"""
User astrological profile service.

Manages user birth data and computed chart information.
Uses Swiss Ephemeris (pyswisseph) for astronomical calculations.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import json

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
except ImportError:
    SWISSEPH_AVAILABLE = False
    logging.warning("pyswisseph not installed. Chart calculations will use fallback.")

logger = logging.getLogger(__name__)

USE_SUPABASE = os.getenv("USE_SUPABASE", "true").lower() == "true"

# =============================================================================
# Constants
# =============================================================================

ZODIAC_SIGNS = [
    "Widder", "Stier", "Zwillinge", "Krebs", 
    "Löwe", "Jungfrau", "Waage", "Skorpion",
    "Schütze", "Steinbock", "Wassermann", "Fische"
]

ZODIAC_SIGNS_EN = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "sonne": swe.SUN if SWISSEPH_AVAILABLE else 0,
    "mond": swe.MOON if SWISSEPH_AVAILABLE else 1,
    "merkur": swe.MERCURY if SWISSEPH_AVAILABLE else 2,
    "venus": swe.VENUS if SWISSEPH_AVAILABLE else 3,
    "mars": swe.MARS if SWISSEPH_AVAILABLE else 4,
    "jupiter": swe.JUPITER if SWISSEPH_AVAILABLE else 5,
    "saturn": swe.SATURN if SWISSEPH_AVAILABLE else 6,
    "uranus": swe.URANUS if SWISSEPH_AVAILABLE else 7,
    "neptun": swe.NEPTUNE if SWISSEPH_AVAILABLE else 8,
    "pluto": swe.PLUTO if SWISSEPH_AVAILABLE else 9,
    "chiron": swe.CHIRON if SWISSEPH_AVAILABLE else 15,
}

ASPECT_TYPES = {
    "konjunktion": 0,
    "sextil": 60,
    "quadrat": 90,
    "trigon": 120,
    "opposition": 180,
}

# Orbs for aspects (degrees)
ASPECT_ORBS = {
    "konjunktion": 8,
    "sextil": 6,
    "quadrat": 8,
    "trigon": 8,
    "opposition": 8,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlanetPosition:
    """Position of a planet in the chart."""
    planet: str
    longitude: float  # 0-360 degrees
    sign: str
    degree_in_sign: float  # 0-30 degrees
    house: Optional[int] = None
    retrograde: bool = False


@dataclass
class Aspect:
    """Aspect between two planets."""
    planet1: str
    planet2: str
    aspect_type: str
    orb: float  # How exact the aspect is
    applying: bool = False  # Is the aspect getting closer?


@dataclass
class UserAstroProfile:
    """Complete astrological profile for a user."""
    user_id: str
    
    # Birth data
    birth_datetime: datetime
    birth_latitude: float
    birth_longitude: float
    birth_location_name: Optional[str] = None
    
    # Core signs
    sun_sign: str = ""
    moon_sign: str = ""
    rising_sign: str = ""
    
    # Detailed positions
    natal_positions: Dict[str, PlanetPosition] = field(default_factory=dict)
    natal_aspects: List[Aspect] = field(default_factory=list)
    house_cusps: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    chart_computed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "birth_datetime": self.birth_datetime.isoformat(),
            "birth_latitude": self.birth_latitude,
            "birth_longitude": self.birth_longitude,
            "birth_location_name": self.birth_location_name,
            "sun_sign": self.sun_sign,
            "moon_sign": self.moon_sign,
            "rising_sign": self.rising_sign,
            "natal_positions": {
                k: {
                    "planet": v.planet,
                    "longitude": v.longitude,
                    "sign": v.sign,
                    "degree_in_sign": v.degree_in_sign,
                    "house": v.house,
                    "retrograde": v.retrograde
                }
                for k, v in self.natal_positions.items()
            },
            "natal_aspects": [
                {
                    "planet1": a.planet1,
                    "planet2": a.planet2,
                    "aspect_type": a.aspect_type,
                    "orb": a.orb,
                    "applying": a.applying
                }
                for a in self.natal_aspects
            ],
            "house_cusps": self.house_cusps,
            "chart_computed_at": self.chart_computed_at.isoformat() if self.chart_computed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserAstroProfile":
        """Create from dictionary."""
        birth_dt = data["birth_datetime"]
        if isinstance(birth_dt, str):
            birth_dt = datetime.fromisoformat(birth_dt.replace('Z', '+00:00'))
        
        profile = cls(
            user_id=data["user_id"],
            birth_datetime=birth_dt,
            birth_latitude=data["birth_latitude"],
            birth_longitude=data["birth_longitude"],
            birth_location_name=data.get("birth_location_name"),
            sun_sign=data.get("sun_sign", ""),
            moon_sign=data.get("moon_sign", ""),
            rising_sign=data.get("rising_sign", ""),
        )
        
        # Parse natal positions
        if data.get("natal_positions"):
            for k, v in data["natal_positions"].items():
                profile.natal_positions[k] = PlanetPosition(
                    planet=v["planet"],
                    longitude=v["longitude"],
                    sign=v["sign"],
                    degree_in_sign=v["degree_in_sign"],
                    house=v.get("house"),
                    retrograde=v.get("retrograde", False)
                )
        
        # Parse aspects
        if data.get("natal_aspects"):
            for a in data["natal_aspects"]:
                profile.natal_aspects.append(Aspect(
                    planet1=a["planet1"],
                    planet2=a["planet2"],
                    aspect_type=a["aspect_type"],
                    orb=a["orb"],
                    applying=a.get("applying", False)
                ))
        
        profile.house_cusps = data.get("house_cusps", {})
        
        if data.get("chart_computed_at"):
            computed_at = data["chart_computed_at"]
            if isinstance(computed_at, str):
                profile.chart_computed_at = datetime.fromisoformat(computed_at.replace('Z', '+00:00'))
            else:
                profile.chart_computed_at = computed_at
        
        return profile


# =============================================================================
# Chart Calculator
# =============================================================================

class ChartCalculator:
    """Calculates astrological charts using Swiss Ephemeris."""
    
    # Default flags for Swiss Ephemeris calculations
    # Will try ephemeris files first, fall back to Moshier if not available
    CALC_FLAGS = 0 if SWISSEPH_AVAILABLE else 0
    
    def __init__(self, ephe_path: Optional[str] = None):
        if SWISSEPH_AVAILABLE:
            # Set ephemeris path - look for files in ephe/ directory relative to project
            if ephe_path is None:
                import os
                # Try to find ephe directory relative to this file
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ephe_path = os.path.join(base_dir, "ephe")
            
            if os.path.isdir(ephe_path):
                swe.set_ephe_path(ephe_path)
                logger.info(f"Using Swiss Ephemeris files from: {ephe_path}")
            else:
                # Fall back to Moshier if no ephemeris directory
                swe.set_ephe_path("")
                logger.warning(f"Ephemeris directory not found at {ephe_path}, using Moshier mode")
    
    def _datetime_to_julian(self, dt: datetime) -> float:
        """Convert datetime to Julian Day."""
        if not SWISSEPH_AVAILABLE:
            # Rough approximation for fallback
            return 2451545.0 + (dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400
        
        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        jd = swe.julday(dt.year, dt.month, dt.day, hour)
        return jd
    
    def _longitude_to_sign(self, longitude: float) -> Tuple[str, float]:
        """Convert ecliptic longitude to zodiac sign and degree."""
        sign_index = int(longitude / 30)
        degree_in_sign = longitude % 30
        return ZODIAC_SIGNS[sign_index], degree_in_sign
    
    def calculate_planet_position(
        self, 
        planet_id: int, 
        julian_day: float
    ) -> Tuple[float, bool]:
        """Calculate planet position and retrograde status."""
        if not SWISSEPH_AVAILABLE:
            # Fallback: return rough estimates
            return (planet_id * 30.0) % 360, False
        
        try:
            # Try with default flags (uses ephemeris files if available)
            result = swe.calc_ut(julian_day, planet_id, self.CALC_FLAGS)
            longitude = result[0][0]
            # Check retrograde (negative speed)
            speed = result[0][3]
            retrograde = speed < 0
            return longitude, retrograde
        except Exception as e:
            # If ephemeris files not found, try Moshier for main planets
            if "not found" in str(e) and planet_id <= swe.PLUTO:
                try:
                    result = swe.calc_ut(julian_day, planet_id, swe.FLG_MOSEPH)
                    longitude = result[0][0]
                    speed = result[0][3]
                    retrograde = speed < 0
                    return longitude, retrograde
                except Exception:
                    pass
            
            # Chiron and other asteroids require ephemeris files
            logger.debug(f"Could not calculate position for planet_id {planet_id}: {e}")
            return None, False
    
    def calculate_houses(
        self,
        julian_day: float,
        latitude: float,
        longitude: float,
        house_system: str = 'P'  # Placidus
    ) -> Tuple[List[float], float]:
        """Calculate house cusps and ascendant."""
        if not SWISSEPH_AVAILABLE:
            # Fallback: equal houses from rough ascendant
            asc = (longitude + 90) % 360
            cusps = [(asc + i * 30) % 360 for i in range(12)]
            return cusps, asc
        
        cusps, ascmc = swe.houses(julian_day, latitude, longitude, house_system.encode())
        ascendant = ascmc[0]
        return list(cusps[1:13]), ascendant  # cusps[0] is unused
    
    def calculate_aspects(
        self,
        positions: Dict[str, PlanetPosition]
    ) -> List[Aspect]:
        """Calculate aspects between planets."""
        aspects = []
        planets = list(positions.keys())
        
        for i, p1 in enumerate(planets):
            for p2 in planets[i+1:]:
                pos1 = positions[p1].longitude
                pos2 = positions[p2].longitude
                
                # Calculate shortest angular distance
                diff = abs(pos1 - pos2)
                if diff > 180:
                    diff = 360 - diff
                
                # Check each aspect type
                for aspect_name, aspect_angle in ASPECT_TYPES.items():
                    orb = abs(diff - aspect_angle)
                    max_orb = ASPECT_ORBS[aspect_name]
                    
                    if orb <= max_orb:
                        aspects.append(Aspect(
                            planet1=p1,
                            planet2=p2,
                            aspect_type=aspect_name,
                            orb=round(orb, 2),
                            applying=False  # Would need speed calc for accuracy
                        ))
                        break
        
        return aspects
    
    def get_house_for_position(
        self,
        longitude: float,
        house_cusps: List[float]
    ) -> int:
        """Determine which house a position falls in."""
        for i in range(12):
            cusp = house_cusps[i]
            next_cusp = house_cusps[(i + 1) % 12]
            
            # Handle wrap-around at 0/360
            if next_cusp < cusp:  # Crosses 0 degrees
                if longitude >= cusp or longitude < next_cusp:
                    return i + 1
            else:
                if cusp <= longitude < next_cusp:
                    return i + 1
        
        return 1  # Default
    
    def compute_full_chart(
        self,
        birth_datetime: datetime,
        latitude: float,
        longitude: float
    ) -> Dict[str, Any]:
        """Compute full birth chart."""
        jd = self._datetime_to_julian(birth_datetime)
        
        # Calculate house cusps and ascendant
        house_cusps, ascendant = self.calculate_houses(jd, latitude, longitude)
        asc_sign, asc_degree = self._longitude_to_sign(ascendant)
        
        # Calculate planetary positions
        positions = {}
        for planet_name, planet_id in PLANETS.items():
            lon, retrograde = self.calculate_planet_position(planet_id, jd)
            
            # Skip planets that couldn't be calculated (e.g., Chiron without ephemeris files)
            if lon is None:
                continue
            
            sign, degree = self._longitude_to_sign(lon)
            house = self.get_house_for_position(lon, house_cusps)
            
            positions[planet_name] = PlanetPosition(
                planet=planet_name,
                longitude=lon,
                sign=sign,
                degree_in_sign=round(degree, 2),
                house=house,
                retrograde=retrograde
            )
        
        # Calculate aspects
        aspects = self.calculate_aspects(positions)
        
        # Build house cusps dict
        house_cusps_dict = {}
        for i, cusp in enumerate(house_cusps):
            sign, degree = self._longitude_to_sign(cusp)
            house_cusps_dict[i + 1] = {
                "sign": sign,
                "degree": round(degree, 2),
                "longitude": round(cusp, 2)
            }
        
        return {
            "positions": positions,
            "aspects": aspects,
            "house_cusps": house_cusps_dict,
            "ascendant": {
                "sign": asc_sign,
                "degree": round(asc_degree, 2),
                "longitude": round(ascendant, 2)
            },
            "sun_sign": positions["sonne"].sign,
            "moon_sign": positions["mond"].sign,
            "rising_sign": asc_sign,
        }


# =============================================================================
# User Profile Service
# =============================================================================

class UserProfileService:
    """Service for managing user astrological profiles."""
    
    def __init__(self, db_client=None):
        """
        Initialize service.
        
        Args:
            db_client: Either asyncpg pool or Supabase client
        """
        self.db = db_client
        self.calculator = ChartCalculator()
        self._use_supabase = USE_SUPABASE
    
    def _get_supabase_client(self):
        """Get Supabase client lazily."""
        if self._use_supabase:
            from agent.supabase_client import supabase_admin_db
            return supabase_admin_db.client
        return None
    
    async def get_profile(self, user_id: str) -> Optional[UserAstroProfile]:
        """Get cached profile from database."""
        if self._use_supabase:
            client = self._get_supabase_client()
            result = client.table("user_profiles") \
                .select("*") \
                .eq("user_id", user_id) \
                .maybe_single() \
                .execute()
            
            if not result.data:
                return None
            
            row = result.data
            return self._row_to_profile(row)
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1",
                user_id
            )
            
            if not row:
                return None
            
            return self._row_to_profile(dict(row))
    
    def _row_to_profile(self, row: Dict[str, Any]) -> UserAstroProfile:
        """Convert database row to UserAstroProfile."""
        birth_dt = row.get("birth_datetime")
        if birth_dt and isinstance(birth_dt, str):
            birth_dt = datetime.fromisoformat(birth_dt.replace('Z', '+00:00'))
        
        if not birth_dt:
            # No birth data yet - return minimal profile
            return UserAstroProfile(
                user_id=row["user_id"],
                birth_datetime=datetime.now(timezone.utc),
                birth_latitude=0.0,
                birth_longitude=0.0
            )
        
        profile = UserAstroProfile(
            user_id=row["user_id"],
            birth_datetime=birth_dt,
            birth_latitude=row.get("birth_latitude", 0.0),
            birth_longitude=row.get("birth_longitude", 0.0),
            birth_location_name=row.get("birth_location_name"),
            sun_sign=row.get("sun_sign", ""),
            moon_sign=row.get("moon_sign", ""),
            rising_sign=row.get("rising_sign", ""),
        )
        
        # Parse JSON fields
        natal_positions = row.get("natal_positions", {})
        if isinstance(natal_positions, str):
            natal_positions = json.loads(natal_positions)
        
        for k, v in (natal_positions or {}).items():
            profile.natal_positions[k] = PlanetPosition(
                planet=v["planet"],
                longitude=v["longitude"],
                sign=v["sign"],
                degree_in_sign=v["degree_in_sign"],
                house=v.get("house"),
                retrograde=v.get("retrograde", False)
            )
        
        natal_aspects = row.get("natal_aspects", [])
        if isinstance(natal_aspects, str):
            natal_aspects = json.loads(natal_aspects)
        
        for a in (natal_aspects or []):
            profile.natal_aspects.append(Aspect(
                planet1=a["planet1"],
                planet2=a["planet2"],
                aspect_type=a["aspect_type"],
                orb=a["orb"],
                applying=a.get("applying", False)
            ))
        
        house_cusps = row.get("house_cusps", {})
        if isinstance(house_cusps, str):
            house_cusps = json.loads(house_cusps)
        profile.house_cusps = {int(k): v for k, v in (house_cusps or {}).items()}
        
        chart_computed = row.get("chart_computed_at")
        if chart_computed:
            if isinstance(chart_computed, str):
                profile.chart_computed_at = datetime.fromisoformat(chart_computed.replace('Z', '+00:00'))
            else:
                profile.chart_computed_at = chart_computed
        
        return profile
    
    async def create_or_update_profile(
        self,
        user_id: str,
        birth_datetime: datetime,
        birth_latitude: float,
        birth_longitude: float,
        birth_location_name: Optional[str] = None
    ) -> UserAstroProfile:
        """Create or update user profile with computed chart."""
        # Compute chart
        logger.info(f"Computing chart for user {user_id}")
        chart_data = self.calculator.compute_full_chart(
            birth_datetime, birth_latitude, birth_longitude
        )
        
        # Build profile
        profile = UserAstroProfile(
            user_id=user_id,
            birth_datetime=birth_datetime,
            birth_latitude=birth_latitude,
            birth_longitude=birth_longitude,
            birth_location_name=birth_location_name,
            sun_sign=chart_data["sun_sign"],
            moon_sign=chart_data["moon_sign"],
            rising_sign=chart_data["rising_sign"],
            natal_positions=chart_data["positions"],
            natal_aspects=chart_data["aspects"],
            house_cusps=chart_data["house_cusps"],
            chart_computed_at=datetime.now(timezone.utc)
        )
        
        # Serialize for database
        natal_positions_json = {
            k: {
                "planet": v.planet,
                "longitude": v.longitude,
                "sign": v.sign,
                "degree_in_sign": v.degree_in_sign,
                "house": v.house,
                "retrograde": v.retrograde
            }
            for k, v in profile.natal_positions.items()
        }
        
        natal_aspects_json = [
            {
                "planet1": a.planet1,
                "planet2": a.planet2,
                "aspect_type": a.aspect_type,
                "orb": a.orb,
                "applying": a.applying
            }
            for a in profile.natal_aspects
        ]
        
        if self._use_supabase:
            client = self._get_supabase_client()
            data = {
                "user_id": user_id,
                "birth_datetime": birth_datetime.isoformat(),
                "birth_latitude": birth_latitude,
                "birth_longitude": birth_longitude,
                "birth_location_name": birth_location_name,
                "sun_sign": profile.sun_sign,
                "moon_sign": profile.moon_sign,
                "rising_sign": profile.rising_sign,
                "natal_positions": natal_positions_json,
                "natal_aspects": natal_aspects_json,
                "house_cusps": profile.house_cusps,
                "chart_computed_at": profile.chart_computed_at.isoformat()
            }
            client.table("user_profiles").upsert(data, on_conflict="user_id").execute()
        else:
            # Fallback to asyncpg
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_profiles (
                        user_id, birth_datetime, birth_latitude, birth_longitude,
                        birth_location_name, sun_sign, moon_sign, rising_sign,
                        natal_positions, natal_aspects, house_cusps, chart_computed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (user_id) DO UPDATE SET
                        birth_datetime = EXCLUDED.birth_datetime,
                        birth_latitude = EXCLUDED.birth_latitude,
                        birth_longitude = EXCLUDED.birth_longitude,
                        birth_location_name = EXCLUDED.birth_location_name,
                        sun_sign = EXCLUDED.sun_sign,
                        moon_sign = EXCLUDED.moon_sign,
                        rising_sign = EXCLUDED.rising_sign,
                        natal_positions = EXCLUDED.natal_positions,
                        natal_aspects = EXCLUDED.natal_aspects,
                        house_cusps = EXCLUDED.house_cusps,
                        chart_computed_at = EXCLUDED.chart_computed_at,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    user_id, birth_datetime, birth_latitude, birth_longitude,
                    birth_location_name, profile.sun_sign, profile.moon_sign,
                    profile.rising_sign, json.dumps(natal_positions_json),
                    json.dumps(natal_aspects_json), json.dumps(profile.house_cusps),
                    profile.chart_computed_at
                )
        
        logger.info(f"Profile created/updated for user {user_id}: {profile.sun_sign} Sun, {profile.moon_sign} Moon, {profile.rising_sign} Rising")
        return profile
    
    async def get_or_create_profile(
        self,
        user_id: str,
        birth_datetime: Optional[datetime] = None,
        birth_latitude: Optional[float] = None,
        birth_longitude: Optional[float] = None,
        birth_location_name: Optional[str] = None
    ) -> Optional[UserAstroProfile]:
        """Get existing profile or create new one if birth data provided."""
        profile = await self.get_profile(user_id)
        
        if profile and profile.sun_sign:
            return profile
        
        if birth_datetime and birth_latitude is not None and birth_longitude is not None:
            return await self.create_or_update_profile(
                user_id, birth_datetime, birth_latitude, birth_longitude, birth_location_name
            )
        
        return None
    
    async def get_all_user_ids(self) -> List[str]:
        """Get all user IDs with profiles."""
        if self._use_supabase:
            client = self._get_supabase_client()
            result = client.table("user_profiles").select("user_id").execute()
            return [row["user_id"] for row in result.data] if result.data else []
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            rows = await conn.fetch("SELECT user_id FROM user_profiles")
            return [row["user_id"] for row in rows]
    
    async def get_users_by_sun_sign(self, sun_sign: str) -> List[str]:
        """Get all user IDs with a specific sun sign."""
        if self._use_supabase:
            client = self._get_supabase_client()
            result = client.table("user_profiles") \
                .select("user_id") \
                .eq("sun_sign", sun_sign) \
                .execute()
            return [row["user_id"] for row in result.data] if result.data else []
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                "SELECT user_id FROM user_profiles WHERE sun_sign = $1",
                sun_sign
            )
            return [row["user_id"] for row in rows]
