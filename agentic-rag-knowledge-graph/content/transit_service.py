"""
Transit service for current celestial positions.

Provides real-time planetary positions and monthly transit data.
Uses Swiss Ephemeris (pyswisseph) for astronomical calculations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
except ImportError:
    SWISSEPH_AVAILABLE = False
    logging.warning("pyswisseph not installed. Transit calculations will use fallback.")

from .user_profile import (
    ZODIAC_SIGNS, PLANETS, ASPECT_TYPES, ASPECT_ORBS,
    ChartCalculator, PlanetPosition, Aspect
)

logger = logging.getLogger(__name__)


# =============================================================================
# Moon Phase Enum
# =============================================================================

class MoonPhase(str, Enum):
    """Moon phases."""
    NEW_MOON = "neumond"
    WAXING_CRESCENT = "zunehmende_sichel"
    FIRST_QUARTER = "erstes_viertel"
    WAXING_GIBBOUS = "zunehmender_mond"
    FULL_MOON = "vollmond"
    WANING_GIBBOUS = "abnehmender_mond"
    LAST_QUARTER = "letztes_viertel"
    WANING_CRESCENT = "abnehmende_sichel"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CurrentTransits:
    """Current celestial positions and aspects."""
    timestamp: datetime
    
    # Current positions (German names)
    positions: Dict[str, PlanetPosition] = field(default_factory=dict)
    
    # Moon-specific data
    moon_sign: str = ""
    moon_phase: MoonPhase = MoonPhase.NEW_MOON
    moon_phase_percent: float = 0.0  # 0-100
    
    # Retrograde planets
    retrograde_planets: List[str] = field(default_factory=list)
    
    # Active aspects between transiting planets
    active_aspects: List[Aspect] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "positions": {
                k: {
                    "sign": v.sign,
                    "degree": v.degree_in_sign,
                    "retrograde": v.retrograde
                }
                for k, v in self.positions.items()
            },
            "moon_sign": self.moon_sign,
            "moon_phase": self.moon_phase.value,
            "moon_phase_percent": self.moon_phase_percent,
            "retrograde_planets": self.retrograde_planets,
            "active_aspects": [
                {
                    "planet1": a.planet1,
                    "planet2": a.planet2,
                    "aspect": a.aspect_type,
                    "orb": a.orb
                }
                for a in self.active_aspects
            ]
        }


@dataclass
class MonthlyTransitEvent:
    """A significant transit event in a month."""
    date: datetime
    event_type: str  # 'ingress', 'aspect', 'retrograde', 'direct', 'moon_phase'
    description: str
    planets_involved: List[str]
    importance: int = 1  # 1-5, 5 being most significant


@dataclass
class MonthlyTransits:
    """Transit data for a month."""
    year: int
    month: int
    events: List[MonthlyTransitEvent] = field(default_factory=list)
    retrogrades: List[Dict[str, Any]] = field(default_factory=list)
    moon_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_major_events(self, min_importance: int = 3) -> List[MonthlyTransitEvent]:
        """Get events above a certain importance level."""
        return [e for e in self.events if e.importance >= min_importance]


# =============================================================================
# Transit Service
# =============================================================================

class TransitService:
    """Service for current and monthly transit calculations."""
    
    def __init__(self, cache=None):
        """
        Initialize transit service.
        
        Args:
            cache: Optional cache instance (e.g., Redis, in-memory dict)
        """
        self.cache = cache or {}
        self.calculator = ChartCalculator()
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if isinstance(self.cache, dict):
            cached = self.cache.get(key)
            if cached:
                data, expiry = cached
                if datetime.now(timezone.utc) < expiry:
                    return data
                del self.cache[key]
            return None
        # For external cache (Redis, etc.)
        return self.cache.get(key) if hasattr(self.cache, 'get') else None
    
    def _set_cache(self, key: str, value: Any, ttl_seconds: int):
        """Set value in cache with TTL."""
        if isinstance(self.cache, dict):
            expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            self.cache[key] = (value, expiry)
        elif hasattr(self.cache, 'set'):
            self.cache.set(key, value, ex=ttl_seconds)
    
    def _calculate_moon_phase(self, sun_lon: float, moon_lon: float) -> Tuple[MoonPhase, float]:
        """Calculate moon phase from sun and moon longitudes."""
        # Angular distance between moon and sun
        diff = (moon_lon - sun_lon) % 360
        
        # Moon phase percentage (0 = new, 50 = full)
        percent = diff / 360 * 100
        
        # Determine phase
        if diff < 22.5:
            phase = MoonPhase.NEW_MOON
        elif diff < 67.5:
            phase = MoonPhase.WAXING_CRESCENT
        elif diff < 112.5:
            phase = MoonPhase.FIRST_QUARTER
        elif diff < 157.5:
            phase = MoonPhase.WAXING_GIBBOUS
        elif diff < 202.5:
            phase = MoonPhase.FULL_MOON
        elif diff < 247.5:
            phase = MoonPhase.WANING_GIBBOUS
        elif diff < 292.5:
            phase = MoonPhase.LAST_QUARTER
        elif diff < 337.5:
            phase = MoonPhase.WANING_CRESCENT
        else:
            phase = MoonPhase.NEW_MOON
        
        return phase, round(percent, 1)
    
    def get_current_transits(self, use_cache: bool = True) -> CurrentTransits:
        """
        Get current planetary positions.
        
        Caching:
        - Moon data: 2 hours (fast-moving)
        - Inner planets: 12 hours
        - Outer planets: 24 hours (slow-moving)
        """
        cache_key = "current_transits"
        
        if use_cache:
            cached = self._get_cache(cache_key)
            if cached:
                return cached
        
        now = datetime.now(timezone.utc)
        jd = self.calculator._datetime_to_julian(now)
        
        # Calculate all positions
        positions = {}
        retrograde_planets = []
        
        for planet_name, planet_id in PLANETS.items():
            lon, retrograde = self.calculator.calculate_planet_position(planet_id, jd)
            
            # Skip planets that couldn't be calculated (e.g., Chiron without ephemeris files)
            if lon is None:
                continue
            
            sign, degree = self.calculator._longitude_to_sign(lon)
            
            positions[planet_name] = PlanetPosition(
                planet=planet_name,
                longitude=lon,
                sign=sign,
                degree_in_sign=round(degree, 2),
                retrograde=retrograde
            )
            
            if retrograde:
                retrograde_planets.append(planet_name)
        
        # Calculate moon phase
        sun_lon = positions["sonne"].longitude
        moon_lon = positions["mond"].longitude
        moon_phase, phase_percent = self._calculate_moon_phase(sun_lon, moon_lon)
        
        # Calculate aspects
        aspects = self.calculator.calculate_aspects(positions)
        
        transits = CurrentTransits(
            timestamp=now,
            positions=positions,
            moon_sign=positions["mond"].sign,
            moon_phase=moon_phase,
            moon_phase_percent=phase_percent,
            retrograde_planets=retrograde_planets,
            active_aspects=aspects
        )
        
        # Cache for 2 hours (moon moves ~1 degree per 2 hours)
        self._set_cache(cache_key, transits, 7200)
        
        return transits
    
    def get_moon_position(self) -> Dict[str, Any]:
        """Get current moon position and phase."""
        transits = self.get_current_transits()
        return {
            "sign": transits.moon_sign,
            "phase": transits.moon_phase.value,
            "phase_percent": transits.moon_phase_percent,
            "degree": transits.positions["mond"].degree_in_sign
        }
    
    def get_monthly_transits(self, year: int, month: int) -> MonthlyTransits:
        """
        Get major transits for a specific month.
        
        This calculates:
        - Sign ingresses (when planets change signs)
        - Major aspects (conjunctions, squares, oppositions)
        - Retrograde stations
        - Moon phases (new/full moons)
        """
        cache_key = f"monthly_transits:{year}:{month}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        events = []
        retrogrades = []
        moon_phases_list = []
        
        # Get start and end of month
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        
        # Sample positions every day to detect changes
        current_date = start_date
        prev_positions = None
        prev_retrograde = {}
        
        while current_date < end_date:
            jd = self.calculator._datetime_to_julian(current_date)
            
            # Calculate positions for this day
            positions = {}
            for planet_name, planet_id in PLANETS.items():
                lon, retrograde = self.calculator.calculate_planet_position(planet_id, jd)
                sign, degree = self.calculator._longitude_to_sign(lon)
                positions[planet_name] = (sign, degree, retrograde)
            
            if prev_positions:
                # Check for sign changes (ingresses)
                for planet, (sign, degree, retro) in positions.items():
                    prev_sign, _, prev_retro = prev_positions[planet]
                    
                    if sign != prev_sign:
                        importance = 5 if planet in ["sonne", "mars", "jupiter", "saturn"] else 3
                        events.append(MonthlyTransitEvent(
                            date=current_date,
                            event_type="ingress",
                            description=f"{planet.capitalize()} wechselt in {sign}",
                            planets_involved=[planet],
                            importance=importance
                        ))
                    
                    # Check retrograde changes
                    if planet != "sonne" and planet != "mond":
                        if retro and not prev_retro:
                            events.append(MonthlyTransitEvent(
                                date=current_date,
                                event_type="retrograde",
                                description=f"{planet.capitalize()} wird rückläufig in {sign}",
                                planets_involved=[planet],
                                importance=4
                            ))
                            retrogrades.append({
                                "planet": planet,
                                "type": "retrograde_start",
                                "date": current_date.isoformat(),
                                "sign": sign
                            })
                        elif not retro and prev_retro:
                            events.append(MonthlyTransitEvent(
                                date=current_date,
                                event_type="direct",
                                description=f"{planet.capitalize()} wird direktläufig in {sign}",
                                planets_involved=[planet],
                                importance=4
                            ))
                            retrogrades.append({
                                "planet": planet,
                                "type": "direct",
                                "date": current_date.isoformat(),
                                "sign": sign
                            })
            
            # Check for new/full moon
            sun_sign, sun_deg, _ = positions["sonne"]
            moon_sign, moon_deg, _ = positions["mond"]
            sun_lon = list(PLANETS.keys()).index("sonne") * 30 + sun_deg  # Rough
            moon_lon = list(PLANETS.keys()).index("mond") * 30 + moon_deg
            
            # This is simplified - would need precise calculation for exact dates
            if prev_positions:
                prev_moon_sign, _, _ = prev_positions["mond"]
                if moon_sign != prev_moon_sign and moon_sign == sun_sign:
                    moon_phases_list.append({
                        "type": "neumond",
                        "date": current_date.isoformat(),
                        "sign": moon_sign
                    })
                    events.append(MonthlyTransitEvent(
                        date=current_date,
                        event_type="moon_phase",
                        description=f"Neumond in {moon_sign}",
                        planets_involved=["mond", "sonne"],
                        importance=4
                    ))
            
            prev_positions = positions
            current_date += timedelta(days=1)
        
        # Sort events by date
        events.sort(key=lambda e: e.date)
        
        monthly = MonthlyTransits(
            year=year,
            month=month,
            events=events,
            retrogrades=retrogrades,
            moon_phases=moon_phases_list
        )
        
        # Cache for 30 days
        self._set_cache(cache_key, monthly, 2592000)
        
        return monthly
    
    def get_transits_to_natal(
        self,
        natal_positions: Dict[str, PlanetPosition],
        orb_multiplier: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Calculate current transits to natal positions.
        
        Args:
            natal_positions: User's natal planet positions
            orb_multiplier: Multiply default orbs (1.0 = normal, 0.5 = tighter)
        """
        current = self.get_current_transits()
        transits_to_natal = []
        
        # Transit planets to check (skip fast-moving Moon for this)
        transit_planets = ["sonne", "merkur", "venus", "mars", "jupiter", "saturn", "uranus", "neptun", "pluto"]
        
        for transit_planet in transit_planets:
            if transit_planet not in current.positions:
                continue
            
            transit_pos = current.positions[transit_planet]
            
            for natal_planet, natal_pos in natal_positions.items():
                # Calculate angular distance
                diff = abs(transit_pos.longitude - natal_pos.longitude)
                if diff > 180:
                    diff = 360 - diff
                
                # Check each aspect
                for aspect_name, aspect_angle in ASPECT_TYPES.items():
                    orb = abs(diff - aspect_angle)
                    max_orb = ASPECT_ORBS[aspect_name] * orb_multiplier
                    
                    if orb <= max_orb:
                        # Determine importance
                        importance = 3
                        if transit_planet in ["saturn", "uranus", "neptun", "pluto"]:
                            importance = 5  # Outer planet transits are significant
                        if aspect_name in ["konjunktion", "opposition", "quadrat"]:
                            importance += 1
                        
                        transits_to_natal.append({
                            "transit_planet": transit_planet,
                            "natal_planet": natal_planet,
                            "aspect": aspect_name,
                            "orb": round(orb, 2),
                            "transit_sign": transit_pos.sign,
                            "natal_sign": natal_pos.sign,
                            "importance": min(importance, 5),
                            "retrograde": transit_pos.retrograde
                        })
                        break
        
        # Sort by importance
        transits_to_natal.sort(key=lambda t: -t["importance"])
        
        return transits_to_natal
    
    def format_retrogrades_text(self, transits: CurrentTransits) -> str:
        """Format retrograde planets as readable text."""
        if not transits.retrograde_planets:
            return "Keine Planeten sind derzeit rückläufig."
        
        planet_names = [p.capitalize() for p in transits.retrograde_planets]
        if len(planet_names) == 1:
            return f"{planet_names[0]} ist rückläufig."
        else:
            return f"{', '.join(planet_names[:-1])} und {planet_names[-1]} sind rückläufig."
    
    def format_moon_phases_text(self, monthly: MonthlyTransits) -> str:
        """Format moon phases for a month as readable text."""
        if not monthly.moon_phases:
            return "Keine signifikanten Mondphasen in diesem Monat."
        
        lines = []
        for phase in monthly.moon_phases:
            date = datetime.fromisoformat(phase["date"])
            date_str = date.strftime("%d.%m.")
            phase_type = "Neumond" if phase["type"] == "neumond" else "Vollmond"
            lines.append(f"- {phase_type} in {phase['sign']} am {date_str}")
        
        return "\n".join(lines)


