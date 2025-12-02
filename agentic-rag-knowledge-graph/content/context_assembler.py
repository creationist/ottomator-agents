"""
Context assembler for content generation.

Combines data from multiple sources (user profile, transits, ontology)
to create the full context needed for prompt templates.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .content_types import ContentType, CONTENT_TEMPLATES
from .user_profile import UserAstroProfile, UserProfileService
from .transit_service import TransitService, CurrentTransits, MonthlyTransits
from .prompts import (
    format_monthly_general_prompt,
    format_monthly_personal_prompt,
    format_moon_reflection_prompt,
)

logger = logging.getLogger(__name__)


class ContentContextAssembler:
    """
    Assembles all context needed for content generation.
    
    Combines:
    - User profile data (birth chart, natal positions)
    - Transit data (current positions, monthly events)
    - Ontology data (sign meanings, planet descriptions)
    """
    
    def __init__(
        self,
        user_service: UserProfileService,
        transit_service: TransitService,
        ontology=None  # AstrologyOntology instance
    ):
        self.users = user_service
        self.transits = transit_service
        self.ontology = ontology
    
    def _get_ontology_context(
        self,
        concepts: List[str],
        max_chars: int = 1000
    ) -> str:
        """
        Get ontology descriptions for a list of concepts.
        
        Args:
            concepts: List of concept names (signs, planets, etc.)
            max_chars: Maximum characters for the context
        """
        if not self.ontology:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for concept in concepts:
            try:
                entity = self.ontology.get_entity(concept.lower())
                if entity:
                    desc = f"**{entity.name}**: {entity.description}"
                    if entity.keywords:
                        desc += f" (Schlüsselwörter: {', '.join(entity.keywords[:5])})"
                    
                    if total_chars + len(desc) > max_chars:
                        break
                    
                    context_parts.append(desc)
                    total_chars += len(desc)
            except Exception as e:
                logger.debug(f"Could not get ontology for {concept}: {e}")
        
        return "\n".join(context_parts)
    
    async def assemble_monthly_general_context(
        self,
        year: int,
        month: int
    ) -> Dict[str, Any]:
        """
        Assemble context for general monthly content.
        
        This content is the same for all users.
        """
        # Get monthly transit data
        monthly = self.transits.get_monthly_transits(year, month)
        
        # Get ontology context for prominent signs/planets this month
        concepts_to_lookup = []
        for event in monthly.events[:5]:
            concepts_to_lookup.extend(event.planets_involved)
        
        ontology_context = self._get_ontology_context(list(set(concepts_to_lookup)))
        
        # Format the prompt
        prompt = format_monthly_general_prompt(
            year=year,
            month=month,
            monthly_transits=[
                {
                    "date": e.date.strftime("%d.%m."),
                    "description": e.description,
                    "event_type": e.event_type
                }
                for e in monthly.events if e.importance >= 3
            ],
            retrogrades=monthly.retrogrades,
            moon_phases=monthly.moon_phases,
            ontology_context=ontology_context
        )
        
        return {
            "content_type": ContentType.MONTHLY_GENERAL,
            "prompt": prompt,
            "year": year,
            "month": month,
            "metadata": {
                "event_count": len(monthly.events),
                "retrograde_count": len(monthly.retrogrades)
            }
        }
    
    async def assemble_monthly_personal_context(
        self,
        user_id: str,
        year: int,
        month: int
    ) -> Optional[Dict[str, Any]]:
        """
        Assemble context for personalized monthly content.
        
        Requires user profile with birth chart data.
        """
        # Get user profile
        profile = await self.users.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return None
        
        # Get monthly transits
        monthly = self.transits.get_monthly_transits(year, month)
        
        # Get transits to natal positions
        transits_to_natal = self.transits.get_transits_to_natal(profile.natal_positions)
        
        # Get ontology context for user's signs and active transits
        concepts = [
            profile.sun_sign,
            profile.moon_sign,
            profile.rising_sign
        ]
        for t in transits_to_natal[:3]:
            concepts.append(t.get("transit_planet", ""))
        
        ontology_context = self._get_ontology_context(list(set(concepts)))
        
        # Format natal positions for prompt
        natal_positions_dict = {
            k: {
                "sign": v.sign,
                "degree_in_sign": v.degree_in_sign,
                "house": v.house,
                "retrograde": v.retrograde
            }
            for k, v in profile.natal_positions.items()
        }
        
        # Find moon phases relevant to user's sign
        moon_phases_for_sign = [
            m for m in monthly.moon_phases
            if m.get("sign", "").lower() == profile.sun_sign.lower()
            or m.get("sign", "").lower() == profile.moon_sign.lower()
        ]
        
        # Format the prompt
        prompt = format_monthly_personal_prompt(
            year=year,
            month=month,
            sun_sign=profile.sun_sign,
            moon_sign=profile.moon_sign,
            rising_sign=profile.rising_sign,
            natal_positions=natal_positions_dict,
            transits_to_natal=transits_to_natal,
            moon_phases_for_sign=moon_phases_for_sign,
            ontology_context=ontology_context
        )
        
        return {
            "content_type": ContentType.MONTHLY_PERSONAL,
            "prompt": prompt,
            "user_id": user_id,
            "year": year,
            "month": month,
            "metadata": {
                "sun_sign": profile.sun_sign,
                "moon_sign": profile.moon_sign,
                "rising_sign": profile.rising_sign,
                "transit_count": len(transits_to_natal)
            }
        }
    
    async def assemble_moon_reflection_context(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Assemble context for moon reflection questions.
        
        Based on current Moon position and user's natal Moon.
        """
        # Get user profile
        profile = await self.users.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return None
        
        # Get current transits
        current = self.transits.get_current_transits()
        
        # Get natal Moon data
        natal_moon = profile.natal_positions.get("mond")
        if not natal_moon:
            logger.warning(f"No natal moon data for user {user_id}")
            natal_moon_sign = profile.moon_sign or "Unbekannt"
            natal_moon_house = None
        else:
            natal_moon_sign = natal_moon.sign
            natal_moon_house = natal_moon.house
        
        # Format moon aspects
        moon_aspects_list = [
            a for a in current.active_aspects
            if "mond" in (a.planet1, a.planet2)
        ]
        moon_aspects_text = ", ".join([
            f"{a.planet1.capitalize()} {a.aspect_type} {a.planet2.capitalize()}"
            for a in moon_aspects_list[:3]
        ]) if moon_aspects_list else ""
        
        # Format natal moon aspects
        natal_moon_aspects_list = [
            a for a in profile.natal_aspects
            if "mond" in (a.planet1, a.planet2)
        ]
        natal_moon_aspects_text = ", ".join([
            f"{a.aspect_type} {a.planet2 if a.planet1 == 'mond' else a.planet1}".capitalize()
            for a in natal_moon_aspects_list[:3]
        ]) if natal_moon_aspects_list else ""
        
        # Get ontology context for the moon signs
        concepts = [current.moon_sign, natal_moon_sign, "mond"]
        moon_sign_context = self._get_ontology_context(concepts)
        
        # Format the prompt
        prompt = format_moon_reflection_prompt(
            transit_moon_sign=current.moon_sign,
            moon_phase=current.moon_phase.value,
            moon_aspects=moon_aspects_text,
            natal_moon_sign=natal_moon_sign,
            natal_moon_house=natal_moon_house,
            natal_moon_aspects=natal_moon_aspects_text,
            moon_sign_context=moon_sign_context
        )
        
        return {
            "content_type": ContentType.MOON_REFLECTION,
            "prompt": prompt,
            "user_id": user_id,
            "metadata": {
                "transit_moon_sign": current.moon_sign,
                "moon_phase": current.moon_phase.value,
                "natal_moon_sign": natal_moon_sign,
                "natal_moon_house": natal_moon_house
            }
        }
    
    async def assemble_context(
        self,
        content_type: ContentType,
        user_id: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Main entry point - assemble context for any content type.
        
        Args:
            content_type: Type of content to generate
            user_id: User ID (required for personalized content)
            year: Year (for monthly content, defaults to current)
            month: Month (for monthly content, defaults to current)
        """
        # Default to current year/month
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        
        template = CONTENT_TEMPLATES[content_type]
        
        # Check if user data is required
        if template.is_personalized and not user_id:
            logger.error(f"User ID required for personalized content type: {content_type}")
            return None
        
        # Route to appropriate assembler
        if content_type == ContentType.MONTHLY_GENERAL:
            return await self.assemble_monthly_general_context(year, month)
        
        elif content_type == ContentType.MONTHLY_PERSONAL:
            return await self.assemble_monthly_personal_context(user_id, year, month)
        
        elif content_type == ContentType.MOON_REFLECTION:
            return await self.assemble_moon_reflection_context(user_id)
        
        else:
            logger.error(f"Unknown content type: {content_type}")
            return None



