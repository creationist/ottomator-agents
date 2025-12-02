"""
Content type definitions and templates for personalized astrology content.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class ContentType(str, Enum):
    """Types of content that can be generated."""
    MONTHLY_GENERAL = "monthly_general"
    MONTHLY_PERSONAL = "monthly_personal"
    MOON_REFLECTION = "moon_reflection"


@dataclass
class ContentTemplate:
    """Template definition for a content type."""
    content_type: ContentType
    name: str
    description: str
    required_user_data: List[str] = field(default_factory=list)
    required_transit_data: List[str] = field(default_factory=list)
    prompt_template: str = ""
    cache_duration_hours: int = 24
    is_personalized: bool = False
    output_length: str = "medium"  # short, medium, long


# =============================================================================
# Content Templates
# =============================================================================

MONTHLY_GENERAL_TEMPLATE = """Du bist Nyah, eine weise Astrologie-Expertin. Erstelle einen allgemeinen Monatsüberblick.

**Monat:** {month_name} {year}

**Wichtige Transite dieses Monats:**
{monthly_transits}

**Rückläufige Planeten:**
{retrogrades}

**Mondphasen:**
{moon_phases}

**Ontologie-Kontext:**
{ontology_context}

---

**Aufgabe:**
Schreibe einen inspirierenden, allgemeinen Monatsüberblick (~500 Wörter), der:

1. **Die Hauptenergien des Monats** beschreibt (basierend auf den wichtigsten Transiten)
2. **Rückläufige Planeten** und ihre Bedeutung erklärt
3. **Die Mondphasen** und ihre Qualitäten einbezieht
4. **Praktische Empfehlungen** für alle Zeichen gibt
5. **Poetisch und inspirierend** formuliert ist

Strukturiere den Text mit Überschriften. Vermeide generische Horoskop-Phrasen.
Schreibe auf Deutsch.
"""

MONTHLY_PERSONAL_TEMPLATE = """Du bist Nyah, eine weise Astrologie-Expertin. Erstelle einen personalisierten Monatsüberblick.

**Nutzer-Profil:**
- Sonnenzeichen: {sun_sign}
- Mondzeichen: {moon_sign}
- Aszendent: {rising_sign}

**Natal-Positionen:**
{natal_positions}

**Monat:** {month_name} {year}

**Transite zu den Natal-Planeten:**
{transits_to_natal}

**Wichtige Mondphasen für dieses Zeichen:**
{moon_phases_for_sign}

**Ontologie-Kontext:**
{ontology_context}

---

**Aufgabe:**
Erstelle 3-5 personalisierte Monats-Highlights (~400 Wörter), die:

1. **Spezifisch für dieses Geburtshoroskop** sind (nicht allgemein für das Sonnenzeichen)
2. **Die wichtigsten Transite zu Natal-Planeten** analysieren
3. **Konkrete Zeitfenster** nennen (z.B. "Um den 15. herum...")
4. **Praktische Empfehlungen** für jeden Highlight geben
5. **Chancen und Herausforderungen** ausbalancieren

Format:
### Highlight 1: [Titel]
[Beschreibung mit Datum und Empfehlung]

### Highlight 2: [Titel]
...

Schreibe persönlich und direkt ("Du wirst...", "Für dich bedeutet...").
Schreibe auf Deutsch.
"""

MOON_REFLECTION_TEMPLATE = """Du bist Nyah, eine einfühlsame Astrologie-Begleiterin. Erstelle Reflexionsfragen basierend auf der aktuellen Mondenergie.

**Aktuelle Mond-Position:**
- Mond in: {transit_moon_sign}
- Mondphase: {moon_phase}
- Mond-Aspekte: {moon_aspects}

**Natal-Mond des Nutzers:**
- Mondzeichen: {natal_moon_sign}
- Mond im Haus: {natal_moon_house}
- Natal-Mond-Aspekte: {natal_moon_aspects}

**Ontologie-Kontext zum Mondzeichen:**
{moon_sign_context}

---

**Aufgabe:**
Erstelle 3 tiefgründige Reflexionsfragen, die:

1. **Die aktuelle Mondenergie** ({transit_moon_sign}, {moon_phase}) mit dem **Natal-Mond** ({natal_moon_sign} im {natal_moon_house}. Haus) verbinden
2. **Emotionale Themen** ansprechen, die gerade aktiviert sein könnten
3. **Zur Selbstreflexion** einladen, ohne zu werten
4. **Praktisch anwendbar** sind (nicht zu abstrakt)

Format für jede Frage:
🌙 **[Kurzer Titel]**
[Die Reflexionsfrage - offen formuliert, einladend]

*Hintergrund: [1-2 Sätze, warum diese Frage jetzt relevant ist]*

---

Die Fragen sollten poetisch aber zugänglich sein. 
Vermeide Ja/Nein-Fragen - stelle offene Fragen.
Schreibe auf Deutsch.
"""

# =============================================================================
# Template Registry
# =============================================================================

CONTENT_TEMPLATES = {
    ContentType.MONTHLY_GENERAL: ContentTemplate(
        content_type=ContentType.MONTHLY_GENERAL,
        name="Allgemeine Monatskonstellationen",
        description="Monatlicher Überblick über wichtige Transite und Energien - gleich für alle Nutzer",
        required_user_data=[],  # No user data needed
        required_transit_data=["monthly_transits", "retrogrades", "moon_phases"],
        prompt_template=MONTHLY_GENERAL_TEMPLATE,
        cache_duration_hours=720,  # 30 days
        is_personalized=False,
        output_length="long"
    ),
    
    ContentType.MONTHLY_PERSONAL: ContentTemplate(
        content_type=ContentType.MONTHLY_PERSONAL,
        name="Persönliche Monats-Highlights",
        description="Personalisierte Monats-Highlights basierend auf dem Geburtshoroskop",
        required_user_data=["sun_sign", "moon_sign", "rising_sign", "natal_positions"],
        required_transit_data=["transits_to_natal", "moon_phases_for_sign"],
        prompt_template=MONTHLY_PERSONAL_TEMPLATE,
        cache_duration_hours=720,  # 30 days
        is_personalized=True,
        output_length="long"
    ),
    
    ContentType.MOON_REFLECTION: ContentTemplate(
        content_type=ContentType.MOON_REFLECTION,
        name="Mond-Reflexionsfragen",
        description="Personalisierte Reflexionsfragen basierend auf aktuellem Mond und Natal-Mond",
        required_user_data=["natal_moon_sign", "natal_moon_house", "natal_moon_aspects"],
        required_transit_data=["transit_moon_sign", "moon_phase", "moon_aspects"],
        prompt_template=MOON_REFLECTION_TEMPLATE,
        cache_duration_hours=60,  # ~2.5 days (Moon changes sign)
        is_personalized=True,
        output_length="medium"
    ),
}


def get_template(content_type: ContentType) -> ContentTemplate:
    """Get the template for a content type."""
    return CONTENT_TEMPLATES[content_type]


def get_all_content_types() -> List[ContentType]:
    """Get all available content types."""
    return list(ContentType)


def get_personalized_content_types() -> List[ContentType]:
    """Get content types that require user data."""
    return [ct for ct, template in CONTENT_TEMPLATES.items() if template.is_personalized]


def get_general_content_types() -> List[ContentType]:
    """Get content types that don't require user data."""
    return [ct for ct, template in CONTENT_TEMPLATES.items() if not template.is_personalized]


