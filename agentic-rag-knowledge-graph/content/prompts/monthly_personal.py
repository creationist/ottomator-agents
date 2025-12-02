"""
Prompt template for personalized monthly highlights.

This content is PERSONALIZED for each user based on their birth chart.
Analyzes how the month's transits affect their specific natal positions.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

# German month names
MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April",
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
}


MONTHLY_PERSONAL_SYSTEM_PROMPT = """Du bist Nyah, eine einfühlsame Astrologie-Beraterin, die personalisierte Monats-Einblicke erstellt.

Du analysierst, wie die aktuellen Transite das individuelle Geburtshoroskop eines Menschen beeinflussen. Deine Aufgabe ist es, 3-5 spezifische "Highlights" für den Monat zu identifizieren.

**Dein Ansatz:**
- Fokussiere auf die WICHTIGSTEN Transite zu Natal-Planeten
- Nenne konkrete Zeitfenster ("Um den 15. herum...")
- Erkläre die psychologische Bedeutung
- Gib praktische Handlungsempfehlungen
- Balanciere Herausforderungen und Chancen

**Wichtig:**
- Schreibe persönlich und direkt ("Du wirst...", "Für dich...")
- Vermeide allgemeine Sonnenzeichen-Horoskope
- Beziehe das gesamte Geburtshoroskop ein
- Sei spezifisch, nicht vage"""


def format_natal_positions(natal_positions: Dict[str, Any]) -> str:
    """Format natal positions for the prompt."""
    if not natal_positions:
        return "Keine Natal-Positionen verfügbar."
    
    lines = []
    for planet, data in natal_positions.items():
        if isinstance(data, dict):
            sign = data.get("sign", "?")
            degree = data.get("degree_in_sign", data.get("degree", "?"))
            house = data.get("house", "?")
            lines.append(f"- {planet.capitalize()}: {sign} {degree}° (Haus {house})")
        else:
            lines.append(f"- {planet.capitalize()}: {data}")
    
    return "\n".join(lines)


def format_transits_to_natal(transits: List[Dict[str, Any]]) -> str:
    """Format transits to natal positions for the prompt."""
    if not transits:
        return "Keine signifikanten Transite zu Natal-Planeten."
    
    lines = []
    for t in transits[:8]:  # Limit to top 8
        transit_planet = t.get("transit_planet", "").capitalize()
        natal_planet = t.get("natal_planet", "").capitalize()
        aspect = t.get("aspect", "").capitalize()
        orb = t.get("orb", 0)
        importance = t.get("importance", 3)
        retrograde = " (rückläufig)" if t.get("retrograde") else ""
        
        importance_marker = "⭐" * min(importance, 5)
        lines.append(f"- {transit_planet} {aspect} Natal-{natal_planet} (Orb: {orb}°){retrograde} {importance_marker}")
    
    return "\n".join(lines)


def format_monthly_personal_prompt(
    year: int,
    month: int,
    sun_sign: str,
    moon_sign: str,
    rising_sign: str,
    natal_positions: Dict[str, Any],
    transits_to_natal: List[Dict[str, Any]],
    moon_phases_for_sign: List[Dict[str, Any]] = None,
    ontology_context: str = ""
) -> str:
    """
    Format the prompt for personalized monthly content generation.
    
    Args:
        year: The year
        month: The month (1-12)
        sun_sign: User's sun sign
        moon_sign: User's moon sign
        rising_sign: User's rising sign (ascendant)
        natal_positions: User's natal planet positions
        transits_to_natal: Current transits aspecting natal planets
        moon_phases_for_sign: Moon phases relevant to user's sign
        ontology_context: Additional context from astrology ontology
    """
    month_name = MONTH_NAMES.get(month, str(month))
    
    natal_text = format_natal_positions(natal_positions)
    transits_text = format_transits_to_natal(transits_to_natal)
    
    # Format moon phases for sign
    if moon_phases_for_sign:
        moon_text = "\n".join([
            f"- {m.get('type', 'Mondphase').capitalize()} in {m.get('sign', '?')} ({m.get('date', '')[:10]})"
            for m in moon_phases_for_sign
        ])
    else:
        moon_text = "Keine besonderen Mondphasen für dein Zeichen."
    
    return f"""Erstelle personalisierte Monats-Highlights für {month_name} {year}.

**Nutzer-Profil:**
- Sonnenzeichen: {sun_sign}
- Mondzeichen: {moon_sign}
- Aszendent: {rising_sign}

**Natal-Positionen:**
{natal_text}

**Aktuelle Transite zu Natal-Planeten:**
{transits_text}

**Wichtige Mondphasen für dieses Zeichen:**
{moon_text}

**Astrologischer Kontext:**
{ontology_context if ontology_context else "Kein zusätzlicher Kontext."}

---

**Aufgabe:**
Erstelle 3-5 personalisierte Monats-Highlights (~400-500 Wörter gesamt).

**Für jeden Highlight:**
1. **Titel** - Prägnant und aussagekräftig
2. **Zeitraum** - Wann ist dieser Transit am stärksten?
3. **Was passiert** - Welche Natal-Punkte werden aktiviert?
4. **Bedeutung** - Was bedeutet das psychologisch/praktisch?
5. **Empfehlung** - Was kannst du tun?

**Format für jeden Highlight:**

### 🌟 Highlight 1: [Titel]
**Zeitraum:** [z.B. "Um den 10.-15. herum" oder "Ganzer Monat"]

[2-3 Sätze die erklären, was astrologisch passiert und wie es den Nutzer betrifft]

**Empfehlung:** [1-2 konkrete Handlungstipps]

---

**Wichtige Hinweise:**
- Beziehe dich auf die SPEZIFISCHEN Transite zu den Natal-Positionen
- Nicht nur aufs Sonnenzeichen fokussieren - beziehe Mond, Aszendent und andere Planeten ein
- Nutze die Aspekt-Informationen (Konjunktion, Quadrat, Trigon, etc.)
- Beachte ob Planeten rückläufig sind
- Priorisiere nach Wichtigkeit (⭐ = wichtiger Transit)

Schreibe auf Deutsch, persönlich und ermutigend."""


