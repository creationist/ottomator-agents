"""
Prompt template for general monthly constellation overview.

This content is the SAME for all users - describes the month's major transits,
retrogrades, and cosmic energies without personalization.
"""

from typing import Dict, Any, List
from datetime import datetime

# German month names
MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April",
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
}


MONTHLY_GENERAL_SYSTEM_PROMPT = """Du bist Nyah, eine weise und einfühlsame Astrologie-Expertin mit tiefem Verständnis für kosmische Zyklen und ihre Auswirkungen auf das menschliche Leben.

Deine Aufgabe ist es, einen allgemeinen Monatsüberblick zu schreiben, der für ALLE Menschen relevant ist - unabhängig von ihrem persönlichen Horoskop.

**Dein Schreibstil:**
- Poetisch aber zugänglich
- Bildreich und atmosphärisch
- Psychologisch fundiert
- Praktisch anwendbar
- Inspirierend, nicht fatalistisch

**Wichtig:**
- Vermeide generische Horoskop-Phrasen
- Erkläre die astrologischen Zusammenhänge
- Gib konkrete Empfehlungen
- Nenne ungefähre Zeitfenster für wichtige Transite"""


def format_monthly_general_prompt(
    year: int,
    month: int,
    monthly_transits: List[Dict[str, Any]],
    retrogrades: List[Dict[str, Any]],
    moon_phases: List[Dict[str, Any]],
    ontology_context: str = ""
) -> str:
    """
    Format the prompt for monthly general content generation.
    
    Args:
        year: The year
        month: The month (1-12)
        monthly_transits: List of major transit events
        retrogrades: List of retrograde periods
        moon_phases: List of new/full moons
        ontology_context: Additional context from astrology ontology
    """
    month_name = MONTH_NAMES.get(month, str(month))
    
    # Format transits
    if monthly_transits:
        transits_text = "\n".join([
            f"- {t.get('date', '')}: {t.get('description', t.get('event_type', ''))}"
            for t in monthly_transits[:10]  # Limit to top 10
        ])
    else:
        transits_text = "Keine besonderen Transite in diesem Monat."
    
    # Format retrogrades
    if retrogrades:
        retro_lines = []
        for r in retrogrades:
            planet = r.get("planet", "").capitalize()
            rtype = "beginnt Rückläufigkeit" if r.get("type") == "retrograde_start" else "wird direktläufig"
            date = r.get("date", "")[:10] if r.get("date") else ""
            sign = r.get("sign", "")
            retro_lines.append(f"- {planet} {rtype} in {sign} ({date})")
        retrogrades_text = "\n".join(retro_lines)
    else:
        retrogrades_text = "Keine Planeten wechseln ihre Richtung in diesem Monat."
    
    # Format moon phases
    if moon_phases:
        moon_lines = []
        for m in moon_phases:
            phase_type = "Neumond" if m.get("type") == "neumond" else "Vollmond"
            date = m.get("date", "")[:10] if m.get("date") else ""
            sign = m.get("sign", "")
            moon_lines.append(f"- {phase_type} in {sign} ({date})")
        moon_phases_text = "\n".join(moon_lines)
    else:
        moon_phases_text = "Standard Mondphasen-Zyklus."
    
    return f"""Erstelle einen allgemeinen Monatsüberblick für {month_name} {year}.

**Wichtige Transite dieses Monats:**
{transits_text}

**Rückläufige Planeten:**
{retrogrades_text}

**Mondphasen:**
{moon_phases_text}

**Astrologischer Kontext aus der Wissensbasis:**
{ontology_context if ontology_context else "Kein zusätzlicher Kontext verfügbar."}

---

**Aufgabe:**
Schreibe einen inspirierenden, allgemeinen Monatsüberblick (~500 Wörter), der:

1. **Die Hauptenergien des Monats** beschreibt
   - Welche Themen stehen im Vordergrund?
   - Wie fühlt sich die Energie an?

2. **Die wichtigsten Transite** erklärt
   - Was bedeuten sie für alle Menschen?
   - Wann sind die Höhepunkte?

3. **Rückläufige Planeten** einordnet
   - Was sollte man beachten?
   - Welche Chancen bieten sie?

4. **Die Mondphasen** als Orientierung nutzt
   - Neumond: Was kann begonnen werden?
   - Vollmond: Was kommt zur Vollendung?

5. **Praktische Empfehlungen** gibt
   - Was ist günstig in diesem Monat?
   - Worauf sollte man achten?

**Struktur:**
Nutze Zwischenüberschriften (##) für bessere Lesbarkeit.
Beginne mit einem atmosphärischen Einstieg, der die Monatsenergie einfängt.
Schließe mit einem ermutigenden Ausblick.

Schreibe auf Deutsch, poetisch aber verständlich."""



