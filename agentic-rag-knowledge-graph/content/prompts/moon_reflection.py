"""
Prompt template for moon-based reflection questions.

This content is PERSONALIZED based on:
- Current Moon position (sign + phase)
- User's natal Moon (sign + house)

Generated every ~2.5 days when the Moon changes sign.
"""

from typing import Dict, Any, Optional


MOON_REFLECTION_SYSTEM_PROMPT = """Du bist Nyah, eine einfühlsame Begleiterin für emotionale Selbstreflexion durch die Linse der Astrologie.

Du erstellst tiefgründige Reflexionsfragen, die auf der aktuellen Mondenergie basieren und diese mit dem persönlichen Mondzeichen des Nutzers verbinden.

**Dein Ansatz:**
- Fragen, die zur Introspektion einladen
- Verbindung zwischen kosmischen und persönlichen Themen
- Poetisch aber zugänglich
- Niemals wertend oder beängstigend
- Praktisch anwendbar im Alltag

**Die Qualität der Fragen:**
- OFFEN formuliert (keine Ja/Nein-Fragen)
- Emotionale Tiefe ohne zu überwältigen
- Konkreter Bezug zur aktuellen Mondenergie
- Verbindung zum Natal-Mond des Nutzers"""


def format_moon_reflection_prompt(
    transit_moon_sign: str,
    moon_phase: str,
    moon_aspects: str = "",
    natal_moon_sign: str = "",
    natal_moon_house: int = None,
    natal_moon_aspects: str = "",
    moon_sign_context: str = ""
) -> str:
    """
    Format the prompt for moon reflection questions.
    
    Args:
        transit_moon_sign: Current sign the Moon is in
        moon_phase: Current moon phase (neumond, vollmond, etc.)
        moon_aspects: Current aspects the Moon makes
        natal_moon_sign: User's natal Moon sign
        natal_moon_house: User's natal Moon house (1-12)
        natal_moon_aspects: User's natal Moon aspects
        moon_sign_context: Ontology context about the moon signs
    """
    
    # Translate moon phase to German description
    phase_descriptions = {
        "neumond": "Neumond - Zeit des Neubeginns und der Intention",
        "zunehmende_sichel": "Zunehmende Sichel - Zeit des Wachstums",
        "erstes_viertel": "Erstes Viertel - Zeit der Entscheidung",
        "zunehmender_mond": "Zunehmender Mond - Zeit der Manifestation",
        "vollmond": "Vollmond - Zeit der Vollendung und Klarheit",
        "abnehmender_mond": "Abnehmender Mond - Zeit der Integration",
        "letztes_viertel": "Letztes Viertel - Zeit des Loslassens",
        "abnehmende_sichel": "Abnehmende Sichel - Zeit der Stille und Reflexion",
    }
    
    phase_description = phase_descriptions.get(moon_phase, moon_phase.replace("_", " ").title())
    
    # Format house information
    house_text = f"im {natal_moon_house}. Haus" if natal_moon_house else "(Haus unbekannt)"
    
    # House meaning hints (simplified)
    house_themes = {
        1: "Selbstbild, Identität, Körper",
        2: "Werte, Besitz, Selbstwert",
        3: "Kommunikation, Lernen, Geschwister",
        4: "Familie, Wurzeln, Zuhause",
        5: "Kreativität, Freude, Kinder",
        6: "Alltag, Gesundheit, Dienst",
        7: "Beziehungen, Partnerschaft",
        8: "Transformation, gemeinsame Ressourcen",
        9: "Philosophie, Reisen, höheres Lernen",
        10: "Berufung, Öffentlichkeit, Status",
        11: "Freundschaft, Gruppen, Zukunftsvisionen",
        12: "Unbewusstes, Spiritualität, Rückzug",
    }
    
    house_theme = house_themes.get(natal_moon_house, "") if natal_moon_house else ""
    
    return f"""Erstelle personalisierte Mond-Reflexionsfragen.

**Aktuelle Mond-Position:**
- Mond in: {transit_moon_sign}
- Mondphase: {phase_description}
- Aktuelle Mond-Aspekte: {moon_aspects if moon_aspects else "Keine besonderen Aspekte"}

**Natal-Mond des Nutzers:**
- Mondzeichen: {natal_moon_sign}
- Position: {house_text}
{f"- Hausthemen: {house_theme}" if house_theme else ""}
- Natal-Aspekte: {natal_moon_aspects if natal_moon_aspects else "Keine Informationen"}

**Kontext zu den Mondzeichen:**
{moon_sign_context if moon_sign_context else "Kein zusätzlicher Kontext."}

---

**Aufgabe:**
Erstelle 3 tiefgründige Reflexionsfragen, die:

1. **Die aktuelle Mondenergie** ({transit_moon_sign}) mit dem **Natal-Mond** ({natal_moon_sign} {house_text}) verbinden

2. **Zur Mondphase passen:**
   - {phase_description}

3. **Emotionale Themen** ansprechen, die gerade aktiviert sein könnten

4. **Praktisch anwendbar** sind - der Nutzer sollte darüber nachdenken oder journalen können

**Format für jede Frage:**

🌙 **[Kurzer, poetischer Titel]**

[Die Reflexionsfrage - offen formuliert, einladend, 1-2 Sätze]

*[1-2 Sätze Hintergrund: Warum ist diese Frage jetzt relevant? Wie verbindet sie Transit-Mond und Natal-Mond?]*

---

**Beispiel-Qualität einer guten Frage:**

🌙 **Emotionale Wahrheit**

Wenn du dir erlauben würdest, ganz ehrlich mit deinen Gefühlen zu sein - was würdest du fühlen, das du vielleicht verdrängt hast?

*Der Mond in Skorpion (Transit) aktiviert dein emotionales Bedürfnis nach Sicherheit (Natal-Mond in Stier). Diese Spannung lädt dich ein, unter die Oberfläche zu schauen.*

---

**Hinweise:**
- Die Fragen sollten unterschiedliche Aspekte des Lebens berühren
- Mindestens eine Frage sollte sich auf die Mondphase beziehen
- Vermeide Ja/Nein-Fragen
- Halte die Fragen offen, nicht suggestiv
- Schreibe auf Deutsch, poetisch aber zugänglich"""



