# Klangrestaurator

Mac-App zur Restaurierung von Mitschnitten, in denen sich **Sprache und Musik abwechseln** –
z. B. Lesungen mit Cello, Vorträge mit musikalischen Zwischenspielen, Gottesdienste.

- **Sprache** wird mit einer Sprach-KI (DeepFilterNet3) entrauscht, Raumhall wird dabei mit reduziert.
  Optional gibt es eine zusätzliche klassische Hallreduktion (WPE).
- **Musik** wird nicht durch die Sprach-KI geschickt, weil die das Instrument zerstören würde.
  Stattdessen gibt es eine schonende spektrale Rauschminderung mit einem Rauschprofil aus den
  leisen Stellen in der Nähe. Der Raumklang bleibt erhalten.
- **Automatische Erkennung**, wo gesprochen und wo musiziert wird. In der Zeitleiste kannst du sie per Rechtsklick korrigieren.
- **A/B-Vorschau** an jeder Stelle, **Export als 24-bit WAV** (48 kHz oder Original-Samplerate).

## Starten

Voraussetzung ist macOS (Apple Silicon oder Intel). Das Python-Tool `uv` wird bei Bedarf automatisch installiert.

1. Doppelklick auf **`Klangrestaurator.command`**
   (beim ersten Mal: Rechtsklick → Öffnen, weil die Datei nicht signiert ist).
   Der erste Start richtet alles ein und dauert 1–2 Minuten.
2. Datei ins Fenster ziehen. Beim ersten Öffnen einer Datei wird die KI einmal über die ganze Aufnahme
   gerechnet (etwa ¼ der Spieldauer). Das Ergebnis wird zwischengespeichert, danach geht alles schnell.
3. Musikpassagen (orange) und Sprache (blau) prüfen und Fehler per Rechtsklick korrigieren.
4. „Vorschau berechnen“, mit Leertaste abspielen und mit Tab zwischen A/B umschalten.
5. „Exportieren…“

Beim allerersten Lauf wird das DeepFilterNet-Programm (~35 MB) von GitHub geladen und unter
`~/Library/Application Support/Klangrestaurator/` gespeichert.

## Voreinstellungen

| Preset     | Sprache        | Hallreduktion | Musik |
|------------|----------------|---------------|-------|
| Natürlich  | −20 dB         | –             | 70 %  |
| Kräftig    | −30 dB         | WPE           | 85 %  |
| Sanft      | −12 dB         | –             | 50 %  |

## Kommandozeile

```bash
uv run klangrestaurator-cli ZOOM0001.WAV                    # Natürlich, 48 kHz
uv run klangrestaurator-cli *.WAV --preset Kräftig --keep-rate
uv run klangrestaurator-cli ZOOM0001.WAV --segments         # nur Musikpassagen auflisten
```

## Als richtige .app bauen

```bash
./scripts/build_app.sh        # → dist/Klangrestaurator.app
```

## Zwischenspeicher

Analysen liegen unter `~/Library/Caches/Klangrestaurator/` (ca. 2× die Größe der Aufnahme als 48-kHz-Float).
Du kannst den Ordner jederzeit löschen.
