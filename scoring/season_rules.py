"""Season-specific fantasy scoring parameters.

Scoring rules change between seasons (e.g. the captain multiplier), so
every parameter that can vary by season must be looked up here rather
than hardcoded in scoring functions.
"""

SEASON_RULES = {
    2023: {"captain_multiplier": 2.0},
    2024: {"captain_multiplier": 1.5},
    2025: {"captain_multiplier": 2.0},
    # 2026-27 season. Confirmed ruleset, verified against the live app at
    # season launch (captain x2) — no longer the July-2026 placeholder
    # assumption that the 2025-26 rules would just carry over.
    2026: {"captain_multiplier": 2.0},
}


def get_rules(season):
    rules = SEASON_RULES.get(season)
    if rules is None:
        raise ValueError(
            f"No scoring rules defined for season {season}. "
            f"Known seasons: {sorted(SEASON_RULES)}"
        )
    return rules
