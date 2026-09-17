"""
Live geo-coordinate data enrichment.

Wires in two real, free, unauthenticated public APIs so the system can pull
regional ground truth for a location instead of relying purely on
user-typed numbers:

1. ISRIC SoilGrids (https://rest.isric.org) - global soil property predictions
   at 250m resolution. We pull soil organic carbon and pH at 0-5cm depth.
2. GBIF Occurrence API (https://api.gbif.org) - the Global Biodiversity
   Information Facility's species occurrence records. We use occurrence
   count and distinct species count within a bounding box as a proxy for
   observed species richness.

Both calls are wrapped in short timeouts with graceful fallback: if the
network is unavailable or the API errors, the caller falls back to
whatever the user supplied manually. This matters because the bonus
requirement is "geo-coordinates or spatial context," not "the system is
unusable without internet."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"

REQUEST_TIMEOUT_SECONDS = 6


@dataclass
class GeoEnrichment:
    latitude: float
    longitude: float
    soil_organic_carbon_pct: Optional[float] = None
    soil_ph: Optional[float] = None
    observed_species_count: Optional[int] = None
    occurrence_count: Optional[int] = None
    source_notes: Optional[str] = None


def fetch_soilgrids(lat: float, lon: float) -> dict:
    """
    Query ISRIC SoilGrids for soil organic carbon (soc) and pH (phh2o) at
    the 0-5cm depth interval. SoilGrids returns soc in dg/kg and phh2o in
    pH*10; both are converted to human-readable units.
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["soc", "phh2o"],
        "depth": "0-5cm",
        "value": "mean",
    }
    resp = requests.get(SOILGRIDS_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()

    result = {"soil_organic_carbon_pct": None, "soil_ph": None}
    layers = payload.get("properties", {}).get("layers", [])
    for layer in layers:
        name = layer.get("name")
        depths = layer.get("depths", [])
        if not depths:
            continue
        mean_value = depths[0].get("values", {}).get("mean")
        if mean_value is None:
            continue
        if name == "soc":
            # SoilGrids soc unit is dg/kg (decigrams per kg) -> convert to %
            result["soil_organic_carbon_pct"] = round(mean_value / 1000.0, 3)
        elif name == "phh2o":
            # SoilGrids phh2o unit is pH*10
            result["soil_ph"] = round(mean_value / 10.0, 2)
    return result


def fetch_gbif_richness(lat: float, lon: float, radius_deg: float = 0.25) -> dict:
    """
    Query GBIF occurrence records in a small bounding box around the
    coordinate and return the distinct species count and total occurrence
    count as a coarse, fast proxy for observed species richness. This is
    intentionally simple (a proper richness estimator would use rarefaction);
    it is presented as an indicative signal, not a precise index.
    """
    params = {
        "decimalLatitude": f"{lat - radius_deg},{lat + radius_deg}",
        "decimalLongitude": f"{lon - radius_deg},{lon + radius_deg}",
        "limit": 300,
        "hasCoordinate": "true",
    }
    resp = requests.get(GBIF_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()

    results = payload.get("results", [])
    species_keys = {r.get("speciesKey") for r in results if r.get("speciesKey")}
    return {
        "observed_species_count": len(species_keys),
        "occurrence_count": payload.get("count", len(results)),
    }


def enrich_from_coordinates(lat: float, lon: float) -> GeoEnrichment:
    """
    Best-effort enrichment: try both APIs independently, keep whatever
    succeeds, and never raise. Any partial or total failure is recorded in
    source_notes so the conversation layer can be transparent with the user
    about which numbers are live vs. missing.
    """
    enrichment = GeoEnrichment(latitude=lat, longitude=lon)
    notes = []

    try:
        soil = fetch_soilgrids(lat, lon)
        enrichment.soil_organic_carbon_pct = soil.get("soil_organic_carbon_pct")
        enrichment.soil_ph = soil.get("soil_ph")
        notes.append("soil: ISRIC SoilGrids v2.0 (0-5cm mean prediction)")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is best-effort enrichment
        logger.warning("SoilGrids lookup failed: %s", exc)
        notes.append("soil: SoilGrids unavailable, using manual input")

    try:
        richness = fetch_gbif_richness(lat, lon)
        enrichment.observed_species_count = richness.get("observed_species_count")
        enrichment.occurrence_count = richness.get("occurrence_count")
        notes.append("biodiversity: GBIF occurrence records, ~0.5deg bounding box")
    except Exception as exc:  # noqa: BLE001
        logger.warning("GBIF lookup failed: %s", exc)
        notes.append("biodiversity: GBIF unavailable, using manual input")

    enrichment.source_notes = "; ".join(notes)
    return enrichment
