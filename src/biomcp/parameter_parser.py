"""Parameter parsing and validation for BioMCP."""

import json
import logging
from typing import Any

from biomcp.exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


class ParameterParser:
    """Handles parameter parsing and validation for search requests."""

    @staticmethod
    def parse_list_param(
        param: str | list[str] | None, param_name: str
    ) -> list[str] | None:
        """Convert various input formats to lists.

        Handles:
        - JSON arrays: '["item1", "item2"]' -> ['item1', 'item2']
        - Comma-separated: 'item1, item2' -> ['item1', 'item2']
        - Single values: 'item' -> ['item']
        - None values: None -> None
        - Already parsed lists: ['item'] -> ['item']

        Args:
            param: The parameter to parse
            param_name: Name of the parameter for error messages

        Returns:
            Parsed list or None

        Raises:
            InvalidParameterError: If parameter cannot be parsed
        """
        if param is None:
            return None

        if isinstance(param, str):
            # First try to parse as JSON array
            if param.startswith("["):
                try:
                    parsed = json.loads(param)
                    if not isinstance(parsed, list):
                        raise InvalidParameterError(
                            param_name,
                            param,
                            "JSON array or comma-separated string",
                        )
                    return parsed
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Failed to parse {param_name} as JSON: {e}")

            # If it's a comma-separated string, split it
            if "," in param:
                return [item.strip() for item in param.split(",")]

            # Otherwise return as single-item list
            return [param]

        # If it's already a list, validate and return as-is
        if isinstance(param, list):
            # Validate all items are strings
            if not all(isinstance(item, str) for item in param):
                raise InvalidParameterError(
                    param_name, param, "list of strings"
                )
            return param

        # Invalid type
        raise InvalidParameterError(
            param_name, param, "string, list of strings, or None"
        )

    @staticmethod
    def normalize_phase(phase: str | None) -> str | None:
        """Normalize phase values for clinical trials.

        Converts various formats to standard enum values:
        - "Phase 3" -> "PHASE3"
        - "phase 3" -> "PHASE3"
        - "PHASE 3" -> "PHASE3"
        - "phase3" -> "PHASE3"
        - "PHASEI" -> "PHASE1"
        - "PHASE I" -> "PHASE1"
        - "PHASEII" -> "PHASE2"
        - "PHASE II" -> "PHASE2"
        - "PHASEIII" -> "PHASE3"
        - "PHASE III" -> "PHASE3"
        - "PHASEIV" -> "PHASE4"
        - "PHASE IV" -> "PHASE4"
        - "I" -> "PHASE1"
        - "II" -> "PHASE2"
        - "III" -> "PHASE3"
        - "IV" -> "PHASE4"
        - "1" -> "PHASE1"
        - "2" -> "PHASE2"
        - "3" -> "PHASE3"
        - "4" -> "PHASE4"

        Args:
            phase: Phase value to normalize

        Returns:
            Normalized phase value or None
        """
        if phase is None:
            return None

        # Convert to uppercase and remove spaces and parentheses
        normalized = phase.upper().replace(" ", "").strip("()")

        # Map Roman numerals to Arabic numerals
        roman_mapping = {
            "PHASEI": "PHASE1",
            "PHASEII": "PHASE2",
            "PHASEIII": "PHASE3",
            "PHASEIV": "PHASE4",
            "I": "PHASE1",
            "II": "PHASE2",
            "III": "PHASE3",
            "IV": "PHASE4",
            "1": "PHASE1",
            "2": "PHASE2",
            "3": "PHASE3",
            "4": "PHASE4",
        }

        # Check if it's a Roman numeral format or standalone number
        if normalized in roman_mapping:
            normalized = roman_mapping[normalized]
        # If it starts with PHASE followed by a Roman numeral or number, normalize it
        elif normalized.startswith("PHASE"):
            # Extract the part after "PHASE"
            phase_suffix = normalized[5:]  # Remove "PHASE" prefix
            if phase_suffix in ["I", "II", "III", "IV", "1", "2", "3", "4"]:
                # Map Roman numerals to numbers
                suffix_mapping = {
                    "I": "1", "II": "2", "III": "3", "IV": "4",
                    "1": "1", "2": "2", "3": "3", "4": "4"
                }
                normalized = f"PHASE{suffix_mapping.get(phase_suffix, phase_suffix)}"

        # Validate it matches expected pattern
        valid_phases = [
            "EARLY_PHASE1",
            "PHASE1",
            "PHASE2",
            "PHASE3",
            "PHASE4",
            "NOT_APPLICABLE",
        ]
        if normalized not in valid_phases:
            # Try to be helpful with common mistakes
            if "EARLY" in normalized and ("1" in normalized or "I" in normalized):
                return "EARLY_PHASE1"
            if "NOT" in normalized and "APPLICABLE" in normalized:
                return "NOT_APPLICABLE"

            raise InvalidParameterError(
                "phase", phase, f"one of: {', '.join(valid_phases)}"
            )

        return normalized

    @staticmethod
    def validate_page_params(page: int, page_size: int) -> tuple[int, int]:
        """Validate pagination parameters.

        Args:
            page: Page number (minimum 1)
            page_size: Results per page (1-100)

        Returns:
            Validated (page, page_size) tuple

        Raises:
            InvalidParameterError: If parameters are invalid
        """
        if page < 1:
            raise InvalidParameterError("page", page, "integer >= 1")

        if page_size < 1 or page_size > 100:
            raise InvalidParameterError(
                "page_size", page_size, "integer between 1 and 100"
            )

        return page, page_size

    @staticmethod
    def parse_search_params(
        params: dict[str, Any], domain: str
    ) -> dict[str, Any]:
        """Parse and validate all search parameters for a domain.

        Args:
            params: Raw parameters dictionary
            domain: Domain being searched

        Returns:
            Validated parameters dictionary
        """
        parsed: dict[str, Any] = {}

        # Common list parameters
        list_params = [
            "genes",
            "diseases",
            "variants",
            "chemicals",
            "keywords",
            "conditions",
            "interventions",
        ]

        for param_name in list_params:
            if param_name in params and params[param_name] is not None:
                parsed[param_name] = ParameterParser.parse_list_param(
                    params[param_name], param_name
                )

        # Domain-specific parameters
        if (
            domain == "trial"
            and "phase" in params
            and params.get("phase") is not None
        ):
            parsed["phase"] = ParameterParser.normalize_phase(
                params.get("phase")
            )

        # Pass through other parameters
        for key, value in params.items():
            if key not in parsed and key not in list_params and key != "phase":
                parsed[key] = value

        return parsed
