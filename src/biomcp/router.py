"""Unified search and fetch tools for BioMCP.

This module provides the main MCP tools for searching and fetching biomedical data
across different domains (articles, trials, variants) with integrated sequential
thinking capabilities.
"""

import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from biomcp.constants import (
    DEFAULT_PAGE_NUMBER,
    DEFAULT_PAGE_SIZE,
    DEFAULT_TITLE,
    ERROR_DOMAIN_REQUIRED,
    ESTIMATED_ADDITIONAL_RESULTS,
    MAX_RESULTS_PER_DOMAIN_DEFAULT,
    TRIAL_DETAIL_SECTIONS,
    VALID_DOMAINS,
)
from biomcp.core import mcp_app
from biomcp.domain_handlers import get_domain_handler
from biomcp.exceptions import (
    InvalidDomainError,
    InvalidParameterError,
    QueryParsingError,
    ResultParsingError,
    SearchExecutionError,
)
from biomcp.integrations.biothings_client import BioThingsClient
from biomcp.metrics import track_performance
from biomcp.parameter_parser import ParameterParser
from biomcp.query_parser import QueryParser
from biomcp.query_router import QueryRouter, execute_routing_plan
from biomcp.thinking_tracker import get_thinking_reminder
from biomcp.trials import getter as trial_getter

logger = logging.getLogger(__name__)


def format_results(
    results: list[dict], domain: str, page: int, page_size: int, total: int
) -> dict:
    """Format search results according to OpenAI MCP search semantics.

    Converts domain-specific result formats into a standardized structure with:
    - id: Unique identifier for the result (required)
    - title: Human-readable title (required)
    - text: Brief preview or summary of the content (required)
    - url: Link to the full resource (optional but recommended for citations)

    Note: The OpenAI MCP specification does NOT require metadata in search results.
    Metadata should only be included in fetch results.

    Args:
        results: Raw results from domain-specific search
        domain: Type of results ('article', 'trial', or 'variant')
        page: Current page number (for internal tracking only)
        page_size: Number of results per page (for internal tracking only)
        total: Total number of results available (for internal tracking only)

    Returns:
        Dictionary with results array following OpenAI MCP format:
        {"results": [{"id", "title", "text", "url"}, ...]}

    Raises:
        InvalidDomainError: If domain is not recognized
    """
    logger.debug(f"Formatting {len(results)} results for domain: {domain}")

    formatted_data = []

    # Get the appropriate handler
    try:
        handler_class = get_domain_handler(domain)
    except ValueError:
        raise InvalidDomainError(domain, VALID_DOMAINS) from None

    # Format each result
    for result in results:
        try:
            formatted_result = handler_class.format_result(result)
            # Ensure the result has the required OpenAI MCP fields
            openai_result = {
                "id": formatted_result.get("id", ""),
                "title": formatted_result.get("title", DEFAULT_TITLE),
                "text": formatted_result.get(
                    "snippet", formatted_result.get("text", "")
                ),
                "url": formatted_result.get("url", ""),
            }
            # Note: OpenAI MCP spec doesn't require metadata in search results
            # Only include it if explicitly needed for enhanced functionality
            formatted_data.append(openai_result)
        except Exception as e:
            logger.warning(f"Failed to format result in domain {domain}: {e}")
            # Skip malformed results
            continue

    # Add thinking reminder if needed (as first result)
    reminder = get_thinking_reminder()
    if reminder and formatted_data:
        reminder_result = {
            "id": "thinking-reminder",
            "title": "⚠️ Research Best Practice Reminder",
            "text": reminder,
            "url": "",
        }
        formatted_data.insert(0, reminder_result)

    # Return OpenAI MCP compliant format
    return {"results": formatted_data}


# ────────────────────────────
# Unified SEARCH tool
class BioDomainSearchInput(BaseModel):
    """Input schema for biomedical literature and clinical trial searches."""

    query: str = Field(
        description=(
            "MANDATORY FIELD-BASED QUERY SYNTAX. Query MUST contain field prefixes (disease:, gene:, chemical:, trials.) or it will return ZERO results.\n\n"
            "✅ CORRECT EXAMPLES:\n"
            "- disease:\"mild cognitive impairment\" AND \"multicenter trial\" AND recruitment\n"
            "- gene:BRAF AND disease:melanoma AND (resistance OR resistant)\n"
            "- chemical:pembrolizumab AND trials.phase:3\n"
            "- trials.condition:diabetes AND trials.intervention:metformin\n\n"
            "❌ WRONG (will return NO results):\n"
            "- \"regulatory compliance AND multi-center trials\" ← Missing disease: prefix\n"
            "- \"BRAF mutations in melanoma\" ← Missing gene: prefix\n\n"
            "REQUIRED SYNTAX RULES:\n"
            "1. Start with field prefix: disease:, gene:, chemical:, drug:, or trials.\n"
            "2. Quote multi-word phrases: disease:\"mild cognitive impairment\" not disease:mild cognitive impairment\n"
            "3. Use uppercase AND/OR: disease:MCI AND recruitment, not disease:MCI and recruitment\n"
            "4. Keep focused (3-5 concepts): Avoid 7+ AND terms\n\n"
            "FIELD PREFIX REFERENCE:\n"
            "- disease:NAME - For disease/condition research (e.g., disease:MCI, disease:\"Alzheimer's disease\")\n"
            "- gene:SYMBOL - For gene research (e.g., gene:BRAF, gene:TP53)\n"
            "- chemical:NAME or drug:NAME - For drug research (e.g., chemical:pembrolizumab)\n"
            "- trials.condition:NAME - For trial condition filters\n"
            "- trials.intervention:NAME - For trial intervention filters\n"
            "- trials.phase:N - For trial phase (1, 2, 3, 4)\n"
            "- articles.date:YYYY-MM-DD..YYYY-MM-DD - For date ranges\n\n"
            "QUERY TEMPLATES BY SCENARIO:\n"
            "Multicenter trials: disease:\"[condition]\" AND \"multicenter trial\" AND (\"regulatory compliance\" OR harmonization)\n"
            "Recruitment challenges: disease:\"[condition]\" AND recruitment AND \"diverse populations\" AND (global OR multinational)\n"
            "Gene-disease research: gene:[SYMBOL] AND disease:\"[condition]\" AND (mechanism OR pathway)\n"
            "Drug efficacy: chemical:\"[drug]\" AND disease:\"[condition]\" AND trials.phase:[N]\n\n"
            "REMEMBER: Without field prefix (disease:, gene:, etc.), query routing fails and returns ZERO results."
        )
    )

    api_key: str | None = Field(
        default=None,
        description="NCI API key for NCI-specific domain searches. Get free key at: https://clinicaltrialsapi.cancer.gov/"
    )


# ────────────────────────────
@mcp_app.tool()
@track_performance("biomcp.search")
async def biodomain_search(  # noqa: C901
    query: str,
    api_key: str | None = None
) -> dict:
    """Search biomedical literature, clinical trials, genetic variants, genes, drugs, and diseases.

    This tool searches across PubMed/PubTator3, ClinicalTrials.gov, MyVariant.info, and BioThings databases.

    ⚠️ CRITICAL: Query parameter MUST use field-based syntax (see query field description for details).
    Queries without field prefixes (disease:, gene:, chemical:, trials.) will return ZERO results.

    Returns results in format: {"results": [{"id", "title", "text", "url"}, ...]}
    """
    logger.info(f"Search called with query={query}")

    # Determine search mode
    if query and query.strip():
        # Validate query has field syntax - critical for proper routing
        required_field_prefixes = [
            "gene:", "disease:", "chemical:", "drug:",
            "trials.", "articles.", "variants."
        ]
        has_field_prefix = any(field in query for field in required_field_prefixes)

        if not has_field_prefix:
            logger.warning(f"Query missing required field prefix: {query}")
            return {
                "results": [],
                "error": "Invalid query syntax: Missing field prefix",
                "message": (
                    "Query MUST contain at least one field prefix for proper routing. "
                    "Without field prefixes (disease:, gene:, chemical:, trials., etc.), "
                    "the query will not route to appropriate databases and will return no results."
                ),
                "examples": {
                    "disease_research": "disease:\"mild cognitive impairment\" AND recruitment",
                    "gene_research": "gene:BRAF AND disease:melanoma",
                    "trial_search": "trials.condition:diabetes AND trials.phase:3",
                    "drug_research": "chemical:pembrolizumab AND disease:melanoma"
                },
                "your_query": query,
                "hint": (
                    "Start your query with one of: disease:, gene:, chemical:, drug:, "
                    "trials.condition:, trials.intervention:, articles.title:, variants.gene:"
                )
            }

        # Check if this is a unified query (contains field syntax like "gene:" or "AND")
        is_unified_query = any(
            marker in query for marker in [":", " AND ", " OR "]
        )

        logger.info(f"Using unified query mode: {query}")
        return await _unified_search(
            query=query,
            max_results_per_domain=MAX_RESULTS_PER_DOMAIN_DEFAULT,
            domains=None,
        )

# ────────────────────────────
# Unified FETCH tool
# ────────────────────────────
@mcp_app.tool()
@track_performance("biomcp.fetch")
async def biodomain_fetch(  # noqa: C901
    id: Annotated[  # noqa: A002
        str,
        "PMID / NCT ID / Variant ID / DOI / Gene ID / Drug ID / Disease ID / NCI Organization ID / NCI Intervention ID / NCI Disease ID / FDA Report ID / FDA Set ID / FDA MDR Key / FDA Application Number / FDA Recall Number",
    ],
    domain: Annotated[
        Literal[
            "article",
            "trial",
            "variant",
            "gene",
            "drug",
            "disease",
            "nci_organization",
            "nci_intervention",
            "nci_biomarker",
            "nci_disease",
            "fda_adverse",
            "fda_label",
            "fda_device",
            "fda_approval",
            "fda_recall",
            "fda_shortage",
        ]
        | None,
        Field(
            description="Domain of the record (auto-detected if not provided)"
        ),
    ] = None,
    call_benefit: Annotated[
        str | None,
        Field(
            description="Brief explanation of why this fetch is being performed and expected benefit. Helps provide context for analytics and improves result relevance."
        ),
    ] = None,
    detail: Annotated[
        Literal[
            "protocol", "locations", "outcomes", "references", "all", "full"
        ]
        | None,
        "Specific section to retrieve (trials) or 'full' (articles)",
    ] = None,
    api_key: Annotated[
        str | None,
        Field(
            description="NCI API key for fetching NCI records (nci_organization, nci_intervention, nci_disease). Required for NCI fetches. Get a free key at: https://clinicaltrialsapi.cancer.gov/"
        ),
    ] = None,
) -> dict:
    """Fetch comprehensive details for a specific biomedical record.

    This tool retrieves full information for articles, clinical trials, genetic variants,
    genes, drugs, or diseases using their unique identifiers. It returns data in a
    standardized format suitable for detailed analysis and research.

    ## IDENTIFIER FORMATS:
    - Articles: PMID (PubMed ID) - e.g., "35271234" OR DOI - e.g., "10.1101/2024.01.20.23288905"
    - Trials: NCT ID (ClinicalTrials.gov ID) - e.g., "NCT04280705"
    - Variants: HGVS notation or dbSNP ID - e.g., "chr7:g.140453136A>T" or "rs121913254"
    - Genes: Gene symbol or Entrez ID - e.g., "BRAF" or "673"
    - Drugs: Drug name or ID - e.g., "imatinib" or "DB00619"
    - Diseases: Disease name or ID - e.g., "melanoma" or "MONDO:0005105"
    - NCI Organizations: NCI organization ID - e.g., "NCI-2011-03337"
    - NCI Interventions: NCI intervention ID - e.g., "INT123456"
    - NCI Diseases: NCI disease ID - e.g., "C4872"

    The domain is automatically detected from the ID format if not provided:
    - NCT* → trial
    - Contains "/" with numeric prefix (DOI) → article
    - Pure numeric → article (PMID)
    - rs* or contains ':' or 'g.' → variant
    - For genes, drugs, diseases: manual specification recommended

    ## DOMAIN-SPECIFIC OPTIONS:

    ### Articles (domain="article"):
    - Returns full article metadata, abstract, and full text when available
    - Supports both PubMed articles (via PMID) and Europe PMC preprints (via DOI)
    - Includes annotations for genes, diseases, chemicals, and variants (PubMed only)
    - detail="full" attempts to retrieve full text content (PubMed only)

    ### Clinical Trials (domain="trial"):
    - detail=None or "protocol": Core study information
    - detail="locations": Study sites and contact information
    - detail="outcomes": Primary/secondary outcomes and results
    - detail="references": Related publications and citations
    - detail="all": Complete trial record with all sections

    ### Variants (domain="variant"):
    - Returns comprehensive variant information including:
      - Clinical significance and interpretations
      - Population frequencies
      - Gene/protein effects
      - External database links
    - detail parameter is ignored (always returns full data)

    ### Genes (domain="gene"):
    - Returns gene information from MyGene.info including:
      - Gene symbol, name, and type
      - Entrez ID and Ensembl IDs
      - Gene summary and aliases
      - RefSeq information
    - detail parameter is ignored (always returns full data)

    ### Drugs (domain="drug"):
    - Returns drug/chemical information from MyChem.info including:
      - Drug name and trade names
      - Chemical formula and structure IDs
      - Clinical indications
      - Mechanism of action
      - External database links (DrugBank, PubChem, ChEMBL)
    - detail parameter is ignored (always returns full data)

    ### Diseases (domain="disease"):
    - Returns disease information from MyDisease.info including:
      - Disease name and definition
      - MONDO ontology ID
      - Disease synonyms
      - Cross-references to other databases
      - Associated phenotypes
    - detail parameter is ignored (always returns full data)

    ### NCI Organizations (domain="nci_organization"):
    - Returns organization information from NCI database including:
      - Organization name and type
      - Full address and contact information
      - Research focus areas
      - Associated clinical trials
    - Requires NCI API key
    - detail parameter is ignored (always returns full data)

    ### NCI Interventions (domain="nci_intervention"):
    - Returns intervention information from NCI database including:
      - Intervention name and type
      - Synonyms and alternative names
      - Mechanism of action (for drugs)
      - FDA approval status
      - Associated clinical trials
    - Requires NCI API key
    - detail parameter is ignored (always returns full data)

    ### NCI Diseases (domain="nci_disease"):
    - Returns disease information from NCI controlled vocabulary including:
      - Preferred disease name
      - Disease category and classification
      - All known synonyms
      - Cross-reference codes (ICD, SNOMED)
    - Requires NCI API key
    - detail parameter is ignored (always returns full data)

    ## RETURN FORMAT:
    All fetch operations return a standardized format:
    ```json
    {
        "id": "unique_identifier",
        "title": "Record title or name",
        "text": "Full content or comprehensive description",
        "url": "Link to original source",
        "metadata": {
            // Domain-specific additional fields
        }
    }
    ```

    ## EXAMPLES:

    Fetch article by PMID (domain auto-detected):
    ```
    await fetch(id="35271234")
    ```

    Fetch article by DOI (domain auto-detected):
    ```
    await fetch(id="10.1101/2024.01.20.23288905")
    ```

    Fetch complete trial information (domain auto-detected):
    ```
    await fetch(
        id="NCT04280705",
        detail="all"
    )
    ```

    Fetch variant with clinical interpretations:
    ```
    await fetch(id="rs121913254")
    ```

    Explicitly specify domain (optional):
    ```
    await fetch(
        domain="variant",
        id="chr7:g.140453136A>T"
    )
    ```
    """
    # Auto-detect domain if not provided
    if domain is None:
        # Try to infer domain from ID format
        if id.upper().startswith("NCT"):
            domain = "trial"
            logger.info(f"Auto-detected domain 'trial' from NCT ID: {id}")
        elif "/" in id and id.split("/")[0].replace(".", "").isdigit():
            # DOI format (e.g., 10.1038/nature12373) - treat as article
            domain = "article"
            logger.info(f"Auto-detected domain 'article' from DOI: {id}")
        elif id.isdigit():
            # Numeric ID - likely PMID
            domain = "article"
            logger.info(
                f"Auto-detected domain 'article' from numeric ID: {id}"
            )
        elif id.startswith("rs") or ":" in id or "g." in id:
            # rsID or HGVS notation
            domain = "variant"
            logger.info(f"Auto-detected domain 'variant' from ID format: {id}")
        else:
            # Default to article if we can't determine
            domain = "article"
            logger.warning(
                f"Could not auto-detect domain for ID '{id}', defaulting to 'article'"
            )

    logger.info(f"Fetch called for {domain} with id={id}, detail={detail}")

    if domain == "article":
        logger.debug("Fetching article details")
        try:
            from biomcp.articles.fetch import _article_details

            # The _article_details function handles both PMIDs and DOIs
            result_str = await _article_details(
                call_benefit=call_benefit
                or "Fetching article details via MCP tool",
                pmid=id,
            )
        except Exception as e:
            logger.error(f"Article fetch failed: {e}")
            raise SearchExecutionError("article", e) from e

        # Parse and return the first article
        try:
            articles = (
                json.loads(result_str)
                if isinstance(result_str, str)
                else result_str
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse article fetch results: {e}")
            raise ResultParsingError("article", e) from e

        if not articles:
            return {"error": "Article not found"}

        article = articles[0]

        # Check if the article is actually an error response
        if "error" in article:
            return {"error": article["error"]}

        # Format according to OpenAI MCP standard
        full_text = article.get("full_text", "")
        abstract = article.get("abstract", "")
        text_content = full_text if full_text else abstract

        return {
            "id": str(article.get("pmid", id)),
            "title": article.get("title", DEFAULT_TITLE),
            "text": text_content,
            "url": article.get(
                "url", f"https://pubmed.ncbi.nlm.nih.gov/{id}/"
            ),
            "metadata": {
                "pmid": article.get("pmid"),
                "journal": article.get("journal"),
                "authors": article.get("authors"),
                "year": article.get("year"),
                "doi": article.get("doi"),
                "annotations": article.get("annotations", {}),
                "is_preprint": article.get("is_preprint", False),
                "preprint_source": article.get("preprint_source"),
            },
        }

    elif domain == "trial":
        logger.debug(f"Fetching trial details for section: {detail}")

        # Validate detail parameter
        if detail is not None and detail not in TRIAL_DETAIL_SECTIONS:
            raise InvalidParameterError(
                "detail",
                detail,
                f"one of: {', '.join(TRIAL_DETAIL_SECTIONS)} or None",
            )

        try:
            # Always fetch protocol for basic info - get JSON format
            protocol_json = await trial_getter.get_trial(
                nct_id=id,
                module=trial_getter.Module.PROTOCOL,
                output_json=True,
            )

            # Parse the JSON response
            try:
                protocol_data = json.loads(protocol_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse protocol JSON for {id}: {e}")
                return {
                    "id": id,
                    "title": f"Clinical Trial {id}",
                    "text": f"Error parsing trial data: {e}",
                    "url": f"https://clinicaltrials.gov/study/{id}",
                    "metadata": {
                        "nct_id": id,
                        "error": f"JSON parse error: {e}",
                    },
                }

            # Check for errors in the response
            if "error" in protocol_data:
                return {
                    "id": id,
                    "title": f"Clinical Trial {id}",
                    "text": protocol_data.get(
                        "details",
                        protocol_data.get("error", "Trial not found"),
                    ),
                    "url": f"https://clinicaltrials.gov/study/{id}",
                    "metadata": {
                        "nct_id": id,
                        "error": protocol_data.get("error"),
                    },
                }

            # Build comprehensive text description
            text_parts = []

            # Extract protocol section data from the API response
            protocol_section = protocol_data.get("protocolSection", {})

            # Extract basic info from the protocol section
            id_module = protocol_section.get("identificationModule", {})
            status_module = protocol_section.get("statusModule", {})
            desc_module = protocol_section.get("descriptionModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            design_module = protocol_section.get("designModule", {})
            arms_module = protocol_section.get("armsInterventionsModule", {})

            # Add basic protocol info to text
            title = id_module.get("briefTitle", f"Clinical Trial {id}")
            text_parts.append(f"Study Title: {title}")

            # Conditions
            conditions = conditions_module.get("conditions", [])
            if conditions:
                text_parts.append(f"\nConditions: {', '.join(conditions)}")

            # Interventions
            interventions = []
            for intervention in arms_module.get("interventions", []):
                interventions.append(intervention.get("name", ""))
            if interventions:
                text_parts.append(f"Interventions: {', '.join(interventions)}")

            # Phase
            phases = design_module.get("phases", [])
            if phases:
                text_parts.append(f"Phase: {', '.join(phases)}")

            # Status
            overall_status = status_module.get("overallStatus", "N/A")
            text_parts.append(f"Status: {overall_status}")

            # Summary
            brief_summary = desc_module.get(
                "briefSummary", "No summary available"
            )
            text_parts.append(f"\nSummary: {brief_summary}")

            # Prepare metadata
            metadata = {"nct_id": id, "protocol": protocol_data}

            if detail in ("all", "locations", "outcomes", "references"):
                # Fetch additional sections as needed
                if detail == "all" or detail == "locations":
                    try:
                        locations_json = await trial_getter.get_trial(
                            nct_id=id,
                            module=trial_getter.Module.LOCATIONS,
                            output_json=True,
                        )
                        locations_data = json.loads(locations_json)
                        if "error" not in locations_data:
                            # Extract locations from the protocol section
                            locations_module = locations_data.get(
                                "protocolSection", {}
                            ).get("contactsLocationsModule", {})
                            locations_list = locations_module.get(
                                "locations", []
                            )
                            metadata["locations"] = locations_list
                            if locations_list:
                                text_parts.append(
                                    f"\n\nLocations: {len(locations_list)} study sites"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch locations for {id}: {e}"
                        )
                        metadata["locations"] = []

                if detail == "all" or detail == "outcomes":
                    try:
                        outcomes_json = await trial_getter.get_trial(
                            nct_id=id,
                            module=trial_getter.Module.OUTCOMES,
                            output_json=True,
                        )
                        outcomes_data = json.loads(outcomes_json)
                        if "error" not in outcomes_data:
                            # Extract outcomes from the protocol section
                            outcomes_module = outcomes_data.get(
                                "protocolSection", {}
                            ).get("outcomesModule", {})
                            primary_outcomes = outcomes_module.get(
                                "primaryOutcomes", []
                            )
                            secondary_outcomes = outcomes_module.get(
                                "secondaryOutcomes", []
                            )
                            metadata["outcomes"] = {
                                "primary_outcomes": primary_outcomes,
                                "secondary_outcomes": secondary_outcomes,
                            }
                            if primary_outcomes:
                                text_parts.append(
                                    f"\n\nPrimary Outcomes: {len(primary_outcomes)} measures"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch outcomes for {id}: {e}"
                        )
                        metadata["outcomes"] = {}

                if detail == "all" or detail == "references":
                    try:
                        references_json = await trial_getter.get_trial(
                            nct_id=id,
                            module=trial_getter.Module.REFERENCES,
                            output_json=True,
                        )
                        references_data = json.loads(references_json)
                        if "error" not in references_data:
                            # Extract references from the protocol section
                            references_module = references_data.get(
                                "protocolSection", {}
                            ).get("referencesModule", {})
                            references_list = references_module.get(
                                "references", []
                            )
                            metadata["references"] = references_list
                            if references_list:
                                text_parts.append(
                                    f"\n\nReferences: {len(references_list)} publications"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch references for {id}: {e}"
                        )
                        metadata["references"] = []

            # Return OpenAI MCP compliant format
            return {
                "id": id,
                "title": title,
                "text": "\n".join(text_parts),
                "url": f"https://clinicaltrials.gov/study/{id}",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Trial fetch failed: {e}")
            raise SearchExecutionError("trial", e) from e

    elif domain == "variant":
        logger.debug("Fetching variant details")
        try:
            from biomcp.variants.getter import get_variant

            result_str = await get_variant(
                variant_id=id,
                output_json=True,
                include_external=True,
            )
        except Exception as e:
            logger.error(f"Variant fetch failed: {e}")
            raise SearchExecutionError("variant", e) from e

        try:
            variant_response = (
                json.loads(result_str)
                if isinstance(result_str, str)
                else result_str
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse variant fetch results: {e}")
            raise ResultParsingError("variant", e) from e

        # get_variant returns a list, extract the first variant
        if isinstance(variant_response, list) and variant_response:
            variant_data = variant_response[0]
        elif isinstance(variant_response, dict):
            variant_data = variant_response
        else:
            return {"error": "Variant not found"}

        # Build comprehensive text description
        text_parts = []

        # Basic variant info
        text_parts.append(f"Variant: {variant_data.get('_id', id)}")

        # Gene information
        if variant_data.get("gene"):
            gene_info = variant_data["gene"]
            text_parts.append(
                f"\nGene: {gene_info.get('symbol', 'Unknown')} ({gene_info.get('name', '')})"
            )

        # Clinical significance
        if variant_data.get("clinvar"):
            clinvar = variant_data["clinvar"]
            if clinvar.get("clinical_significance"):
                text_parts.append(
                    f"\nClinical Significance: {clinvar['clinical_significance']}"
                )
            if clinvar.get("review_status"):
                text_parts.append(f"Review Status: {clinvar['review_status']}")

        # dbSNP info
        if variant_data.get("dbsnp"):
            dbsnp = variant_data["dbsnp"]
            if dbsnp.get("rsid"):
                text_parts.append(f"\ndbSNP: {dbsnp['rsid']}")

        # CADD scores
        if variant_data.get("cadd"):
            cadd = variant_data["cadd"]
            if cadd.get("phred"):
                text_parts.append(f"\nCADD Score: {cadd['phred']}")

        # Allele frequencies
        if variant_data.get("gnomad_exome"):
            gnomad = variant_data["gnomad_exome"]
            if gnomad.get("af", {}).get("af"):
                text_parts.append(
                    f"\nGnomAD Allele Frequency: {gnomad['af']['af']:.6f}"
                )

        # External links
        if variant_data.get("external_links"):
            links = variant_data["external_links"]
            text_parts.append(
                f"\n\nExternal Resources: {len(links)} database links available"
            )

        # Check for external data indicators
        if variant_data.get("tcga"):
            text_parts.append("\n\nTCGA Data: Available")
        if variant_data.get("1000genomes"):
            text_parts.append("\n1000 Genomes Data: Available")

        # Determine best URL
        url = variant_data.get("url", "")
        if not url and variant_data.get("dbsnp", {}).get("rsid"):
            url = f"https://www.ncbi.nlm.nih.gov/snp/{variant_data['dbsnp']['rsid']}"
        elif not url:
            url = f"https://myvariant.info/v1/variant/{id}"

        # Return OpenAI MCP compliant format
        return {
            "id": variant_data.get("_id", id),
            "title": f"Variant {variant_data.get('_id', id)}",
            "text": "\n".join(text_parts),
            "url": url,
            "metadata": variant_data,
        }

    elif domain == "gene":
        logger.debug("Fetching gene details")
        try:
            client = BioThingsClient()
            gene_info = await client.get_gene_info(id)

            if not gene_info:
                return {"error": f"Gene {id} not found"}

            # Build comprehensive text description
            text_parts = []
            text_parts.append(f"Gene: {gene_info.symbol} ({gene_info.name})")

            if gene_info.entrezgene:
                text_parts.append(f"\nEntrez ID: {gene_info.entrezgene}")

            if gene_info.type_of_gene:
                text_parts.append(f"Type: {gene_info.type_of_gene}")

            if gene_info.summary:
                text_parts.append(f"\nSummary: {gene_info.summary}")

            if gene_info.alias:
                text_parts.append(f"\nAliases: {', '.join(gene_info.alias)}")

            # URL
            url = (
                f"https://www.genenames.org/data/gene-symbol-report/#!/symbol/{gene_info.symbol}"
                if gene_info.symbol
                else ""
            )

            # Return OpenAI MCP compliant format
            return {
                "id": str(gene_info.gene_id),
                "title": f"{gene_info.symbol}: {gene_info.name}"
                if gene_info.symbol and gene_info.name
                else gene_info.symbol or gene_info.name or DEFAULT_TITLE,
                "text": "\n".join(text_parts),
                "url": url,
                "metadata": gene_info.model_dump(),
            }

        except Exception as e:
            logger.error(f"Gene fetch failed: {e}")
            raise SearchExecutionError("gene", e) from e

    elif domain == "drug":
        logger.debug("Fetching drug details")
        try:
            client = BioThingsClient()
            drug_info = await client.get_drug_info(id)

            if not drug_info:
                return {"error": f"Drug {id} not found"}

            # Build comprehensive text description
            text_parts = []
            text_parts.append(f"Drug: {drug_info.name}")

            if drug_info.drugbank_id:
                text_parts.append(f"\nDrugBank ID: {drug_info.drugbank_id}")

            if drug_info.formula:
                text_parts.append(f"Formula: {drug_info.formula}")

            if drug_info.tradename:
                text_parts.append(
                    f"\nTrade Names: {', '.join(drug_info.tradename)}"
                )

            if drug_info.description:
                text_parts.append(f"\nDescription: {drug_info.description}")

            if drug_info.indication:
                text_parts.append(f"\nIndication: {drug_info.indication}")

            if drug_info.mechanism_of_action:
                text_parts.append(
                    f"\nMechanism of Action: {drug_info.mechanism_of_action}"
                )

            # URL
            url = ""
            if drug_info.drugbank_id:
                url = f"https://www.drugbank.ca/drugs/{drug_info.drugbank_id}"
            elif drug_info.pubchem_cid:
                url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{drug_info.pubchem_cid}"

            # Return OpenAI MCP compliant format
            return {
                "id": drug_info.drug_id,
                "title": drug_info.name or drug_info.drug_id or DEFAULT_TITLE,
                "text": "\n".join(text_parts),
                "url": url,
                "metadata": drug_info.model_dump(),
            }

        except Exception as e:
            logger.error(f"Drug fetch failed: {e}")
            raise SearchExecutionError("drug", e) from e

    elif domain == "disease":
        logger.debug("Fetching disease details")
        try:
            client = BioThingsClient()
            disease_info = await client.get_disease_info(id)

            if not disease_info:
                return {"error": f"Disease {id} not found"}

            # Build comprehensive text description
            text_parts = []
            text_parts.append(f"Disease: {disease_info.name}")

            if disease_info.mondo and isinstance(disease_info.mondo, dict):
                mondo_id = disease_info.mondo.get("id")
                if mondo_id:
                    text_parts.append(f"\nMONDO ID: {mondo_id}")

            if disease_info.definition:
                text_parts.append(f"\nDefinition: {disease_info.definition}")

            if disease_info.synonyms:
                text_parts.append(
                    f"\nSynonyms: {', '.join(disease_info.synonyms[:5])}"
                )
                if len(disease_info.synonyms) > 5:
                    text_parts.append(
                        f"  ... and {len(disease_info.synonyms) - 5} more"
                    )

            if disease_info.phenotypes:
                text_parts.append(
                    f"\nAssociated Phenotypes: {len(disease_info.phenotypes)}"
                )

            # URL
            url = ""
            if disease_info.mondo and isinstance(disease_info.mondo, dict):
                mondo_id = disease_info.mondo.get("id")
                if mondo_id:
                    url = f"https://monarchinitiative.org/disease/{mondo_id}"

            # Return OpenAI MCP compliant format
            return {
                "id": disease_info.disease_id,
                "title": disease_info.name
                or disease_info.disease_id
                or DEFAULT_TITLE,
                "text": "\n".join(text_parts),
                "url": url,
                "metadata": disease_info.model_dump(),
            }

        except Exception as e:
            logger.error(f"Disease fetch failed: {e}")
            raise SearchExecutionError("disease", e) from e

    elif domain == "nci_organization":
        logger.debug("Fetching NCI organization details")
        try:
            from biomcp.organizations import get_organization
            from biomcp.organizations.getter import format_organization_details

            org_data = await get_organization(
                org_id=id,
                api_key=api_key,
            )

            # Format the details
            formatted_text = format_organization_details(org_data)

            # Return OpenAI MCP compliant format
            return {
                "id": id,
                "title": org_data.get("name", "Unknown Organization"),
                "text": formatted_text,
                "url": "",  # NCI doesn't provide direct URLs
                "metadata": org_data,
            }

        except Exception as e:
            logger.error(f"NCI organization fetch failed: {e}")
            raise SearchExecutionError("nci_organization", e) from e

    elif domain == "nci_intervention":
        logger.debug("Fetching NCI intervention details")
        try:
            from biomcp.interventions import get_intervention
            from biomcp.interventions.getter import format_intervention_details

            intervention_data = await get_intervention(
                intervention_id=id,
                api_key=api_key,
            )

            # Format the details
            formatted_text = format_intervention_details(intervention_data)

            # Return OpenAI MCP compliant format
            return {
                "id": id,
                "title": intervention_data.get("name", "Unknown Intervention"),
                "text": formatted_text,
                "url": "",  # NCI doesn't provide direct URLs
                "metadata": intervention_data,
            }

        except Exception as e:
            logger.error(f"NCI intervention fetch failed: {e}")
            raise SearchExecutionError("nci_intervention", e) from e

    elif domain == "nci_disease":
        logger.debug("Fetching NCI disease details")
        try:
            from biomcp.diseases import get_disease_by_id

            disease_data = await get_disease_by_id(
                disease_id=id,
                api_key=api_key,
            )

            # Build text description
            text_parts = []
            text_parts.append(
                f"Disease: {disease_data.get('name', 'Unknown Disease')}"
            )

            if disease_data.get("category"):
                text_parts.append(f"\nCategory: {disease_data['category']}")

            if disease_data.get("synonyms"):
                synonyms = disease_data["synonyms"]
                if isinstance(synonyms, list) and synonyms:
                    text_parts.append(f"\nSynonyms: {', '.join(synonyms[:5])}")
                    if len(synonyms) > 5:
                        text_parts.append(
                            f"  ... and {len(synonyms) - 5} more"
                        )

            if disease_data.get("codes"):
                codes = disease_data["codes"]
                if isinstance(codes, dict):
                    code_items = [
                        f"{system}: {code}" for system, code in codes.items()
                    ]
                    if code_items:
                        text_parts.append(f"\nCodes: {', '.join(code_items)}")

            # Return OpenAI MCP compliant format
            return {
                "id": id,
                "title": disease_data.get(
                    "name",
                    disease_data.get("preferred_name", "Unknown Disease"),
                ),
                "text": "\n".join(text_parts),
                "url": "",  # NCI doesn't provide direct URLs
                "metadata": disease_data,
            }

        except Exception as e:
            logger.error(f"NCI disease fetch failed: {e}")
            raise SearchExecutionError("nci_disease", e) from e

    # Note: nci_biomarker doesn't support fetching by ID, only searching

    # OpenFDA domains
    elif domain == "fda_adverse":
        from biomcp.openfda import get_adverse_event

        result = await get_adverse_event(id, api_key=api_key)
        return {
            "title": f"FDA Adverse Event Report {id}",
            "text": result,
            "url": "",
            "metadata": {"report_id": id, "domain": "fda_adverse"},
        }

    elif domain == "fda_label":
        from biomcp.openfda import get_drug_label

        result = await get_drug_label(id, api_key=api_key)
        return {
            "title": f"FDA Drug Label {id}",
            "text": result,
            "url": "",
            "metadata": {"set_id": id, "domain": "fda_label"},
        }

    elif domain == "fda_device":
        from biomcp.openfda import get_device_event

        result = await get_device_event(id, api_key=api_key)
        return {
            "title": f"FDA Device Event {id}",
            "text": result,
            "url": "",
            "metadata": {"mdr_report_key": id, "domain": "fda_device"},
        }

    elif domain == "fda_approval":
        from biomcp.openfda import get_drug_approval

        result = await get_drug_approval(id, api_key=api_key)
        return {
            "title": f"FDA Drug Approval {id}",
            "text": result,
            "url": "",
            "metadata": {"application_number": id, "domain": "fda_approval"},
        }

    elif domain == "fda_recall":
        from biomcp.openfda import get_drug_recall

        result = await get_drug_recall(id, api_key=api_key)
        return {
            "title": f"FDA Drug Recall {id}",
            "text": result,
            "url": "",
            "metadata": {"recall_number": id, "domain": "fda_recall"},
        }

    elif domain == "fda_shortage":
        from biomcp.openfda import get_drug_shortage

        result = await get_drug_shortage(id, api_key=api_key)
        return {
            "title": f"FDA Drug Shortage - {id}",
            "text": result,
            "url": "",
            "metadata": {"drug": id, "domain": "fda_shortage"},
        }

    # Invalid domain
    raise InvalidDomainError(domain, VALID_DOMAINS)


# Internal function for unified search
async def _unified_search(  # noqa: C901
    query: str,
    max_results_per_domain: int = MAX_RESULTS_PER_DOMAIN_DEFAULT,
    domains: list[str] | None = None,
    explain_query: bool = False,
) -> dict:
    """Internal unified search implementation.

    Parses the unified query language and routes to appropriate domain tools.
    Supports field-based syntax like 'gene:BRAF AND trials.phase:3'.

    Args:
        query: Unified query string with field syntax
        max_results_per_domain: Limit results per domain
        domains: Optional list to filter which domains to search
        explain_query: If True, return query parsing explanation

    Returns:
        Dictionary with results organized by domain

    Raises:
        QueryParsingError: If query cannot be parsed
        SearchExecutionError: If search execution fails
    """
    logger.info(f"Unified search with query: {query}")
    # Parse the query
    try:
        parser = QueryParser()
        parsed = parser.parse(query)
    except Exception as e:
        logger.error(f"Failed to parse query: {e}")
        raise QueryParsingError(query, e) from e

    # Route to appropriate tools
    router = QueryRouter()
    plan = router.route(parsed)

    # Filter domains if specified
    if domains:
        filtered_tools = []
        for tool in plan.tools_to_call:
            if (
                ("article" in tool and "articles" in domains)
                or ("trial" in tool and "trials" in domains)
                or ("variant" in tool and "variants" in domains)
            ):
                filtered_tools.append(tool)
        plan.tools_to_call = filtered_tools

    # Return explanation if requested
    if explain_query:
        return {
            "original_query": query,
            "parsed_structure": {
                "cross_domain_fields": parsed.cross_domain_fields,
                "domain_specific_fields": parsed.domain_specific_fields,
                "terms": [
                    {
                        "field": term.field,
                        "operator": term.operator.value,
                        "value": term.value,
                        "domain": term.domain,
                    }
                    for term in parsed.terms
                ],
            },
            "routing_plan": {
                "tools_to_call": plan.tools_to_call,
                "field_mappings": plan.field_mappings,
            },
            "schema": parser.get_schema(),
        }

    # Execute the search plan
    try:
        results = await execute_routing_plan(plan, output_json=True)
    except Exception as e:
        logger.error(f"Failed to execute search plan: {e}")
        raise SearchExecutionError("unified", e) from e

    # Format unified results - collect all results into a single array
    all_results = []

    for domain, result_str in results.items():
        if isinstance(result_str, dict) and "error" in result_str:
            logger.warning(f"Error in domain {domain}: {result_str['error']}")
            continue

        try:
            data = (
                json.loads(result_str)
                if isinstance(result_str, str)
                else result_str
            )

            # Get the appropriate handler for formatting
            handler_class = get_domain_handler(
                domain.rstrip("s")
            )  # Remove trailing 's'

            # Process and format each result
            # Handle both list format and dict format (for articles with cBioPortal data)
            items_to_process = []
            cbioportal_summary = None

            if isinstance(data, list):
                items_to_process = data[:max_results_per_domain]
            elif isinstance(data, dict):
                # Handle unified search format with cBioPortal data
                if "articles" in data:
                    items_to_process = data["articles"][
                        :max_results_per_domain
                    ]
                    cbioportal_summary = data.get("cbioportal_summary")
                else:
                    # Single item dict
                    items_to_process = [data]

            # Add cBioPortal summary as first result if available
            if cbioportal_summary and domain == "articles":
                try:
                    # Extract gene name from parsed query or summary
                    gene_name = parsed.cross_domain_fields.get("gene", "")
                    if not gene_name and "Summary for " in cbioportal_summary:
                        # Try to extract from summary title
                        import re

                        match = re.search(
                            r"Summary for (\w+)", cbioportal_summary
                        )
                        if match:
                            gene_name = match.group(1)

                    cbio_result = {
                        "id": f"cbioportal_summary_{gene_name or 'gene'}",
                        "title": f"cBioPortal Summary for {gene_name or 'Gene'}",
                        "text": cbioportal_summary[:5000],  # Limit text length
                        "url": f"https://www.cbioportal.org/results?gene_list={gene_name}"
                        if gene_name
                        else "",
                    }
                    all_results.append(cbio_result)
                except Exception as e:
                    logger.warning(f"Failed to format cBioPortal summary: {e}")

            for item in items_to_process:
                try:
                    formatted_result = handler_class.format_result(item)
                    # Ensure OpenAI MCP format
                    openai_result = {
                        "id": formatted_result.get("id", ""),
                        "title": formatted_result.get("title", DEFAULT_TITLE),
                        "text": formatted_result.get(
                            "snippet", formatted_result.get("text", "")
                        ),
                        "url": formatted_result.get("url", ""),
                    }
                    # Note: For unified search, we can optionally include domain in metadata
                    # This helps distinguish between result types
                    all_results.append(openai_result)
                except Exception as e:
                    logger.warning(
                        f"Failed to format result in domain {domain}: {e}"
                    )
                    continue

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse results for domain {domain}: {e}")
            continue

    logger.info(
        f"Unified search completed with {len(all_results)} total results"
    )

    # Return OpenAI MCP compliant format
    return {"results": all_results}
