"""
Entity and relationship extraction from text chunks using an LLM.

The extractor sends each chunk to the configured LLM with a structured prompt
and parses the JSON response into Entity and Relationship objects.  Robustness
mechanisms include JSON repair for common LLM formatting errors and exponential
backoff on rate-limit responses.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .chunker import Chunk
from .config import LLMConfig
from .graph_store import Entity, Relationship, compute_hash_id


_EXTRACTION_PROMPT = """\
Analyze the text below and extract named entities and their relationships.

You MUST return ONLY a raw JSON object — no markdown fences, no extra text.
Use EXACTLY these field names (no aliases, no alternative keys):

Required schema:
{{
  "entities": [
    {{
      "entity_name": "<name of the entity>",
      "entity_type": "<one of: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, MEDICAL_CONDITION, MEDICATION, PROCEDURE, EVENT, OTHER>",
      "description": "<one sentence describing this entity in context>",
      "attributes": {{}}
    }}
  ],
  "relationships": [
    {{
      "source_entity": "<entity_name of the source — must exactly match one entity_name above>",
      "target_entity": "<entity_name of the target — must exactly match one entity_name above>",
      "relationship_type": "<e.g. HAS_CONDITION, PRESCRIBED, LOCATED_IN, RELATED_TO, IS_A, PART_OF>",
      "description": "<one sentence describing this relationship>",
      "weight": 1.0
    }}
  ]
}}

Example output for the text "Dr. Smith prescribed Metformin to patient John, who has Type 2 Diabetes.":
{{
  "entities": [
    {{"entity_name": "Dr. Smith", "entity_type": "PERSON", "description": "Physician who issued the prescription.", "attributes": {{}}}},
    {{"entity_name": "Metformin", "entity_type": "MEDICATION", "description": "Oral antidiabetic drug prescribed for Type 2 Diabetes.", "attributes": {{}}}},
    {{"entity_name": "John", "entity_type": "PERSON", "description": "Patient diagnosed with Type 2 Diabetes.", "attributes": {{}}}},
    {{"entity_name": "Type 2 Diabetes", "entity_type": "MEDICAL_CONDITION", "description": "Chronic metabolic disorder affecting blood glucose regulation.", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source_entity": "Dr. Smith", "target_entity": "Metformin", "relationship_type": "PRESCRIBED", "description": "Dr. Smith prescribed Metformin.", "weight": 1.0}},
    {{"source_entity": "John", "target_entity": "Type 2 Diabetes", "relationship_type": "HAS_CONDITION", "description": "John is diagnosed with Type 2 Diabetes.", "weight": 1.0}},
    {{"source_entity": "John", "target_entity": "Metformin", "relationship_type": "PRESCRIBED", "description": "Metformin was prescribed to John.", "weight": 1.0}}
  ]
}}

Now extract from this text:
{text}
"""

_SUMMARY_PROMPT = """\
Write a concise, informative summary (3-5 sentences) of the following text.
Focus on the key concepts, main findings, and important entities.
The summary should be self-contained - a reader should be able to understand
the gist without reading the original text.

Text:
{text}

Output only the summary paragraph, no labels.
"""


@dataclass
class ExtractionResult:
    """Result of extracting entities and relationships from a chunk."""

    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    summary: Optional[str] = None


class LLMClient:
    """Thin wrapper around the configured LLM provider (sync)."""

    def __init__(self, config: LLMConfig):
        self._config = config

    def complete(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        if self._config.provider == "openai":
            return self._call_openai(prompt)
        if self._config.provider == "gemini":
            return self._call_gemini(prompt)
        if self._config.provider == "groq":
            return self._call_groq(prompt)
        raise ValueError(f"Unsupported LLM_PROVIDER: '{self._config.provider}'")

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
        )
        response = client.chat.completions.create(
            model=self._config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self._config.gemini_api_key)
        model = genai.GenerativeModel(self._config.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            ),
        )
        return response.text

    def _call_groq(self, prompt: str) -> str:
        from groq import Groq
        client = Groq(api_key=self._config.groq_api_key)
        response = client.chat.completions.create(
            model=self._config.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        return response.choices[0].message.content


class EntityExtractor:
    """
    Extracts entities, relationships, and summaries from text chunks.

    Uses exponential backoff when the provider signals rate limits.  JSON
    repair handles common formatting issues produced by LLMs (single quotes,
    trailing commas, unescaped newlines).
    """

    _MAX_RETRIES = 3
    _BASE_DELAY_SECONDS = 2.0

    def __init__(self, config: Optional[LLMConfig] = None):
        self._config = config or LLMConfig()
        self._client = LLMClient(self._config)

    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities, relationships, and optionally a summary from a chunk."""
        text = chunk.content
        if not text.strip():
            return ExtractionResult()

        try:
            raw = await self._call_with_backoff(
                _EXTRACTION_PROMPT.format(text=text[:4000])
            )
            entities, relationships = self._parse_extraction_response(raw, chunk)
        except Exception as exc:
            print(f"Extraction failed for chunk {chunk.chunk_id}: {exc}")
            return ExtractionResult()

        return ExtractionResult(entities=entities, relationships=relationships)

    async def summarize(self, text: str) -> Optional[str]:
        """Produce a concise summary of the given text."""
        if not text.strip():
            return None
        try:
            return await self._call_with_backoff(
                _SUMMARY_PROMPT.format(text=text[:6000])
            )
        except Exception as exc:
            print(f"Summarization failed: {exc}")
            return None

    async def _call_with_backoff(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                return await loop.run_in_executor(
                    None, lambda p=prompt: self._client.complete(p)
                )
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limit(exc) and attempt < self._MAX_RETRIES - 1:
                    delay = self._BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"Rate limit hit, retrying in {delay:.1f}s ...")
                    await asyncio.sleep(delay)
                elif attempt < self._MAX_RETRIES - 1:
                    await asyncio.sleep(1.0)
                else:
                    raise

        raise last_exc

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "rate" in msg or "429" in msg or "quota" in msg or "limit" in msg

    def _parse_extraction_response(
        self, response: str, chunk: Chunk
    ) -> Tuple[List[Entity], List[Relationship]]:
        json_str = self._extract_json(response)
        if not json_str:
            return [], []

        json_str = self._repair_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            print(f"JSON parse failed after repair: {exc}")
            return [], []

        entities = self._parse_entities(data.get("entities", []), chunk)
        entity_names = {e.entity_name for e in entities}
        relationships = self._parse_relationships(
            data.get("relationships", []), chunk, entity_names
        )
        return entities, relationships

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        # Remove any markdown code fences.
        text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = text.strip().rstrip("`")

        # Find the outermost { ... } using bracket counting.
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # Unclosed object: return everything after the opening brace.
        return text[start:]

    @staticmethod
    def _repair_json(raw: str) -> str:
        # 1. Remove trailing commas before closing brackets.
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        # 2. Ensure property names are double-quoted.
        raw = re.sub(r"(?<=[{,\[\s])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', raw)

        # 3. Replace Python-style None/True/False.
        raw = raw.replace(": None", ": null")
        raw = raw.replace(": True", ": true")
        raw = raw.replace(": False", ": false")

        # 4. Try direct parse; if it fails, truncate the JSON to the last fully
        #    closed top-level list item to recover partial content.
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass

        # Attempt to close incomplete arrays/objects by finding the last valid
        # position where a complete item ends.
        for close_char, open_char in (("]}", "{["), ):
            try:
                # Find last full entity/relationship object end.
                last_brace = raw.rfind("}")
                if last_brace > 0:
                    candidate = raw[: last_brace + 1]
                    # Close any unclosed arrays.
                    open_brackets = candidate.count("[") - candidate.count("]")
                    open_braces = candidate.count("{") - candidate.count("}")
                    candidate += "]" * open_brackets + "}" * open_braces
                    json.loads(candidate)
                    return candidate
            except json.JSONDecodeError:
                pass

        return raw

    def _parse_entities(
        self, items: List[Dict[str, Any]], chunk: Chunk
    ) -> List[Entity]:
        entities: List[Entity] = []
        seen_names: set = set()

        for item in items:
            # Accept both the required field name and common LLM aliases.
            name = (
                item.get("entity_name")
                or item.get("name")
                or item.get("text")
                or item.get("label")
                or ""
            ).strip()
            if not name or name in seen_names:
                continue

            entity_type = (
                item.get("entity_type")
                or item.get("type")
                or item.get("kind")
                or "OTHER"
            ).upper()

            seen_names.add(name)
            entity_id = compute_hash_id(name, prefix="entity-")
            entities.append(
                Entity(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_type=entity_type,
                    description=item.get("description", ""),
                    source_chunks=[chunk.chunk_id],
                    doc_id=chunk.doc_id,
                    file_path=chunk.file_path,
                    attributes=item.get("attributes", {}),
                )
            )

        return entities

    def _parse_relationships(
        self,
        items: List[Dict[str, Any]],
        chunk: Chunk,
        valid_entity_names: set,
    ) -> List[Relationship]:
        relationships: List[Relationship] = []

        for item in items:
            # Accept both required field names and common LLM aliases.
            src = (
                item.get("source_entity")
                or item.get("source")
                or item.get("from")
                or ""
            ).strip()
            tgt = (
                item.get("target_entity")
                or item.get("target")
                or item.get("to")
                or ""
            ).strip()

            if not src or not tgt:
                continue
            # Only drop if both names are truly absent from entities.
            if src not in valid_entity_names and tgt not in valid_entity_names:
                continue

            rel_type = (
                item.get("relationship_type")
                or item.get("type")
                or item.get("relation")
                or "RELATED_TO"
            ).upper()

            rel_id = compute_hash_id(
                f"{src}:{rel_type}:{tgt}:{chunk.chunk_id}", prefix="rel-"
            )

            try:
                weight = float(item.get("weight", 1.0))
            except (ValueError, TypeError):
                weight = 1.0

            relationships.append(
                Relationship(
                    relationship_id=rel_id,
                    source_entity=src,
                    target_entity=tgt,
                    relationship_type=rel_type,
                    description=item.get("description", ""),
                    weight=weight,
                    source_chunk=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    file_path=chunk.file_path,
                )
            )

        return relationships
