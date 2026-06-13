"""JSON Schema (draft 2020-12) validation against the bundled Astel schemas.

The schemas in :mod:`astel_format.schemas` are byte-identical copies of
``docs/specs/schemas/*.json`` (the repo's authoritative source). They
``$ref`` each other by bare filename (e.g. ``manifest.schema.json`` refs
``"layer.schema.json"``), so a :class:`jsonschema.validators.Registry` is
built mapping those filenames -- and their ``$id`` URIs -- to the local
copies, giving fully offline validation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from astel_format.errors import AstelValidationError

_SCHEMA_FILES = (
    "manifest.schema.json",
    "layer.schema.json",
    "buffers.schema.json",
    "provenance.schema.json",
    "quality-report.schema.json",
)


@lru_cache(maxsize=1)
def _load_schemas() -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for filename in _SCHEMA_FILES:
        text = (
            resources.files("astel_format.schemas")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        schemas[filename] = json.loads(text)
    return schemas


@lru_cache(maxsize=1)
def _registry() -> Registry:
    schemas = _load_schemas()
    resources_: list[tuple[str, Resource]] = []
    for filename, schema in schemas.items():
        resource = Resource.from_contents(schema)
        # Register under the bare filename (how sibling schemas $ref each
        # other) and under the schema's own $id (absolute URI), so both
        # resolution styles work offline.
        resources_.append((filename, resource))
        schema_id = schema.get("$id")
        if schema_id:
            resources_.append((schema_id, resource))
    return Registry().with_resources(resources_)


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    schemas = _load_schemas()
    manifest_schema = schemas["manifest.schema.json"]
    return Draft202012Validator(manifest_schema, registry=_registry())


def validate_manifest_dict(manifest: dict[str, Any]) -> None:
    """Validate a manifest dict against ``manifest.schema.json``.

    Raises :class:`AstelValidationError` with all collected error messages
    if validation fails.
    """
    validator = _manifest_validator()
    errors: list[ValidationError] = sorted(
        validator.iter_errors(manifest), key=lambda e: list(e.path)
    )
    if errors:
        messages = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise AstelValidationError(f"manifest failed schema validation: {messages}")
