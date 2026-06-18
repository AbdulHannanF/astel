# MCP server

Astel ships an MCP (Model Context Protocol) server that exposes asset generation as tools for AI agents and IDEs.

## Install

```bash
pip install "astel-sdk[mcp]"
```

## Start

```bash
# stdio transport (Claude Desktop, Continue, etc.)
astel-mcp

# SSE transport (web integrations)
astel-mcp --transport sse --host 127.0.0.1 --port 9000
```

## Environment

```bash
ASTEL_API_URL=http://localhost:8000   # default
ASTEL_API_KEY=your_key                # optional bearer token
```

## Tools

### `generate_asset`

Generate a layered Gaussian splat asset.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text description of the object |
| `modality` | string | `"text"` | `"text"`, `"image"`, or `"video"` |
| `mode` | string | `"refine"` | `"refine"` or `"preview"` |
| `capture_id` | string? | null | For image/video: upload first via `/v1/captures` |

**Returns**

```json
{
  "generation_id": "abc-123",
  "status": "SUCCEEDED",
  "artifacts": ["l3.ply", "l3.spz", "package.astel", "quality-report.json"],
  "billing": { "total_credits": 21, "total_usd": 0.21 },
  "conditioning": "prompt"
}
```

### `get_asset`

Poll a generation by ID.

**Parameters**: `generation_id: string`

**Returns**

```json
{
  "status": "SUCCEEDED",
  "is_ready": true,
  "is_failed": false,
  "artifacts": [{ "name": "l3.ply", "url": "http://localhost:8000/...", "bytes": 12345 }],
  "quality_report_url": "http://localhost:8000/v1/generations/abc-123/artifacts/quality-report.json",
  "billing": { "total_credits": 21, "total_usd": 0.21 }
}
```

### `list_pricing`

Returns the credit price schedule.

## Claude Desktop configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "astel": {
      "command": "astel-mcp",
      "env": {
        "ASTEL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

Then ask Claude: *"Generate a worn brass astrolabe as a 3D Gaussian splat and give me the download URL."*
