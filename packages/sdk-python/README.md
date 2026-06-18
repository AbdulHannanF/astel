# astel-sdk

Python SDK for the Astel API — generate and retrieve layered Gaussian splat assets.

```python
from astel_sdk import AstelClient

client = AstelClient("http://localhost:8000")
gen = client.generate(prompt="a brass astrolabe on a wooden base")
client.download_all_artifacts(gen.id, "out/")
```

See [docs/api-reference.md](../../docs/site/docs/api-reference.md) for full reference.
