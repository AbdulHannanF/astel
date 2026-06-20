# Remote access — drive this box's 4090s from a laptop

Goal: edit/work on a **laptop**, but run **generation on the GPU box**
(`THREADRIPPER-48`, 2×RTX 4090). The GPU producer is a local subprocess on the
box, so the **API must run on the box** — the laptop just talks to it. Two
supported models, plus the firewall + tunnel notes both need.

> This box's LAN address (at time of writing): **`172.14.38.59`** — re-check with
> `ipconfig` (look for the IPv4 of your active adapter) if your network changed.

---

## One-time: open the firewall on the box

LAN clients can't reach the ports until Windows Firewall allows them. Run this
**once, in an elevated PowerShell** (Admin) on the box — from a normal Claude
Code prompt you can do it with the `!` prefix only if that session is elevated;
otherwise open "Windows PowerShell (Admin)":

```powershell
New-NetFirewallRule -DisplayName "Astel API (8000)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
New-NetFirewallRule -DisplayName "Astel Web (5173)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5173 -Profile Private
```

(`-Profile Private` keeps it to home/work networks. Don't open these on a public
network without a tunnel — see below.)

---

## Model A — browser only (simplest, zero laptop setup)

Run the **whole stack on the box**, exposed on the LAN:

```powershell
# on the box (THREADRIPPER-48)
pnpm run up -- -BindHost 0.0.0.0
```

Then on the laptop, open a browser to:

```
http://172.14.38.59:5173
```

That's it. The Vite dev server (on the box) proxies `/v1` to the box's API, the
API runs the **GPU producer**, and generation happens on the 4090s. The laptop
needs nothing installed. Use this if you only want to *use* the product
remotely.

---

## Model B — Claude Code on the laptop, GPU API on the box (recommended for dev)

Keep editing the repo with Claude Code on the laptop (full local speed), and
point only the **generation API** at the box.

**On the box** — run just the API, on the LAN, with the GPU producer:

```powershell
# on the box
cd D:\Astel\services\api
$env:ASTEL_PRODUCER = "gpu"
uv run uvicorn astel_api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

**On the laptop** — run the web app pointed at the box's API. Vite's proxy target
is the `ASTEL_API_URL` env var (see `apps/web/vite.config.ts`):

```bash
# on the laptop, in a clone of this repo
ASTEL_API_URL=http://172.14.38.59:8000 pnpm -C apps/web dev
# open http://localhost:5173 on the laptop — generation runs on the box's 4090s
```

The laptop's browser hits its local Vite, which proxies `/v1` to the box, so
there are no CORS issues. You get the full Studio locally while every splat is
generated on the box.

Getting the repo onto the laptop: `git clone` it (push/pull between box and
laptop through your git remote), or use a shared/synced folder. Claude Code runs
against whatever copy is on the laptop.

---

## Over the internet (not just same LAN): use Tailscale

Don't expose 8000/5173 to the public internet directly. Install
[Tailscale](https://tailscale.com/) on both the box and the laptop (free tier).
Each machine gets a stable `100.x.y.z` address; use that in place of the LAN IP
above and skip the firewall rule entirely (Tailscale traffic is allowed by
default). This works from anywhere, encrypted, no port-forwarding.

---

## Sanity checks

```bash
# from the laptop, confirm the box API is reachable:
curl http://172.14.38.59:8000/healthz        # -> {"status":"ok",...}
curl http://172.14.38.59:8000/v1/generations # -> the produced-asset catalog
```

If `healthz` works from the box itself but not the laptop, it's the firewall
(run the rules above) or you're on a different subnet / "Public" network profile.
