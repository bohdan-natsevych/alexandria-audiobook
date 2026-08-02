# Run Alexandria with Compute Server and UI on Separate Machines

This guide runs all GPU/LLM/TTS work on a **server machine** while the web UI,
project data, clone references, and generated audio stay in the **client
browser**.

The server is intentionally open for this prototype: there is no login or API
token. Use it only on a trusted LAN or a private VPN; do not expose port 4200
directly to the public internet.

## What Lives Where

| Location | Stored or executed there |
|---|---|
| Server | Docker image, GPU models, server configuration, built-in/trained LoRA adapters, and temporary job workspaces |
| Client browser | Script, chunk edits, voice settings, clone references, designed-voice previews, training-dataset ZIPs, and generated WAV chunks |
| During a job | Input uploads and results exist in a server UUID workspace only until they are downloaded or expire |

Jobs expire after 24 hours by default. The browser is the source of truth for a
project, so do not clear its site data while working on a book.

## 1. Prepare the Server Machine

The server needs Docker, a compatible NVIDIA driver, Docker GPU support, and
enough free disk space for model downloads. The supplied Docker image contains
FFmpeg and the Python packages used for TTS, LoRA, and the voice-dataset
preparer.

Clone or copy this repository onto the GPU server, then start it from the
repository root:

```bash
docker compose up --build -d
docker compose logs -f alexandria
```

The first model use downloads model weights, so the initial TTS, Voice Design,
LoRA, or Preparer job can take substantially longer than later jobs.

The server listens on all network interfaces at port 4200. Confirm it is
reachable locally:

```bash
curl http://127.0.0.1:4200/api/system/stats
```

### Persisted Server Data

The Compose configuration persists only these server-side assets:

| Host directory / volume | Purpose |
|---|---|
| `hf_cache` Docker volume | Hugging Face model cache |
| `data/config/` | Shared LLM/TTS configuration |
| `data/lora_models/` | Trained reusable LoRA adapters |
| `data/builtin_lora/` | Downloaded built-in LoRA adapters |

Project files and generated audiobook audio are deliberately not mounted here.

### Open the Firewall

Allow TCP port 4200 from only your trusted client subnet/VPN.

Windows PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "Alexandria compute server" -Direction Inbound -Protocol TCP -LocalPort 4200 -Action Allow
```

Linux with UFW:

```bash
sudo ufw allow from CLIENT_SUBNET to any port 4200 proto tcp
```

Replace `CLIENT_SUBNET` with your LAN subnet, for example
`192.168.1.0/24`. If no firewall is enabled, this step is not needed, but keep
the server on a trusted network.

## 2. Configure the Server’s LLM and TTS

Open the UI once from the server at `http://SERVER_IP:4200`, then use **Setup**
to save the shared server configuration. It is stored in
`data/config/config.json`.

Important: the LLM and any external TTS URL are contacted by the **server
container**, not the client browser.

- If the LLM/TTS service is on another LAN machine, use its LAN address, for
  example `http://192.168.1.50:1234/v1`.
- If the LLM is on the same Docker host, use an address that the container can
  reach (normally the host LAN IP). Do not use `127.0.0.1` or `localhost`;
  inside the container those refer to the Alexandria container itself.
- For built-in TTS, choose `local`. The GPU server loads and runs the TTS model
  directly.
- For external TTS, use a URL reachable from the GPU server container.

The Preparer downloads its faster-whisper model on first use. Its defaults can
be changed by setting `ALEXANDRIA_WHISPER_MODEL` and
`ALEXANDRIA_WHISPER_DEVICE` in the container environment before startup.

## 3. Serve the UI on the Client Machine

Do not double-click `index.html` or open it with a `file://` URL. Serve the
static UI over HTTP so IndexedDB works reliably.

Copy `app/static/` to the client machine, then run one of these commands in
that directory.

Python:

```bash
python -m http.server 8080
```

Node.js:

```bash
npx serve -l 8080
```

Open the UI with the compute server address:

```text
http://CLIENT_IP:8080/?apiBase=http://SERVER_IP:4200
```

For example:

```text
http://192.168.1.20:8080/?apiBase=http://192.168.1.10:4200
```

The `apiBase` value is saved in that browser’s local storage. Later visits can
use `http://CLIENT_IP:8080/` without the query parameter.

To point the UI at another server, open it once with the new `apiBase` value,
or remove the browser local-storage key `alexandria.apiBaseUrl`.

## 4. Client Browser Settings

Use a current Chromium, Edge, Firefox, or Safari browser. When prompted,
allow persistent storage. Alexandria also requests it automatically, but a
browser can still refuse or evict storage under disk pressure.

- Do not use private/incognito mode for projects you want to keep.
- Do not clear browsing data for the UI origin.
- Keep an exported final MP3/M4B/Audacity ZIP outside the browser as a normal
  backup.
- Clone reference files and generated chunks are stored only in the browser
  that uploaded/generated them. Another browser or machine cannot see them
  unless you import/recreate that project there.

## 5. Verify the Cross-Machine Connection

From the client machine, verify the server before opening the UI.

PowerShell:

```powershell
Invoke-RestMethod http://SERVER_IP:4200/api/system/stats
```

Linux/macOS:

```bash
curl http://SERVER_IP:4200/api/system/stats
```

Then open the client URL with `?apiBase=...`. The **Setup** page should load
the server configuration and the system-status area should show server disk and
GPU information.

## 6. Normal Workflow

1. Configure the LLM/TTS server in Setup.
2. Upload a TXT, Markdown, or EPUB source in the client UI.
3. Generate/review the script and configure voices.
4. Generate individual chunks or a TTS batch; returned WAVs are saved in the
   client browser.
5. Use Voice Design, Dataset Builder, LoRA Training, or Preparer as needed.
   These send temporary job inputs to the server and save returned results on
   the client, except trained LoRA adapter weights, which intentionally remain
   reusable server compute assets.
6. Export MP3, M4B, or an Audacity ZIP. The client uploads the required WAVs
   temporarily, downloads the export, and the server workspace expires.

## Troubleshooting

| Symptom | Check |
|---|---|
| UI cannot reach server | Confirm `SERVER_IP`, port 4200, firewall rule, and that `docker compose ps` shows the container running. |
| UI loads but jobs fail immediately | Check `docker compose logs -f alexandria`; verify LLM/TTS URLs are reachable from the server container. |
| Browser shows no old project | It is a different browser profile, origin, or private session. Project data is intentionally browser-local. |
| Clone voice fails | Re-upload the reference in that browser; clone files are not persisted on the compute server. |
| GPU is not used | Confirm Docker GPU support and inspect `docker compose logs -f alexandria`. Built-in TTS must be set to `local`. |
| First run is very slow | Model downloads and GPU warm-up are expected; watch the server logs. |
| Preparer fails | The first run downloads its Whisper model. Ensure the server has internet access, disk space, and enough CPU/GPU memory. |

## Optional: Job Retention

Temporary job workspaces expire after 24 hours. To change that, add this to the
`alexandria` service in `docker-compose.yml` and recreate the container:

```yaml
environment:
  ALEXANDRIA_JOB_TTL_HOURS: "48"
```

This only changes temporary server job retention; it does not move your
browser-owned project or audio to the server.
