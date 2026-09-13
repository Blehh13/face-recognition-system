# Deployment

The application runs as a container. Everything below is free.

**Read the privacy section before putting this on a public URL.** It is not
boilerplate: a face recognition endpoint open to the internet is biometric
data processing, and the defaults here are set accordingly.

---

## Run it locally

```bash
docker build -t facerec .
docker run -p 7860:7860 facerec
# http://127.0.0.1:7860
```

The build compiles dlib from source and takes roughly ten minutes. That build
is also why some hosts fail — see the comparison below.

Without Docker:

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python app.py
```

---

## Configuration

| variable | default | purpose |
|---|---|---|
| `FACEREC_HOST` | `127.0.0.1` | Interface to bind. Loopback unless set. |
| `FACEREC_PORT` | `5000` | Port. The container uses 7860. |
| `FACEREC_SECRET` | dev value | Flask secret key. Set a real one when exposed. |
| `FACEREC_DB` | `database/enrolled_faces.json` | Database path. **A `.sqlite` extension selects the multi-process backend.** |
| `FACEREC_DEMO` | unset | `1` wipes the database at start-up. |

### The database extension is not cosmetic

The JSON store keeps the whole database in memory and rewrites it wholesale.
With two workers:

```
worker A enrols Alice   -> writes {Alice}
worker B enrols Bob     -> writes {Bob}, from its own stale snapshot
on disk                 -> {Bob}.  Alice is gone, and nothing errors.
```

Gunicorn runs several workers by default, so a deployment hits this on the
second request. Use a `.sqlite` path — `src/sqlite_store.py` takes a real write
lock and every read sees committed state. `tests/test_sqlite_store.py` pins
both the failure and the fix.

To carry an existing JSON database over:

```python
from src.sqlite_store import SqliteFaceDatabase
SqliteFaceDatabase("database/enrolled_faces.sqlite").import_json("database/enrolled_faces.json")
```

---

## Privacy: read this before exposing it

This application stores face embeddings of whoever is uploaded. On a public URL
with no authentication, that means **collecting strangers' biometric data**,
which is specially protected under GDPR Article 9 and India's DPDP Act. It is a
different legal proposition from running the same code on your laptop.

The container therefore ships with `FACEREC_DEMO=1`, which empties the database
at start-up so a public instance is a demonstration rather than a face
collection. Keep it that way unless you have added authentication and have a
lawful basis for holding the data.

There is still **no authentication**: anyone who can reach the URL can enrol,
identify and delete. Put a reverse proxy with auth in front of it before it is
anything but a demo.

---

## Hosting options

| host | free tier | works? |
|---|---|---|
| **Hugging Face Spaces** | 16 GB RAM, 2 vCPU, no card | **Recommended.** Enough memory for the dlib build, Docker native, no cold-start eviction. |
| Render | 512 MB RAM | Build usually fails or OOMs compiling dlib, and the free tier sleeps after 15 minutes — the first visitor then waits for a cold start *plus* model loading. |
| Fly.io | small allowance | Works, but asks for a card. |
| Railway | trial credit | Works until the credit runs out. |

### Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space> — SDK **Docker**, hardware **CPU basic (free)**.
2. Add the header below to the top of `README.md` (Spaces requires it):

```yaml
---
title: Face Recognition Identification System
emoji: 🙂
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

3. Push:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space master
```

The first build takes 10–15 minutes, nearly all of it dlib.

**Note on the frontend.** `static/dist/` is gitignored, so the image has no
built frontend and the page will say so. Either build it into the image by
adding a Node stage to the Dockerfile, or commit `static/dist/` on the branch
you push to the Space.

---

## Is deploying worth it?

For an assignment that asked for a GitHub link, honestly: it is optional. A
deployed demo helps only if it is reliably fast when someone clicks it, and the
common failure — a free instance cold-starting for a minute while an
interviewer waits — is worse than no link at all.

Running it locally during the interview, or a short screen recording, costs
nothing and cannot break. Deploy if you want the URL on a CV, not to satisfy
the brief.
