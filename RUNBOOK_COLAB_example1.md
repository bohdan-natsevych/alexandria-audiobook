# Runbook: example1.txt -> Russian audiobook on Google Colab

This runbook turns `example1.txt` (a Russian novel, ~106,000 characters) into a
multi-voice audiobook. The app reads the book with an LLM, finds the
characters, writes a voice description for each one, creates a matching voice
sample, and then reads the whole book with those voices.

Everything runs in Google Colab. Files are kept in your Google Drive, so you can
stop at any point and continue later without starting over (see
"Checkpoints and resuming").

The work is split into two Colab sessions:

| Session | Colab hardware | What happens | Uses GPU quota |
|---|---|---|---|
| A | CPU | Script generation and review (LLM only) | No |
| B | T4 GPU | Character voices, audio generation, final MP3 | Yes |

Doing the LLM steps on a CPU runtime saves your limited free GPU time for the
steps that actually need it.

Notes:
- Screen and menu names below match Google Colab at the time of writing. If a
  label looks slightly different, look for the closest match.
- Never close the browser tab of a running Colab session for long. Free Colab
  disconnects sessions that look idle.

---

## 0. What you need

1. A Google account with **at least 8 GB free in Google Drive**
   (models ~5 GB, audio ~200 MB, app files).
   Check at https://drive.google.com -> "Storage" in the left menu.
2. An **LLM API key**: Gemini (free tier) or DeepSeek (paid, a few dollars) -
   created in step 2. Chat subscriptions (Gemini app plans, Claude Pro) do not
   include API access and cannot be used here.
3. The book file on this PC: `C:\Work\personal\new\example1.txt`.
4. This PC with the project folder `C:\Work\personal\new\alexandria-audiobook`
   (only needed once, to publish the code in step 1).

---

## 1. Publish the code to your GitHub fork (this PC, once)

Colab downloads the app from GitHub, so your branch must be pushed first.

1. Open a terminal (PowerShell or Git Bash) and run:

   ```
   cd C:\Work\personal\new\alexandria-audiobook
   git push mine my-branch
   ```

2. If git asks you to log in, or the push is rejected with a permission
   error, switch the GitHub CLI to your personal account and try again:

   ```
   gh auth switch -u bohdan-natsevych
   git push mine my-branch
   gh auth switch -u Dan-Natsevych_pdfs
   ```

   (The last line switches back to your work account.)

3. Check in a browser that the branch exists:
   https://github.com/bohdan-natsevych/alexandria-audiobook/tree/my-branch

Repeat step 1 whenever you change the code on this PC.

---

## 2. Create an LLM API key (once)

The app needs an LLM to read the book (script, review, character
descriptions). This book needs roughly 130-180 LLM requests in total. Pick ONE
option and note its three values; you enter them in section 4.3.

### Option A - Gemini (free, may take two days)

1. Open https://aistudio.google.com/apikey and sign in with your Google account.
2. Click **Create API key**, then copy the key (a long string starting with
   `AIza`). Keep it private.
3. Values for section 4.3:
   - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
   - API Key: the key you copied
   - Model Name: `gemini-2.5-flash` (current names are listed at
     https://ai.google.dev/gemini-api/docs/models - use any "Flash" model from
     that page if this one is no longer offered)

The free tier limits requests per day. If you hit the limit, the app stops,
keeps its progress, and you continue the next day (see section 7). To avoid the
limit, set up billing for the key in AI Studio (a few dollars for this book).

### Option B - DeepSeek (paid, about 1-3 USD for this book)

1. Open https://platform.deepseek.com and create an account.
2. Open **Top up** and add a small balance (for example 5 USD).
3. Open **API keys** -> **Create new API key**, copy the key (starts with
   `sk-`). It is shown only once. Keep it private.
4. Values for section 4.3:
   - Base URL: `https://api.deepseek.com/v1`
   - API Key: the key you copied
   - Model Name: `deepseek-flash` (or `deepseek-v4-pro` for higher quality,
     about 3x the price). Current names: https://api-docs.deepseek.com/quick_start/pricing

DeepSeek is half price outside its peak hours (peak: 01:00-04:00 and
06:00-10:00 UTC, Monday-Friday). There is no daily request limit to wait for;
if the balance runs out, the app stops and keeps its progress - top up and
continue (section 7).

---

## 3. Open the notebook in Colab

1. Open this link in Chrome (sign in with the same Google account as Drive):

   https://colab.research.google.com/github/bohdan-natsevych/alexandria-audiobook/blob/my-branch/alexandria_colab.ipynb

2. The notebook opens with numbered sections:
   "1. Check GPU Runtime", "2. Mount Google Drive", "3. Install Alexandria",
   "4. Start Alexandria", "5. (Optional) Install Ollama", "6. View Server Logs",
   "7. Stop Server".
3. To run a code cell: hover over it and click the round **play button** on its
   left (or click inside it and press **Ctrl+Enter**). A spinning icon means it
   is running; a green check mark next to it means it finished.
4. Do **not** use "Run all" - in session A the GPU check cell is supposed to fail
   and would stop "Run all".
5. If Colab shows "Warning: This notebook was not authored by Google", click
   **Run anyway**.

---

## 4. Session A - script generation on CPU

### 4.1 Choose the CPU runtime

1. Menu **Runtime -> Change runtime type**.
2. Under **Hardware accelerator** select **CPU**, click **Save**.
3. Click **Connect** (top right) if it is not connected yet. When connected it
   shows RAM and Disk bars.

### 4.2 Start the app

Skip section "1. Check GPU Runtime" in this session (it only works on a GPU).

1. **Section 2 "Mount Google Drive"** - run the cell.
   - A dialog asks "Permit this notebook to access your Google Drive files?"
     -> click **Connect to Google Drive** -> choose your account -> on the
     permissions page click **Select all** (if shown) -> **Continue**.
   - Done when it prints `Alexandria files and TTS models will persist across sessions.`
2. **Section 3 "Install Alexandria"** - before running, fill in the form on the
   right side of the cell:
   - `REPO_URL`: `https://github.com/bohdan-natsevych/alexandria-audiobook.git`
   - `BRANCH`: `my-branch`

   Then run the cell. It takes about 3-6 minutes. Red "ERROR: pip's dependency
   resolver..." lines can appear and are usually harmless; the cell is done when
   it prints `Installation complete.`
3. **Section 4 "Start Alexandria"** - run the cell. When it prints
   `Server is running.`, click the link **"Open Alexandria in a new tab"**.
   Use that tab for everything below (the small embedded view in the notebook
   works too, but is cramped).

### 4.3 Configure the app (Setup tab)

In the Alexandria tab, open **1. Setup**:

1. **LLM Settings (Script Generation)** - enter the three values of the option
   you chose in step 2 (Base URL, API Key, Model Name).
2. **TTS Settings (Voice Generation)**
   - TTS Mode: **Local Model**
   - Device: **Auto (best available)**
   - TTS Language: **Russian**  <- important: this selects the Russian voice
     engine (ESpeech) for all character voices
   - Parallel Workers: `16`
   - Sub-batching: on
   - Max Items/Batch: `12`
   - Compile Codec: off
3. Click **Prompt Settings (Advanced)** to expand it:
   - Chunk Size (characters): leave `3000`
   - Max Tokens (response): `32768`
     (both Gemini and DeepSeek "think" before answering and that hidden
     reasoning counts against this budget; a small value makes answers come
     back cut off.)
   - Leave Top K and Min P at `0`.
4. Click **Save Configuration** at the bottom.

Settings are stored on Drive and are kept for session B.

### 4.4 Generate the script

1. Open **2. Script**.
2. Under "Select Book/Novel", click **Choose File** and pick
   `C:\Work\personal\new\example1.txt`. Wait for `Loaded: example1.txt`.
3. Leave "Single speaker mode" **unchecked**.
4. Click **Generate Annotated Script**.
5. Watch the log window below. It shows `Processing chunk N/37 ...`.
   Expect roughly 20-60 minutes.
6. Done when the log shows `Task script completed successfully.` and
   `Speakers found: ...`.

If the log ends with `Progress is saved; run Generate Script again to resume`
or `failed with return code`, see section 7 ("Resume script generation").

### 4.5 Review the script (recommended)

The review pass fixes wrong speaker labels, which matters in this book because
it has a lot of dialogue.

1. Still in **2. Script**, leave **Context Window** at `4`.
2. Click **Contextual Review (+/- N)**. The note under the button shows the
   estimated number of LLM calls.
3. Done when the log shows `Review complete:` and
   `Task review completed successfully.`

### 4.6 End session A

1. Back in the Colab notebook tab: menu **Runtime -> Disconnect and delete runtime**.

Your script is saved in Drive (`MyDrive/Alexandria/annotated_script.json`).

---

## 5. Session B - voices and audio on T4 GPU

Free GPU time is limited, so start this session when you can keep the browser
open for about 1.5-2 hours.

### 5.1 Choose the GPU runtime and start the app

1. Menu **Runtime -> Change runtime type** -> **T4 GPU** -> **Save**.
   If Colab says no GPU is available ("Cannot connect to GPU backend"), try
   again later; free GPUs are not always available.
2. Run **section 1 "Check GPU Runtime"**. It must print `GPU: Tesla T4`.
3. Run **section 2** (Drive), **section 3** (install - the form keeps your
   `REPO_URL`/`BRANCH` only if you saved a copy of the notebook, so check them
   again before running), and **section 4** (start).
4. Open Alexandria with **"Open Alexandria in a new tab"**.
5. In **1. Setup**, check that TTS Language is still **Russian**.

### 5.2 Create the character voices

1. Open **3. Voices**. You should see one card per character
   (including `NARRATOR`).
2. Switch on **Advanced** next to "Generate Personas". Leave **Batch** at `40`.
   (Advanced mode reads the whole script to collect what the book says about
   each character before writing its voice description.)
3. Click **Generate Personas**. The first run downloads the voice-design model
   (~3.5 GB, saved to Drive for next time).
4. Watch the server log if you want details: in the notebook, run
   **section 6 "View Server Logs"** (click its stop button to stop watching;
   the server keeps running).
5. When it finishes, each card shows **Voice Clone** with a reference audio file
   `designed_voices/..._preview.wav` and a reference text. Click the **play**
   button on a few cards to listen.
6. Do not change the text in the cards while persona generation is running.

If a character's voice sounds wrong:
- Open the **Designer** tab. Fill in **Voice Description** (how the voice
  should sound) and **Sample Text** (one or two Russian sentences the
  character could say), click **Generate Preview** and listen.
- When you like it, type a **Voice Name** and click **Save Voice**.
- Back in **3. Voices**, on that character's card make sure **Voice Clone** is
  selected, then pick the saved voice in the drop-down
  "-- Select voice or enter path manually --". The card fills in the audio file
  and reference text automatically and saves.

### 5.3 Generate the audio

1. Open **4. Editor**. The table lists all chunks; status starts as pending.
2. Click **Render Pending**. The first run downloads the Russian voice model
   (~1.3 GB, saved to Drive).
3. Progress is shown in the table (chunks turn green). Expect roughly
   30-75 minutes for this book on a T4.
4. If you see "CUDA out of memory" errors in the server log (section 6), click
   **Cancel**, lower **Max Items/Batch** in **1. Setup** to `6`, click
   **Save Configuration**, and click **Render Pending** again.
5. When all chunks are green, click **Merge All**. The app switches to
   **5. Result**.

Note: the "Emotion / Style" column is not used by the Russian voice engine;
the voice comes only from each character's reference sample.

### 5.4 Download the audiobook

- Easiest and most reliable for a large file: open Google Drive ->
  `MyDrive/Alexandria/cloned_audiobook.mp3` -> right-click -> **Download**.
- Or in **5. Result**, click **Download MP3**.
- Optional chapters/metadata: expand **M4B Export Options**, fill in Title and
  Author, then click **Export M4B**. The M4B file is also saved in
  `MyDrive/Alexandria/`.

### 5.5 End session B

In the notebook: **Runtime -> Disconnect and delete runtime**. This stops GPU
usage immediately.

---

## 6. Checkpoints and resuming

Every stage saves its progress to Google Drive while it runs. If the session
disconnects, the GPU quota runs out, or the LLM daily limit is hit, start a new
session (section 4.1-4.2 or 5.1) and repeat the **same** action. The app picks up
where it stopped.

| Stage | Saved progress (in `MyDrive/Alexandria/`) | How to continue | You will see in the log |
|---|---|---|---|
| Script generation | `script_progress.json` (after every chunk) | Script tab: choose the same `example1.txt` again, click **Generate Annotated Script** | `Resuming from checkpoint: N/37 chunks already done` |
| Review | `review_progress.json` (after every batch) | Click the **same** review button with the **same** Context Window value | `Resuming from checkpoint: N batch(es) already reviewed` |
| Character voices | `persona_progress.json`, `voice_config.json` (after every character / discovery batch) | Voices tab: same **Advanced** and **Batch** settings, click **Generate Personas** | `Resuming from checkpoint: N speaker(s) done, M discovery batch(es) done` |
| Audio | `chunks.json` and `voicelines/` (after every batch) | Editor tab: click **Render Pending** | Finished chunks stay green and are skipped |

Rules:
- Each progress file is deleted automatically when its stage finishes.
- A checkpoint is only used for the same input and settings. If you change the
  book, Chunk Size, the script (between an interrupted review and its resume),
  or the persona settings, the log says `Ignoring ... checkpoint` and the stage
  starts from the beginning.
- **Generate Annotated Script** and **Review** replace the chunk list. Run them
  before you edit text in the Editor tab or render audio, not after.
- Never click **Regenerate All** unless you want to throw away all audio.

---

## 7. Troubleshooting

**"Progress is saved; run ... again to resume" / "LLM API unavailable"**
The LLM stopped answering. Most often: Gemini free daily limit reached (wait
until it resets the next day) or DeepSeek balance used up (top up at
https://platform.deepseek.com). Sometimes a network error (just retry). Also
check the three LLM values in **1. Setup** if it fails right at the start. Then
resume as in section 6. No work is lost.

**Resume script generation**
After a new session: section 2, 3, 4 of the notebook -> open the app -> Script
tab -> choose `example1.txt` again -> **Generate Annotated Script**.
Choosing the file again is required after every restart; it does not reset
progress.

**"N speaker(s) did not get a generated voice"**
The voice sample for those characters could not be created (for example, GPU
out of memory). Click **Generate Personas** again; only those characters are
redone. This step must run on the T4 GPU session.

**The app tab stops responding**
Run section 6 "View Server Logs" in the notebook and read the last lines. If the
server crashed, run section 4 again and continue (all progress is on Drive).

**"No GPU detected!" in section 1**
The runtime is CPU. Change it to T4 GPU (section 5.1).

**Session disconnected by Colab**
Click **Reconnect**, then run sections 2, 3 and 4 again (and section 1 on GPU).
Then resume the stage you were in (section 6).

**Chunks with status "error" in the Editor**
Click **Render Pending** again; it retries all chunks that are not finished.
To check one chunk, use the play / regenerate buttons in its row.
