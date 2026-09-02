# cvoice

Clone a voice from about twenty seconds of speech, then make it say anything.
Runs entirely on your own hardware — a server that holds the model, and a thin
client that talks to it over HTTP.

Serbian-first, because that is what it was built and measured for. The
underlying model covers 600+ languages, so other languages work; they just
haven't been through the same testing.

```
cvoice                       # record 20 s, audition it, save it as a profile
reci "Ćao Mina, ja sam Bogdan."   # say something in that voice
reci -p sanja "Drugi glas."       # ...or in another
```

## What works today

- **Voice cloning from ~20 s of audio.** No fine-tuning, no training run.
- **Named profiles.** Enrol as many speakers as you like; pick one per line.
- **Automatic take selection.** Generates several attempts, transcribes each
  with Whisper, keeps the one closest to what you asked for. On short lines
  this matters more than it sounds — see *Why several takes* below.
- **Microphone gain calibration** during enrolment (Linux; guided elsewhere).
- **Optional Discord delivery.** If configured, the result is posted as an
  attachment; otherwise you get a `.wav`.
- **Long text** is split at sentence boundaries so a two-minute script does not
  blow up GPU memory.

## What does not exist yet

- No fine-tuning. This is zero-shot cloning only; quality is what the reference
  gives you.
- No streaming. You get the whole clip when it is done.
- No Windows/macOS *server*. The client runs everywhere; the server has only
  been run on Linux with an NVIDIA GPU. It should work on CPU and on Apple
  Silicon, but that is untested and will be slow.
- No web UI. Command line only.
- Take-scoring quality depends on Whisper's Serbian, which is decent but not
  perfect — treat the numbers as a screen, not a verdict.

## Install

Via [ownbox](https://github.com/BogdanStamenovic/ownbox):

```bash
ownbox install cvoice
```

The installer asks whether this machine is a **client**, a **server**, or
**both**. Client installs ask which server to connect to.

Or directly:

```bash
git clone https://github.com/BogdanStamenovic/cvoice
cd cvoice
python3 install.py          # add --role server|client|both to skip the prompt
```

### Server

Needs Python 3.10+ and, realistically, an NVIDIA GPU with ~4 GB free. It has
been run on an RTX 4060 (8 GiB), where generation is roughly 5× faster than
real time and the model holds ~2.2 GiB.

The model (~3 GB) downloads on **first generation**, not at install, so the
install stays fast and you only pay for what you use.

```bash
systemctl --user enable --now cvoiced     # Linux
cvoiced                                    # or run it in the foreground
```

The installer prints a token. The server binds to `127.0.0.1` by default;
if you expose it, put it on a tailnet or behind a reverse proxy rather than
straight onto the internet.

### Client

Needs Python 3.10+ and a microphone for enrolment. No GPU, no torch.

## Configuration

`~/.config/cvoice/config.toml` (`%APPDATA%\cvoice\config.toml` on Windows):

```toml
[client]
server  = "http://myserver:8760"
token   = "..."
profile = "bogdan-stamenovic"   # default voice
takes   = 3

[server]
host     = "127.0.0.1"
port     = 8760
token    = "..."
data_dir = "~/.local/share/cvoice"
language = "sr"

[discord]                        # entirely optional
enabled         = true
bot_token       = "..."
channel_id      = "..."
mention_user_id = "..."
```

The Discord token stays on the **client**. The server never sees it.

## Writing text that sounds right

These are measured, not folklore:

| you write | you get |
|---|---|
| `č ć ž š đ` | correct pronunciation |
| `c z s d` instead | roughly **4× the error rate** — always use diacritics |
| `,` | a breath |
| `...` | a hesitation (~1.5 s in testing) |
| `.` | a closed thought |
| `Sanjaaa` | a drawn-out call (~9% longer than `Sanja`) |

Latin script beats Cyrillic as *input* — Cyrillic input lost the `lj` digraph in
testing. What the reader sees is your business; this is only about what you feed
the model.

## Why several takes

On a four-word sentence, two of three takes came back wrong in testing — one
with the wrong vocative, one slurring two words together. Short lines give the
model very little context to stabilise on. So `reci` generates three by default,
transcribes each, and keeps the closest match. `-1` turns it off.

## Choosing a reference

The model copies **delivery**, not just timbre. A reference read in a calm,
even voice produces calm, even speech no matter what the text says. If you want
something whispered and tired, record the reference whispered and tired.

In testing, a register-matched reference produced 30% slower speech *and* lower
error rates than a neutral one on the same text — so this is not a tradeoff, it
is just better.

Roughly 20 seconds is the useful length. Ten pins the timbre but leaves phrasing
to guesswork; past about 26 seconds there was no further gain.

## Recording a good reference

- **Not Bluetooth.** HFP/mSBC is band-limited and removes exactly the high
  frequencies the speaker encoder relies on. `cvoice` refuses Bluetooth inputs.
- **Quiet room**, mic about a hand-span away and slightly off-axis so plosives
  do not hit the capsule straight on.
- **Do not clip.** On Linux `cvoice` sets the input gain for you and refuses a
  clipped take. Elsewhere it measures and tells you to turn the gain down.
  Clipping is not cosmetic: the encoder learns the distortion as part of the
  voice, and no declipper recovers it afterwards.

## Licence

Code: MIT.

The default model, [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), is
Apache-2.0 for the code and **CC-BY-NC for the weights** — non-commercial use
only. cvoice downloads the weights; it does not redistribute them. If you need
commercial use, point `server.model` at something you are licensed for.

## Don't clone people without asking them

Enrolling your own voice is nobody's business but yours. Enrolling somebody
else's is their decision to make, and it should be an actual conversation, not
a recording taken off a call. Impersonating a real person to deceive someone is
fraud in most places, and this tool makes it easy enough that the ease is not
an excuse.
