# cvoice

Clone a voice from about twenty seconds of speech, then make it say anything.
Runs entirely on your own hardware — a server that holds the model, and a thin
client that talks to it over HTTP.

Serbian-first, because that is what it was built and measured for. The
underlying model covers 600+ languages, so other languages work; they just
haven't been through the same testing.

```
cvoice                       # record 20 s, audition it, save it as a profile
reci "Zdravo, ovo je moj glas."    # say something in that voice
reci -p marko-ilic "Drugi glas."       # ...or in another
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

The first time you run `reci` on a machine that has never been configured, it
asks where the server is and what the password is. Both have defaults — enter
nothing and you get `localhost:8760` with no password, which is what a
single-machine install wants:

```console
$ reci "Zdravo."
cvoice — prvo pokretanje
  Gde je server? Ostavi prazno za lokalni.
  server [localhost:8760]:
  lozinka [bez lozinke]:
```

It is asked once and remembered. `reci --configure` changes it later. A bare
host is accepted as well as a full URL — `archserver`, `archserver:9001`,
`10.0.0.4` and `https://voice.example.com` all work.

## Enrolling from a file or a link

When the voice you want is already recorded, `cvoice` can take it directly
instead of going through the microphone:

```bash
cvoice --from-wav interview.m4a --start 1:24 --duration 18 --denoise \
       --text "exactly what is said in that clip"

cvoice --from-url "https://..." --start 2:05 --duration 20 --denoise
```

The clip is trimmed, downmixed to 48 kHz mono, optionally denoised and
normalised to the same -9 dBFS target the microphone path uses, so a downloaded
reference and a recorded one are handled identically from there on. With a start
and duration, only that window is fetched rather than the whole video.

**Pass `--text`.** The reference transcript is what the model conditions on; if
it does not match the audio, cloning is measurably worse. `cvoice` warns when it
is missing rather than quietly substituting the enrolment passage.

**`--denoise` is a trade, not a free win.** Noise reduction removes some of the
high-frequency detail the speaker encoder relies on. The settings here are
measured rather than chosen by feel - on a test clip with pink noise mixed in:

| filter | SNR | energy above 8 kHz | bandwidth |
|---|---|---|---|
| noisy input | 12.2 dB | 3.65% | 16.3 kHz |
| `afftdn=nr=12` | 22.1 dB | **1.77%** | 12.3 kHz |
| `afftdn=nr=6:tn=1` (used) | 18.1 dB | 3.74% | 14.5 kHz |

The aggressive setting buys the most SNR by cutting 4 kHz off the top and
halving the detail being cloned. The shipped setting keeps nearly all the gain
with the highs intact. The un-denoised version is written alongside as
`ref-raw.wav` so you can compare.

`--yes` with `--name` skips the audition for scripted use.

## Moving a voice between machines

Profiles live on the server. To copy one somewhere else, `cvoice` shells out to
[fiotransfer](https://github.com/BogdanStamenovic/fiotransfer), which uploads to
anonymous temporary file hosts and returns a short code:

```console
$ cvoice profiles share            # no names: pick from a list
Koje profile deliš?
 ❯ ● ana-maric               Ana Marić              10.5s
   ● ana-maric-tiho          Ana Marić (tiho)       26.0s
   ○ marko-ilic              Marko Ilić             21.2s
   ↑↓ move · space select · a all · enter confirm · q cancel

  cvoice-2-profila.tar.gz · 2821 KB
  u:h.uguu.se/HoFGMzex

$ cvoice profiles get u:h.uguu.se/HoFGMzex
Instalirano (2).
  Ana Marić         (ana-maric, 10.47s)
  Ana Marić (tiho)  (ana-maric-tiho, 25.98s)
```

Naming profiles on the command line skips the picker. `--save-only` downloads
the archive without installing; `--name` renames a single-profile archive.
With no terminal to draw a picker on, `share` with no names refuses rather than
guessing which voices to publish.

**The upload is public.** Those hosts are anonymous and unauthenticated: anyone
holding the code can download the reference recording of that person's voice.
Fine for moving your own voice between your own machines. Think twice before
doing it with somebody else's.

## Configuration

`~/.config/cvoice/config.toml` (`%APPDATA%\cvoice\config.toml` on Windows):

```toml
[client]
server  = "http://myserver:8760"
token   = "..."
profile = "ana-maric"           # default voice
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
