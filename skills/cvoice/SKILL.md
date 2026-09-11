---
name: cvoice
description: Speak text aloud in a cloned voice, locally, and manage voice profiles. Use whenever a task needs synthesized speech in a specific person's voice - a spoken message, a voiceover, a TTS clip, an audio reply, "say this as X", "record this line", or enrolling a new voice from a sample. Serbian-first (diacritics matter), but the model covers 600+ languages. Runs on the local cvoice server; nothing is sent to a cloud TTS.
---

# Speak text in a cloned voice (cvoice)

`cvoice` clones a voice from ~20 s of speech and then speaks any text in it, over a local
server. The client is a thin CLI; the model runs on the machine hosting `cvoiced`. Use it to
produce a `.wav` (or, if configured, a Discord attachment) of a line spoken in a chosen voice.

## Speaking a line

```bash
cvoice say -p bogdan-stamenovic "Zdravo, ovo je poruka." -s out.wav -d
```

- `-p SLUG` picks the voice. `-s FILE` writes the `.wav` there. `-d` (dry) keeps it local
  instead of posting to Discord — pass it unless the task is explicitly to deliver over Discord.
- `-t N` sets how many takes to generate before the best is auto-selected (default 3); `-1`
  forces a single take with no check, faster but lower quality on short lines.
- List available voices: `cvoice say -l` or `cvoice profiles list`.

**Diacritics are mandatory for Serbian.** The ear needs them even though written Serbian for
the eye is often stripped to ASCII: write `"šifra"`, not `"sifra"`; `"đ"`, `"č"`, `"ć"`, `"ž"`,
`"š"`. A stripped line is mispronounced. (This is the inverse of outward *written* text, which
is ASCII — here the text is spoken, so it gets the diacritics.)

**Long text** is split at sentence boundaries automatically, so a paragraph is fine in one call.

## Enrolling a new voice

```bash
cvoice --from-wav sample.m4a --duration 18 --text "what is said in the sample"
cvoice --from-url URL --start 1:24 --duration 18 --denoise
cvoice                       # record ~20 s from the microphone and audition it
```

A clean ~20 s sample clones better than a long noisy one. `--denoise` helps a noisy source;
skip it on a clean one. Give `--text` the actual words in the sample when you know them.

## Profiles are personal data

The enrolled voices are real people. Do not create, share, or export a profile of someone
without a reason to, and never publish profile audio — the profiles live outside the repo and
are gitignored precisely because they are third parties' voices. `cvoice profiles share|get`
move a profile between machines; treat that as moving someone's biometric sample.

## Checking the install

```bash
cvoice doctor          # platform, audio, and whether the server is reachable
```

If `doctor` reports the server unreachable, the `cvoiced` half is not running or not installed
on the host — this skill only drives it, it does not start it. See the cvoice README for the
server role.

## What it is and is not

- **Is:** zero-shot voice cloning and TTS, local, Serbian-measured, 600+ languages nominally.
- **Is not:** fine-tuning (quality is whatever the reference gives), streaming (you get the whole
  clip at once), or a way to start/stop the model server (that is `cvoiced`, installed separately
  for a server-role machine).

Exit codes follow the standard convention: 0 success, non-zero failure.
