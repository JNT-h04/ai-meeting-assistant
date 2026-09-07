# Sample Data

Six sample meetings from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (`ES2002a/b/c`, `ES2003a/b/c`, Mix-Headset mix) were used for development and testing.

- The `.wav` files themselves (~350 MB total) are **not committed to git** — see `.gitignore`. Download them from the [AMI Corpus downloader](https://groups.inf.ed.ac.uk/ami/download/) (select the `Mix-Headset` audio for meetings `ES2002a`, `ES2002b`, `ES2002c`, `ES2003a`, `ES2003b`, `ES2003c`) and place them in this folder.
- `diarization/` and `transcripts/` contain small JSON outputs from earlier pipeline runs, kept as examples of the expected output shape.
- `final_transcripts/` and `final_summary.txt` are example end-to-end outputs for `ES2003c` / `ES2003b`.
