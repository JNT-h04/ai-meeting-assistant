# Speaker Attribution

Diarization only produces anonymous speaker IDs (`SPEAKER_00`, `SPEAKER_01`, ...). Names are recovered opportunistically from self-introductions in the transcript.

```mermaid
flowchart TD
    A["ASR segment: '...my name is John Smith...'"] --> B{Regex match?\n'my name is X' / 'this is X'}
    B -- yes --> C[Extract name]
    C --> D[speaker_id -> name\nmapping table]
    B -- no --> E[No update to mapping table]

    D --> F[Every future segment from\nthat speaker_id is relabeled]
    E --> F
    F --> G{Speaker ever\nintroduced themselves?}
    G -- no --> H[Falls back to raw\nSPEAKER_id in transcript]
    G -- yes --> I[Shown as their real name\nin transcript]
```

**Limitation:** if a speaker never says a self-introducing phrase, they stay labeled by their raw diarization ID for the whole transcript.
