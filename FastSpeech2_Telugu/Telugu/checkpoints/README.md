# Telugu checkpoints

Do not commit raw source audio or unverified temporary MFA files here.

The current Drive artifact is:

`models/telugu/telugu_acoustic_model_sat2_recovered.zip`

It is a recoverable MFA 3.4.3 SAT2 GMM-HMM acoustic-model package (22.95 MB). It passed ZIP integrity checking, `mfa model inspect acoustic`, and a one-utterance `mfa align_one` smoke test. The original MFA training command did not exit successfully because the final pronunciation-probability calculation raised `ZeroDivisionError`; treat this as a recovered SAT2 model checkpoint, not a fully completed standard training export.

To use it from this project, copy it to:

`Telugu/checkpoints/telugu_acoustic_model_sat2_recovered.zip`

For GitHub, prefer Git LFS or an externally hosted, access-controlled checkpoint link. The checkpoint does not belong in the regular Git history if your repository has a file-size limit. Do not upload IndicTTS source audio.

No Telugu FastSpeech 2 neural checkpoint exists yet. Add it here only after FastSpeech 2 training has actually completed.
