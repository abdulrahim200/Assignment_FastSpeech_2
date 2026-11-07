# Exact Telugu test sentences entered in notebook cell 56.

TELUGU_TEST_SENTENCE_IDS = ["T1", "T2a", "T2b", "T3", "T4"]
TELUGU_TEST_SENTENCES = [
    "న్యూరోఫార్మాకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
]

# Keep original strings unchanged for assignment compliance. Normalize only in separate preprocessing code if needed.
filled_test_sentences = [(sid, sentence) for sid, sentence in zip(TELUGU_TEST_SENTENCE_IDS, TELUGU_TEST_SENTENCES) if sentence.strip()]
assert len(filled_test_sentences) == 5
for sid, sentence in filled_test_sentences:
    print(f"{sid}: {sentence}")
