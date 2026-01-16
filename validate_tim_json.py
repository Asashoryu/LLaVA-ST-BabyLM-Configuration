import json

json_path = "data/localized_narratives/llava_datasets/all_shards_tim_merged.json"

try:
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_path}")
    # Optionally, print a sample and check required fields
    if len(data) > 0:
        print("Sample entry:")
        print(json.dumps(data[0], indent=2))
    # Check for missing/empty/invalid fields
    required_fields = ["id", "origin", "sample_type", "word_count", "conversations"]
    conversation_fields = ["from", "value"]
    missing = 0
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            print(f"Entry {i} is not a dict!")
            missing += 1
            continue
        for field in required_fields:
            if field not in ex:
                print(f"Entry {i} missing field: {field}")
                missing += 1
        if "word_count" in ex and (not isinstance(ex["word_count"], int) or ex["word_count"] <= 0):
            print(f"Entry {i} has invalid word_count: {ex.get('word_count')}")
            missing += 1
        if "conversations" in ex:
            if not isinstance(ex["conversations"], list) or len(ex["conversations"]) == 0:
                print(f"Entry {i} has invalid or empty conversations")
                missing += 1
            else:
                for j, conv in enumerate(ex["conversations"]):
                    if not isinstance(conv, dict):
                        print(f"Entry {i} conversation {j} is not a dict!")
                        missing += 1
                        continue
                    for cfield in conversation_fields:
                        if cfield not in conv or not conv[cfield]:
                            print(f"Entry {i} conversation {j} missing or empty field: {cfield}")
                            missing += 1
    print(f"Found {missing} problematic entries.")
except Exception as e:
    print(f"Error loading JSON: {e}")
