import os
import pandas as pd

base_path = "C:\\Users\\ragha\\Downloads\\MELD_Dataset\\MELD-RAW\\MELD.Raw"

splits = {
    "train": os.path.join(base_path, "train", "train_sent_emo.csv"),
    "dev": os.path.join(base_path, "dev_sent_emo.csv"),     # fixed
    "test": os.path.join(base_path, "test_sent_emo.csv"),   # fixed
}

all_utterances = []

for split, path in splits.items():
    print(f"Reading: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    df = pd.read_csv(path)
    df = df[df["Emotion"] == "joy"]
    grouped = df.groupby("Dialogue_ID")

    for dialog_id, group in grouped:
        speakers = group["Speaker"].unique()
        if len(speakers) == 2:
            for _, row in group.iterrows():
                all_utterances.append({
                    "Dialogue_ID": dialog_id,
                    "Utterance_ID": row["Utterance_ID"],
                    "Speaker": row["Speaker"],
                    "Utterance": row["Utterance"],
                    "Emotion": row["Emotion"]
                })

output_file = "joy_conversations_2speakers.csv"
df_out = pd.DataFrame(all_utterances)
df_out.to_csv(output_file, index=False)
print(f"\n✅ Saved {len(df_out)} utterances to '{output_file}'")
