import os
import pandas as pd
from collections import defaultdict

# Root path
base_path = os.getcwd()
csv_path = os.path.join(base_path, 'MELD-RAW', 'MELD.Raw', 'train', 'train_sent_emo.csv')
print(f"Reading: {csv_path}")

# Read the CSV
df = pd.read_csv(csv_path)

# Filter only 'joy' emotions
df = df[df['Emotion'] == 'joy']

# Dictionary to store conversation data
conversations = defaultdict(list)

# Group utterances by Dialogue_ID
for _, row in df.iterrows():
    conv_id = row['Dialogue_ID']
    utterance = {
        'Utterance_ID': row['Utterance_ID'],
        'Speaker': row['Speaker'],
        'Utterance': row['Utterance'],
        'Emotion': row['Emotion'],
        'Sentiment': row['Sentiment']
    }
    conversations[conv_id].append(utterance)

# Filter to only conversations between exactly 2 speakers
two_person_joy_conversations = {}
for conv_id, utterances in conversations.items():
    speakers = set([utt['Speaker'] for utt in utterances])
    if len(speakers) == 2:
        two_person_joy_conversations[conv_id] = utterances

# Limit to 1000 conversations
limited_conversations = dict(list(two_person_joy_conversations.items())[:1000])

# Save the filtered data to a new CSV
output_data = []
for conv_id, utterances in limited_conversations.items():
    for utt in utterances:
        row = {
            'Dialogue_ID': conv_id,
            **utt
        }
        output_data.append(row)

output_df = pd.DataFrame(output_data)
output_df.to_csv("joy_conversations_2speakers.csv", index=False)

print(f"Saved {len(limited_conversations)} conversations to 'joy_conversations_2speakers.csv'")
