import pandas as pd

pd.set_option('display.max_rows', None)

df = pd.read_csv("./datasets/horoscope.csv")
df.drop(columns=["date_range", "description", "compatibility", "color", "lucky_number",
                 "lucky_time"], axis=1, inplace=True)

df["current_date"] = pd.to_datetime(df["current_date"])

df_pivot = df.pivot_table(values="mood", index="current_date", columns="sign", aggfunc="first")
df_pivot = df_pivot.iloc[29:-4]
df_pivot.reset_index(inplace=True)
df_pivot.drop(columns="current_date", axis=1, inplace=True)
df_pivot = df_pivot.rename_axis("month_date", axis="columns")

df_pivot.to_csv("./datasets/horoscope.csv", index=False)

all_values = df_pivot.values.flatten().tolist()
unique_values = list(set(all_values))
len(unique_values) #65
#  ['Warm',
#  'Calm',
#  'Responsible',
#  'Joyful',
#  'Cautious',
#  'Grateful',
#  'Surprised',
#  'Social',
#  'Busy',
#  'Creative',
#  'Happy',
#  'Energetic',
#  'Helpful',
#  'Truthful',
#  'Conservative',
#  'Dreamy',
#  'Fun',
#  'Quiet',
#  'Honest',
#  'Pleasant',
#  'Satisfied',
#  'Curious',
#  'Friendly',
#  'Innocent',
#  'Cool',
#  'Successful',
#  'Cherishing',
#  'Touched',
#  'Loved',
#  'Diligent',
#  'Confidence',
#  'Sweet',
#  'Stubborn',
#  'Silly',
#  'Lucky',
#  'Relaxed',
#  'Depressed',
#  'Serious',
#  'Focus',
#  'Generous',
#  'Sincere',
#  'Aggressive',
#  'Humble',
#  'Artistic',
#  'Relieved',
#  'Indifferent',
#  'Hopeful',
#  'Moody',
#  'Persuade',
#  'Stressed',
#  'Independent',
#  'Productive',
#  'Smart',
#  'Mellow',
#  'Refreshed',
#  'Talkative',
#  'Excited',
#  'Attractive',
#  'Pleased',
#  'Patient',
#  'Charming',
#  'Brave',
#  'Thoughtful',
#  'Accomplished',
#  'Peaceful']