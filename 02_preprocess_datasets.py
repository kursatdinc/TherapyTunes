import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

###########
# READ DATA - MENTAL SURVEY
###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
###########


###########
# DROP COLUMNS
###########
mental_survey_df.drop(columns=["Timestamp", "Composer", "Foreign languages", "Frequency [Rap]",
                               "Frequency [Video game music]", "OCD", "Permissions"], inplace=True)
###########


###########
# CHANGE COLUMN NAMES
###########
changed_column_names = {"Age":"age",
                        "Primary streaming service":"streaming_service",
                        "Hours per day":"hours_per_day",
                        "While working":"while_working",
                        "Instrumentalist":"instrumentalist",
                        "Fav genre":"fav_genre",
                        "Exploratory":"exploratory",
                        "BPM":"tempo",
                        "Frequency [Classical]":"frequency_classical",
                        "Frequency [Country]":"frequency_country", 
                        "Frequency [EDM]":"frequency_edm",
                        "Frequency [Folk]":"frequency_folk", 
                        "Frequency [Gospel]":"frequency_gospel", 
                        "Frequency [Hip hop]":"frequency_hiphop",
                        "Frequency [Jazz]":"frequency_jazz", 
                        "Frequency [K pop]":"frequency_kpop", 
                        "Frequency [Latin]":"frequency_latin",
                        "Frequency [Lofi]":"frequency_lofi", 
                        "Frequency [Metal]":"frequency_metal", 
                        "Frequency [Pop]":"frequency_pop",
                        "Frequency [R&B]":"frequency_rnb", 
                        "Frequency [Rock]":"frequency_rock", 
                        "Anxiety":"anxiety", 
                        "Depression":"depression",
                        "Insomnia":"insomnia", 
                        "Music effects":"music_effects"}
mental_survey_df = mental_survey_df.rename(columns=changed_column_names)
###########


###########
# CHANGE VARIABLE NAMES
###########
replaced_streaming = {"Pandora":"Other",
                      "I do not use a streaming service.":"Other",
                      "Other streaming service":"Other"}
mental_survey_df["streaming_service"] = mental_survey_df["streaming_service"].replace(replaced_streaming)

replaced_fav_genres = {"K pop":"K-Pop",
                       "Hip hop":"Hip-Hop",
                       "Rap":"Hip-Hop"}
mental_survey_df["fav_genre"] = mental_survey_df["fav_genre"].replace(replaced_fav_genres)

replaced_selection_columns = ["frequency_classical", "frequency_country", "frequency_edm",
                              "frequency_folk", "frequency_gospel", "frequency_hiphop",
                              "frequency_jazz", "frequency_kpop", "frequency_latin", "frequency_lofi",
                              "frequency_metal", "frequency_pop", "frequency_rnb", "frequency_rock"]
for col in replaced_selection_columns:
    mental_survey_df[col] = mental_survey_df[col].replace({"Very frequently":"Often"})

mental_survey_df["music_effects"] = mental_survey_df["music_effects"].replace({"No effect":"No Effect"})
###########


###########
# CHANGE RARE GENRES
###########
video_game_indices = mental_survey_df[mental_survey_df["fav_genre"] == "Video game music"].index
shuffled_indices = np.random.permutation(video_game_indices)

mental_survey_df.loc[shuffled_indices[:22], "fav_genre"] = "EDM"
mental_survey_df.loc[shuffled_indices[22:], "fav_genre"] = "Lofi"
###########


###########
# FILLING MISSING VALUES
###########
mental_survey_df["age"] = mental_survey_df["age"].fillna(mental_survey_df["age"].median())
mental_survey_df["tempo"] = mental_survey_df["tempo"].fillna(mental_survey_df.groupby("fav_genre")["age"].transform("median"))
mental_survey_df["streaming_service"] = mental_survey_df["streaming_service"].fillna(mental_survey_df["streaming_service"].mode().iloc[0])
mental_survey_df["while_working"] = mental_survey_df["while_working"].fillna(mental_survey_df["while_working"].mode().iloc[0])
mental_survey_df["instrumentalist"] = mental_survey_df["instrumentalist"].fillna(mental_survey_df["instrumentalist"].mode().iloc[0])
mental_survey_df["music_effects"] = mental_survey_df["music_effects"].fillna(mental_survey_df["music_effects"].mode().iloc[0])
###########


###########
# CHANGE COLUMN TYPES
###########
mental_survey_df["anxiety"].value_counts()
mental_survey_df["anxiety"] = mental_survey_df["anxiety"].replace({7.5:7})
mental_survey_df["anxiety"] = mental_survey_df["anxiety"].astype(int)

mental_survey_df["depression"].value_counts()
mental_survey_df["depression"] = mental_survey_df["depression"].replace({3.5:3})
mental_survey_df["depression"] = mental_survey_df["depression"].astype(int)

mental_survey_df["insomnia"].value_counts()
mental_survey_df["insomnia"] = mental_survey_df["insomnia"].replace({3.5:3})
mental_survey_df["insomnia"] = mental_survey_df["insomnia"].astype(int)

mental_survey_df["age"] = mental_survey_df["age"].astype(int)

mental_survey_df.info()
###########


###########
# EXPORT DATA
###########
mental_survey_df.reset_index(drop=True, inplace=True)

mental_survey_df.to_csv("./datasets/mental_final.csv", index=False)

mental_survey_df = pd.read_csv("./datasets/mental_final.csv")
mental_survey_df.head(10)
###########





###########
# READ DATA - SPOTIFY
###########
spotify_df = pd.read_csv("./datasets/spotify_data.csv", index=False)
###########


###########
# GENRE MATCH
###########
deleted_genres = ["afrobeat",
                  "comedy",
                  "french",
                  "german",
                  "groove",
                  "indian",
                  "party",
                  "romance",
                  "sad",
                  "show-tunes",
                  "singer-songwriter",
                  "ska",
                  "swedish",
                  "songwriter"]
spotify_df = spotify_df[~spotify_df["genre"].isin(deleted_genres)]

replaced_genres = { "acoustic":"Classical",
                    "alt-rock":"Rock",
                    "ambient":"Lofi",
                    "black-metal":"Metal",
                    "blues":"Jazz",
                    "breakbeat":"EDM",
                    "cantopop":"Pop",
                    "chicago-house":"EDM",
                    "chill":"Lofi",
                    "classical":"Classical",
                    "club":"EDM",
                    "country":"Country",
                    "dance":"EDM",
                    "death-metal":"Metal",
                    "deep-house":"EDM",
                    "detroit-techno":"EDM",
                    "disco":"EDM",
                    "drum-and-bass":"EDM",
                    "dub":"EDM",
                    "dubstep":"EDM",
                    "edm":"EDM",
                    "electro":"EDM",
                    "electronic":"EDM",
                    "emo":"Rock",
                    "folk":"Folk",
                    "forro":"Latin",
                    "funk":"R&B",
                    "garage":"EDM",
                    "gospel":"Gospel",
                    "goth":"Rock",
                    "grindcore":"Metal",
                    "guitar":"Classical",
                    "hard-rock":"Rock",
                    "hardcore":"Metal",
                    "hardstyle":"Metal",
                    "heavy-metal":"Metal",
                    "hip-hop":"Hip-Hop",
                    "house":"EDM",
                    "indie-pop":"Pop",
                    "industrial":"EDM",
                    "jazz":"Jazz",
                    "k-pop":"K-Pop",
                    "metal":"Metal",
                    "metalcore":"Metal",
                    "minimal-techno":"EDM",
                    "new-age":"EDM",
                    "opera":"Classical",
                    "piano":"Classical",
                    "pop":"Pop",
                    "pop-film":"Pop",
                    "power-pop":"Pop",
                    "progressive-house":"EDM",
                    "psych-rock":"Rock",
                    "punk":"Rock",
                    "punk-rock":"Rock",
                    "rock":"Rock",
                    "rock-n-roll":"Rock",
                    "salsa":"Latin",
                    "samba":"Latin",
                    "sertanejo":"Latin",
                    "sleep":"Lofi",
                    "soul":"R&B",
                    "spanish":"Latin",
                    "tango":"Latin",
                    "techno":"EDM",
                    "trance":"EDM",
                    "trip-hop":"EDM",
                    "dancehall":"EDM",}
spotify_df["genre"] = spotify_df["genre"].replace({replaced_genres})


###########
# EXPORT DATA
###########
spotify_df.reset_index(drop=True, inplace=True)

spotify_df.to_csv("./datasets/spotify_final.csv")

spotify_final_df = pd.read_csv("./datasets/spotify_final.csv")
spotify_final_df.head(10)
###########