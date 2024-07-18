import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)

mental_survey_df.drop(columns=["Timestamp", "Composer", "Foreign languages", "Frequency [Rap]", "Frequency [Video game music]", "OCD", "Permissions"], inplace=True)


replaced_streaming = {"Pandora":"Other",
                      "I do not use a streaming service.":"Other",
                      "Other streaming service":"Other"}
mental_survey_df["Primary streaming service"] = mental_survey_df["Primary streaming service"].replace(replaced_streaming)


mental_survey_df["Age"] = mental_survey_df["Age"].fillna(mental_survey_df["Age"].median())
mental_survey_df["BPM"] = mental_survey_df["BPM"].fillna(mental_survey_df["BPM"].median())
mental_survey_df["Primary streaming service"] = mental_survey_df["Primary streaming service"].fillna(mental_survey_df["Primary streaming service"].mode().iloc[0])
mental_survey_df["While working"] = mental_survey_df["While working"].fillna(mental_survey_df["While working"].mode().iloc[0])
mental_survey_df["Instrumentalist"] = mental_survey_df["Instrumentalist"].fillna(mental_survey_df["Instrumentalist"].mode().iloc[0])
mental_survey_df["Music effects"] = mental_survey_df["Music effects"].fillna(mental_survey_df["Music effects"].mode().iloc[0])


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


video_game_indices = mental_survey_df[mental_survey_df["fav_genre"] == "Video game music"].index
shuffled_indices = np.random.permutation(video_game_indices)

mental_survey_df.loc[shuffled_indices[:22], "fav_genre"] = "EDM"
mental_survey_df.loc[shuffled_indices[22:], "fav_genre"] = "Lofi"

replaced_fav_genres = {"K pop":"K-Pop",
                       "Hip hop":"Hip-Hop",
                       "Rap":"Hip-Hop"}

mental_survey_df["fav_genre"] = mental_survey_df["fav_genre"].replace(replaced_fav_genres)


mental_survey_df.to_csv("./datasets/mental_final.csv")
###########

###########
spotify_df = pd.read_csv("./datasets/spotify_data.csv")
spotify_df.head(10)

mental_survey_df["fav_genre"].value_counts()

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

spotify_df["genre"] = spotify_df["genre"].replace(replaced_genres)

spotify_df["genre"].value_counts()


spotify_df.to_csv("./datasets/spotify_final.csv")
###########

###########
mental_final_df = pd.read_csv("./datasets/mental_final.csv")
mental_final_df.head(10)
###########

###########
spotify_final_df = pd.read_csv("./datasets/spotify_final.csv")
spotify_final_df.head(10)
###########