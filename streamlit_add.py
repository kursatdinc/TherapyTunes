        # def mental_feature_eng(df):
        #     return mental_inpuf_fe_df

        # mental_inpuf_fe_df = mental_feature_eng(mental_input_df)

        models = {"Anxiety": joblib.load("./models/anx_model.pkl"),
                  "Depression": joblib.load("./models/dep_model.pkl"),
                  "Insomnia": joblib.load("./models/ins_model.pkl"),
                  "Obsession": joblib.load("./models/obs_model.pkl")}   

        predictions = {}
        for condition, model in models.items():
            predictions[condition] = model.predict_proba(mental_input_fe_df)[0][1]

        
        st.subheader("Mental Health Predictions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anxiety", f"{predictions["Anxiety"]:.2%}")
        with col2:
            st.metric("Depression", f"{predictions["Depression"]:.2%}")
        with col3:
            st.metric("Insomnia", f"{predictions["Insomnia"]:.2%}")

        
        spoti_input = {}

        spoti_input_df = pd.DataFrame(data=spoti_input, index=[0])

        # def spoti_feature_eng(df):
        #     return spoti_input_fe_df

        # spoti_input_fe_df = spoti_feature_eng(spoti_input_df)

        # spoti_model = joblib.load("./models/spoti_model.pkl")

        # spoti_model_predict = spoti_model.predict(spoti_input_fe_df)

        def get_recommendations(n=3, df, spoti_model_predict, answer_dict):
            recom_pool = df[df["cluster"] = spoti_model_predict and df["pc_segment"] = answer_dict.get("pc_segment") and df["genre"] = answer_dict.get("fav_genre")].iloc[:100]
            
            return recom_pool.sample(n)["track_id"].tolist()


        if st.button("Would You Like Us to Recommend a Song?"):
            st.session_state.show_recommendation = True
            st.session_state.recommendations = get_recommendations(n=3, df, spoti_model_predict, answer_dict)

    
        if st.session_state.get("show_recommendation", False):
            st.write("Here's Your Personalized Song Recommendations")

            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
        
            for i in st.session_state.recommendations:
                with columns[i]:
                    spotify_player(i)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Get New Recommendations"):
                    st.session_state.recommendations = get_recommendations()
                    st.rerun()
            
            with col2:
                if st.button("Start Quiz Again"):
                    st.session_state.question_index = 0
                    st.session_state.quiz_data = get_question(st.session_state.question_index)
                    st.session_state.user_answers = []
                    st.session_state.show_recommendation = False
                    st.session_state.recommendations = []
                    dataset = [{"segment": 11, "track_id": segment_11.sample(1)["track_id"].values[0]},
                               {"segment": 12, "track_id": segment_12.sample(1)["track_id"].values[0]},
                               {"segment": 13, "track_id": segment_13.sample(1)["track_id"].values[0]},
                               {"segment": 21, "track_id": segment_21.sample(1)["track_id"].values[0]},
                               {"segment": 22, "track_id": segment_22.sample(1)["track_id"].values[0]},
                               {"segment": 23, "track_id": segment_23.sample(1)["track_id"].values[0]},
                               {"segment": 31, "track_id": segment_31.sample(1)["track_id"].values[0]},
                               {"segment": 32, "track_id": segment_32.sample(1)["track_id"].values[0]},
                               {"segment": 33, "track_id": segment_33.sample(1)["track_id"].values[0]}]
                    st.session_state.segment_selector = SegmentSelector(dataset)
                    st.rerun()