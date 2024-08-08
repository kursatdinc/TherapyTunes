# <img src="https://github.com/kursatdinc/TherapyTunes/blob/main/images/theraphytunes_logo.jpeg" alt="TherapyTunes logo" width="100" height="100"> TherapyTunes Project
# Therapy Tunes

Therapy Tunes is an innovative project exploring the intersection of music and mental health. Developed collaboratively by three colleagues, this platform offers personalized song recommendations aimed at enhancing users' emotional well-being.

## Project Objective

TherapyTunes approaches recommendation systems from a new perspective, follows a different and more personal path than the systems used so far, and uses the healing power of music when making suggestions to users.

As it is known, people have a hard time evaluating their mental health objectively and are unable to do daily activities that will be good for them. One of the most important of these is listening to music that will be good for our soul.

This is exactly where we step in and detect the emotional states of the users according to their music listening habits and create our first stage recommendation pool based on the emotional states we detect.

Then, in order to capture daily emotional changes, we also process daily horoscope comments and narrow down our suggestion pool one step further.

Finally, in order to get an objective result, we apply users to a mini song selection test to determine their music preferences and apply the final narrowing process to our recommendation pool. In this way, we have access to the most optimum recommendation pool possible and we make suggestions to users from this pool.

In short, TherapyTunes stands out as the most personalized and accurate recommendation system among the recommendation systems developed so far.

## Operational Methodology

1. **Data Collection**: Gather user information to assess anxiety, depression, and insomnia levels, alongside musical preferences.

2. **BPM Determination**: Calculate an ideal BPM (Beats Per Minute) range based on collected data.

3. **Data Analysis**: Analyze an extensive song database using determined BPM and other parameters, employing clustering and PCA (Principal Component Analysis) for song categorization.

4. **Song Recommendation**: Identify the most suitable song segment based on the user's profile and recommend songs accordingly.

5. **Astrological Integration**: Incorporate current horoscope interpretations, extracted via web scraping, to enhance song recommendations.

## Technologies Employed

- Python
- Machine Learning Models:
  - XGBoost
  - AdaBoost
  - SVC
  - RandomForestClassifier
  - LightGBM
- Machine Learning Algorithms:
  - Clustering
  - PCA
- Web Scraping: Beautiful Soup
- Data Analysis and Visualization Tools

## Future Developments

- [ ] Integrate user feedback for continuous improvement of the recommendation system
- [ ] Develop a mobile application
- [ ] Expand the music database
- [ ] Collaborate with mental health professionals to enhance platform efficacy

## Streamlit Link

https://therapytunes.streamlit.app/

## Contact Information

For inquiries or suggestions, please reach out to us via our LinkedIn profiles:

- [Büşra Sürücü](https://www.linkedin.com/in/busrasurucu/)
- [Hilal Alpak](https://www.linkedin.com/in/hilal-alpak/)
- [Kürşat Dinç](https://www.linkedin.com/in/kursatdinc/)
