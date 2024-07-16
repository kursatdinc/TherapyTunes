import requests
from bs4 import BeautifulSoup

sign_list = ["aries", "taurus", "gemini", "cancer", "leo", "virgo","libra",
             "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]

def get_star_ratings(sign, date='today'):

    url = f"https://www.horoscope.com/star-ratings/{date}/{sign}"
    
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    rating_tags = soup.find_all('h3')
    
    desired_categories = ['sex', 'hustle', 'vibe', 'success']
    
    ratings = {}
    for tag in rating_tags:
        category = tag.text.split()[0].lower()  # İlk kelimeyi al (Sex, Hustle, Vibe, Success)
        if category in desired_categories:
            highlighted_stars = tag.find_all('i', class_='icon-star-filled highlight')
            stars = len(highlighted_stars)
            ratings[category] = f"{stars}/5"
    
    return ratings


for sign in sign_list:
    print(sign, get_star_ratings(sign))