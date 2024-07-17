import requests
from bs4 import BeautifulSoup

sign_list = ["aries", "taurus", "gemini", "cancer", "leo", "virgo","libra",
             "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]

def get_star_ratings(sign, date='today'):

    url = f"https://www.horoscope.com/star-ratings/{date}/{sign}"
    
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    rating_tags = soup.find_all("h3")
    
    ratings = {}
    for tag in rating_tags:
        category = tag.text.split()[0].lower()
        if category in ["sex", "hustle", "vibe", "success"]:
            highlighted_stars = tag.find_all("i", class_="icon-star-filled highlight")
            stars = len(highlighted_stars)
            # ratings[category] = f"{stars}/5"
            ratings[category] = f"{stars}"
    
    return ratings

rating  = get_star_ratings("gemini")

int(rating.get("sex"))
int(rating.get("hustle"))
int(rating.get("vibe"))
int(rating.get("success"))

for sign in sign_list:
    print(sign, get_star_ratings(sign))