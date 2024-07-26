import requests
from bs4 import BeautifulSoup

sign_list = ["aries", "taurus", "gemini", "cancer", "leo", "virgo","libra",
             "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]

def get_star_ratings(sign, date="today"):

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
            
            # sex_star = int(ratings.get("sex"))
            hustle_star =ratings.get("hustle")
            vibe_star = ratings.get("vibe")
            # success_star = int(rating.get("success"))
    
    return hustle_star, vibe_star