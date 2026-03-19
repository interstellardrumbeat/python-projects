import requests
from bs4 import BeautifulSoup
import time
import random
import re
import pandas as pd

api_token = 'INSERT GENIUS API TOKEN HERE'
genius_url = 'https://api.genius.com'
headers = {'Authorization': f'Bearer {api_token}'}


def search_artist(artist_name):
    """
    Search for an artist on Genius and return their ID.

    Match results against the provided artist name (case-insensitive).
    Return the ID of the artist whose name is an exact match.

    Parameters
    ----------
    artist_name : str
        The full name of the artist to search for.

    Returns
    -------
    int or None
        The artist ID if a match is found,
        otherwise None.

    Raises
    ------
    requests.HTTPError
        If the API request fails (e.g. invalid token, network error).
    """
    search_url = f"{genius_url}/search"
    params = {'q': artist_name}
    response_artist = requests.get(search_url, headers=headers, params=params)
    response_artist.raise_for_status()
    search_artist_data = response_artist.json()
    for hit in search_artist_data['response']['hits']:
        primary_artist = hit["result"]["primary_artist"]
        if primary_artist['name'].lower().strip() == artist_name.lower():
            return primary_artist['id']
    return None


def get_artist_songs(artist_id, max_songs=20):
    """
    Retrieve a list of the artist's songs from Genius.

    Iterate through the song catalogue of the artist, sorted by popularity.
    Collect metadata until the target count is met or the catalogue ends.
    A random delay between requests is included to avoid rate-limiting.

    Parameters
    ----------
    artist_id : int
        The Genius artist ID, as returned by `search_artist`.
    max_songs : int, optional
        Maximum number of songs to retrieve. Default is 20.

    Returns
    -------
    list of dict
        A list of song metadata dictionaries, containing:

        - ``title`` : str - Song title.
        - ``id`` : int - Genius song ID.
        - ``url`` : str - URL to the song's Genius page.
        - ``release_year`` : int or None - Year the song was released.

    Raises
    ------
    requests.HTTPError
        If any API request fails.
    """
    songs, page = [], 1
    while len(songs) < max_songs:
        songs_url = f"{genius_url}/artists/{artist_id}/songs"
        params = {'page': page, 'per_page': 50, 'sort': 'popularity'}
        response_songs = requests.get(songs_url, headers=headers, params=params)
        response_songs.raise_for_status()
        artist_songs_data = response_songs.json()
        songs_on_page = artist_songs_data['response']['songs']
        if not songs_on_page:
            break
        for song in songs_on_page:
            songs.append({
                'title': song['title'],
                'id': song['id'],
                'url': song['url'],
                'release_year': song['release_date_components']['year'],
            })
            if len(songs) >= max_songs:
                break
        page += 1
        time.sleep(random.uniform(1.0, 3.0))  # Limiting API requests
    return songs


# To be fixed, for cleaner lyrics (right now it contains also song info, unnecessary)
def scrape_lyrics(song_url):
    """
    Scrape and clean the lyrics from the song page.

    Fetch the HTML of the given song URL, extract all
    lyrics containers, strip labels (e.g. ``[Chorus]``)
    and extra blank lines.

    .. note::
        The output may still contain some non-lyric metadata.
        Further cleaning may be required.

    Parameters
    ----------
    song_url : str
        The full URL to the Genius song page.

    Returns
    -------
    str
        The cleaned lyrics as a single string,
        separated by newline characters.
    """
    page = requests.get(song_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    lyrics_divs = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
    lyrics = '\n'.join([div.get_text(separator='\n').strip() for div in lyrics_divs])
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove strings in square brackets, eg. [Chorus]
    lyrics = re.sub(r'\n{2,}', '\n', lyrics).strip()  # Removes double new line with only one new line
    return lyrics


def scraper_to_csv():
    """
    The full scraping pipeline. Save results to a CSV file.

    Prompt the user for the number of songs per artist and the desired
    output filename. Iterate over a list of artists (hard coded),
    fetches their top songs via the Genius API, scrape the lyrics for
    each song, and write all collected data to a CSV file using pandas.
    Songs that fail during scraping are skipped with a printed warning.
    Include random delays between requests to respect rate limits.

    .. note::
        This function performs console I/O via ``input()`` and ``print()``.
        The list of artists is hardcoded and includes: The Clash, Sex Pistols,
        Dead Kennedys, Black Flag, Bad Religion, NOFX, Green Day, Rancid,
        and Fugazi.

    Returns
    -------
    str
        The filename of the saved CSV file, intended for use eventually in a
        different pipeline.

    The output CSV contains the following columns:

    - ``artist`` : str - Artist name.
    - ``song_title`` : str - Song title.
    - ``release_year`` : int or None - Year of release.
    - ``lyrics`` : str - Scraped and cleaned lyrics.
    """
    artists = [
        "The Clash",
        "Sex Pistols",
        "Dead Kennedys",
        "Black Flag",
        "Bad Religion",
        "NOFX",
        "Green Day",
        "Rancid",
        "Fugazi"
    ]
    
    # Calls all previous functions and records data on csv file
    data_in_dict = []
    number_of_songs = input("How many songs per artist you want to store in the final data file?\n > ").strip()
    for artist in artists:
        print(f"\nFinding artist: {artist}")
        artist_id = search_artist(artist)
        if not artist_id:
            print(f"Artist {artist} not found.")
            continue
        songs = get_artist_songs(artist_id, max_songs=int(number_of_songs))
        print(f"Collected top {len(songs)} songs for {artist}.")
        for song in songs:
            try:
                print(f"Scraping lyrics for: {song['title']}")
                lyrics = scrape_lyrics(song['url'])
                # Appends to keep in-memory and only do one I/O process at the end
                data_in_dict.append({
                    'artist': artist,
                    'song_title': song['title'],
                    'release_year': song['release_year'],
                    'lyrics': lyrics
                })
                time.sleep(random.uniform(1.0, 3.0))  # Limiting API requests
            except Exception as e:
                print(f"Error scraping {song['title']}: {e}")
    df = pd.DataFrame(data_in_dict)
    output_name = input("\nHow would you like to name the output csv file? Write also the format, eg .csv\n > ").strip()
    df.to_csv(output_name, index=False)
    print(f"\nSaved data to \"{output_name}\" file.\n")
    return output_name  # Eventually, to use it in a pipeline


if __name__ == '__main__':
    scraper_to_csv()
