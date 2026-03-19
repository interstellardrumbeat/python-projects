# Lyrics Scraper

A Python script that fetches song lyrics for a curated list of artists (currently Punk-Rock, hard-coded) using the [Genius API](https://genius.com/api-clients) and exports them to a CSV file (eventually for analysis of diffferent kind).

## Artists Included

| Artist | Genre |
|---|---|
| The Clash | Punk / Punk Rock |
| Sex Pistols | Punk / Punk Rock |
| Dead Kennedys | Punk / Hardcore Punk |
| Black Flag | Hardcore / Punk |
| Bad Religion | Punk / Punk Rock |
| NOFX | Punk / Punk Rock |
| Green Day | Punk Rock / Pop Punk |
| Rancid | Ska Punk / Punk / Punk Rock |
| Fugazi | Post-Hardcore |

## Overview

- Searches for artists by name using the Genius API
- Retrieves top songs sorted by popularity
- Scrapes and cleans lyrics from Genius
- Exports all data to a CSV file with artist, song title, release year, and lyrics

## Requirements

- Python 3.8+
- A [Genius API token](https://genius.com/api-clients)
- pandas
- BeautfulSoup4

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Open the script and replace the placeholder with your Genius API token:

```
api_token = 'INSERT GENIUS API TOKEN HERE'
```

## How to use

Run the script from the command line:

```
python lyrics-scraper.py
```

You will be prompted to enter:
1. The number of songs per artist you want to collect
2. The name of the output CSV file (e.g. `lyrics.csv`)

### Example output (console)

```
Finding artist: The Clash
Collected top 10 songs for The Clash.
Scraping lyrics for: London Calling
Scraping lyrics for: Should I Stay or Should I Go
...
Saved data to "lyrics.csv" file.
```

### Output Format

The output CSV will contain the following data:

| Column | Type | Description |
|---|---|---|
| `artist` | str | Artist name |
| `song_title` | str | Song title |
| `release_year` | int | Year the song was released |
| `lyrics` | str | Scraped and cleaned lyrics |

## Roadmap

- [ ] Better Lyrics cleaning - Right now the final Lyrics may still contain some non-lyric metadata
- [ ] Not hard-coded artist list - Editing it requires modifying the script directly
