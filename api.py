import waybackpy
import datetime as datetime

# Initialize the API
url = "https://news.mingpao.com/"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Get all available snapshots for 2018-2024 (This will take a long time)
# Note: This only gets metadata, not the full HTML content.
archive = waybackpy.Url(url, user_agent)

# Example: Get closest snapshot to a specific date
target_date = datetime.datetime(2020, 1, 15)
closest_snapshot = archive.closest(target_date)

if closest_snapshot:
    print(f"Found snapshot: {closest_snapshot.archive_url}")
    # You can then download the HTML from this URL
else:
    print("No snapshot found for this date.")