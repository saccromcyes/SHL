import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm
import pandas as pd
import re

BASE_URL = "https://www.shl.com"
CATALOG_URL_TEMPLATE = "https://www.shl.com/products/product-catalog/?start={}&type=1&type=1"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_html(url):
    """Fetch HTML content of a URL with a short delay and retry."""
    time.sleep(1)
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        time.sleep(2)
        res = requests.get(url, headers=HEADERS)
    return res.text

def parse_catalog_page(html):
    """Parse catalog page and extract basic assessment info."""
    soup = BeautifulSoup(html, 'html.parser')
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.select('tr[data-entity-id]')
    data = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        name_tag = cols[0].find('a')
        name = name_tag.text.strip()
        relative_url = name_tag['href']
        url = BASE_URL + relative_url

        remote = "Yes" if cols[1].find('span', class_='-yes') else "No"
        adaptive = "Yes" if cols[2].find('span', class_='-yes') else "No"
        test_types = ','.join([span.text.strip() for span in cols[3].find_all('span')])

        data.append({
            "Assessment Name": name,
            "URL": url,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive,
            "Test Type(s)": test_types
        })

    return data

def parse_detail_page(url):
    """Visit the detail page twice and extract description, job levels, and duration."""
    # First visit
    fetch_html(url)
    time.sleep(1)  # Wait before the second visit
    html = fetch_html(url)

    soup = BeautifulSoup(html, 'html.parser')
    blocks = soup.select('.product-catalogue-training-calendar__row')

    details = {"Description": "", "Job levels": "", "Duration": ""}
    for block in blocks:
        h4 = block.find("h4")
        p = block.find("p")
        if not h4 or not p:
            continue

        title = h4.text.strip()
        content = p.text.strip()

        if title == "Description":
            details["Description"] = content
        elif title == "Job levels":
            details["Job levels"] = content
        elif title == "Assessment length":
            details["Duration"] = content.replace("Approximate Completion Time in minutes =", "").strip() + " mins"

    return details

def clean_text(text):
    """Remove regulatory disclaimer and collapse whitespace."""
    if not isinstance(text, str):
        return text
    text = re.sub(r'Your use of this assessment.*?shl-us-regulatory-compliance/?', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    start = 0
    all_data = []

    while True:
        catalog_url = CATALOG_URL_TEMPLATE.format(start)
        html = fetch_html(catalog_url)
        page_data = parse_catalog_page(html)

        if not page_data:  # No more rows found, stop
            break

        for item in tqdm(page_data, desc=f"Detail pages for start={start}", leave=False):
            detail_info = parse_detail_page(item["URL"])
            item.update(detail_info)
            all_data.append(item)

        start += 12  # Move to next page

    # Clean Data and Write to CSV
    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Rename columns for consistency
    df.columns = [
        'name', 'url', 'remote_support', 'adaptive_support',
        'test_type', 'description', 'roles', 'duration'
    ]

    # Clean descriptions
    df['description'] = df['description'].apply(clean_text)

    # Save cleaned data
    df.to_csv('try_shl_assessments.csv', index=False)
  


    print("✅ Scraping completed. Data saved to shl_assessments.csv.")

if __name__ == "__main__":
    main()
