from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json

from config import Config

def setup_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    return driver

def extract_issue_data(driver, issue_number):
    url = f"https://www.deeplearning.ai/the-batch/issue-{issue_number}/"
    print(url)
    driver.get(url)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    main_content = soup.find("div", class_="prose--styled")

    if not main_content:
        time.sleep(5)

    if not main_content:
        print(f"❌ No main content found for issue {issue_number}")
        return None

    news_anchor = main_content.find(lambda tag: tag.name in ["h1", "h2"] and tag.get("id") == "news")

    if not news_anchor:
        print(f"⚠️ No 'news' anchor found in issue {issue_number}, get all")
        elements_to_process = main_content
    else:
        elements_to_process = list(news_anchor.next_siblings)

    articles_data = []
    current_article = {
        "issue": issue_number,
        "url": url,
        "title": None,
        "text": "",
        "images": [],
        "captions": []
    }

    for element in elements_to_process:
        if isinstance(element, str):
            continue

        if element.name in ["h1", "h2", "h3"]:
            current_article["title"] = element.get_text(strip=True)

        elif element.name == "p":
            text = element.get_text(strip=True)
            if text:
                current_article["text"] += text + "\n"

        elif element.name == "ul":
            for li in element.find_all("li"):
                li_text = li.get_text(strip=True)
                if li_text:
                    current_article["text"] += "• " + li_text + "\n"

        elif element.name == "figure":
            img_tag = element.find("img")
            if img_tag:
                img_src = img_tag.get("src")
                if img_src:
                    current_article["images"].append(img_src)

            caption_tag = element.find("figcaption")
            if caption_tag:
                caption = caption_tag.get_text(strip=True)
            else:
                # fallback — сусідній <p>
                next_p = element.find_next_sibling("p")
                caption = next_p.get_text(strip=True) if next_p else ""

            current_article["captions"].append(caption)

        elif element.name == "hr":
            if current_article["title"] is not None or current_article["text"] is not None:
                if 'a message from' not in current_article["title"].lower() or 'new from' not in current_article["title"].lower():
                    articles_data.append(current_article)
                current_article = {
                    "issue": issue_number,
                    "url": url,
                    "title": None,
                    "text": "",
                    "images": [],
                    "captions": []
                }

    if current_article["title"] or current_article["text"]:
        if 'a message from' not in current_article["title"].lower() or 'new from' not in current_article[
            "title"].lower():
            articles_data.append(current_article)

    return articles_data

def scrape_all_issues(start=1, end=239, headless=True):
    driver = setup_driver(headless)
    all_issues = []

    for issue_number in range(start, end + 1, 1):
        try:
            print(f"Scraping Issue #{issue_number}")
            data = extract_issue_data(driver, issue_number)
            all_issues.extend(data)
        except Exception as e:
            print(f"Failed to scrape issue {issue_number}: {e}")
            continue

    driver.quit()
    return all_issues


if __name__ == "__main__":
    results = scrape_all_issues(start=1, end=298)

    with open(f"../{Config.parsed_full_data_path}", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Scraping complete. Data saved to 'the_batch_articles.json'")
