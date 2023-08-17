import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


def get_driver(driver="chrome", headless=False, userdata_dir="selenium", cookie_dir=""):
    if driver == 'firefox':
        inst = GeckoDriverManager(version='v0.31.0').install()
        options = webdriver.FirefoxOptions()
        options.accept_insecure_certs = True

        if headless:
            options.headless = True
            options.add_argument("--use-gl=angle")
            options.add_argument("--width=2560")
            options.add_argument("--height=1440")

        profile = webdriver.FirefoxProfile(
            '/Users/sean/Library/Application Support/Firefox/Profiles/rvdb4tpz.default-release-1681846257167')

        driver = webdriver.Firefox(executable_path=inst, options=options, firefox_profile=profile)

        # if cookie_dir:
        #
        #     print("Loaded {} cookies".format(cookie_cnt))
    else:
        option = webdriver.ChromeOptions()
        option.add_argument("user-data-dir=/tmp/{}".format(userdata_dir))
        # option.add_argument("user-data-dir=/Users/sean/Library/Application Support/Google/Chrome/Default")
        option.add_argument('disable-infobars')

        if headless:
            option.add_argument('--headless')
            option.add_argument('--no-sandbox')
            option.add_argument('--disable-gpu')
            option.add_argument('--disable-dev-shm-usage')
            option.add_argument('--window-size=2560,1440')

        inst = ChromeDriverManager().install()
        driver = webdriver.Chrome(options=option)
    return driver


def get_neurips_main_html(driver=None, year=2022):
    if driver is None:
        driver = get_driver(headless=False)
    main_lik = f"https://openreview.net/group?id=NeurIPS.cc/{year}/Conference"
    driver.get(main_lik)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "note")))
    with open(cache_folder / f"NeurIPS{year}.html", "w") as f:
        f.write(driver.page_source)


def save_html(link, out_file, wait_class='note_content_field'):
    driver.get(link)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, wait_class)))
    with open(out_file, "w") as f:
        f.write(driver.page_source)


def retry_request(url, max_retries=5):
    for i in range(max_retries):
        try:
            return requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        except Exception as e:
            print(e)
    return None


def extract_neurips_main_html():
    with open(cache_folder / "NeurIPS2022.html", 'r') as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, 'html.parser')

    accepted_papers = [a for a in soup.find('div', {'id': 'accepted-papers'}).find_all('a', href=True) if
                       'forum?id=' in a['href']]
    rejected_papers = [a for a in soup.find_all('a', href=True) if 'forum?id=' in a['href'] if a not in accepted_papers]

    res = {}

    def extract_paper_info(papers, is_accepted=True):
        for paper in papers:

            link = "https://openreview.net/" + paper['href']
            pdf_link = link.replace("forum?id=", "pdf?id=")
            paper_id = paper['href'].split("=")[-1]
            title = paper.text.strip().replace(" ", "_").replace("/", "_").replace("?", "_").replace('"', "_")
            cache_path = Path("cache/{}".format('accepted' if is_accepted else 'rejected'))
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_html_path = cache_path / "{}.html".format(title)
            cache_pdf_path = cache_path / "{}.pdf".format(title)

            paper_res = {
                'link': link,
                'pdf_link': pdf_link,
                'paper_id': paper_id,
                'title': title,
                'is_accepted': is_accepted
            }

            if not cache_pdf_path.exists():
                pdf_response = retry_request(pdf_link)
                if pdf_response is None:
                    print("Error: Cannot download pdf")
                else:
                    with open(cache_pdf_path, "wb") as f:
                        f.write(pdf_response.content)

            if not cache_html_path.exists():
                save_html(link, cache_html_path)

            with open(cache_html_path, 'r') as f:
                html_text = f.read()
            paper_soup = BeautifulSoup(html_text, 'html.parser')

            span = [s for s in paper_soup.find_all('span', {'class': 'note_content_field'}) if s.text == 'Metareview: ']
            meta_review = span[0].next_sibling.text.strip() if len(span) > 0 else None
            confidence = span[0].parent.previous_sibling.text
            recommendation = span[0].parent.previous_sibling.previous_sibling.text
            paper_res['meta_review'] = recommendation + "\n" + confidence + "\n" + meta_review if meta_review else None

            # div with class "note panel" whose first child with text "Official Review of Paper"
            reviewers_res = []
            reviewers = [s for s in paper_soup.find_all('div', {'class': 'note panel'}) if s.text.strip().startswith(
                "Official Review of Paper")]
            # combine all text in div with class "note_contents"
            for r in reviewers:
                review = '\n'.join([rr.text.strip() for rr in r.find_all('div', {'class': 'note_contents'}) if rr.text.strip() != 'Official Review of Paper'])
                reviewers_res.append(review)
            paper_res['reviews'] = reviewers_res
            res[title] = paper_res

    extract_paper_info(accepted_papers, is_accepted=True)
    extract_paper_info(rejected_papers, is_accepted=False)
    #     save res
    with open(cache_folder / "_NeurIPS2022.json", 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    # driver = get_driver(headless=False)
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)
    # get_neurips_main_html()
    extract_neurips_main_html()
