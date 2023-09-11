import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time
from tqdm import tqdm

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_driver(driver="chrome", headless=False, userdata_dir="selenium"):
    """
    Get the webdriver
    :param driver: driver name
    :param headless: whether to run in headless mode
    :param userdata_dir: user data dir
    :return:
    """
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


def get_neurips_pages(driver=None, year=2022):
    """
    Get the openreview pages
    :param driver: webdriver
    :param year: NeurIPS year
    :return:
    """
    if driver is None:
        driver = get_driver(headless=False)
    main_lik = f"https://openreview.net/group?id=NeurIPS.cc/{year}/Conference"
    driver.get(main_lik)

    def remove_overlay():
        overlay = driver.find_elements(By.ID, 'flash-message-container')
        if len(overlay) > 0:
            overlay = overlay[0]
            driver.execute_script("""
                                var element = arguments[0];
                                element.parentNode.removeChild(element);
                                """, overlay)

    # for accepted papers
    page = 1
    while True:
        print(f"Collecting page {page}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "note")))
        out = cache_folder / 'pages'
        out.mkdir(exist_ok=True)
        with open(out / f"NeurIPS{year}_{page}.html", "w") as f:
            f.write(driver.page_source)
        # li class="  right-arrow"class="  right-arrow"
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        wait = WebDriverWait(driver, 10)
        next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[@class='  right-arrow']")))
        if 'disabled' in next_btn.get_attribute('class'):
            break
        a_link = next_btn.find_element(By.TAG_NAME, 'a')
        try:
            a_link.click()
            time.sleep(2)
        except Exception:
            print()
        page += 1

    # for rejected papers
    # todo - it has bugs (Do manually)
    # driver.get(main_lik + "#rejected-papers-opted-in-public")
    # time.sleep(2)
    # page = 1
    # while True:
    #     print(f"Collecting page {page}")
    #     WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "note")))
    #     out = cache_folder / 'pages'
    #     out.mkdir(exist_ok=True)
    #     with open(out / f"NeurIPS{year}_{page}_reject.html", "w") as f:
    #         f.write(driver.page_source)
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #     time.sleep(1)
    #     wait = WebDriverWait(driver, 3)
    #     # it has bugs
    #     # next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[@class='  right-arrow']/a")))
    #     # if 'disabled' in next_btn.get_attribute('class'):
    #     #     break
    #     # next_btn.click()
    #     # time.sleep(2)
    #     page += 1


def get_paper(link, out_file, wait_class='note_content_field', driver=None):
    """
    Get the paper page
    :param link: link of OpenReview paper
    :param out_file: output file
    :param wait_class: html class to wait for
    :param driver: webdriver
    :return:
    """
    if driver is None:
        driver = get_driver(headless=False)
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


def extract_paper_info(paper, is_accepted=True, driver=None):
    """
    Extract paper info
    :param paper:
    :param is_accepted:
    :param driver:
    :return:
    """
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
        'pub_url': 'https://openreview.net/forum?id={}'.format(paper_id),
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
        get_paper(link, cache_html_path, driver=driver)

    with open(cache_html_path, 'r') as f:
        html_text = f.read()
    paper_soup = BeautifulSoup(html_text, 'html.parser')

    span = [s for s in paper_soup.find_all('span', {'class': 'note_content_field'}) if
            s.text == 'Metareview: ']
    meta_review = span[0].next_sibling.text.strip() if len(span) > 0 else None
    confidence = span[0].parent.previous_sibling.text
    recommendation = span[0].parent.previous_sibling.previous_sibling.text
    paper_res[
        'meta_review'] = recommendation + "\n" + confidence + "\n" + meta_review if meta_review else None

    # div with class "note panel" whose first child with text "Official Review of Paper"

    reviewers = [s for s in paper_soup.find_all('div', {'class': 'note panel'}) if
                 s.text.strip().startswith(
                     "Official Review of Paper")]
    reviewers_parsed = [[{x.text.replace(': ', '').strip(): x.next_sibling.text.strip()} for x in
                         r.find_all('span', {'class': 'note_content_field'})] for r in reviewers]
    reviewers_parsed = [{k: v for d in r for k, v in d.items()} for r in reviewers_parsed]
    reviewers_text = []
    for idx_r, r in enumerate(reviewers):
        review = '\n'.join([rr.text.strip() for rr in r.find_all('div', {'class': 'note_contents'}) if
                            rr.text.strip() != 'Official Review of Paper'])
        review = 'Reviewer {}: \n'.format(idx_r + 1) + review + '\n\n'
        reviewers_text.append(review)
    paper_res['reviews'] = reviewers_text
    paper_res['reviews_parsed'] = reviewers_parsed
    paper_res['rating_avg'] = round(
        sum([float(x['Rating'].split(':')[0]) for x in reviewers_parsed]) / len(reviewers_parsed),
        3) if len(
        reviewers_parsed) > 0 else None
    paper_res['confidence_avg'] = round(
        sum([float(x['Confidence'].split(':')[0]) for x in reviewers_parsed]) / len(reviewers_parsed),
        3) if len(reviewers_parsed) > 0 else None
    paper_res['soundness_avg'] = round(
        sum([float(x['Soundness'].split(':')[0].split(' ')[0]) for x in reviewers_parsed]) / len(
            reviewers_parsed), 3) if len(reviewers_parsed) > 0 else None
    paper_res['presentation_avg'] = round(
        sum([float(x['Presentation'].split(':')[0].split(' ')[0]) for x in reviewers_parsed]) / len(
            reviewers_parsed), 3) if len(reviewers_parsed) > 0 else None
    paper_res['contribution_avg'] = round(
        sum([float(x['Contribution'].split(':')[0].split(' ')[0]) for x in reviewers_parsed]) / len(
            reviewers_parsed), 3) if len(reviewers_parsed) > 0 else None
    return {title: paper_res}


def extract_neurips_main_pages():
    """
    Extract reviews information to raw.json
    :return:
    """
    driver = get_driver(headless=False)

    pages = list((cache_folder / 'pages').glob("NeurIPS*.html"))
    pages_accepted = [p for p in pages if 'reject' not in p.name]
    pages_rejected = [p for p in pages if 'reject' in p.name]

    res = {}

    def process_pages(pages_set, is_accepted=True):
        for page in tqdm(pages_set, total=len(pages_set)):
            with open(page, 'r') as f:
                html_text = f.read()
            soup = BeautifulSoup(html_text, 'html.parser')

            accepted_papers = [a for a in soup.find('div', {'id': 'accepted-papers'}).find_all('a', href=True) if
                               'forum?id=' in a['href']]
            rejected_papers = [a for a in soup.find_all('a', href=True) if 'forum?id=' in a['href'] if
                               a not in accepted_papers]

            papers = accepted_papers if is_accepted else rejected_papers

            for paper in tqdm(papers, total=len(papers)):
                paper_res = extract_paper_info(paper, is_accepted=is_accepted, driver=driver)
                res.update(paper_res)

    process_pages(pages_accepted, is_accepted=True)
    process_pages(pages_rejected, is_accepted=False)

    with open(cache_folder / "raw.json", 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)
    get_neurips_pages()
    extract_neurips_main_pages()
