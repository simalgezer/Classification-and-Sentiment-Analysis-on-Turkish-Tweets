from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time
import re

SCROLL_PAUSE_TIME = 2

### ELEMENTS' XPATHS ###
CHROME_DRIVER_PATH = '/Users/ardaakdere/Desktop/PythonEnvs/twitter_tweet/chromedriver'
SEARCH_INPUT_XPATH = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/label/div[2]/div/input'
TWITTER_PROFILE_XPATH = '//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[2]/div/div/div/div/div[2]/div/div[2]/div/div/div[4]/div'

### TWITTER URLS ###
TWITTER_LOGIN_URL = 'https://twitter.com/login'
TWITTER_HOME_URL = 'https://twitter.com/home'

print("""
___________ __ _________________________     _________________________________________________
7      77  V  V  77     77     77      7     7     77     77  _  77  _  77     77     77  _  7
!__  __!|  |  |  ||  ___!|  ___!!__  __!     |  ___!|  ___!|    _||  _  ||  -  ||  ___!|    _|
  7  7  |  !  !  ||  __|_|  __|_  7  7       !__   7|  7___|  _ \ |  7  ||  ___!|  __|_|  _ \ 
  |  |  |        ||     7|     7  |  |       7     ||     7|  7  ||  |  ||  7   |     7|  7  |
  !__!  !________!!_____!!_____!  !__!       !_____!!_____!!__!__!!__!__!!__!   !_____!!__!__!                                      
""")

class TwitterScraper:

  def __init__(self, count) -> None:
      self.tweets = []
      self.hashtag = ''
      self.tweet_count = count

      self.last_height = 0
      self.new_height = 0

      self.pass_counter = 0

  def get_driver(self):
    driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH)
    driver.implicitly_wait(8)
    driver.get(TWITTER_LOGIN_URL)
    return driver

  def wait_for_login(self, driver):
    input('Twitter\'a giriş yaptıktan sonra bir tuşa basınız:')
    driver.get(TWITTER_HOME_URL)

  def check_islogin(self, driver):
    try:
      profile = driver.find_element_by_xpath(TWITTER_PROFILE_XPATH)
    except NoSuchElementException:
      return False
    if profile:
      return True
    
  def prepare_hashtag(self, hashtag):
    hashtag = hashtag.strip()
    hashtag = hashtag.replace('#', '')
    hashtag = hashtag.lower()
    return hashtag

  def search_hashtag(self, driver, hashtag):
    driver.get(f'https://twitter.com/search?q=%23'+hashtag+f'%20lang%3Atr&src=typed_query')
  
  def collect_tweets(self, driver):
    counter = 0
    while True:
      counter += 1
      try:
        tweet = driver.find_element_by_xpath(f'/html/body/div[1]/div/div/div[2]/main/div/div/div/div/div/div[2]/div/section/div/div/div[{counter}]/div/div/article/div/div/div/div[2]/div[2]/div[2]/div[1]').text
        self.save_tweet(tweet)
      except:
        print('Passed')
      
      if counter % 2 == 0:
        self.scroll_page(driver)

      if counter == 8:
        counter = 0
      
      if len(self.tweets) == self.tweet_count or self.pass_counter > 15:
        break
  
  def prepare_tweet(self, tweet):
    tweet = tweet.strip()
    tweet = tweet.lower()
    #tweet = tweet.replace('\n', ' ')
    tweet = re.sub(r'\n+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet

  def save_tweet(self, tweet):
    tweet = self.prepare_tweet(tweet)
    if tweet in self.tweets:
        self.pass_counter += 1
        print('pass counter + 1')
    else:
        self.tweets.append(tweet)
        print(f'{len(self.tweets)} *Saved', end="-")
  
  def scroll_page(self, driver):

    # Get scroll height
    self.last_height = driver.execute_script("return document.body.scrollHeight")

    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    self.new_height = driver.execute_script("return document.body.scrollHeight")
    if self.new_height == self.last_height:
        print('Sayfanın Sonu')
        time.sleep(2)
    self.last_height = self.new_height
  
  def save_to_csv(self):
    df = pd.DataFrame({'tweet': self.tweets})
    df.to_csv(self.hashtag+'.csv')

  def run(self):
    driver = self.get_driver()
    self.wait_for_login(driver)
    if self.check_islogin(driver):
      for hashtag in hashtags_list:
        self.pass_counter = 0
        self.tweets = []
        self.hashtag = hashtag[1:]
        self.search_hashtag(driver, self.hashtag)
        self.collect_tweets(driver)
        self.save_to_csv()
        print(self.hashtag+' SAVED!')
      time.sleep(5)

with open('hashtags.txt', 'r') as hash_file:
  hashtags_list = hash_file.readlines()


ts = TwitterScraper(count=50)
ts.run()