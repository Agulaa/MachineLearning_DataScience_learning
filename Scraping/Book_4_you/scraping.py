import urllib.request
from bs4 import  BeautifulSoup
import re
import smtplib
from email.message import EmailMessage


def get_response(url):
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    return response

def get_soup(response):
    soup = BeautifulSoup(response, 'html.parser')
    return soup

def get_title(soup):
    div_title = soup.find_all("div", class_="dotd-title")
    title = div_title[0].get_text()
    title = title.replace('\t', '').replace('\n', '').lower()
    return title


def chceck_title(title):
    pattern = 'machine learning|data science|python|r|data analysis|analysis'
    if re.search(pattern, title):
        print(title)
        send_email()


def send_email():
    sent_from = 'mail@gmail.com'
    password = 'password'
    send_to = 'mail@gmail.com'
    message = "Download book from page: https://www.packtpub.com/packt/offers/free-learning"
    msg = EmailMessage()
    msg['Subject'] = 'Free books'
    msg['From'] = sent_from
    msg['To'] = send_to
    msg.set_content(message)
    try:
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.ehlo()
        server.starttls()
        server.login(sent_from,password)
        server.send_message(msg)
        print('send mail')
        server.quit()
    except:
        print('something wrong')

def do():
    url = 'https://www.packtpub.com/packt/offers/free-learning'
    response = get_response(url)
    soup = get_soup(response)
    title = get_title(soup)
    chceck_title(title)
